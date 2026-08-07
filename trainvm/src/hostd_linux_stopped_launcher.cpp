#include "trainvm/hostd_linux_stopped_launcher.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <grp.h>
#include <linux/capability.h>
#include <linux/sched.h>
#include <poll.h>
#include <ranges>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace trainvm {
namespace {

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) : value_(value) {}
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  ~Descriptor() { reset(); }

  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;

  [[nodiscard]] int get() const { return value_; }
  [[nodiscard]] int release() { return std::exchange(value_, -1); }
  void reset() noexcept {
    if (value_ >= 0) (void)::close(value_);
    value_ = -1;
  }

 private:
  int value_;
};

[[noreturn]] void reject(std::string message) {
  throw LinuxStoppedLauncherError(std::move(message));
}

std::string system_error(std::string_view action) {
  return std::string(action) + ": " + std::strerror(errno);
}

bool canonical_absolute_path(std::string_view value) {
  if (value.empty() || value.size() > 4096U || value.front() != '/' ||
      value.back() == '/' || value.contains("//")) {
    return value == "/";
  }
  std::size_t begin = 1U;
  while (begin < value.size()) {
    const std::size_t end = value.find('/', begin);
    const std::string_view part = value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin);
    if (part.empty() || part == "." || part == "..") return false;
    if (end == std::string_view::npos) break;
    begin = end + 1U;
  }
  return true;
}

std::string read_bounded_file(int directory, const char* name,
                              std::size_t maximum) {
  Descriptor file(::openat(directory, name,
                           O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0) reject(system_error("could not open child proc evidence"));
  struct stat status {};
  if (::fstat(file.get(), &status) != 0 || !S_ISREG(status.st_mode)) {
    reject("child proc evidence is not a regular procfs file");
  }
  std::string result;
  result.reserve(maximum + 1U);
  std::array<char, 512U> buffer{};
  while (result.size() <= maximum) {
    const std::size_t wanted =
        std::min(buffer.size(), maximum + 1U - result.size());
    const ssize_t count = ::read(file.get(), buffer.data(), wanted);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) reject(system_error("could not read child proc evidence"));
    if (count == 0) break;
    result.append(buffer.data(), static_cast<std::size_t>(count));
  }
  if (result.empty() || result.size() > maximum) {
    reject("child proc evidence is empty or exceeds its bound");
  }
  return result;
}

struct ProcStatIdentity final {
  std::uint64_t starttime{};
  std::int32_t nice{};

  bool operator==(const ProcStatIdentity&) const = default;
};

ProcStatIdentity parse_stat_identity(std::string_view value) {
  const std::size_t close = value.rfind(") ");
  if (value.empty() || value.front() < '1' || close == std::string_view::npos) {
    reject("child proc stat has no canonical comm boundary");
  }
  value.remove_prefix(close + 2U);
  std::array<std::string_view, 20U> fields{};
  std::size_t count = 0U;
  while (!value.empty() && count < fields.size()) {
    while (!value.empty() && value.front() == ' ') value.remove_prefix(1U);
    if (value.empty()) break;
    const std::size_t separator = value.find(' ');
    fields[count++] = value.substr(0U, separator);
    if (separator == std::string_view::npos) break;
    value.remove_prefix(separator + 1U);
  }
  // stat field 3 (state) is fields[0]; starttime is field 22.
  if (count != fields.size() || fields[0].size() != 1U) {
    reject("child proc stat is truncated before starttime");
  }
  std::uint64_t starttime = 0U;
  const auto parsed = std::from_chars(fields[19U].data(),
                                      fields[19U].data() + fields[19U].size(),
                                      starttime);
  if (parsed.ec != std::errc{} ||
      parsed.ptr != fields[19U].data() + fields[19U].size() ||
      starttime == 0U) {
    reject("child proc starttime is malformed");
  }
  std::int32_t nice = 0;
  const auto nice_parsed = std::from_chars(
      fields[16U].data(), fields[16U].data() + fields[16U].size(), nice);
  if (nice_parsed.ec != std::errc{} ||
      nice_parsed.ptr != fields[16U].data() + fields[16U].size() ||
      nice < -20 || nice > 19) {
    reject("child proc nice level is malformed");
  }
  return {.starttime = starttime, .nice = nice};
}

std::uint64_t parse_starttime(std::string_view value) {
  return parse_stat_identity(value).starttime;
}

std::string parse_cgroup(std::string_view value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.remove_suffix(1U);
  }
  constexpr std::string_view prefix = "0::";
  if (!value.starts_with(prefix) || value.contains('\n') || value.contains('\r')) {
    reject("child has no singular unified cgroup-v2 membership");
  }
  const std::string_view path = value.substr(prefix.size());
  if (!canonical_absolute_path(path)) {
    reject("child unified cgroup path is not canonical");
  }
  return std::string(path);
}

bool pidfd_alive(int descriptor) {
  pollfd item{.fd = descriptor, .events = POLLIN, .revents = 0};
  const int status = ::poll(&item, 1U, 0);
  return status == 0 ||
         (status > 0 && (item.revents & (POLLIN | POLLHUP | POLLERR)) == 0);
}

bool install_inherited_worker_descriptors(std::optional<int> code_fd,
                                          int worker_bootstrap_fd) noexcept {
  if ((code_fd && *code_fd <= kLinuxWorkerBootstrapDescriptor) ||
      worker_bootstrap_fd <= kLinuxWorkerBootstrapDescriptor)
    return false;
  if (code_fd &&
      ::dup3(*code_fd, kLinuxWorkerCodeDescriptor, 0) < 0)
    return false;
  return ::dup3(worker_bootstrap_fd, kLinuxWorkerBootstrapDescriptor, 0) >= 0;
}

bool install_inherited_profiled_worker_descriptors(
    std::optional<int> code_fd, int worker_bootstrap_fd,
    int profiler_authority_fd, int target_executable_fd) noexcept {
  if ((code_fd && *code_fd <= kLinuxProfilerTargetExecutableDescriptor) ||
      worker_bootstrap_fd <= kLinuxProfilerTargetExecutableDescriptor ||
      profiler_authority_fd <= kLinuxProfilerTargetExecutableDescriptor ||
      target_executable_fd <= kLinuxProfilerTargetExecutableDescriptor) {
    return false;
  }
  if (code_fd && ::dup3(*code_fd, kLinuxWorkerCodeDescriptor, 0) < 0)
    return false;
  return ::dup3(worker_bootstrap_fd, kLinuxWorkerBootstrapDescriptor, 0) >= 0 &&
         ::dup3(profiler_authority_fd, kLinuxProfilerAuthorityDescriptor, 0) >=
             0 &&
         ::dup3(target_executable_fd,
                kLinuxProfilerTargetExecutableDescriptor, 0) >= 0;
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::string> compose_exec_arguments(
    const LinuxStoppedLaunchSpec& spec) {
  std::vector<std::string> result;
  const std::size_t profiler_arguments =
      spec.profiler ? spec.profiler->arguments.size() + 1U : 0U;
  result.reserve(1U + profiler_arguments + spec.arguments.size());
  if (spec.profiler) {
    result.push_back(spec.profiler->executable_name);
    result.insert(result.end(), spec.profiler->arguments.begin(),
                  spec.profiler->arguments.end());
    result.push_back("/proc/self/fd/" +
                     std::to_string(kLinuxProfilerTargetExecutableDescriptor));
  } else {
    result.push_back(spec.executable_name);
  }
  result.insert(result.end(), spec.arguments.begin(), spec.arguments.end());
  return result;
}

bool install_worker_credentials(
    const LinuxWorkerCredentialSpec& credentials) noexcept {
  if (credentials.uid == 0U || credentials.gid == 0U ||
      !credentials.no_new_privileges ||
      credentials.supplementary_gids.size() > kMaximumSupplementaryGroups ||
      ::prctl(PR_SET_NO_NEW_PRIVS, 1UL, 0UL, 0UL, 0UL) != 0 ||
      ::prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0UL, 0UL, 0UL) != 0) {
    return false;
  }
  // setgroups(2) needs CAP_SETGID. When the sealed set is empty the authority
  // is privileged and drops the groups outright; otherwise the sealed set names
  // the groups it already carries and the check below proves none were gained.
  if (credentials.supplementary_gids.empty() && ::setgroups(0U, nullptr) != 0) {
    return false;
  }
  // Same-value setresuid/setresgid are permitted unprivileged and still prove
  // the running identity is the one that was sealed.
  if (::setresgid(credentials.gid, credentials.gid, credentials.gid) != 0 ||
      ::setresuid(credentials.uid, credentials.uid, credentials.uid) != 0) {
    return false;
  }
  struct __user_cap_header_struct header {
    .version = _LINUX_CAPABILITY_VERSION_3, .pid = 0
  };
  std::array<struct __user_cap_data_struct, 2U> capabilities{};
  if (::syscall(SYS_capset, &header, capabilities.data()) != 0 ||
      ::prctl(PR_SET_DUMPABLE, 0UL, 0UL, 0UL, 0UL) != 0 ||
      ::geteuid() != credentials.uid || ::getuid() != credentials.uid ||
      ::getegid() != credentials.gid || ::getgid() != credentials.gid ||
      ::prctl(PR_GET_NO_NEW_PRIVS, 0UL, 0UL, 0UL, 0UL) != 1) {
    return false;
  }
  // Exactly the sealed set, in either direction. With an empty sealed set this
  // is the previous `getgroups(0, nullptr) != 0` check unchanged.
  std::array<gid_t, kMaximumSupplementaryGroups> observed{};
  const int observed_count =
      ::getgroups(static_cast<int>(observed.size()), observed.data());
  if (observed_count < 0 ||
      static_cast<std::size_t>(observed_count) !=
          credentials.supplementary_gids.size()) {
    return false;
  }
  std::sort(observed.begin(), observed.begin() + observed_count);
  for (int index = 0; index < observed_count; ++index) {
    const auto slot = static_cast<std::size_t>(index);
    if (observed[slot] == 0U || observed[slot] != credentials.supplementary_gids[slot])
      return false;
  }
  (void)::umask(0077);
  return true;
}

bool install_process_priority(std::optional<std::int32_t> nice) noexcept {
  if (!nice) return true;
  if (*nice < -20 || *nice > 19 ||
      ::setpriority(PRIO_PROCESS, 0, *nice) != 0) {
    return false;
  }
  errno = 0;
  const int observed = ::getpriority(PRIO_PROCESS, 0);
  return errno == 0 && observed == *nice;
}

std::optional<std::string_view> status_field(std::string_view status,
                                             std::string_view name) {
  const std::string prefix = std::string(name) + ":";
  std::size_t begin = 0U;
  std::optional<std::string_view> result;
  while (begin < status.size()) {
    const std::size_t end = status.find('\n', begin);
    std::string_view line = status.substr(
        begin, (end == std::string_view::npos ? status.size() : end) - begin);
    if (line.starts_with(prefix)) {
      if (result) return std::nullopt;
      line.remove_prefix(prefix.size());
      while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
        line.remove_prefix(1U);
      result = line;
    }
    if (end == std::string_view::npos) break;
    begin = end + 1U;
  }
  return result;
}

bool four_ids_equal(std::string_view field, std::uint64_t expected) {
  for (std::size_t index = 0U; index < 4U; ++index) {
    while (!field.empty() && (field.front() == ' ' || field.front() == '\t'))
      field.remove_prefix(1U);
    const std::size_t end = field.find_first_of(" \t");
    const std::string_view token = field.substr(0U, end);
    std::uint64_t value = 0U;
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
        value != expected) {
      return false;
    }
    field = end == std::string_view::npos ? std::string_view{}
                                          : field.substr(end);
  }
  while (!field.empty() && (field.front() == ' ' || field.front() == '\t'))
    field.remove_prefix(1U);
  return field.empty();
}

bool zero_hex(std::string_view value) {
  return !value.empty() &&
         std::ranges::all_of(value, [](char character) {
           return character == '0';
         });
}

// The worker's supplementary groups must equal the sealed set exactly — neither
// a subset nor a superset. gid 0 may never appear, whether or not the worker
// shares the authority identity.
bool supplementary_groups_equal(std::string_view field,
                                const std::vector<gid_t>& expected) {
  std::vector<gid_t> observed;
  observed.reserve(expected.size());
  while (!field.empty()) {
    while (!field.empty() && (field.front() == ' ' || field.front() == '\t'))
      field.remove_prefix(1U);
    if (field.empty()) break;
    const std::size_t end = field.find_first_of(" \t");
    const std::string_view token = field.substr(0U, end);
    std::uint64_t value = 0U;
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
        value == 0U ||
        value > static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max()) ||
        observed.size() >= kMaximumSupplementaryGroups) {
      return false;
    }
    observed.push_back(static_cast<gid_t>(value));
    field = end == std::string_view::npos ? std::string_view{}
                                          : field.substr(end);
  }
  std::ranges::sort(observed);
  return std::ranges::adjacent_find(observed) == observed.end() &&
         observed == expected;
}

bool worker_status_has_credentials(
    std::string_view status, const LinuxWorkerCredentialSpec& expected) {
  const auto uid = status_field(status, "Uid");
  const auto gid = status_field(status, "Gid");
  const auto groups = status_field(status, "Groups");
  const auto no_new_privileges = status_field(status, "NoNewPrivs");
  const auto effective = status_field(status, "CapEff");
  const auto permitted = status_field(status, "CapPrm");
  const auto inheritable = status_field(status, "CapInh");
  const auto ambient = status_field(status, "CapAmb");
  return expected.uid > 0U && expected.gid > 0U &&
         expected.no_new_privileges && uid && gid && groups &&
         no_new_privileges && effective && permitted && inheritable && ambient &&
         four_ids_equal(*uid, expected.uid) &&
         four_ids_equal(*gid, expected.gid) &&
         supplementary_groups_equal(*groups, expected.supplementary_gids) &&
         *no_new_privileges == "1" && zero_hex(*effective) &&
         zero_hex(*permitted) && zero_hex(*inheritable) && zero_hex(*ambient);
}

void require_descriptor_identity(const LinuxStoppedLaunchSpec& spec) {
  struct stat cgroup {};
  struct stat executable {};
  struct stat code {};
  struct stat bootstrap {};
  struct stat working_directory {};
  struct stat profiler_executable {};
  struct stat profiler_authority {};
  const int seals = ::fcntl(spec.executable_fd, F_GET_SEALS);
  const int code_seals = spec.code_fd ? ::fcntl(*spec.code_fd, F_GET_SEALS) : 0;
  const int bootstrap_seals = ::fcntl(spec.worker_bootstrap_fd, F_GET_SEALS);
  const int profiler_executable_seals = spec.profiler
      ? ::fcntl(spec.profiler->executable_fd, F_GET_SEALS)
      : 0;
  const int profiler_authority_seals = spec.profiler
      ? ::fcntl(spec.profiler->authority_fd, F_GET_SEALS)
      : 0;
  constexpr int required_seals =
      F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  const bool profiler_executable_valid = !spec.profiler ||
      (::fstat(spec.profiler->executable_fd, &profiler_executable) == 0 &&
       S_ISREG(profiler_executable.st_mode) &&
       (spec.profiler->execute_from_source
            ? static_cast<std::uint64_t>(profiler_executable.st_dev) ==
                      spec.profiler->source_device &&
                  static_cast<std::uint64_t>(profiler_executable.st_ino) ==
                      spec.profiler->source_inode &&
                  static_cast<std::uint64_t>(profiler_executable.st_size) ==
                      spec.profiler->source_size &&
                  static_cast<std::uint32_t>(profiler_executable.st_mode) ==
                      spec.profiler->source_mode &&
                  static_cast<std::uint32_t>(profiler_executable.st_uid) ==
                      spec.profiler->source_uid &&
                  static_cast<std::uint32_t>(profiler_executable.st_gid) ==
                      spec.profiler->source_gid
            : profiler_executable_seals >= 0 &&
                  (profiler_executable_seals & required_seals) ==
                      required_seals));
  if (spec.cgroup_fd < 0 || spec.executable_fd < 0 ||
      spec.worker_bootstrap_fd < 0 ||
      spec.working_directory_fd < 0 ||
      ::fstat(spec.cgroup_fd, &cgroup) != 0 || !S_ISDIR(cgroup.st_mode) ||
      static_cast<std::uint64_t>(cgroup.st_dev) !=
          spec.expected_cgroup_device ||
      static_cast<std::uint64_t>(cgroup.st_ino) !=
          spec.expected_cgroup_inode ||
      ::fstat(spec.executable_fd, &executable) != 0 ||
      !S_ISREG(executable.st_mode) || seals < 0 ||
      (seals & required_seals) != required_seals ||
      (spec.code_fd &&
       (::fstat(*spec.code_fd, &code) != 0 || !S_ISREG(code.st_mode) ||
        code_seals < 0 || (code_seals & required_seals) != required_seals)) ||
      ::fstat(spec.worker_bootstrap_fd, &bootstrap) != 0 ||
      !S_ISREG(bootstrap.st_mode) || bootstrap_seals < 0 ||
      (bootstrap_seals & required_seals) != required_seals ||
      !profiler_executable_valid ||
      (spec.profiler &&
       (
        ::fstat(spec.profiler->authority_fd, &profiler_authority) != 0 ||
        !S_ISREG(profiler_authority.st_mode) || profiler_authority_seals < 0 ||
        (profiler_authority_seals & required_seals) != required_seals)) ||
      ::fstat(spec.working_directory_fd, &working_directory) != 0 ||
      !S_ISDIR(working_directory.st_mode)) {
    reject("stopped launch descriptors do not match sealed authority");
  }
}

void validate_spec(const LinuxStoppedLaunchSpec& spec) {
  if (spec.launch_id.empty() || spec.launch_id.size() > 1024U ||
      !canonical_absolute_path(spec.expected_cgroup_path) ||
      spec.expected_cgroup_device == 0U || spec.expected_cgroup_inode == 0U ||
      spec.executable_name.empty() || spec.executable_name.size() > 4096U ||
      !valid_digest(spec.executable_digest) ||
      // The launcher may target a uid it already is, or any uid if it is root.
      // In the unprivileged case this is strictly stronger than the old
      // "must be root" rule; in the privileged case it is identical, so no
      // capability is silently gained.
      (::geteuid() != 0U && ::geteuid() != spec.credentials.uid) ||
      spec.credentials.uid == 0U ||
      spec.credentials.gid == 0U || !spec.credentials.no_new_privileges ||
      spec.credentials.supplementary_gids.size() >
          kMaximumSupplementaryGroups ||
      (spec.nice && (*spec.nice < -20 || *spec.nice > 19)) ||
      spec.arguments.size() > 256U) {
    reject("stopped launch specification is malformed or unbounded");
  }
  if (spec.profiler &&
      (spec.profiler->executable_fd < 0 || spec.profiler->authority_fd < 0 ||
       !canonical_absolute_path(spec.profiler->executable_name) ||
       !valid_digest(spec.profiler->executable_digest) ||
       (spec.profiler->execute_from_source &&
        (spec.profiler->source_device == 0U ||
         spec.profiler->source_inode == 0U ||
         spec.profiler->source_size == 0U ||
         (spec.profiler->source_mode & S_IFMT) != S_IFREG ||
         (spec.profiler->source_mode & 0111U) == 0U ||
         (spec.profiler->source_mode & (S_IWGRP | S_IWOTH)) != 0U)) ||
       spec.profiler->arguments.empty() ||
       spec.profiler->arguments.size() + spec.arguments.size() + 2U > 256U)) {
    reject("stopped profiler wrapper specification is malformed or unbounded");
  }
  const std::string bootstrap_argument =
      "--trainvm-bootstrap-fd=" +
      std::to_string(kLinuxWorkerBootstrapDescriptor);
  if (spec.arguments.empty() || spec.arguments.back() != bootstrap_argument ||
      (spec.code_fd &&
       (spec.code_argument_index >= spec.arguments.size() ||
        spec.arguments.at(spec.code_argument_index) !=
            "/proc/self/fd/" +
                std::to_string(kLinuxWorkerCodeDescriptor)))) {
    reject("stopped launch inherited-descriptor argv ABI is invalid");
  }
  std::size_t argument_bytes = spec.executable_name.size();
  for (const std::string& argument : spec.arguments) {
    if (argument.empty() || argument.size() > 4096U ||
        argument.find('\0') != std::string::npos ||
        argument_bytes > 65'536U - argument.size()) {
      reject("stopped launch argv is malformed or unbounded");
    }
    argument_bytes += argument.size();
  }
  if (spec.profiler) {
    if (argument_bytes > 65'536U - spec.profiler->executable_name.size())
      reject("stopped profiler wrapper argv exceeds its byte bound");
    argument_bytes += spec.profiler->executable_name.size();
    for (const std::string& argument : spec.profiler->arguments) {
      if (argument.empty() || argument.size() > 4096U ||
          argument.find('\0') != std::string::npos ||
          argument_bytes > 65'536U - argument.size()) {
        reject("stopped profiler wrapper argv is malformed or unbounded");
      }
      argument_bytes += argument.size();
    }
  }
  require_descriptor_identity(spec);
}

}  // namespace

LinuxStoppedChild::LinuxStoppedChild(LinuxStoppedChildIdentity identity,
                                     int pidfd, int gate_fd) noexcept
    : identity_(std::move(identity)), pidfd_(pidfd), gate_fd_(gate_fd) {}

LinuxStoppedChild::LinuxStoppedChild(LinuxStoppedChild&& other) noexcept
    : identity_(std::move(other.identity_)),
      pidfd_(std::exchange(other.pidfd_, -1)),
      gate_fd_(std::exchange(other.gate_fd_, -1)),
      released_(other.released_), reaped_(other.reaped_),
      exit_observation_(other.exit_observation_) {
  other.reaped_ = true;
}

LinuxStoppedChild& LinuxStoppedChild::operator=(
    LinuxStoppedChild&& other) noexcept {
  if (this != &other) {
    terminate_and_reap();
    identity_ = std::move(other.identity_);
    pidfd_ = std::exchange(other.pidfd_, -1);
    gate_fd_ = std::exchange(other.gate_fd_, -1);
    released_ = other.released_;
    reaped_ = other.reaped_;
    exit_observation_ = other.exit_observation_;
    other.reaped_ = true;
  }
  return *this;
}

LinuxStoppedChild::~LinuxStoppedChild() { terminate_and_reap(); }

const LinuxStoppedChildIdentity& LinuxStoppedChild::identity() const {
  return identity_;
}

bool LinuxStoppedChild::released() const { return released_; }

bool LinuxStoppedChild::exited() const {
  return reaped_ || pidfd_ < 0 || !pidfd_alive(pidfd_);
}

void LinuxStoppedChild::release_to_exec() {
  if (reaped_ || pidfd_ < 0 || gate_fd_ < 0 || released_ ||
      !pidfd_alive(pidfd_)) {
    reject("stopped child cannot be released from its current state");
  }
  const char release = 'G';
  ssize_t written = 0;
  do {
    written = ::write(gate_fd_, &release, 1U);
  } while (written < 0 && errno == EINTR);
  if (written != 1) reject(system_error("could not release stopped child"));
  (void)::close(gate_fd_);
  gate_fd_ = -1;
  released_ = true;
}

void LinuxStoppedChild::terminate_and_reap() noexcept {
  (void)reap(true, false);
}

LinuxChildExitObservation LinuxStoppedChild::wait_and_reap() {
  const auto observation = reap(false, true);
  if (!observation) reject("child wait produced no terminal observation");
  return *observation;
}

LinuxChildExitObservation LinuxStoppedChild::terminate_and_observe() {
  const auto observation = reap(true, true);
  if (!observation) reject("child termination produced no terminal observation");
  return *observation;
}

std::optional<LinuxChildExitObservation> LinuxStoppedChild::reap(
    bool terminate, bool fail_on_error) {
  if (exit_observation_) return exit_observation_;
  if (gate_fd_ >= 0) {
    (void)::close(gate_fd_);
    gate_fd_ = -1;
  }
  if (pidfd_ < 0) {
    reaped_ = true;
    if (fail_on_error) reject("child pidfd authority is unavailable");
    return std::nullopt;
  }
#ifdef SYS_pidfd_send_signal
  if (terminate && !reaped_ && pidfd_alive(pidfd_)) {
    (void)::syscall(SYS_pidfd_send_signal, pidfd_, SIGKILL, nullptr, 0U);
  }
#endif
  siginfo_t information {};
  int status = 0;
  do {
    status = ::waitid(P_PIDFD, static_cast<id_t>(pidfd_), &information,
                      WEXITED);
  } while (status != 0 && errno == EINTR);
  if (status != 0) {
    if (fail_on_error) reject(system_error("pidfd waitid failed"));
  } else {
    exit_observation_ = LinuxChildExitObservation{
        .wait_code = information.si_code,
        .wait_status = information.si_status};
  }
  (void)::close(pidfd_);
  pidfd_ = -1;
  reaped_ = true;
  return exit_observation_;
}

LinuxStoppedChild LinuxStoppedLauncherKernel::spawn_stopped(
    const LinuxStoppedLaunchSpec& spec) const {
  validate_spec(spec);
  Descriptor inherited_executable(
      ::fcntl(spec.executable_fd, F_DUPFD_CLOEXEC, 64));
  Descriptor inherited_code(
      spec.code_fd ? ::fcntl(*spec.code_fd, F_DUPFD_CLOEXEC, 64) : -1);
  Descriptor inherited_bootstrap(
      ::fcntl(spec.worker_bootstrap_fd, F_DUPFD_CLOEXEC, 64));
  Descriptor inherited_profiler_executable(
      spec.profiler
          ? ::fcntl(spec.profiler->executable_fd, F_DUPFD_CLOEXEC, 64)
          : -1);
  Descriptor inherited_profiler_authority(
      spec.profiler ? ::fcntl(spec.profiler->authority_fd, F_DUPFD_CLOEXEC, 64)
                    : -1);
  if (inherited_executable.get() < 0 ||
      (spec.code_fd && inherited_code.get() < 0) ||
      inherited_bootstrap.get() < 0 ||
      (spec.profiler && (inherited_profiler_executable.get() < 0 ||
                         inherited_profiler_authority.get() < 0))) {
    reject(system_error("could not duplicate inherited worker descriptor"));
  }
  std::vector<std::string> argument_storage = compose_exec_arguments(spec);
  std::vector<char*> arguments;
  arguments.reserve(argument_storage.size() + 1U);
  for (const std::string& argument : argument_storage) {
    arguments.push_back(const_cast<char*>(argument.c_str()));
  }
  arguments.push_back(nullptr);
  std::array<int, 2U> gate{-1, -1};
  std::array<int, 2U> ready{-1, -1};
  if (::pipe2(gate.data(), O_CLOEXEC) != 0) {
    reject(system_error("could not create stopped-child gate"));
  }
  if (::pipe2(ready.data(), O_CLOEXEC) != 0) {
    const int saved_errno = errno;
    (void)::close(gate[0]);
    (void)::close(gate[1]);
    errno = saved_errno;
    reject(system_error("could not create stopped-child readiness pipe"));
  }
  Descriptor gate_read(gate[0]);
  Descriptor gate_write(gate[1]);
  Descriptor ready_read(ready[0]);
  Descriptor ready_write(ready[1]);
  int pidfd = -1;
  struct clone_args clone {};
  clone.flags = CLONE_INTO_CGROUP | CLONE_PIDFD;
  clone.pidfd = reinterpret_cast<std::uint64_t>(&pidfd);
  clone.exit_signal = SIGCHLD;
  clone.cgroup = static_cast<std::uint64_t>(spec.cgroup_fd);
  const long result = ::syscall(SYS_clone3, &clone, sizeof(clone));
  if (result < 0) reject(system_error("clone3 stopped launch failed"));
  if (result == 0) {
    (void)::close(gate_write.release());
    (void)::close(ready_read.release());
    const bool prepared =
        ::fchdir(spec.working_directory_fd) == 0 &&
        (spec.profiler
             ? install_inherited_profiled_worker_descriptors(
                   spec.code_fd ? std::optional<int>{inherited_code.get()}
                                : std::nullopt,
                   inherited_bootstrap.get(),
                   inherited_profiler_authority.get(),
                   inherited_executable.get())
             : install_inherited_worker_descriptors(
                   spec.code_fd ? std::optional<int>{inherited_code.get()}
                                : std::nullopt,
                   inherited_bootstrap.get())) &&
        install_process_priority(spec.nice) &&
        install_worker_credentials(spec.credentials);
    const char ready_command = 'R';
    ssize_t ready_count = 0;
    if (prepared) {
      do {
        ready_count = ::write(ready_write.get(), &ready_command, 1U);
      } while (ready_count < 0 && errno == EINTR);
    }
    ready_write.reset();
    if (!prepared || ready_count != 1) ::_exit(124);
    char command = 0;
    ssize_t count = 0;
    do {
      count = ::read(gate_read.get(), &command, 1U);
    } while (count < 0 && errno == EINTR);
    if (count != 1 || command != 'G') {
      ::_exit(125);
    }
    char* const environment[] = {nullptr};
    const int launch_executable = spec.profiler
        ? inherited_profiler_executable.get()
        : inherited_executable.get();
    (void)::execveat(launch_executable, "", arguments.data(), environment,
                     AT_EMPTY_PATH);
    ::_exit(126);
  }
  gate_read.reset();
  ready_write.reset();
  LinuxStoppedChild child({}, pidfd, gate_write.release());
  pollfd ready_poll{.fd = ready_read.get(), .events = POLLIN, .revents = 0};
  int ready_status = 0;
  do {
    ready_status = ::poll(&ready_poll, 1U, 5'000);
  } while (ready_status < 0 && errno == EINTR);
  char ready_command = 0;
  ssize_t ready_count = 0;
  if (ready_status == 1 && (ready_poll.revents & POLLIN) != 0) {
    do {
      ready_count = ::read(ready_read.get(), &ready_command, 1U);
    } while (ready_count < 0 && errno == EINTR);
  }
  ready_read.reset();
  if (ready_count != 1 || ready_command != 'R' || !pidfd_alive(pidfd)) {
    reject("child failed its bounded pre-exec credential transition");
  }
  const pid_t pid = static_cast<pid_t>(result);
  const std::string proc_name = std::to_string(pid);
  Descriptor process(::openat(AT_FDCWD, ("/proc/" + proc_name).c_str(),
                              O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (pidfd < 0 || process.get() < 0 || !pidfd_alive(pidfd)) {
    reject("clone3 did not return one live pidfd-bound child");
  }
  const std::string stat_before = read_bounded_file(process.get(), "stat", 4096U);
  const std::string cgroup_before =
      read_bounded_file(process.get(), "cgroup", 4096U);
  const std::string status_before =
      read_bounded_file(process.get(), "status", 16U << 10U);
  const std::string stat_after = read_bounded_file(process.get(), "stat", 4096U);
  const std::string cgroup_after =
      read_bounded_file(process.get(), "cgroup", 4096U);
  const std::string status_after =
      read_bounded_file(process.get(), "status", 16U << 10U);
  const ProcStatIdentity stat_identity_before =
      parse_stat_identity(stat_before);
  const ProcStatIdentity stat_identity_after =
      parse_stat_identity(stat_after);
  const std::string cgroup_path = parse_cgroup(cgroup_before);
  if (stat_identity_before != stat_identity_after ||
      (spec.nice && stat_identity_before.nice != *spec.nice) ||
      cgroup_path != parse_cgroup(cgroup_after) ||
      cgroup_path != spec.expected_cgroup_path ||
      !worker_status_has_credentials(status_before, spec.credentials) ||
      !worker_status_has_credentials(status_after, spec.credentials) ||
      !pidfd_alive(pidfd)) {
    reject("stopped child identity changed or missed its protected cgroup");
  }
  child.identity_ = LinuxStoppedChildIdentity{
      .host_pid = pid,
      .process_starttime_ticks = stat_identity_before.starttime,
      .cgroup_path = cgroup_path,
      .cgroup_device = spec.expected_cgroup_device,
      .cgroup_inode = spec.expected_cgroup_inode,
      .executable_digest = spec.executable_digest,
      .uid = spec.credentials.uid,
      .gid = spec.credentials.gid,
      .no_new_privileges = true,
      .nice = stat_identity_before.nice,
  };
  return child;
}

namespace hostd_linux_stopped_launcher_test_seam {

std::uint64_t parse_proc_starttime(std::string_view stat) {
  return parse_starttime(stat);
}

std::int32_t parse_proc_nice(std::string_view stat) {
  return parse_stat_identity(stat).nice;
}

std::string parse_unified_cgroup(std::string_view cgroup) {
  return parse_cgroup(cgroup);
}

bool install_inherited_worker_descriptors(
    std::optional<int> code_fd, int worker_bootstrap_fd) noexcept {
  return trainvm::install_inherited_worker_descriptors(code_fd,
                                                        worker_bootstrap_fd);
}

bool install_inherited_profiled_worker_descriptors(
    std::optional<int> code_fd, int worker_bootstrap_fd,
    int profiler_authority_fd, int target_executable_fd) noexcept {
  return trainvm::install_inherited_profiled_worker_descriptors(
      code_fd, worker_bootstrap_fd, profiler_authority_fd,
      target_executable_fd);
}

std::vector<std::string> compose_exec_arguments(
    const LinuxStoppedLaunchSpec& spec) {
  return trainvm::compose_exec_arguments(spec);
}

bool worker_status_has_credentials(
    std::string_view status, const LinuxWorkerCredentialSpec& expected) {
  return trainvm::worker_status_has_credentials(status, expected);
}

}  // namespace hostd_linux_stopped_launcher_test_seam

}  // namespace trainvm
