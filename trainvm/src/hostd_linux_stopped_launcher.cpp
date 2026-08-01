#include "trainvm/hostd_linux_stopped_launcher.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <linux/sched.h>
#include <poll.h>
#include <ranges>
#include <signal.h>
#include <string>
#include <sys/stat.h>
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

std::uint64_t parse_starttime(std::string_view value) {
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
  return starttime;
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

void require_descriptor_identity(const LinuxStoppedLaunchSpec& spec) {
  struct stat cgroup {};
  struct stat executable {};
  struct stat working_directory {};
  const int seals = ::fcntl(spec.executable_fd, F_GET_SEALS);
  constexpr int required_seals =
      F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  if (spec.cgroup_fd < 0 || spec.executable_fd < 0 ||
      spec.working_directory_fd < 0 ||
      ::fstat(spec.cgroup_fd, &cgroup) != 0 || !S_ISDIR(cgroup.st_mode) ||
      static_cast<std::uint64_t>(cgroup.st_dev) !=
          spec.expected_cgroup_device ||
      static_cast<std::uint64_t>(cgroup.st_ino) !=
          spec.expected_cgroup_inode ||
      ::fstat(spec.executable_fd, &executable) != 0 ||
      !S_ISREG(executable.st_mode) || seals < 0 ||
      (seals & required_seals) != required_seals ||
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
      spec.executable_digest.size() != 71U ||
      !spec.executable_digest.starts_with("sha256:") ||
      !std::ranges::all_of(spec.executable_digest.substr(7U), [](char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'a' && value <= 'f');
      }) ||
      spec.arguments.size() > 256U) {
    reject("stopped launch specification is malformed or unbounded");
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
  std::vector<char*> arguments;
  arguments.reserve(spec.arguments.size() + 2U);
  arguments.push_back(const_cast<char*>(spec.executable_name.c_str()));
  for (const std::string& argument : spec.arguments) {
    arguments.push_back(const_cast<char*>(argument.c_str()));
  }
  arguments.push_back(nullptr);
  std::array<int, 2U> gate{-1, -1};
  if (::pipe2(gate.data(), O_CLOEXEC) != 0) {
    reject(system_error("could not create stopped-child gate"));
  }
  Descriptor gate_read(gate[0]);
  Descriptor gate_write(gate[1]);
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
    char command = 0;
    ssize_t count = 0;
    do {
      count = ::read(gate_read.get(), &command, 1U);
    } while (count < 0 && errno == EINTR);
    if (count != 1 || command != 'G' ||
        ::fchdir(spec.working_directory_fd) != 0) {
      ::_exit(125);
    }
    char* const environment[] = {nullptr};
    (void)::execveat(spec.executable_fd, "", arguments.data(), environment,
                     AT_EMPTY_PATH);
    ::_exit(126);
  }
  gate_read.reset();
  LinuxStoppedChild child({}, pidfd, gate_write.release());
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
  const std::string stat_after = read_bounded_file(process.get(), "stat", 4096U);
  const std::string cgroup_after =
      read_bounded_file(process.get(), "cgroup", 4096U);
  const std::uint64_t starttime = parse_starttime(stat_before);
  const std::string cgroup_path = parse_cgroup(cgroup_before);
  if (starttime != parse_starttime(stat_after) ||
      cgroup_path != parse_cgroup(cgroup_after) ||
      cgroup_path != spec.expected_cgroup_path || !pidfd_alive(pidfd)) {
    reject("stopped child identity changed or missed its protected cgroup");
  }
  child.identity_ = LinuxStoppedChildIdentity{
      .host_pid = pid,
      .process_starttime_ticks = starttime,
      .cgroup_path = cgroup_path,
      .cgroup_device = spec.expected_cgroup_device,
      .cgroup_inode = spec.expected_cgroup_inode,
      .executable_digest = spec.executable_digest,
  };
  return child;
}

namespace hostd_linux_stopped_launcher_test_seam {

std::uint64_t parse_proc_starttime(std::string_view stat) {
  return parse_starttime(stat);
}

std::string parse_unified_cgroup(std::string_view cgroup) {
  return parse_cgroup(cgroup);
}

}  // namespace hostd_linux_stopped_launcher_test_seam

}  // namespace trainvm
