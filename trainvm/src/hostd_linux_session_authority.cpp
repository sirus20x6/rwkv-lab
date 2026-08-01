#include "trainvm/hostd_linux_session_authority.hpp"

#include <fcntl.h>
#include <asm-generic/socket.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <openssl/sha.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <limits>
#include <mutex>
#include <ranges>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

constexpr std::size_t kEntropyBytes = 32U;
constexpr std::size_t kMaximumProcFileBytes = 64U * 1024U;
constexpr std::size_t kMaximumPurposeBytes = 64U;
constexpr std::string_view kNonceDerivationDomain =
    "trainvm.hostd-linux-session-nonce/v1";

[[nodiscard]] pid_t current_thread_id() noexcept {
  const long value = ::syscall(SYS_gettid);
  if (value <= 0 ||
      static_cast<unsigned long>(value) >
          static_cast<unsigned long>(std::numeric_limits<pid_t>::max()))
    return -1;
  return static_cast<pid_t>(value);
}

[[nodiscard]] bool is_creator_task(pid_t creator_pid,
                                   pid_t creator_tid) noexcept {
  return creator_pid > 0 && creator_tid > 0 && ::getpid() == creator_pid &&
         current_thread_id() == creator_tid;
}

[[noreturn]] void reject(std::string_view message) {
  throw HostdSessionChallengeRejected(std::string(message));
}

class Descriptor final {
public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }
  Descriptor(Descriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor &operator=(Descriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }

private:
  int value_;
};

std::string trim_ascii(std::string value) {
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                            value.back() == '\r' || value.back() == '\n'))
    value.pop_back();
  std::size_t begin = 0U;
  while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t' ||
                                  value[begin] == '\r' || value[begin] == '\n'))
    ++begin;
  value.erase(0U, begin);
  return value;
}

bool canonical_boot_id(std::string_view value) {
  if (value.size() != 36U)
    return false;
  for (std::size_t index = 0U; index < value.size(); ++index) {
    const bool hyphen =
        index == 8U || index == 13U || index == 18U || index == 23U;
    const bool hex = (value[index] >= '0' && value[index] <= '9') ||
                     (value[index] >= 'a' && value[index] <= 'f');
    if (hyphen ? value[index] != '-' : !hex)
      return false;
  }
  return true;
}

bool valid_purpose(std::string_view value) {
  return !value.empty() && value.size() <= kMaximumPurposeBytes &&
         std::ranges::all_of(value, [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= '0' && character <= '9') || character == '_' ||
                  character == '-';
         });
}

std::string hex_digest(std::string_view framed) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  if (::SHA256(reinterpret_cast<const unsigned char *>(framed.data()),
               framed.size(), digest.data()) == nullptr)
    reject("Linux session nonce derivation failed");
  constexpr std::array digits{'0', '1', '2', '3', '4', '5', '6', '7',
                              '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string result;
  result.reserve(digest.size() * 2U);
  for (const unsigned char byte : digest) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

std::optional<std::uint64_t> parse_unsigned(std::string_view value) {
  if (value.empty())
    return std::nullopt;
  std::uint64_t parsed = 0U;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
    return std::nullopt;
  return parsed;
}

std::vector<std::string_view> whitespace_tokens(std::string_view value) {
  std::vector<std::string_view> result;
  std::size_t begin = 0U;
  while (begin < value.size()) {
    while (begin < value.size() &&
           (value[begin] == ' ' || value[begin] == '\t'))
      ++begin;
    if (begin == value.size())
      break;
    std::size_t end = begin;
    while (end < value.size() && value[end] != ' ' && value[end] != '\t')
      ++end;
    result.push_back(value.substr(begin, end - begin));
    begin = end;
  }
  return result;
}

std::optional<std::uint64_t> parse_proc_stat_impl(std::string_view stat,
                                                  pid_t expected_pid) {
  if (expected_pid <= 0 || stat.empty() || stat.size() > kMaximumProcFileBytes)
    return std::nullopt;
  const auto open = stat.find(" (");
  const auto close = stat.rfind(") ");
  if (open == std::string_view::npos || close == std::string_view::npos ||
      close <= open + 1U)
    return std::nullopt;
  const auto parsed_pid = parse_unsigned(stat.substr(0U, open));
  if (!parsed_pid || *parsed_pid != static_cast<std::uint64_t>(expected_pid))
    return std::nullopt;
  const auto fields = whitespace_tokens(stat.substr(close + 2U));
  // fields[0] is field 3 (state); Linux starttime is field 22.
  if (fields.size() < 20U || fields[0].size() != 1U)
    return std::nullopt;
  const auto starttime = parse_unsigned(fields[19U]);
  return starttime && *starttime > 0U ? starttime : std::nullopt;
}

std::optional<std::array<std::uint64_t, 4U>>
status_id_line(std::string_view status, std::string_view name) {
  std::optional<std::array<std::uint64_t, 4U>> result;
  std::size_t begin = 0U;
  while (begin < status.size()) {
    const auto end = status.find('\n', begin);
    const std::string_view line = status.substr(
        begin, (end == std::string_view::npos ? status.size() : end) - begin);
    if (line.starts_with(name)) {
      if (result)
        return std::nullopt;
      const auto values = whitespace_tokens(line.substr(name.size()));
      if (values.size() != 4U)
        return std::nullopt;
      std::array<std::uint64_t, 4U> parsed{};
      for (std::size_t index = 0U; index < parsed.size(); ++index) {
        const auto value = parse_unsigned(values[index]);
        if (!value)
          return std::nullopt;
        parsed[index] = *value;
      }
      result = parsed;
    }
    if (end == std::string_view::npos)
      break;
    begin = end + 1U;
  }
  return result;
}

std::optional<std::pair<uid_t, gid_t>>
parse_proc_status_impl(std::string_view status) {
  if (status.empty() || status.size() > kMaximumProcFileBytes)
    return std::nullopt;
  const auto uids = status_id_line(status, "Uid:");
  const auto gids = status_id_line(status, "Gid:");
  if (!uids || !gids || (*uids)[1U] > std::numeric_limits<uid_t>::max() ||
      (*gids)[1U] > std::numeric_limits<gid_t>::max())
    return std::nullopt;
  return std::pair{static_cast<uid_t>((*uids)[1U]),
                   static_cast<gid_t>((*gids)[1U])};
}

class RealLinuxSessionKernel final : public IHostdLinuxSessionKernel {
public:
  explicit RealLinuxSessionKernel(HostdLinuxSessionKernelConfig config)
      : enforcement_grade_(config.enforcement_grade),
        creator_pid_(::getpid()),
        creator_tid_(current_thread_id()),
        proc_(::open("/proc", O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)) {
    if (config.api_version != kHostdLinuxSessionAuthorityApiVersion ||
        (enforcement_grade_ != HostdLinuxSessionEnforcementGrade::
                                   cooperative_namespace_observation &&
         enforcement_grade_ != HostdLinuxSessionEnforcementGrade::
                                   strict_host_namespaces_and_socket_pidfd) ||
        creator_pid_ <= 0 || creator_tid_ <= 0 || proc_.get() < 0)
      throw HostdSessionChallengeError(
          "Linux session procfs configuration is unavailable");
    struct statfs filesystem{};
    struct stat status{};
    if (::fstatfs(proc_.get(), &filesystem) != 0 ||
        filesystem.f_type != PROC_SUPER_MAGIC ||
        ::fstat(proc_.get(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != 0U || (status.st_mode & (S_IWGRP | S_IWOTH)) != 0)
      throw HostdSessionChallengeError(
          "Linux session procfs root validation failed");
    // Prove openat2 is available up front; there is no path-resolution
    // fallback.
    Descriptor probe(open_proc("sys/kernel/random/boot_id", O_RDONLY));
    if (probe.get() < 0)
      throw HostdSessionChallengeError(
          "Linux session procfs cannot be pinned with openat2");
    observed_namespaces_ = observe_namespaces();
    if (observe_namespaces() != observed_namespaces_)
      throw HostdSessionChallengeError(
          "Linux host namespace observation was torn");
    expected_namespaces_ = observed_namespaces_;
    if (enforcement_grade_ ==
        HostdLinuxSessionEnforcementGrade::
            strict_host_namespaces_and_socket_pidfd) {
      if (!config.expected_host_namespaces ||
          !valid_namespace_policy(*config.expected_host_namespaces) ||
          *config.expected_host_namespaces != observed_namespaces_)
        throw HostdSessionChallengeError(
            "strict Linux host namespace policy is absent or inexact");
      expected_namespaces_ = *config.expected_host_namespaces;
    } else if (config.expected_host_namespaces) {
      throw HostdSessionChallengeError(
          "cooperative Linux namespace grade cannot claim host policy");
    }
  }

  HostdLinuxSessionEnforcementGrade enforcement_grade() const override {
    attest_context();
    return enforcement_grade_;
  }

  HostdLinuxRandomRead getrandom_bytes(void *buffer,
                                       std::size_t count) override {
    attest_context();
    errno = 0;
    const ssize_t result =
        static_cast<ssize_t>(::syscall(SYS_getrandom, buffer, count, 0U));
    return {.count = result, .error_number = result < 0 ? errno : 0};
  }

  HostdLinuxClockRead clock_boottime() override {
    attest_context();
    timespec value{};
    errno = 0;
    if (::clock_gettime(CLOCK_BOOTTIME, &value) != 0)
      return {.success = false, .value = {}, .error_number = errno};
    return {.success = true, .value = value, .error_number = 0};
  }

  HostdLinuxBootIdRead read_boot_id() override {
    attest_context();
    const auto value = read_proc("sys/kernel/random/boot_id", 256U);
    if (!value)
      return {.success = false, .value = {}, .error_number = errno};
    return {.success = true, .value = trim_ascii(*value), .error_number = 0};
  }

  HostdLinuxPeerKernelObservation observe_process(pid_t pid) override {
    attest_context();
    HostdLinuxPeerKernelObservation result{.pid = pid};
    if (pid <= 0) {
      result.error_number = EINVAL;
      return result;
    }
    Descriptor pidfd;
#ifdef SYS_pidfd_open
    errno = 0;
    pidfd = Descriptor(static_cast<int>(::syscall(SYS_pidfd_open, pid, 0U)));
    if (pidfd.get() < 0 && errno != ENOSYS && errno != EINVAL) {
      result.error_number = errno;
      return result;
    }
#endif
    result.pidfd_available = pidfd.get() >= 0;
    if (enforcement_grade_ ==
            HostdLinuxSessionEnforcementGrade::
                strict_host_namespaces_and_socket_pidfd &&
        !result.pidfd_available) {
      result.error_number = ENOTSUP;
      return result;
    }
    result.pidfd_alive_before =
        !result.pidfd_available || pidfd_alive(pidfd.get());
    if (!result.pidfd_alive_before) {
      result.error_number = ESRCH;
      return result;
    }

    const std::string pid_name = std::to_string(pid);
    Descriptor directory(open_proc(pid_name, O_PATH | O_DIRECTORY));
    if (directory.get() < 0) {
      result.error_number = errno;
      return result;
    }
    struct stat before{};
    struct stat after{};
    if (::fstat(directory.get(), &before) != 0) {
      result.error_number = errno;
      return result;
    }
    result.process_directory_device_before = before.st_dev;
    result.process_directory_inode_before = before.st_ino;
    const auto stat_before = read_at(directory.get(), "stat", 4096U);
    const auto status_before =
        read_at(directory.get(), "status", kMaximumProcFileBytes);
    const auto stat_after = read_at(directory.get(), "stat", 4096U);
    const auto status_after =
        read_at(directory.get(), "status", kMaximumProcFileBytes);
    if (!stat_before || !status_before || !stat_after || !status_after ||
        ::fstat(directory.get(), &after) != 0) {
      result.error_number = errno == 0 ? ESRCH : errno;
      return result;
    }
    result.process_directory_device_after = after.st_dev;
    result.process_directory_inode_after = after.st_ino;
    const auto start_before = parse_proc_stat_impl(*stat_before, pid);
    const auto start_after = parse_proc_stat_impl(*stat_after, pid);
    const auto credentials_before = parse_proc_status_impl(*status_before);
    const auto credentials_after = parse_proc_status_impl(*status_after);
    result.pidfd_alive_after =
        !result.pidfd_available || pidfd_alive(pidfd.get());
    if (!start_before || !start_after || !credentials_before ||
        !credentials_after) {
      result.error_number = EPROTO;
      return result;
    }
    result.process_starttime_ticks_before = *start_before;
    result.process_starttime_ticks_after = *start_after;
    result.effective_uid_before = credentials_before->first;
    result.effective_gid_before = credentials_before->second;
    result.effective_uid_after = credentials_after->first;
    result.effective_gid_after = credentials_after->second;
    result.complete = result.pidfd_alive_before && result.pidfd_alive_after;
    return result;
  }

private:
  [[nodiscard]] static bool
  valid_namespace_identity(const HostdLinuxNamespaceIdentity &identity) {
    return identity.device != 0U && identity.inode != 0U;
  }

  [[nodiscard]] static bool
  valid_namespace_policy(const HostdLinuxHostNamespacePolicy &policy) {
    return valid_namespace_identity(policy.mount_namespace) &&
           valid_namespace_identity(policy.pid_namespace) &&
           valid_namespace_identity(policy.time_namespace) &&
           valid_namespace_identity(policy.time_for_children_namespace);
  }

  [[nodiscard]] HostdLinuxNamespaceIdentity
  observe_namespace(int process_directory, std::string_view name) const {
    const std::string relative = "ns/" + std::string(name);
    const int opened = ::openat(process_directory, relative.c_str(),
                                O_RDONLY | O_CLOEXEC);
    if (opened < 0)
      throw HostdSessionChallengeError(
          "Linux process namespace identity is unavailable");
    Descriptor descriptor(opened);
    struct stat status{};
    struct statfs filesystem{};
    if (::fstat(descriptor.get(), &status) != 0 ||
        ::fstatfs(descriptor.get(), &filesystem) != 0 ||
        filesystem.f_type != NSFS_MAGIC || status.st_dev == 0 ||
        status.st_ino == 0)
      throw HostdSessionChallengeError(
          "Linux process namespace identity is invalid");
    return {.device = static_cast<std::uint64_t>(status.st_dev),
            .inode = static_cast<std::uint64_t>(status.st_ino)};
  }

  [[nodiscard]] HostdLinuxHostNamespacePolicy observe_namespaces() const {
    const std::string task = std::to_string(creator_pid_) + "/task/" +
                             std::to_string(creator_tid_);
    Descriptor process(open_proc(task, O_PATH | O_DIRECTORY));
    if (process.get() < 0)
      throw HostdSessionChallengeError(
          "Linux creator process namespace directory is unavailable");
    return {.mount_namespace = observe_namespace(process.get(), "mnt"),
            .pid_namespace = observe_namespace(process.get(), "pid"),
            .time_namespace = observe_namespace(process.get(), "time"),
            .time_for_children_namespace =
                observe_namespace(process.get(), "time_for_children")};
  }

  void attest_context() const {
    // Linux mount and time namespaces are task scoped. Reject before any
    // namespace, procfs, clock, entropy, or peer observation so a caller on a
    // sibling thread can never borrow the creator task's authority.
    if (!is_creator_task(creator_pid_, creator_tid_))
      throw HostdSessionChallengeError(
          "Linux session authority crossed fork, thread, or namespace identity");
    const auto before = observe_namespaces();
    const auto after = observe_namespaces();
    if (before != expected_namespaces_ || after != expected_namespaces_)
      throw HostdSessionChallengeError(
          "Linux session authority crossed fork, thread, or namespace identity");
  }

  [[nodiscard]] int open_proc(std::string_view relative,
                              std::uint64_t flags) const {
    if (relative.empty() || relative.starts_with('/') ||
        relative.find('\0') != std::string_view::npos)
      return -1;
    open_how how{};
    how.flags = flags | O_CLOEXEC | O_NOFOLLOW;
    how.resolve = RESOLVE_IN_ROOT | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV;
    return static_cast<int>(::syscall(SYS_openat2, proc_.get(),
                                      std::string(relative).c_str(), &how,
                                      sizeof(how)));
  }

  [[nodiscard]] static int
  open_relative(int directory, std::string_view relative, std::uint64_t flags) {
    open_how how{};
    how.flags = flags | O_CLOEXEC | O_NOFOLLOW;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV;
    return static_cast<int>(::syscall(SYS_openat2, directory,
                                      std::string(relative).c_str(), &how,
                                      sizeof(how)));
  }

  [[nodiscard]] static std::optional<std::string>
  read_descriptor(int descriptor, std::size_t maximum) {
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode))
      return std::nullopt;
    std::string result;
    std::array<char, 1024U> buffer{};
    for (;;) {
      const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
      if (count < 0) {
        if (errno == EINTR)
          continue;
        return std::nullopt;
      }
      if (count == 0)
        break;
      if (result.size() + static_cast<std::size_t>(count) > maximum) {
        errno = EFBIG;
        return std::nullopt;
      }
      result.append(buffer.data(), static_cast<std::size_t>(count));
    }
    struct stat after{};
    if (::fstat(descriptor, &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino) {
      errno = ESTALE;
      return std::nullopt;
    }
    return result;
  }

  [[nodiscard]] std::optional<std::string>
  read_proc(std::string_view relative, std::size_t maximum) const {
    Descriptor file(open_proc(relative, O_RDONLY));
    return file.get() < 0 ? std::nullopt : read_descriptor(file.get(), maximum);
  }

  [[nodiscard]] static std::optional<std::string>
  read_at(int directory, std::string_view relative, std::size_t maximum) {
    Descriptor file(open_relative(directory, relative, O_RDONLY));
    return file.get() < 0 ? std::nullopt : read_descriptor(file.get(), maximum);
  }

  [[nodiscard]] static bool pidfd_alive(int pidfd) {
    pollfd descriptor{.fd = pidfd, .events = POLLIN, .revents = 0};
    int status = 0;
    do {
      status = ::poll(&descriptor, 1U, 0);
    } while (status < 0 && errno == EINTR);
    return status == 0;
  }

  HostdLinuxSessionEnforcementGrade enforcement_grade_{};
  pid_t creator_pid_{};
  pid_t creator_tid_{};
  Descriptor proc_;
  HostdLinuxHostNamespacePolicy observed_namespaces_;
  HostdLinuxHostNamespacePolicy expected_namespaces_;
};

HostdSocketPeerInstance
validate_observation(const HostdLinuxSocketPeerCredentials &credentials,
                     const HostdLinuxPeerKernelObservation &observed) {
  if (credentials.pid <= 0 || !observed.complete ||
      observed.pid != credentials.pid ||
      observed.effective_uid_before != credentials.uid ||
      observed.effective_uid_after != credentials.uid ||
      observed.effective_gid_before != credentials.gid ||
      observed.effective_gid_after != credentials.gid ||
      observed.process_starttime_ticks_before == 0U ||
      observed.process_starttime_ticks_before !=
          observed.process_starttime_ticks_after ||
      observed.process_directory_device_before == 0U ||
      observed.process_directory_inode_before == 0U ||
      observed.process_directory_device_before !=
          observed.process_directory_device_after ||
      observed.process_directory_inode_before !=
          observed.process_directory_inode_after ||
      (observed.pidfd_available &&
       (!observed.pidfd_alive_before || !observed.pidfd_alive_after)))
    reject("Linux socket peer process instance is missing, reused, or torn");
  return {.uid = credentials.uid,
          .gid = credentials.gid,
          .pid = credentials.pid,
          .process_starttime_ticks = observed.process_starttime_ticks_before};
}

} // namespace

std::shared_ptr<IHostdLinuxSessionKernel>
make_hostd_linux_session_kernel(HostdLinuxSessionKernelConfig config) {
  return std::make_shared<RealLinuxSessionKernel>(std::move(config));
}

struct HostdLinuxCSPRNGNonceSource::Implementation final {
  std::shared_ptr<IHostdLinuxSessionKernel> kernel;
  std::uint64_t maximum_tokens{};
  std::uint64_t counter{};
  bool poisoned{};
  pid_t creator_pid{};
  pid_t creator_tid{};
  std::set<std::string> issued;
  std::mutex mutex;
};

HostdLinuxCSPRNGNonceSource::HostdLinuxCSPRNGNonceSource(
    std::shared_ptr<IHostdLinuxSessionKernel> kernel,
    std::uint64_t maximum_tokens)
    : implementation_(std::make_unique<Implementation>()) {
  if (!kernel || maximum_tokens == 0U || maximum_tokens > 16'777'216U)
    throw HostdSessionChallengeError(
        "Linux session nonce source configuration is invalid");
  implementation_->kernel = std::move(kernel);
  implementation_->maximum_tokens = maximum_tokens;
  implementation_->creator_pid = ::getpid();
  implementation_->creator_tid = current_thread_id();
  if (implementation_->creator_pid <= 0 || implementation_->creator_tid <= 0)
    throw HostdSessionChallengeError(
        "Linux session nonce source creator task is unavailable");
}

HostdLinuxCSPRNGNonceSource::~HostdLinuxCSPRNGNonceSource() = default;

std::string
HostdLinuxCSPRNGNonceSource::next_hex_256(std::string_view purpose) {
  if (!is_creator_task(implementation_->creator_pid,
                       implementation_->creator_tid))
    reject("Linux session nonce source cannot cross fork or creator thread");
  std::scoped_lock lock(implementation_->mutex);
  if (implementation_->poisoned || !valid_purpose(purpose) ||
      implementation_->counter >= implementation_->maximum_tokens)
    reject("Linux session nonce source is poisoned or exhausted");
  std::array<unsigned char, kEntropyBytes> entropy{};
  std::size_t offset = 0U;
  try {
    while (offset < entropy.size()) {
      const HostdLinuxRandomRead read =
          implementation_->kernel->getrandom_bytes(entropy.data() + offset,
                                                   entropy.size() - offset);
      if (read.count < 0 && read.error_number == EINTR)
        continue;
      if (read.count <= 0 ||
          static_cast<std::size_t>(read.count) > entropy.size() - offset) {
        implementation_->poisoned = true;
        reject("Linux getrandom failed closed");
      }
      offset += static_cast<std::size_t>(read.count);
    }
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    implementation_->poisoned = true;
    reject("Linux getrandom authority is unavailable");
  }
  try {
    std::string framed(kNonceDerivationDomain);
    framed.push_back('\0');
    framed.append(purpose);
    framed.push_back('\0');
    for (int shift = 56; shift >= 0; shift -= 8)
      framed.push_back(
          static_cast<char>((implementation_->counter >> shift) & 0xffU));
    framed.push_back('\0');
    framed.append(reinterpret_cast<const char *>(entropy.data()),
                  entropy.size());
    const std::string token = hex_digest(framed);
    ++implementation_->counter;
    if (implementation_->issued.insert(token).second)
      return token;
    implementation_->poisoned = true;
    reject("Linux session nonce derivation collided");
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    implementation_->poisoned = true;
    reject("Linux session nonce derivation failed closed");
  }
}

struct HostdLinuxBoottimeSource::Implementation final {
  std::string boot_id;
  std::shared_ptr<IHostdLinuxSessionKernel> kernel;
  std::optional<std::int64_t> high_water_ns;
  bool poisoned{};
  pid_t creator_pid{};
  pid_t creator_tid{};
  std::mutex mutex;
};

HostdLinuxBoottimeSource::HostdLinuxBoottimeSource(
    std::string expected_boot_id,
    std::shared_ptr<IHostdLinuxSessionKernel> kernel)
    : implementation_(std::make_unique<Implementation>()) {
  if (!kernel || !canonical_boot_id(expected_boot_id))
    throw HostdSessionChallengeError(
        "Linux boottime source configuration is invalid");
  implementation_->creator_pid = ::getpid();
  implementation_->creator_tid = current_thread_id();
  if (implementation_->creator_pid <= 0 || implementation_->creator_tid <= 0)
    throw HostdSessionChallengeError(
        "Linux boottime source creator task is unavailable");
  HostdLinuxBootIdRead observed;
  try {
    observed = kernel->read_boot_id();
  } catch (...) {
    throw HostdSessionChallengeError(
        "Linux boottime source boot identity is unavailable");
  }
  if (!observed.success || observed.value != expected_boot_id)
    throw HostdSessionChallengeError(
        "Linux boottime source boot identity does not match");
  implementation_->boot_id = std::move(expected_boot_id);
  implementation_->kernel = std::move(kernel);
}

HostdLinuxBoottimeSource::~HostdLinuxBoottimeSource() = default;

HostdSessionChallengeTime HostdLinuxBoottimeSource::now() {
  if (!is_creator_task(implementation_->creator_pid,
                       implementation_->creator_tid))
    reject("Linux boottime source cannot cross fork or creator thread");
  std::scoped_lock lock(implementation_->mutex);
  if (implementation_->poisoned)
    reject("Linux boottime source is poisoned");
  try {
    const HostdLinuxBootIdRead before = implementation_->kernel->read_boot_id();
    const HostdLinuxClockRead clock = implementation_->kernel->clock_boottime();
    const HostdLinuxBootIdRead after = implementation_->kernel->read_boot_id();
    if (!before.success || !after.success || !clock.success ||
        before.value != implementation_->boot_id ||
        after.value != implementation_->boot_id || clock.value.tv_sec < 0 ||
        clock.value.tv_nsec < 0 || clock.value.tv_nsec >= 1'000'000'000L) {
      implementation_->poisoned = true;
      reject("Linux boottime observation is invalid or torn");
    }
    if (clock.value.tv_sec >
        (std::numeric_limits<std::int64_t>::max() - clock.value.tv_nsec) /
            1'000'000'000LL) {
      implementation_->poisoned = true;
      reject("Linux boottime observation overflows nanoseconds");
    }
    const std::int64_t now_ns =
        static_cast<std::int64_t>(clock.value.tv_sec) * 1'000'000'000LL +
        static_cast<std::int64_t>(clock.value.tv_nsec);
    if (implementation_->high_water_ns &&
        now_ns < *implementation_->high_water_ns) {
      implementation_->poisoned = true;
      reject("Linux CLOCK_BOOTTIME regressed");
    }
    implementation_->high_water_ns = now_ns;
    return {.boot_id = implementation_->boot_id, .boottime_ns = now_ns};
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    implementation_->poisoned = true;
    reject("Linux boottime authority is unavailable");
  }
}

struct HostdLinuxPeerProcessObserver::Implementation final {
  std::shared_ptr<IHostdLinuxSessionKernel> kernel;
  pid_t creator_pid{};
  pid_t creator_tid{};
  std::mutex mutex;
};

HostdLinuxPeerProcessObserver::HostdLinuxPeerProcessObserver(
    std::shared_ptr<IHostdLinuxSessionKernel> kernel)
    : implementation_(std::make_unique<Implementation>()) {
  if (!kernel)
    throw HostdSessionChallengeError(
        "Linux peer process observer configuration is invalid");
  implementation_->kernel = std::move(kernel);
  implementation_->creator_pid = ::getpid();
  implementation_->creator_tid = current_thread_id();
  if (implementation_->creator_pid <= 0 || implementation_->creator_tid <= 0)
    throw HostdSessionChallengeError(
        "Linux peer process observer creator task is unavailable");
}

HostdLinuxPeerProcessObserver::~HostdLinuxPeerProcessObserver() = default;

HostdSocketPeerInstance HostdLinuxPeerProcessObserver::observe(
    const HostdLinuxSocketPeerCredentials &credentials) {
  if (!is_creator_task(implementation_->creator_pid,
                       implementation_->creator_tid))
    reject("Linux peer observer cannot cross fork or creator thread");
  std::scoped_lock lock(implementation_->mutex);
  try {
    return validate_observation(
        credentials, implementation_->kernel->observe_process(credentials.pid));
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("Linux peer process observation is unavailable");
  }
}

HostdSocketPeerInstance HostdLinuxPeerProcessObserver::reobserve(
    const HostdLinuxSocketPeerCredentials &credentials,
    const HostdSocketPeerInstance &expected_instance) {
  const HostdSocketPeerInstance observed = observe(credentials);
  if (observed != expected_instance)
    reject("Linux socket peer PID was reused or credentials changed");
  return observed;
}

struct HostdLinuxBoundSocketPeer::Implementation final {
  Descriptor socket;
  Descriptor peer_pidfd;
  std::shared_ptr<IHostdLinuxSessionKernel> kernel;
  HostdLinuxSessionEnforcementGrade grade{};
  HostdLinuxSocketPeerCredentials credentials;
  HostdSocketPeerInstance expected_instance;
  pid_t creator_pid{};
  pid_t creator_tid{};
  mutable std::mutex mutex;
};

namespace {

Descriptor duplicate_descriptor(int descriptor) {
  if (descriptor < 0)
    reject("Linux socket peer descriptor is invalid");
  const int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0)
    reject("Linux socket peer descriptor cannot be pinned");
  return Descriptor(duplicate);
}

HostdLinuxSocketPeerCredentials socket_peer_credentials(int descriptor) {
  sockaddr_storage local{};
  sockaddr_storage remote{};
  socklen_t local_length = sizeof(local);
  socklen_t remote_length = sizeof(remote);
  if (::getsockname(descriptor, reinterpret_cast<sockaddr *>(&local),
                    &local_length) != 0 ||
      ::getpeername(descriptor, reinterpret_cast<sockaddr *>(&remote),
                    &remote_length) != 0 || local.ss_family != AF_UNIX ||
      remote.ss_family != AF_UNIX)
    reject("Linux peer descriptor is not a connected Unix socket");
  ucred credentials{};
  socklen_t length = sizeof(credentials);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &credentials, &length) !=
          0 ||
      length != sizeof(credentials) || credentials.pid <= 0)
    reject("Linux socket peer credentials are unavailable");
  return {.uid = credentials.uid,
          .gid = credentials.gid,
          .pid = credentials.pid};
}

Descriptor socket_peer_pidfd(int descriptor) {
#ifdef SO_PEERPIDFD
  int pidfd = -1;
  socklen_t length = sizeof(pidfd);
  const int status =
      ::getsockopt(descriptor, SOL_SOCKET, SO_PEERPIDFD, &pidfd, &length);
  if (status == 0 && length == sizeof(pidfd) && pidfd >= 0) {
    const int flags = ::fcntl(pidfd, F_GETFD);
    if (flags < 0 || ::fcntl(pidfd, F_SETFD, flags | FD_CLOEXEC) != 0) {
      (void)::close(pidfd);
      reject("Linux socket peer pidfd cannot be protected");
    }
    return Descriptor(pidfd);
  }
  if (status == 0 && pidfd >= 0)
    (void)::close(pidfd);
#else
  (void)descriptor;
#endif
  return Descriptor();
}

bool live_pidfd(int descriptor) {
  if (descriptor < 0)
    return false;
  pollfd observed{.fd = descriptor, .events = POLLIN, .revents = 0};
  int status = 0;
  do {
    status = ::poll(&observed, 1U, 0);
  } while (status < 0 && errno == EINTR);
  return status == 0;
}

} // namespace

HostdLinuxBoundSocketPeer::HostdLinuxBoundSocketPeer(
    std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation)) {}

HostdLinuxBoundSocketPeer::~HostdLinuxBoundSocketPeer() = default;
HostdLinuxBoundSocketPeer::HostdLinuxBoundSocketPeer(
    HostdLinuxBoundSocketPeer &&) noexcept = default;
HostdLinuxBoundSocketPeer &HostdLinuxBoundSocketPeer::operator=(
    HostdLinuxBoundSocketPeer &&) noexcept = default;

const HostdSocketPeerInstance &HostdLinuxBoundSocketPeer::instance() const {
  if (!implementation_ ||
      !is_creator_task(implementation_->creator_pid,
                       implementation_->creator_tid))
    reject("Linux bound socket peer cannot cross fork or creator thread");
  return implementation_->expected_instance;
}

HostdLinuxSessionEnforcementGrade
HostdLinuxBoundSocketPeer::enforcement_grade() const {
  if (!implementation_ ||
      !is_creator_task(implementation_->creator_pid,
                       implementation_->creator_tid))
    reject("Linux bound socket peer cannot cross fork or creator thread");
  return implementation_->grade;
}

HostdSocketPeerInstance HostdLinuxBoundSocketPeer::reobserve() {
  if (!implementation_ ||
      !is_creator_task(implementation_->creator_pid,
                       implementation_->creator_tid))
    reject("Linux bound socket peer cannot cross fork or creator thread");
  std::scoped_lock lock(implementation_->mutex);
  const auto credentials =
      socket_peer_credentials(implementation_->socket.get());
  if (credentials != implementation_->credentials ||
      (implementation_->peer_pidfd.get() >= 0 &&
       !live_pidfd(implementation_->peer_pidfd.get())))
    reject("Linux bound socket peer credentials or pidfd changed");
  HostdSocketPeerInstance observed;
  try {
    observed = validate_observation(
        credentials,
        implementation_->kernel->observe_process(credentials.pid));
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("Linux bound socket peer observation is unavailable");
  }
  if (observed != implementation_->expected_instance ||
      (implementation_->peer_pidfd.get() >= 0 &&
       !live_pidfd(implementation_->peer_pidfd.get())))
    reject("Linux bound socket peer process instance was substituted");
  return observed;
}

HostdLinuxBoundSocketPeer make_hostd_linux_bound_socket_peer(
    int accepted_socket_fd, std::shared_ptr<IHostdLinuxSessionKernel> kernel,
    HostdLinuxSessionEnforcementGrade enforcement_grade) {
  const pid_t creator_pid = ::getpid();
  const pid_t creator_tid = current_thread_id();
  if (!kernel || creator_pid <= 0 || creator_tid <= 0 ||
      (enforcement_grade != HostdLinuxSessionEnforcementGrade::
                                  cooperative_namespace_observation &&
       enforcement_grade != HostdLinuxSessionEnforcementGrade::
                                  strict_host_namespaces_and_socket_pidfd))
    throw HostdSessionChallengeError(
        "Linux bound socket peer configuration is invalid");
  try {
    if (enforcement_grade ==
            HostdLinuxSessionEnforcementGrade::
                strict_host_namespaces_and_socket_pidfd &&
        kernel->enforcement_grade() != enforcement_grade)
      reject("strict Linux socket peer requires a strict namespace kernel");
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("Linux socket peer namespace enforcement grade is unavailable");
  }
  auto implementation =
      std::make_unique<HostdLinuxBoundSocketPeer::Implementation>();
  implementation->socket = duplicate_descriptor(accepted_socket_fd);
  implementation->peer_pidfd =
      socket_peer_pidfd(implementation->socket.get());
  if (enforcement_grade ==
          HostdLinuxSessionEnforcementGrade::
              strict_host_namespaces_and_socket_pidfd &&
      implementation->peer_pidfd.get() < 0)
    reject("strict Linux socket peer requires SO_PEERPIDFD");
  implementation->kernel = std::move(kernel);
  implementation->grade = enforcement_grade;
  implementation->creator_pid = creator_pid;
  implementation->creator_tid = creator_tid;
  implementation->credentials =
      socket_peer_credentials(implementation->socket.get());
  if (implementation->peer_pidfd.get() >= 0 &&
      !live_pidfd(implementation->peer_pidfd.get()))
    reject("Linux socket peer pidfd is not live");
  try {
    implementation->expected_instance = validate_observation(
        implementation->credentials,
        implementation->kernel->observe_process(
            implementation->credentials.pid));
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("Linux socket peer process observation is unavailable");
  }
  if (implementation->peer_pidfd.get() >= 0 &&
      !live_pidfd(implementation->peer_pidfd.get()))
    reject("Linux socket peer exited during process observation");
  return HostdLinuxBoundSocketPeer(std::move(implementation));
}

namespace hostd_linux_session_test_seam {

std::optional<std::uint64_t> parse_proc_stat_starttime(std::string_view stat,
                                                       pid_t expected_pid) {
  return parse_proc_stat_impl(stat, expected_pid);
}

std::optional<std::pair<uid_t, gid_t>>
parse_proc_status_effective_credentials(std::string_view status) {
  return parse_proc_status_impl(status);
}

} // namespace hostd_linux_session_test_seam

} // namespace trainvm
