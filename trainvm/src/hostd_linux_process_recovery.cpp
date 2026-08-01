#include "trainvm/hostd_linux_process_recovery.hpp"

#include <fcntl.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <openssl/evp.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumProcBytes = 64U * 1024U;

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept { return std::exchange(value_, -1); }

 private:
  int value_;
};

bool pidfd_alive(int descriptor) noexcept {
  pollfd item{.fd = descriptor, .events = POLLIN, .revents = 0};
  const int status = ::poll(&item, 1U, 0);
  return status == 0 ||
         (status > 0 && (item.revents & (POLLIN | POLLHUP | POLLERR)) == 0);
}

std::optional<std::uint64_t> unsigned_value(std::string_view value) {
  std::uint64_t result{};
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size())
    return std::nullopt;
  return result;
}

std::vector<std::string_view> tokens(std::string_view value) {
  std::vector<std::string_view> result;
  std::size_t begin = 0U;
  while (begin < value.size()) {
    while (begin < value.size() &&
           (value[begin] == ' ' || value[begin] == '\t'))
      ++begin;
    if (begin == value.size()) break;
    std::size_t end = begin;
    while (end < value.size() && value[end] != ' ' && value[end] != '\t')
      ++end;
    result.push_back(value.substr(begin, end - begin));
    begin = end;
  }
  return result;
}

std::uint64_t parse_starttime(std::string_view value) {
  const std::size_t open = value.find(" (");
  const std::size_t close = value.rfind(") ");
  if (open == std::string_view::npos || close == std::string_view::npos ||
      close <= open + 1U)
    throw HostLedgerError("process recovery proc stat is malformed");
  const auto fields = tokens(value.substr(close + 2U));
  if (fields.size() < 20U || fields.front().size() != 1U)
    throw HostLedgerError("process recovery proc stat is incomplete");
  const auto result = unsigned_value(fields[19U]);
  if (!result || *result == 0U)
    throw HostLedgerError("process recovery starttime is invalid");
  return *result;
}

std::string parse_cgroup(std::string_view value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r'))
    value.remove_suffix(1U);
  constexpr std::string_view prefix = "0::";
  if (!value.starts_with(prefix) || value.contains('\n') ||
      value.contains('\r'))
    throw HostLedgerError("process recovery cgroup membership is malformed");
  std::string result(value.substr(prefix.size()));
  if (result.empty() || result.front() != '/')
    throw HostLedgerError("process recovery cgroup path is not absolute");
  return result;
}

std::string read_file(int directory, const char* name) {
  Descriptor file(::openat(directory, name,
                           O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0) throw HostLedgerError("process recovery proc read failed");
  std::array<char, kMaximumProcBytes + 1U> bytes{};
  ssize_t count = 0;
  do {
    count = ::read(file.get(), bytes.data(), bytes.size());
  } while (count < 0 && errno == EINTR);
  if (count <= 0 || count > static_cast<ssize_t>(kMaximumProcBytes))
    throw HostLedgerError("process recovery proc value is outside its bound");
  return std::string(bytes.data(), static_cast<std::size_t>(count));
}

std::string executable_digest(int process) {
  Descriptor executable(::openat(process, "exe", O_RDONLY | O_CLOEXEC));
  struct stat status {};
  if (executable.get() < 0 || ::fstat(executable.get(), &status) != 0 ||
      !S_ISREG(status.st_mode))
    throw HostLedgerError("process recovery executable is unavailable");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw HostLedgerError("process recovery executable hash init failed");
  std::array<unsigned char, 1U << 20U> bytes{};
  while (true) {
    ssize_t count = 0;
    do {
      count = ::read(executable.get(), bytes.data(), bytes.size());
    } while (count < 0 && errno == EINTR);
    if (count < 0 ||
        (count > 0 &&
         EVP_DigestUpdate(context.get(), bytes.data(),
                          static_cast<std::size_t>(count)) != 1))
      throw HostLedgerError("process recovery executable hash failed");
    if (count == 0) break;
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32U)
    throw HostLedgerError("process recovery executable hash final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  result.reserve(71U);
  for (std::size_t index = 0U; index < digest_size; ++index) {
    const unsigned char byte = digest[index];
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

bool cgroup_identity_matches(const HostProcessSpawnRequest& expected) {
  Descriptor root(::open("/sys/fs/cgroup",
                         O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (root.get() < 0)
    throw HostLedgerError("process recovery cgroup root is unavailable");
  struct statfs filesystem {};
  if (::fstatfs(root.get(), &filesystem) != 0 ||
      filesystem.f_type != CGROUP2_SUPER_MAGIC)
    throw HostLedgerError("process recovery requires cgroup v2");
  Descriptor cgroup;
  if (expected.cgroup_path == "/") {
    cgroup = Descriptor(::fcntl(root.get(), F_DUPFD_CLOEXEC, 3));
  } else {
    const std::string relative = expected.cgroup_path.substr(1U);
    struct open_how how {};
    how.flags = O_PATH | O_DIRECTORY | O_CLOEXEC;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS |
                  RESOLVE_NO_SYMLINKS | RESOLVE_NO_XDEV;
    cgroup = Descriptor(static_cast<int>(::syscall(
        SYS_openat2, root.get(), relative.c_str(), &how, sizeof(how))));
  }
  struct stat status {};
  if (cgroup.get() < 0 || ::fstat(cgroup.get(), &status) != 0)
    throw HostLedgerError("process recovery cgroup is unavailable");
  return static_cast<std::uint64_t>(status.st_dev) == expected.cgroup_device &&
         static_cast<std::uint64_t>(status.st_ino) == expected.cgroup_inode;
}

LinuxProcessRecoveryResult result(LinuxProcessRecoveryDisposition disposition,
                                  std::string detail) {
  return LinuxProcessRecoveryResult(disposition, std::move(detail));
}

}  // namespace

LinuxRecoveredProcess::LinuxRecoveredProcess(HostProcessSpawnRequest identity,
                                             int pidfd) noexcept
    : identity_(std::move(identity)), pidfd_(pidfd) {}

LinuxRecoveredProcess::LinuxRecoveredProcess(
    LinuxRecoveredProcess&& other) noexcept
    : identity_(std::move(other.identity_)),
      pidfd_(std::exchange(other.pidfd_, -1)) {}

LinuxRecoveredProcess& LinuxRecoveredProcess::operator=(
    LinuxRecoveredProcess&& other) noexcept {
  if (this != &other) {
    if (pidfd_ >= 0) (void)::close(pidfd_);
    identity_ = std::move(other.identity_);
    pidfd_ = std::exchange(other.pidfd_, -1);
  }
  return *this;
}

LinuxRecoveredProcess::~LinuxRecoveredProcess() {
  if (pidfd_ >= 0) (void)::close(pidfd_);
}

const HostProcessSpawnRequest& LinuxRecoveredProcess::identity() const noexcept {
  return identity_;
}

bool LinuxRecoveredProcess::alive() const noexcept {
  return pidfd_ >= 0 && pidfd_alive(pidfd_);
}

LinuxProcessRecoveryResult LinuxProcessRecoveryProbe::observe(
    const HostProcessSpawnRequest& expected) const {
  if (seal_host_process_spawn_request(expected) != expected)
    throw HostLedgerError("process recovery identity is not canonical");
#ifndef SYS_pidfd_open
  return result(LinuxProcessRecoveryDisposition::observation_failed,
                "pidfd_open is unavailable");
#else
  errno = 0;
  Descriptor pidfd(static_cast<int>(
      ::syscall(SYS_pidfd_open, expected.host_pid, 0U)));
  if (pidfd.get() < 0) {
    if (errno == ESRCH)
      return result(LinuxProcessRecoveryDisposition::already_gone,
                    "recorded process no longer exists");
    return result(LinuxProcessRecoveryDisposition::observation_failed,
                  "pidfd_open failed");
  }
  if (!pidfd_alive(pidfd.get()))
    return result(LinuxProcessRecoveryDisposition::already_gone,
                  "recorded process is already terminal");
  const std::string proc_path = "/proc/" + std::to_string(expected.host_pid);
  Descriptor process(
      ::open(proc_path.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (process.get() < 0)
    return result(LinuxProcessRecoveryDisposition::observation_failed,
                  "recorded process directory is unavailable");
  try {
    const std::uint64_t start_before = parse_starttime(read_file(process.get(), "stat"));
    const std::string cgroup_before = parse_cgroup(read_file(process.get(), "cgroup"));
    const std::string digest = executable_digest(process.get());
    const bool cgroup_identity = cgroup_identity_matches(expected);
    const std::uint64_t start_after = parse_starttime(read_file(process.get(), "stat"));
    const std::string cgroup_after = parse_cgroup(read_file(process.get(), "cgroup"));
    if (!pidfd_alive(pidfd.get()))
      return result(LinuxProcessRecoveryDisposition::already_gone,
                    "recorded process exited during observation");
    if (start_before != expected.process_starttime_ticks ||
        start_after != expected.process_starttime_ticks ||
        cgroup_before != expected.cgroup_path ||
        cgroup_after != expected.cgroup_path || !cgroup_identity ||
        digest != expected.executable_digest) {
      return result(LinuxProcessRecoveryDisposition::identity_mismatch,
                    "live process identity differs from the spawn receipt");
    }
    LinuxRecoveredProcess recovered_process(expected, pidfd.release());
    return LinuxProcessRecoveryResult(
        LinuxProcessRecoveryDisposition::exact_live_process,
        "exact live process identity recovered", std::move(recovered_process));
  } catch (const std::exception&) {
    return result(LinuxProcessRecoveryDisposition::observation_failed,
                  "process identity observation failed");
  }
#endif
}

namespace hostd_linux_process_recovery_test_seam {

std::uint64_t parse_proc_starttime(std::string_view value) {
  return parse_starttime(value);
}

std::string parse_unified_cgroup(std::string_view value) {
  return parse_cgroup(value);
}

}  // namespace hostd_linux_process_recovery_test_seam

}  // namespace trainvm
