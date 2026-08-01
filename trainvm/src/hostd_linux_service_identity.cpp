#include "trainvm/hostd_linux_service_identity.hpp"

#include <fcntl.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <ranges>
#include <utility>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumRoles = 256U;
constexpr std::size_t kMaximumIdentityBytes = 192U;

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

[[noreturn]] void reject(std::string_view message) {
  throw HostdLinuxServiceIdentityError(std::string(message));
}

pid_t current_tid() noexcept {
  const long value = ::syscall(SYS_gettid);
  if (value <= 0 || static_cast<unsigned long>(value) >
                        static_cast<unsigned long>(
                            std::numeric_limits<pid_t>::max()))
    return -1;
  return static_cast<pid_t>(value);
}

bool valid_identity(std::string_view value) {
  return !value.empty() && value.size() <= kMaximumIdentityBytes &&
         std::ranges::all_of(value, [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') ||
                  character == '.' || character == '_' || character == '-';
         });
}

bool valid_cgroup_path(std::string_view value) {
  if (value.empty() || value.size() > 2048U || value.front() != '/' ||
      (value.size() > 1U && value.back() == '/') ||
      value.find("//") != std::string_view::npos ||
      value.find('\0') != std::string_view::npos)
    return false;
  std::size_t begin = 1U;
  while (begin < value.size()) {
    const std::size_t end = value.find('/', begin);
    const std::string_view part = value.substr(
        begin, (end == std::string_view::npos ? value.size() : end) - begin);
    if (part.empty() || part == "." || part == ".." ||
        !std::ranges::all_of(part, [](unsigned char character) {
          return (character >= 'a' && character <= 'z') ||
                 (character >= 'A' && character <= 'Z') ||
                 (character >= '0' && character <= '9') ||
                 character == '.' || character == '_' || character == '-' ||
                 character == ':' || character == '@' || character == '\\';
        }))
      return false;
    if (end == std::string_view::npos)
      break;
    begin = end + 1U;
  }
  return true;
}

std::optional<std::string> parse_cgroup(std::string_view value,
                                        std::size_t maximum) {
  if (value.empty() || value.size() > maximum)
    return std::nullopt;
  if (value.back() == '\n')
    value.remove_suffix(1U);
  if (value.empty() || value.find('\n') != std::string_view::npos ||
      !value.starts_with("0::"))
    return std::nullopt;
  const std::string_view path = value.substr(3U);
  return valid_cgroup_path(path)
             ? std::optional<std::string>(std::string(path))
             : std::nullopt;
}

int open_beneath(int root, std::string_view relative, std::uint64_t flags) {
  if (relative.empty() || relative.starts_with('/') ||
      relative.find('\0') != std::string_view::npos)
    return -1;
  open_how how{};
  how.flags = flags | O_CLOEXEC | O_NOFOLLOW;
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS |
                RESOLVE_NO_SYMLINKS | RESOLVE_NO_XDEV;
  return static_cast<int>(::syscall(SYS_openat2, root,
                                    std::string(relative).c_str(), &how,
                                    sizeof(how)));
}

std::optional<std::string> read_file_at(int directory,
                                        std::string_view relative,
                                        std::size_t maximum) {
  Descriptor file(open_beneath(directory, relative, O_RDONLY));
  if (file.get() < 0)
    return std::nullopt;
  struct stat before{};
  if (::fstat(file.get(), &before) != 0 || !S_ISREG(before.st_mode))
    return std::nullopt;
  std::string result;
  std::array<char, 1024U> buffer{};
  for (;;) {
    const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0)
      return std::nullopt;
    if (count == 0)
      break;
    if (result.size() + static_cast<std::size_t>(count) > maximum)
      return std::nullopt;
    result.append(buffer.data(), static_cast<std::size_t>(count));
  }
  struct stat after{};
  if (::fstat(file.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino)
    return std::nullopt;
  return result;
}

bool same_inode(const struct stat &left, const struct stat &right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_mode == right.st_mode && left.st_uid == right.st_uid &&
         left.st_gid == right.st_gid;
}

struct PinnedRole final {
  HostdLinuxServiceRole role;
  Descriptor directory;
  struct stat identity{};
};

} // namespace

struct HostdLinuxServiceIdentityAuthority::Implementation final {
  HostdLinuxServiceIdentityConfig config;
  pid_t creator_pid{};
  pid_t creator_tid{};
  Descriptor proc;
  Descriptor cgroup;
  std::map<std::string, PinnedRole> roles;

  void attest_task() const {
    if (::getpid() != creator_pid || current_tid() != creator_tid)
      reject("Linux service identity authority crossed process or thread");
  }

  Descriptor open_role(std::string_view path) const {
    if (path == "/") {
      const int duplicate = ::fcntl(cgroup.get(), F_DUPFD_CLOEXEC, 3);
      return Descriptor(duplicate);
    }
    return Descriptor(open_beneath(cgroup.get(), path.substr(1U),
                                   O_PATH | O_DIRECTORY));
  }

  void reattest_role(const PinnedRole &pinned) const {
    Descriptor reopened = open_role(pinned.role.cgroup_path);
    struct stat current{};
    struct stat held{};
    if (reopened.get() < 0 || ::fstat(reopened.get(), &current) != 0 ||
        ::fstat(pinned.directory.get(), &held) != 0 ||
        !same_inode(current, pinned.identity) ||
        !same_inode(held, pinned.identity))
      reject("configured service cgroup identity changed");
  }
};

HostdLinuxServiceIdentityAuthority::HostdLinuxServiceIdentityAuthority(
    HostdLinuxServiceIdentityConfig config)
    : implementation_(std::make_unique<Implementation>()) {
  if (config.api_version != kHostdLinuxServiceIdentityApiVersion ||
      config.roles.empty() || config.roles.size() > kMaximumRoles ||
      config.maximum_cgroup_file_bytes < 4U ||
      config.maximum_cgroup_file_bytes > 64U * 1024U)
    reject("Linux service identity configuration is invalid");
  implementation_->creator_pid = ::getpid();
  implementation_->creator_tid = current_tid();
  implementation_->proc =
      Descriptor(::open("/proc", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  implementation_->cgroup = Descriptor(
      ::open("/sys/fs/cgroup", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  struct statfs procfs{};
  struct statfs cgroupfs{};
  struct stat proc_status{};
  struct stat cgroup_status{};
  if (implementation_->creator_pid <= 0 || implementation_->creator_tid <= 0 ||
      implementation_->proc.get() < 0 || implementation_->cgroup.get() < 0 ||
      ::fstatfs(implementation_->proc.get(), &procfs) != 0 ||
      procfs.f_type != PROC_SUPER_MAGIC ||
      ::fstatfs(implementation_->cgroup.get(), &cgroupfs) != 0 ||
      cgroupfs.f_type != CGROUP2_SUPER_MAGIC ||
      ::fstat(implementation_->proc.get(), &proc_status) != 0 ||
      ::fstat(implementation_->cgroup.get(), &cgroup_status) != 0 ||
      proc_status.st_uid != 0U || cgroup_status.st_uid != 0U)
    reject("host procfs or cgroup-v2 authority is unavailable");
  implementation_->config = std::move(config);
  for (const auto &role : implementation_->config.roles) {
    if (!valid_cgroup_path(role.cgroup_path) ||
        !valid_identity(role.service_identity) ||
        (role.access != HostdSessionAccess::grant_release &&
         role.access != HostdSessionAccess::release_only) ||
        implementation_->roles.contains(role.cgroup_path))
      reject("Linux service role is invalid or duplicated");
    Descriptor directory = implementation_->open_role(role.cgroup_path);
    struct stat identity{};
    if (directory.get() < 0 || ::fstat(directory.get(), &identity) != 0 ||
        !S_ISDIR(identity.st_mode) || identity.st_uid != 0U)
      reject("configured service cgroup cannot be pinned");
    implementation_->roles.emplace(
        role.cgroup_path,
        PinnedRole{.role = role,
                   .directory = std::move(directory),
                   .identity = identity});
  }
}

HostdLinuxServiceIdentityAuthority::~HostdLinuxServiceIdentityAuthority() =
    default;

HostdMutationServiceAuthorization
HostdLinuxServiceIdentityAuthority::authorize(
    const HostdSocketPeerInstance &peer) {
  implementation_->attest_task();
  if (peer.pid <= 0 || peer.process_starttime_ticks == 0U)
    reject("service peer process identity is invalid");
  const std::string pid = std::to_string(peer.pid);
  Descriptor process(open_beneath(implementation_->proc.get(), pid,
                                  O_PATH | O_DIRECTORY));
  struct stat before{};
  struct stat after{};
  if (process.get() < 0 || ::fstat(process.get(), &before) != 0)
    reject("service peer proc identity is unavailable");
  const auto stat_before = read_file_at(process.get(), "stat", 4096U);
  const auto cgroup_before = read_file_at(
      process.get(), "cgroup", implementation_->config.maximum_cgroup_file_bytes);
  const auto cgroup_after = read_file_at(
      process.get(), "cgroup", implementation_->config.maximum_cgroup_file_bytes);
  const auto stat_after = read_file_at(process.get(), "stat", 4096U);
  if (!stat_before || !stat_after || !cgroup_before || !cgroup_after ||
      ::fstat(process.get(), &after) != 0 || !same_inode(before, after))
    reject("service peer proc evidence is torn");
  const auto start_before =
      hostd_linux_session_test_seam::parse_proc_stat_starttime(*stat_before,
                                                               peer.pid);
  const auto start_after =
      hostd_linux_session_test_seam::parse_proc_stat_starttime(*stat_after,
                                                               peer.pid);
  const auto path_before = parse_cgroup(
      *cgroup_before, implementation_->config.maximum_cgroup_file_bytes);
  const auto path_after = parse_cgroup(
      *cgroup_after, implementation_->config.maximum_cgroup_file_bytes);
  if (!start_before || !start_after ||
      *start_before != peer.process_starttime_ticks ||
      *start_after != peer.process_starttime_ticks || !path_before ||
      path_before != path_after)
    reject("service peer cgroup or process instance is inexact");
  const auto role = implementation_->roles.find(*path_before);
  if (role == implementation_->roles.end() ||
      role->second.role.expected_uid != peer.uid ||
      role->second.role.expected_gid != peer.gid)
    reject("service peer has no configured cgroup role");
  implementation_->reattest_role(role->second);
  return {.service_identity = role->second.role.service_identity,
          .access = role->second.role.access,
          .service_identity_enforced = true};
}

namespace hostd_linux_service_identity_test_seam {

std::optional<std::string>
parse_unified_cgroup_path(std::string_view value, std::size_t maximum_bytes) {
  return parse_cgroup(value, maximum_bytes);
}

} // namespace hostd_linux_service_identity_test_seam

} // namespace trainvm
