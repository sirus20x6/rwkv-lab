#include "trainvm/hostd_linux_cgroup_authority.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <string_view>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

[[noreturn]] void reject(std::string message) {
  throw LinuxCgroupAuthorityError(std::move(message));
}

std::string system_error(std::string_view action) {
  return std::string(action) + ": " + std::strerror(errno);
}

bool canonical_absolute_path(std::string_view value) {
  if (value == "/") return true;
  if (value.empty() || value.size() > 4096U || value.front() != '/' ||
      value.back() == '/' || value.contains("//")) {
    return false;
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

int openat2_directory(int directory, const char* path,
                      std::uint64_t resolve) {
  struct open_how how {};
  how.flags = O_PATH | O_DIRECTORY | O_CLOEXEC;
  how.resolve = resolve;
  const long result = ::syscall(SYS_openat2, directory, path, &how,
                                sizeof(how));
  if (result < 0) reject(system_error("could not pin cgroup directory"));
  return static_cast<int>(result);
}

LinuxAllocationCgroupIdentity identity_of(int descriptor,
                                          std::string unified_path,
                                          uid_t owner_uid,
                                          gid_t owner_gid) {
  struct stat status {};
  struct statfs filesystem {};
  if (::fstat(descriptor, &status) != 0 || !S_ISDIR(status.st_mode) ||
      ::fstatfs(descriptor, &filesystem) != 0 ||
      filesystem.f_type != CGROUP2_SUPER_MAGIC || status.st_uid != owner_uid ||
      status.st_gid != owner_gid ||
      (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    reject("cgroup directory has unsafe identity, ownership, or filesystem");
  }
  return {.unified_path = std::move(unified_path),
          .device = static_cast<std::uint64_t>(status.st_dev),
          .inode = static_cast<std::uint64_t>(status.st_ino)};
}

bool cgroup_is_empty(int descriptor) {
  const int file =
      ::openat(descriptor, "cgroup.procs", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (file < 0) reject(system_error("could not inspect cgroup membership"));
  std::array<char, 2U> bytes{};
  ssize_t count = 0;
  do {
    count = ::read(file, bytes.data(), bytes.size());
  } while (count < 0 && errno == EINTR);
  const int saved_errno = errno;
  (void)::close(file);
  errno = saved_errno;
  if (count < 0) reject(system_error("could not read cgroup membership"));
  return count == 0;
}

std::string cgroup_name(const std::string& allocation_id,
                        const std::string& launch_id) {
  if (allocation_id.empty() || allocation_id.size() > 1024U ||
      launch_id.empty() || launch_id.size() > 1024U) {
    reject("allocation or launch ID is empty or unbounded");
  }
  std::string digest_input("trainvm.linux-allocation-cgroup/v1");
  digest_input += '\0';
  digest_input += allocation_id;
  digest_input += '\n';
  digest_input += launch_id;
  return "launch-" + sha256_hex(digest_input).substr(0U, 32U);
}

}  // namespace

struct LinuxCgroupAuthority::Implementation final {
  LinuxCgroupAuthorityConfig config;
  int root{-1};
  LinuxAllocationCgroupIdentity root_identity;

  ~Implementation() {
    if (root >= 0) (void)::close(root);
  }

  void reattest() const {
    const int reopened = openat2_directory(
        AT_FDCWD, config.root_path.c_str(),
        RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS);
    struct Close final {
      int value;
      ~Close() { (void)::close(value); }
    } close{reopened};
    const auto observed = identity_of(reopened, config.root_unified_path,
                                      config.expected_owner_uid,
                                      config.expected_owner_gid);
    if (observed != root_identity) {
      reject("cgroup authority root identity changed");
    }
  }
};

LinuxAllocationCgroup::LinuxAllocationCgroup(
    LinuxAllocationCgroupIdentity identity, int descriptor,
    int parent_descriptor, std::string name, bool remove_on_destroy) noexcept
    : identity_(std::move(identity)), descriptor_(descriptor),
      parent_descriptor_(parent_descriptor), name_(std::move(name)),
      remove_on_destroy_(remove_on_destroy) {}

LinuxAllocationCgroup::LinuxAllocationCgroup(
    LinuxAllocationCgroup&& other) noexcept
    : identity_(std::move(other.identity_)),
      descriptor_(std::exchange(other.descriptor_, -1)),
      parent_descriptor_(std::exchange(other.parent_descriptor_, -1)),
      name_(std::move(other.name_)),
      remove_on_destroy_(other.remove_on_destroy_) {
  other.remove_on_destroy_ = false;
}

LinuxAllocationCgroup& LinuxAllocationCgroup::operator=(
    LinuxAllocationCgroup&& other) noexcept {
  if (this != &other) {
    close_and_maybe_remove();
    identity_ = std::move(other.identity_);
    descriptor_ = std::exchange(other.descriptor_, -1);
    parent_descriptor_ = std::exchange(other.parent_descriptor_, -1);
    name_ = std::move(other.name_);
    remove_on_destroy_ = other.remove_on_destroy_;
    other.remove_on_destroy_ = false;
  }
  return *this;
}

LinuxAllocationCgroup::~LinuxAllocationCgroup() {
  close_and_maybe_remove();
}

void LinuxAllocationCgroup::close_and_maybe_remove() noexcept {
  if (descriptor_ >= 0) (void)::close(descriptor_);
  descriptor_ = -1;
  if (remove_on_destroy_ && parent_descriptor_ >= 0 && !name_.empty()) {
    (void)::unlinkat(parent_descriptor_, name_.c_str(), AT_REMOVEDIR);
  }
  if (parent_descriptor_ >= 0) (void)::close(parent_descriptor_);
  parent_descriptor_ = -1;
  remove_on_destroy_ = false;
}

const LinuxAllocationCgroupIdentity& LinuxAllocationCgroup::identity() const {
  return identity_;
}

int LinuxAllocationCgroup::duplicate_fd() const {
  const int duplicate = ::fcntl(descriptor_, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0) reject(system_error("could not duplicate cgroup fd"));
  return duplicate;
}

bool LinuxAllocationCgroup::empty() const {
  if (descriptor_ < 0) reject("allocation cgroup descriptor is unavailable");
  return cgroup_is_empty(descriptor_) && cgroup_is_empty(descriptor_);
}

void LinuxAllocationCgroup::remove_if_empty() {
  struct stat status {};
  if (descriptor_ < 0 || parent_descriptor_ < 0 || name_.empty() ||
      ::fstat(descriptor_, &status) != 0 ||
      static_cast<std::uint64_t>(status.st_dev) != identity_.device ||
      static_cast<std::uint64_t>(status.st_ino) != identity_.inode ||
      !empty()) {
    reject("allocation cgroup is not identically pinned and empty");
  }
  if (::unlinkat(parent_descriptor_, name_.c_str(), AT_REMOVEDIR) != 0) {
    reject(system_error("could not remove terminal allocation cgroup"));
  }
  (void)::close(descriptor_);
  descriptor_ = -1;
  name_.clear();
  remove_on_destroy_ = false;
}

void LinuxAllocationCgroup::retain_for_durable_intent() noexcept {
  remove_on_destroy_ = false;
}

LinuxCgroupAuthority::LinuxCgroupAuthority(LinuxCgroupAuthorityConfig config)
    : implementation_(std::make_unique<Implementation>()) {
  if (config.root_path.empty() || !config.root_path.is_absolute() ||
      config.root_path.lexically_normal() != config.root_path ||
      !canonical_absolute_path(config.root_unified_path)) {
    reject("cgroup authority configuration is not canonical");
  }
  implementation_->config = std::move(config);
  implementation_->root = openat2_directory(
      AT_FDCWD, implementation_->config.root_path.c_str(),
      RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS);
  implementation_->root_identity = identity_of(
      implementation_->root, implementation_->config.root_unified_path,
      implementation_->config.expected_owner_uid,
      implementation_->config.expected_owner_gid);
}

LinuxCgroupAuthority::~LinuxCgroupAuthority() = default;

LinuxAllocationCgroup LinuxCgroupAuthority::open_or_create(
    const std::string& allocation_id, const std::string& launch_id) const {
  implementation_->reattest();
  const std::string name = cgroup_name(allocation_id, launch_id);
  bool created = false;
  if (::mkdirat(implementation_->root, name.c_str(), 0750) == 0) {
    created = true;
  } else if (errno != EEXIST) {
    reject(system_error("could not create allocation cgroup"));
  }
  int descriptor = -1;
  try {
    descriptor = openat2_directory(
        implementation_->root, name.c_str(),
        RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS |
            RESOLVE_NO_XDEV);
    const std::string unified_path =
        implementation_->config.root_unified_path == "/"
            ? "/" + name
            : implementation_->config.root_unified_path + "/" + name;
    auto identity = identity_of(
        descriptor, unified_path,
        implementation_->config.expected_owner_uid,
        implementation_->config.expected_owner_gid);
    if (!cgroup_is_empty(descriptor)) {
      reject("existing allocation cgroup is nonempty and requires audit");
    }
    const int parent =
        ::fcntl(implementation_->root, F_DUPFD_CLOEXEC, 3);
    if (parent < 0) reject(system_error("could not retain cgroup parent"));
    return LinuxAllocationCgroup(std::move(identity), descriptor, parent, name,
                                 created);
  } catch (...) {
    if (descriptor >= 0) (void)::close(descriptor);
    if (created) {
      (void)::unlinkat(implementation_->root, name.c_str(), AT_REMOVEDIR);
    }
    throw;
  }
}

LinuxAllocationCgroup LinuxCgroupAuthority::open_existing_for_recovery(
    const std::string& allocation_id, const std::string& launch_id,
    const LinuxAllocationCgroupIdentity& expected) const {
  implementation_->reattest();
  const std::string name = cgroup_name(allocation_id, launch_id);
  int descriptor = -1;
  try {
    descriptor = openat2_directory(
        implementation_->root, name.c_str(),
        RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS |
            RESOLVE_NO_XDEV);
    const std::string unified_path =
        implementation_->config.root_unified_path == "/"
            ? "/" + name
            : implementation_->config.root_unified_path + "/" + name;
    auto identity = identity_of(
        descriptor, unified_path,
        implementation_->config.expected_owner_uid,
        implementation_->config.expected_owner_gid);
    if (identity != expected) {
      reject("recovery cgroup identity differs from durable spawn evidence");
    }
    const int parent = ::fcntl(implementation_->root, F_DUPFD_CLOEXEC, 3);
    if (parent < 0) reject(system_error("could not retain cgroup parent"));
    return LinuxAllocationCgroup(std::move(identity), descriptor, parent, name,
                                 false);
  } catch (...) {
    if (descriptor >= 0) (void)::close(descriptor);
    throw;
  }
}

LinuxTerminalCgroupCleanupDisposition
LinuxCgroupAuthority::cleanup_terminal_or_confirm_absent(
    const std::string& allocation_id, const std::string& launch_id,
    const LinuxAllocationCgroupIdentity& expected) const {
  implementation_->reattest();
  const std::string name = cgroup_name(allocation_id, launch_id);
  struct open_how how {};
  how.flags = O_PATH | O_DIRECTORY | O_CLOEXEC;
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
                RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV;
  const long opened = ::syscall(SYS_openat2, implementation_->root,
                                name.c_str(), &how, sizeof(how));
  if (opened < 0) {
    if (errno == ENOENT)
      return LinuxTerminalCgroupCleanupDisposition::already_absent;
    reject(system_error("could not pin terminal recovery cgroup"));
  }
  const int descriptor = static_cast<int>(opened);
  struct Close final {
    int value;
    ~Close() {
      if (value >= 0) (void)::close(value);
    }
  } close{descriptor};
  const std::string unified_path =
      implementation_->config.root_unified_path == "/"
          ? "/" + name
          : implementation_->config.root_unified_path + "/" + name;
  const LinuxAllocationCgroupIdentity identity = identity_of(
      descriptor, unified_path, implementation_->config.expected_owner_uid,
      implementation_->config.expected_owner_gid);
  if (identity != expected || !cgroup_is_empty(descriptor) ||
      !cgroup_is_empty(descriptor)) {
    reject("terminal recovery cgroup is mismatched or nonempty");
  }
  if (::unlinkat(implementation_->root, name.c_str(), AT_REMOVEDIR) != 0) {
    reject(system_error("could not remove terminal recovery cgroup"));
  }
  return LinuxTerminalCgroupCleanupDisposition::removed;
}

namespace hostd_linux_cgroup_authority_test_seam {

std::string allocation_cgroup_name(const std::string& allocation_id,
                                   const std::string& launch_id) {
  return cgroup_name(allocation_id, launch_id);
}

}  // namespace hostd_linux_cgroup_authority_test_seam

}  // namespace trainvm
