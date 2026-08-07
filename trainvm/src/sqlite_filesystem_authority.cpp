#include "trainvm/sqlite_filesystem_authority.hpp"

#include <sqlite3.h>

#include <dirent.h>
#include <fcntl.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumPathBytes = 4096U;
constexpr std::size_t kMaximumFilenameBytes = 128U;
constexpr unsigned int kProtectedDirectoryMode = 0700U;
constexpr unsigned int kProtectedFileMode = 0600U;
constexpr uid_t kNobodyUid = 65534U;

class FileDescriptor final {
public:
  explicit FileDescriptor(int value = -1) noexcept : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }
  FileDescriptor(FileDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }

private:
  int value_;
};

[[noreturn]] void throw_errno(std::string_view action) {
  const int error = errno;
  throw SqliteAuthorityError(std::string(action) + ": " + std::strerror(error));
}

bool safe_filename(std::string_view value) {
  if (value.empty() || value.size() > kMaximumFilenameBytes || value == "." ||
      value == "..") {
    return false;
  }
  for (const char character : value) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    if (!alphanumeric && character != '.' && character != '_' &&
        character != '-') {
      return false;
    }
  }
  return true;
}

SqliteFileIdentity identity(const struct stat &status) {
  return {.device = static_cast<std::uint64_t>(status.st_dev),
          .inode = static_cast<std::uint64_t>(status.st_ino),
          .size = status.st_size < 0
                      ? 0U
                      : static_cast<std::uint64_t>(status.st_size),
          .mode = static_cast<std::uint32_t>(status.st_mode),
          .owner_uid = static_cast<std::uint32_t>(status.st_uid),
          .owner_gid = static_cast<std::uint32_t>(status.st_gid),
          .link_count = static_cast<std::uint64_t>(status.st_nlink)};
}

bool same_pinned_inode(const SqliteFileIdentity &left,
                       const SqliteFileIdentity &right) {
  return left.device == right.device && left.inode == right.inode &&
         left.mode == right.mode && left.owner_uid == right.owner_uid &&
         left.owner_gid == right.owner_gid &&
         left.link_count == right.link_count;
}

int open_beneath_raw(int directory, std::string_view name, std::uint64_t flags,
                     mode_t mode = 0U) {
  const std::string owned(name);
  struct open_how how{};
  how.flags = flags;
  how.mode = mode;
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS;
  const long result =
      ::syscall(SYS_openat2, directory, owned.c_str(), &how, sizeof(how));
  if (result < 0)
    return -1;
  if (static_cast<unsigned long>(result) >
      static_cast<unsigned long>(std::numeric_limits<int>::max())) {
    (void)::close(static_cast<int>(result));
    errno = EMFILE;
    return -1;
  }
  return static_cast<int>(result);
}

FileDescriptor open_beneath(int directory, std::string_view name,
                            std::uint64_t flags, mode_t mode = 0U) {
  const int result = open_beneath_raw(directory, name, flags, mode);
  if (result < 0) {
    const int error = errno;
    const std::string owned(name);
    errno = error;
    throw_errno("secure openat2 failed for " + owned);
  }
  return FileDescriptor(result);
}

void require_directory_policy(const struct stat &status,
                              const SqliteAuthorityConfig &config,
                              bool final_directory) {
  if (!S_ISDIR(status.st_mode)) {
    throw SqliteAuthorityError("authority ancestry contains a non-directory");
  }
  const auto permissions = static_cast<unsigned int>(status.st_mode) & 07777U;
  const bool root_owned = status.st_uid == 0U;
  const bool authority_owned = status.st_uid == config.expected_owner_uid;
  const bool cooperative_sandbox_ancestor =
      !final_directory &&
      config.enforcement_grade ==
          SqliteAuthorityEnforcementGrade::cooperative_test &&
      status.st_uid == kNobodyUid;
  if (!root_owned && !authority_owned && !cooperative_sandbox_ancestor) {
    throw SqliteAuthorityError("authority ancestry has an unexpected owner");
  }
  if (final_directory) {
    if (!authority_owned || status.st_gid != config.expected_owner_gid) {
      throw SqliteAuthorityError(
          "authority directory must be owner-controlled mode 0700");
    }
    if (permissions == kProtectedDirectoryMode)
      return;
    // strict_filesystem is the deployed grade and requires the exact 0700
    // authority directory. cooperative_test is diagnostic isolation only, never
    // a boundary against a same-UID process, so it accepts an owner-controlled
    // directory that is merely not group- or world-writable. Relaxing it here
    // keeps hermetic tests runnable under a default-mode temporary directory
    // without loosening any deployed check.
    if (config.enforcement_grade !=
            SqliteAuthorityEnforcementGrade::cooperative_test ||
        (permissions & 0022U) != 0U) {
      throw SqliteAuthorityError(
          "authority directory must be owner-controlled mode 0700");
    }
    return;
  }
  const bool group_or_world_writable = (permissions & 0022U) != 0U;
  if (!group_or_world_writable)
    return;
  const bool sticky_root_directory =
      (root_owned || cooperative_sandbox_ancestor) &&
      (permissions & static_cast<unsigned int>(S_ISVTX)) != 0U;
  if (config.enforcement_grade ==
          SqliteAuthorityEnforcementGrade::cooperative_test &&
      sticky_root_directory) {
    return;
  }
  throw SqliteAuthorityError(
      "authority ancestry is writable outside the authority owner");
}

struct OpenedParent final {
  FileDescriptor descriptor;
  SqliteFileIdentity identity;
};

OpenedParent open_authority_parent(const SqliteAuthorityConfig &config) {
  const auto parent = config.ledger_path.parent_path();
  FileDescriptor current(
      ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (current.get() < 0)
    throw_errno("could not open filesystem root");
  struct stat root_status{};
  if (::fstat(current.get(), &root_status) != 0) {
    throw_errno("could not inspect filesystem root");
  }
  require_directory_policy(root_status, config, parent == "/");

  const auto relative = parent.relative_path();
  std::size_t component_index = 0U;
  const std::size_t component_count =
      static_cast<std::size_t>(std::distance(relative.begin(), relative.end()));
  for (const auto &component_path : relative) {
    const std::string component = component_path.string();
    if (!safe_filename(component)) {
      throw SqliteAuthorityError(
          "authority path contains a noncanonical component");
    }
    auto next =
        open_beneath(current.get(), component,
                     static_cast<std::uint64_t>(O_RDONLY | O_DIRECTORY |
                                                O_CLOEXEC | O_NOFOLLOW));
    struct stat status{};
    if (::fstat(next.get(), &status) != 0) {
      throw_errno("could not inspect authority path component");
    }
    ++component_index;
    require_directory_policy(status, config,
                             component_index == component_count);
    current = std::move(next);
  }
  struct stat final_status{};
  if (::fstat(current.get(), &final_status) != 0) {
    throw_errno("could not inspect authority directory");
  }
  return {.descriptor = std::move(current), .identity = identity(final_status)};
}

std::string filesystem_name(long filesystem_type,
                            SqliteAuthorityEnforcementGrade grade) {
  switch (filesystem_type) {
  case EXT4_SUPER_MAGIC:
    return "ext-family";
  case XFS_SUPER_MAGIC:
    return "xfs";
  case BTRFS_SUPER_MAGIC:
    return "btrfs";
  case 0xF2F52010L:
    return "f2fs";
  case 0x2FC12FC1L:
    return "zfs";
  case TMPFS_MAGIC:
    if (grade == SqliteAuthorityEnforcementGrade::cooperative_test)
      return "tmpfs";
    break;
  case RAMFS_MAGIC:
    if (grade == SqliteAuthorityEnforcementGrade::cooperative_test)
      return "ramfs";
    break;
  case OVERLAYFS_SUPER_MAGIC:
    if (grade == SqliteAuthorityEnforcementGrade::cooperative_test) {
      return "overlayfs";
    }
    break;
  default:
    break;
  }
  throw SqliteAuthorityError(
      "authority directory is not on a supported local filesystem");
}

// Best-effort listing of every name in the authority directory that resolves to
// `inode`. Purely diagnostic: it runs only on the failure path, and any error
// reading the directory yields no suffix rather than masking the real fault.
std::string names_sharing_inode(int parent_descriptor, ino_t inode) {
  if (parent_descriptor < 0) return "";
  const int duplicated = ::dup(parent_descriptor);
  if (duplicated < 0) return "";
  DIR *directory = ::fdopendir(duplicated);
  if (directory == nullptr) {
    ::close(duplicated);
    return "";
  }
  std::string names;
  ::rewinddir(directory);
  while (const dirent *entry = ::readdir(directory)) {
    if (entry->d_ino != inode) continue;
    if (std::string_view(entry->d_name) == "." ||
        std::string_view(entry->d_name) == "..") {
      continue;
    }
    names += names.empty() ? "" : ", ";
    names += entry->d_name;
  }
  ::closedir(directory);
  return names.empty() ? "" : ("; names in this directory sharing inode " +
                               std::to_string(
                                   static_cast<unsigned long long>(inode)) +
                               ": " + names);
}

// Four independent properties, reported separately. They used to share one
// message -- "must be an owned 0600 singleton regular file" -- which named
// ownership and permissions whatever had actually gone wrong, sending readers
// after a permissions bug when the real violation was a link count or a file
// type. A diagnostic that does not say which condition failed, or what it saw
// instead, is a measurement that looks valid while describing the wrong thing.
void validate_regular_file(const struct stat &status,
                           const SqliteAuthorityConfig &config,
                           std::string_view description,
                           int parent_descriptor = -1) {
  const auto prefix = std::string(description) + " ";
  if (!S_ISREG(status.st_mode)) {
    throw SqliteAuthorityError(
        prefix + "must be a regular file, but its mode is " +
        std::to_string(static_cast<unsigned int>(status.st_mode) & S_IFMT) +
        " (S_IFMT); a symlink, directory or device was substituted");
  }
  if (status.st_nlink != 1) {
    // Name the other links rather than only counting them. "another name
    // refers to these same bytes" tells a reader nothing actionable; which
    // name does, and it is the difference between diagnosing a planted alias
    // and diagnosing a transient one this process created itself.
    std::string aliases = names_sharing_inode(parent_descriptor, status.st_ino);
    throw SqliteAuthorityError(
        prefix + "must have exactly one link, but has " +
        std::to_string(static_cast<unsigned long long>(status.st_nlink)) +
        "; the authority directory does not solely control these bytes" +
        aliases);
  }
  if (status.st_uid != config.expected_owner_uid ||
      status.st_gid != config.expected_owner_gid) {
    throw SqliteAuthorityError(
        prefix + "must be owned by uid " +
        std::to_string(config.expected_owner_uid) + " gid " +
        std::to_string(config.expected_owner_gid) + ", but is owned by uid " +
        std::to_string(status.st_uid) + " gid " + std::to_string(status.st_gid));
  }
  const auto mode = static_cast<unsigned int>(status.st_mode) & 07777U;
  if (mode != kProtectedFileMode) {
    // Octal, because that is how the expectation is written and how anybody
    // reading this will compare it against `ls -l`.
    std::array<char, 8> observed{};
    std::snprintf(observed.data(), observed.size(), "%04o", mode);
    throw SqliteAuthorityError(
        prefix + "must be mode 0600, but is mode " + observed.data());
  }
}

// Narrow an authority file that is already owner-owned, regular, unaliased, and
// not group- or world-writable down to the protected 0600 mode, so that the
// strict validation above accepts it. This only ever removes access, so it can
// never widen a planted file, and anything failing those prior properties is
// rejected rather than repaired. A database adopted from a run that predates
// this policy is 0644 under the usual umask, and stock SQLite derives -wal and
// -journal modes from the database file, so normalizing the database at
// adoption also normalizes every auxiliary SQLite subsequently creates.

struct OpenedFile final {
  FileDescriptor descriptor;
  SqliteFileIdentity identity;
};

OpenedFile create_or_open_file(int parent, std::string_view name,
                               const SqliteAuthorityConfig &config) {
  bool created = false;
  FileDescriptor path_descriptor;
  int opened =
      open_beneath_raw(parent, name,
                       static_cast<std::uint64_t>(O_RDWR | O_CREAT | O_EXCL |
                                                  O_CLOEXEC | O_NOFOLLOW),
                       kProtectedFileMode);
  if (opened >= 0) {
    created = true;
  } else {
    const int create_error = errno;
    if (create_error != EEXIST) {
      errno = create_error;
      throw_errno("could not securely create authority file");
    }
    path_descriptor = open_beneath(
        parent, name,
        static_cast<std::uint64_t>(O_PATH | O_CLOEXEC | O_NOFOLLOW));
    struct stat path_descriptor_status{};
    if (::fstat(path_descriptor.get(), &path_descriptor_status) != 0) {
      throw_errno("could not inspect existing authority file");
    }
    validate_regular_file(path_descriptor_status, config, name);
    opened = open_beneath_raw(
        parent, name,
        static_cast<std::uint64_t>(O_RDWR | O_CLOEXEC | O_NOFOLLOW));
    if (opened < 0)
      throw_errno("could not securely open authority file");
  }
  FileDescriptor descriptor(opened);
  if (created) {
    struct stat created_status{};
    if (::fstat(descriptor.get(), &created_status) != 0) {
      throw_errno("could not inspect newly created authority file");
    }
    if (created_status.st_uid != config.expected_owner_uid ||
        created_status.st_gid != config.expected_owner_gid) {
      if (::fchown(descriptor.get(), config.expected_owner_uid,
                   config.expected_owner_gid) != 0) {
        throw_errno("could not assign authority file ownership");
      }
    }
    if (::fchmod(descriptor.get(), kProtectedFileMode) != 0 ||
        ::fsync(descriptor.get()) != 0 || ::fsync(parent) != 0) {
      throw_errno("could not durably protect newly created authority file");
    }
  }
  struct stat descriptor_status{};
  if (::fstat(descriptor.get(), &descriptor_status) != 0) {
    throw_errno("could not inspect authority file");
  }
  validate_regular_file(descriptor_status, config, name);
  if (path_descriptor.get() >= 0) {
    struct stat path_descriptor_status{};
    if (::fstat(path_descriptor.get(), &path_descriptor_status) != 0) {
      throw_errno("could not re-inspect existing authority file");
    }
    if (!same_pinned_inode(identity(path_descriptor_status),
                           identity(descriptor_status))) {
      throw SqliteAuthorityError(
          "authority file changed while it was being opened");
    }
  }
  struct stat path_status{};
  const std::string owned_name(name);
  if (::fstatat(parent, owned_name.c_str(), &path_status,
                AT_SYMLINK_NOFOLLOW) != 0) {
    throw_errno("could not re-inspect authority file path");
  }
  if (!same_pinned_inode(identity(descriptor_status), identity(path_status))) {
    throw SqliteAuthorityError(
        "authority file changed while it was being opened");
  }
  return {.descriptor = std::move(descriptor),
          .identity = identity(descriptor_status)};
}

} // namespace

struct SqliteFilesystemAuthority::Implementation final {
  SqliteAuthorityConfig config;
  std::string database_name;
  std::string lock_name;
  FileDescriptor parent;
  FileDescriptor database;
  FileDescriptor lock;
  SqliteFileIdentity parent_identity;
  SqliteFileIdentity database_identity;
  SqliteFileIdentity lock_identity;
  std::string filesystem;
  bool protected_filesystem_boundary{};
  std::shared_ptr<SqliteAuthorityVfs> sqlite_vfs;
  mutable std::mutex mutex;

  SqliteAuthorityAttestation attest() const {
    auto reopened_parent = open_authority_parent(config);
    struct stat held_parent_status{};
    struct stat held_database_status{};
    struct stat held_lock_status{};
    if (::fstat(parent.get(), &held_parent_status) != 0 ||
        ::fstat(database.get(), &held_database_status) != 0 ||
        ::fstat(lock.get(), &held_lock_status) != 0) {
      throw_errno("could not inspect pinned authority descriptors");
    }
    validate_regular_file(held_database_status, config, "database file");
    validate_regular_file(held_lock_status, config, "lock file");
    if (!same_pinned_inode(parent_identity, identity(held_parent_status)) ||
        !same_pinned_inode(parent_identity, reopened_parent.identity) ||
        !same_pinned_inode(database_identity, identity(held_database_status)) ||
        !same_pinned_inode(lock_identity, identity(held_lock_status))) {
      throw SqliteAuthorityError("a pinned authority inode changed");
    }
    auto reopened_database = open_beneath(
        reopened_parent.descriptor.get(), database_name,
        static_cast<std::uint64_t>(O_RDWR | O_CLOEXEC | O_NOFOLLOW));
    auto reopened_lock = open_beneath(
        reopened_parent.descriptor.get(), lock_name,
        static_cast<std::uint64_t>(O_RDWR | O_CLOEXEC | O_NOFOLLOW));
    struct stat reopened_database_status{};
    struct stat reopened_lock_status{};
    if (::fstat(reopened_database.get(), &reopened_database_status) != 0 ||
        ::fstat(reopened_lock.get(), &reopened_lock_status) != 0) {
      throw_errno("could not inspect reopened authority files");
    }
    validate_regular_file(reopened_database_status, config, "database file");
    validate_regular_file(reopened_lock_status, config, "lock file");
    if (!same_pinned_inode(database_identity,
                           identity(reopened_database_status)) ||
        !same_pinned_inode(lock_identity, identity(reopened_lock_status))) {
      throw SqliteAuthorityError("authority pathname inode was replaced");
    }
    return {.api_version = std::string(kSqliteAuthorityApiVersion),
            .ledger_path = config.ledger_path,
            .enforcement_grade = config.enforcement_grade,
            .protected_filesystem_boundary = protected_filesystem_boundary,
            .filesystem_name = filesystem,
            .authority_directory = identity(held_parent_status),
            .database_file = identity(held_database_status),
            .lock_file = identity(held_lock_status)};
  }
};

SqliteFilesystemAuthority
SqliteFilesystemAuthority::acquire(SqliteAuthorityConfig config) {
  if (sqlite3_libversion_number() < kSqliteAuthorityMinimumVersionNumber) {
    throw SqliteAuthorityError(
        "SQLite is older than the auxiliary-path validated-at floor 3.53.3");
  }
  const std::string native_path = config.ledger_path.string();
  const bool grade_is_valid =
      config.enforcement_grade ==
          SqliteAuthorityEnforcementGrade::strict_filesystem ||
      config.enforcement_grade ==
          SqliteAuthorityEnforcementGrade::strict_privileged_filesystem ||
      config.enforcement_grade ==
          SqliteAuthorityEnforcementGrade::cooperative_test;
  const bool has_posix_double_slash_prefix =
      native_path.size() > 1U && native_path[0] == '/' && native_path[1] == '/';
  if (config.api_version != kSqliteAuthorityApiVersion || !grade_is_valid ||
      !config.ledger_path.is_absolute() || config.ledger_path.empty() ||
      native_path.size() > kMaximumPathBytes || has_posix_double_slash_prefix ||
      config.ledger_path.lexically_normal() != config.ledger_path) {
    throw SqliteAuthorityError(
        "SQLite authority requires one canonical absolute path and v1 API");
  }
  if (config.enforcement_grade ==
      SqliteAuthorityEnforcementGrade::strict_filesystem) {
    const uid_t effective_uid = ::geteuid();
    if (effective_uid == 0U || effective_uid == kNobodyUid ||
        effective_uid != config.expected_owner_uid ||
        ::getegid() != config.expected_owner_gid) {
      throw SqliteAuthorityError(
          "strict SQLite authority requires a dedicated non-root, non-nobody "
          "effective UID/GID matching the authority owner");
    }
  }
  if (config.enforcement_grade ==
      SqliteAuthorityEnforcementGrade::strict_privileged_filesystem) {
    if (::geteuid() != 0U || config.expected_owner_uid != 0U ||
        ::getegid() != config.expected_owner_gid) {
      throw SqliteAuthorityError(
          "privileged strict SQLite authority requires effective uid 0 and "
          "the configured authority group");
    }
  }
  const std::string database_name = config.ledger_path.filename().string();
  if (!safe_filename(database_name)) {
    throw SqliteAuthorityError(
        "SQLite database filename is not canonical or bounded");
  }
  const std::string lock_name = database_name + ".authority.lock";
  if (!safe_filename(lock_name)) {
    throw SqliteAuthorityError(
        "SQLite authority lock filename is not canonical");
  }
  auto parent = open_authority_parent(config);
  struct statfs filesystem_status{};
  if (::fstatfs(parent.descriptor.get(), &filesystem_status) != 0) {
    throw_errno("could not inspect authority filesystem");
  }
  const std::string filesystem =
      filesystem_name(filesystem_status.f_type, config.enforcement_grade);
  auto database =
      create_or_open_file(parent.descriptor.get(), database_name, config);
  auto lock = create_or_open_file(parent.descriptor.get(), lock_name, config);
  if (::flock(lock.descriptor.get(), LOCK_EX | LOCK_NB) != 0) {
    if (errno == EWOULDBLOCK || errno == EAGAIN) {
      throw SqliteAuthorityError(
          "another SQLite filesystem authority already owns this lock");
    }
    throw_errno("could not acquire host-global SQLite authority flock");
  }
  auto implementation = std::make_unique<Implementation>();
  implementation->config = std::move(config);
  implementation->database_name = database_name;
  implementation->lock_name = lock_name;
  implementation->parent = std::move(parent.descriptor);
  implementation->database = std::move(database.descriptor);
  implementation->lock = std::move(lock.descriptor);
  implementation->parent_identity = parent.identity;
  implementation->database_identity = database.identity;
  implementation->lock_identity = lock.identity;
  implementation->filesystem = filesystem;
  implementation->protected_filesystem_boundary =
      implementation->config.enforcement_grade !=
      SqliteAuthorityEnforcementGrade::cooperative_test;
  // The pinned parent descriptor closes pathname races on the database and
  // lock files, but SQLite opens `-wal`, `-shm`, and `-journal` itself, by
  // concatenated pathname and without any identity check. The authority VFS
  // extends the same pinned-descriptor discipline to those auxiliaries.
  implementation->sqlite_vfs = SqliteAuthorityVfs::create(
      implementation->parent.get(), implementation->database_name,
      implementation->config.expected_owner_uid);
  SqliteFilesystemAuthority result(std::move(implementation));
  (void)result.attest_before_open();
  return result;
}

SqliteFilesystemAuthority::SqliteFilesystemAuthority(
    std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation)) {}

SqliteFilesystemAuthority::~SqliteFilesystemAuthority() = default;
SqliteFilesystemAuthority::SqliteFilesystemAuthority(
    SqliteFilesystemAuthority &&) noexcept = default;
SqliteFilesystemAuthority &SqliteFilesystemAuthority::operator=(
    SqliteFilesystemAuthority &&) noexcept = default;

SqliteAuthorityAttestation
SqliteFilesystemAuthority::attest_before_open() const {
  if (!implementation_) {
    throw SqliteAuthorityError("SQLite authority has been moved from");
  }
  std::scoped_lock guard(implementation_->mutex);
  return implementation_->attest();
}

SqliteAuthorityAttestation
SqliteFilesystemAuthority::attest_after_open() const {
  return attest_before_open();
}

std::vector<SqliteAuxiliaryFile>
SqliteFilesystemAuthority::validate_auxiliary_files() const {
  if (!implementation_) {
    throw SqliteAuthorityError("SQLite authority has been moved from");
  }
  std::scoped_lock guard(implementation_->mutex);
  (void)implementation_->attest();
  std::vector<SqliteAuxiliaryFile> result;
  for (const std::string_view suffix : {"-wal", "-shm", "-journal"}) {
    const std::string name =
        implementation_->database_name + std::string(suffix);
    struct stat path_status{};
    if (::fstatat(implementation_->parent.get(), name.c_str(), &path_status,
                  AT_SYMLINK_NOFOLLOW) != 0) {
      if (errno == ENOENT)
        continue;
      throw_errno("could not inspect SQLite auxiliary path");
    }
    // Name the suffix. "SQLite auxiliary file" alone left the reader unable to
    // tell a -wal problem from a -journal one, which are created by different
    // code paths and mean different things.
    validate_regular_file(path_status, implementation_->config,
                          "SQLite auxiliary file " + name,
                          implementation_->parent.get());
    auto descriptor = open_beneath(
        implementation_->parent.get(), name,
        static_cast<std::uint64_t>(O_RDWR | O_CLOEXEC | O_NOFOLLOW));
    struct stat descriptor_status{};
    if (::fstat(descriptor.get(), &descriptor_status) != 0) {
      throw_errno("could not inspect SQLite auxiliary descriptor");
    }
    validate_regular_file(descriptor_status, implementation_->config,
                          "SQLite auxiliary file " + name + " (as opened)");
    if (!same_pinned_inode(identity(path_status),
                           identity(descriptor_status))) {
      throw SqliteAuthorityError(
          "SQLite auxiliary path changed during validation");
    }
    result.push_back({.suffix = std::string(suffix),
                      .identity = identity(descriptor_status)});
  }
  return result;
}

const std::filesystem::path &SqliteFilesystemAuthority::ledger_path() const {
  if (!implementation_) {
    throw SqliteAuthorityError("SQLite authority has been moved from");
  }
  return implementation_->config.ledger_path;
}

const SqliteAuthorityVfs &SqliteFilesystemAuthority::sqlite_vfs() const {
  if (!implementation_ || !implementation_->sqlite_vfs) {
    throw SqliteAuthorityError("SQLite authority has been moved from");
  }
  return *implementation_->sqlite_vfs;
}

bool SqliteFilesystemAuthority::is_protected_filesystem_boundary() const {
  return implementation_ && implementation_->protected_filesystem_boundary;
}

int SqliteFilesystemAuthority::duplicate_database_fd() const {
  if (!implementation_) {
    throw SqliteAuthorityError("SQLite authority has been moved from");
  }
  const int duplicated =
      ::fcntl(implementation_->database.get(), F_DUPFD_CLOEXEC, 3);
  if (duplicated < 0)
    throw_errno("could not duplicate pinned database fd");
  return duplicated;
}

} // namespace trainvm
