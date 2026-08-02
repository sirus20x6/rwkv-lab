#include "trainvm/sqlite_authority_vfs.hpp"

#include <sqlite3.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>

namespace trainvm {
namespace {

// Every probe this VFS performs refuses to traverse a final-component symlink
// and never leaks across exec.
constexpr int kProbeFlags = O_CLOEXEC | O_NOFOLLOW;
constexpr mode_t kAuthorityFileMode = S_IRUSR | S_IWUSR;

struct InodeIdentity final {
  std::uint64_t device{};
  std::uint64_t inode{};

  bool operator==(const InodeIdentity&) const = default;
};

InodeIdentity identity_of(const struct stat& status) {
  return {.device = static_cast<std::uint64_t>(status.st_dev),
          .inode = static_cast<std::uint64_t>(status.st_ino)};
}

// An authority auxiliary must be an unaliased regular file owned by the
// authority identity and unwritable by anyone else. `st_nlink == 1` is the
// check that refuses a hardlinked alias, which no other predicate catches.
bool safe_authority_inode(const struct stat& status, uid_t owner) {
  return S_ISREG(status.st_mode) && status.st_uid == owner &&
         status.st_nlink == 1U && (status.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}

bool plain_filename(std::string_view name) {
  return !name.empty() && name != "." && name != ".." &&
         name.find('/') == std::string_view::npos &&
         name.find('\0') == std::string_view::npos;
}

}  // namespace

struct SqliteAuthorityVfs::Implementation final {
  sqlite3_vfs vfs{};
  sqlite3_vfs* parent{nullptr};
  int directory_descriptor{-1};
  uid_t expected_owner_uid{};
  bool registered{false};
  std::string name;
  std::string database;
  std::string prefix;
  std::string path;

  mutable std::mutex mutex;
  SqliteAuthorityVfsStatistics statistics;

  void count_open() {
    const std::scoped_lock guard(mutex);
    ++statistics.guarded_opens;
  }
  void count_delete() {
    const std::scoped_lock guard(mutex);
    ++statistics.guarded_deletes;
  }
  void count_shared_memory_guard() {
    const std::scoped_lock guard(mutex);
    ++statistics.shared_memory_guards;
  }
  void count_rejected_name() {
    const std::scoped_lock guard(mutex);
    ++statistics.rejected_names;
  }
  void count_rejected_identity() {
    const std::scoped_lock guard(mutex);
    ++statistics.rejected_identities;
  }
  void count_rejected_substitution() {
    const std::scoped_lock guard(mutex);
    ++statistics.rejected_substitutions;
  }

  // Maps a pathname SQLite derived from `path` back to the plain filename it
  // must have inside the pinned directory. Anything that is not a recognized
  // auxiliary of exactly this database is refused: the namespace is closed, so
  // an unexpected name is evidence of a bug or an attack, never routine.
  [[nodiscard]] bool resolve(const char* pathname, std::string* out) const {
    if (pathname == nullptr) return false;
    const std::string_view view(pathname);
    if (!view.starts_with(prefix)) return false;
    const std::string_view base = view.substr(prefix.size());
    if (!plain_filename(base)) return false;
    if (base == database) {
      out->assign(base);
      return true;
    }
    if (!base.starts_with(database)) return false;
    const std::string_view suffix = base.substr(database.size());
    const bool recognized =
        suffix == "-journal" || suffix == "-wal" || suffix == "-shm" ||
        // Super journals exist only for multi-database transactions, but the
        // namespace must still admit them rather than fail an honest commit.
        (suffix.starts_with("-mj") &&
         suffix.find_first_not_of("0123456789abcdefABCDEF", 3U) ==
             std::string_view::npos);
    if (!recognized) return false;
    out->assign(base);
    return true;
  }
};

namespace {

using Implementation = SqliteAuthorityVfs::Implementation;

Implementation* owner_of(sqlite3_vfs* vfs) {
  return static_cast<Implementation*>(vfs->pAppData);
}

// A thin wrapper around the parent VFS's file object. It exists for exactly one
// reason: the unix VFS opens the `-shm` wal-index file inside `xShmMap`, by a
// pathname it derives internally, never through `xOpen`. Wrapping the database
// file is the only interception point that layer offers.
struct AuthorityFile final {
  sqlite3_file base{};
  Implementation* owner{nullptr};
  bool guard_shared_memory{false};
  bool shared_memory_pinned{false};
  InodeIdentity shared_memory_identity{};
};

constexpr std::size_t kParentAlignment = alignof(std::max_align_t);
constexpr std::size_t kParentOffset =
    ((sizeof(AuthorityFile) + kParentAlignment - 1U) / kParentAlignment) *
    kParentAlignment;

sqlite3_file* parent_file(sqlite3_file* file) {
  return reinterpret_cast<sqlite3_file*>(reinterpret_cast<char*>(file) +
                                         kParentOffset);
}

AuthorityFile* authority_file(sqlite3_file* file) {
  return reinterpret_cast<AuthorityFile*>(file);
}

const sqlite3_io_methods* parent_methods(sqlite3_file* file) {
  return parent_file(file)->pMethods;
}

int authority_close(sqlite3_file* file) {
  sqlite3_file* parent = parent_file(file);
  const int result =
      parent->pMethods != nullptr ? parent->pMethods->xClose(parent) : SQLITE_OK;
  std::destroy_at(authority_file(file));
  return result;
}

int authority_read(sqlite3_file* file, void* buffer, int amount,
                   sqlite3_int64 offset) {
  return parent_methods(file)->xRead(parent_file(file), buffer, amount, offset);
}

int authority_write(sqlite3_file* file, const void* buffer, int amount,
                    sqlite3_int64 offset) {
  return parent_methods(file)->xWrite(parent_file(file), buffer, amount, offset);
}

int authority_truncate(sqlite3_file* file, sqlite3_int64 size) {
  return parent_methods(file)->xTruncate(parent_file(file), size);
}

int authority_sync(sqlite3_file* file, int flags) {
  return parent_methods(file)->xSync(parent_file(file), flags);
}

int authority_file_size(sqlite3_file* file, sqlite3_int64* size) {
  return parent_methods(file)->xFileSize(parent_file(file), size);
}

int authority_lock(sqlite3_file* file, int level) {
  return parent_methods(file)->xLock(parent_file(file), level);
}

int authority_unlock(sqlite3_file* file, int level) {
  return parent_methods(file)->xUnlock(parent_file(file), level);
}

int authority_check_reserved_lock(sqlite3_file* file, int* result) {
  return parent_methods(file)->xCheckReservedLock(parent_file(file), result);
}

int authority_file_control(sqlite3_file* file, int operation, void* argument) {
  return parent_methods(file)->xFileControl(parent_file(file), operation,
                                            argument);
}

int authority_sector_size(sqlite3_file* file) {
  return parent_methods(file)->xSectorSize(parent_file(file));
}

int authority_device_characteristics(sqlite3_file* file) {
  return parent_methods(file)->xDeviceCharacteristics(parent_file(file));
}

// Pins the wal-index inode before the unix VFS opens it by pathname, so a
// planted symlink fails closed instead of being followed.
int prepare_shared_memory(AuthorityFile* self) {
  Implementation* owner = self->owner;
  const std::string name = owner->database + "-shm";
  const int probe =
      ::openat(owner->directory_descriptor, name.c_str(),
               O_RDWR | O_CREAT | kProbeFlags, kAuthorityFileMode);
  if (probe < 0) {
    owner->count_rejected_identity();
    return SQLITE_IOERR_SHMOPEN;
  }
  struct stat status {};
  if (::fstat(probe, &status) != 0 ||
      !safe_authority_inode(status, owner->expected_owner_uid)) {
    (void)::close(probe);
    owner->count_rejected_identity();
    return SQLITE_IOERR_SHMOPEN;
  }
  self->shared_memory_identity = identity_of(status);
  self->shared_memory_pinned = true;
  (void)::close(probe);
  owner->count_shared_memory_guard();
  return SQLITE_OK;
}

// Re-checks that the pinned wal-index name still resolves to the pinned inode.
// SQLite's dead-man-switch protocol guarantees the `-shm` file is not unlinked
// while any connection has it mapped, so a changed inode here is a live
// substitution, not routine churn.
int verify_shared_memory(AuthorityFile* self) {
  Implementation* owner = self->owner;
  const std::string name = owner->database + "-shm";
  struct stat status {};
  if (::fstatat(owner->directory_descriptor, name.c_str(), &status,
                AT_SYMLINK_NOFOLLOW) != 0 ||
      !safe_authority_inode(status, owner->expected_owner_uid) ||
      identity_of(status) != self->shared_memory_identity) {
    owner->count_rejected_substitution();
    return SQLITE_IOERR_SHMOPEN;
  }
  return SQLITE_OK;
}

int authority_shm_map(sqlite3_file* file, int page, int page_size, int extend,
                      void volatile** mapped) {
  AuthorityFile* self = authority_file(file);
  if (self->guard_shared_memory) {
    if (!self->shared_memory_pinned) {
      const int prepared = prepare_shared_memory(self);
      if (prepared != SQLITE_OK) return prepared;
    } else {
      const int verified = verify_shared_memory(self);
      if (verified != SQLITE_OK) return verified;
    }
  }
  const int result = parent_methods(file)->xShmMap(parent_file(file), page,
                                                   page_size, extend, mapped);
  if (self->guard_shared_memory && result == SQLITE_OK) {
    const int verified = verify_shared_memory(self);
    if (verified != SQLITE_OK) return verified;
  }
  return result;
}

int authority_shm_lock(sqlite3_file* file, int offset, int count, int flags) {
  return parent_methods(file)->xShmLock(parent_file(file), offset, count, flags);
}

void authority_shm_barrier(sqlite3_file* file) {
  parent_methods(file)->xShmBarrier(parent_file(file));
}

int authority_shm_unmap(sqlite3_file* file, int delete_flag) {
  AuthorityFile* self = authority_file(file);
  const int result =
      parent_methods(file)->xShmUnmap(parent_file(file), delete_flag);
  if (delete_flag != 0) {
    // The wal-index was unlinked; the next mapping legitimately gets a new
    // inode, so the pin must be re-established rather than compared.
    self->shared_memory_pinned = false;
  }
  return result;
}

int authority_fetch(sqlite3_file* file, sqlite3_int64 offset, int amount,
                    void** page) {
  return parent_methods(file)->xFetch(parent_file(file), offset, amount, page);
}

int authority_unfetch(sqlite3_file* file, sqlite3_int64 offset, void* page) {
  return parent_methods(file)->xUnfetch(parent_file(file), offset, page);
}

// SQLite dispatches on the advertised iVersion, so the wrapper must advertise
// exactly what the parent implements rather than a fixed maximum.
constexpr sqlite3_io_methods make_methods(int version) {
  return {
      .iVersion = version,
      .xClose = &authority_close,
      .xRead = &authority_read,
      .xWrite = &authority_write,
      .xTruncate = &authority_truncate,
      .xSync = &authority_sync,
      .xFileSize = &authority_file_size,
      .xLock = &authority_lock,
      .xUnlock = &authority_unlock,
      .xCheckReservedLock = &authority_check_reserved_lock,
      .xFileControl = &authority_file_control,
      .xSectorSize = &authority_sector_size,
      .xDeviceCharacteristics = &authority_device_characteristics,
      .xShmMap = version >= 2 ? &authority_shm_map : nullptr,
      .xShmLock = version >= 2 ? &authority_shm_lock : nullptr,
      .xShmBarrier = version >= 2 ? &authority_shm_barrier : nullptr,
      .xShmUnmap = version >= 2 ? &authority_shm_unmap : nullptr,
      .xFetch = version >= 3 ? &authority_fetch : nullptr,
      .xUnfetch = version >= 3 ? &authority_unfetch : nullptr,
  };
}

constexpr sqlite3_io_methods kMethodsV1 = make_methods(1);
constexpr sqlite3_io_methods kMethodsV2 = make_methods(2);
constexpr sqlite3_io_methods kMethodsV3 = make_methods(3);

const sqlite3_io_methods* methods_for(int version) {
  if (version >= 3) return &kMethodsV3;
  if (version == 2) return &kMethodsV2;
  return &kMethodsV1;
}

int authority_open(sqlite3_vfs* vfs, sqlite3_filename pathname,
                   sqlite3_file* file, int flags, int* out_flags) {
  Implementation* owner = owner_of(vfs);
  file->pMethods = nullptr;

  // A null pathname is an anonymous temporary file: SQLite unlinks it
  // immediately, it holds no authority state, and it has no name for anyone to
  // race. It is the one open that legitimately leaves the namespace, and
  // refusing it would break honest spill-to-disk queries.
  if (pathname == nullptr) {
    AuthorityFile* anonymous = std::construct_at(authority_file(file));
    anonymous->owner = owner;
    sqlite3_file* temporary = parent_file(file);
    temporary->pMethods = nullptr;
    const int temporary_result = owner->parent->xOpen(owner->parent, pathname,
                                                      temporary, flags, out_flags);
    if (temporary_result != SQLITE_OK) {
      std::destroy_at(anonymous);
      return temporary_result;
    }
    file->pMethods = methods_for(temporary->pMethods->iVersion);
    return SQLITE_OK;
  }

  std::string base;
  if (!owner->resolve(pathname, &base)) {
    owner->count_rejected_name();
    return SQLITE_CANTOPEN;
  }
  if ((flags & SQLITE_OPEN_DELETEONCLOSE) != 0) {
    // Delete-on-close unlinks by pathname inside the parent VFS, outside the
    // pinned descriptor. No named authority file uses it.
    owner->count_rejected_name();
    return SQLITE_CANTOPEN;
  }

  const bool read_only = (flags & SQLITE_OPEN_READONLY) != 0;
  int probe_flags = kProbeFlags | (read_only ? O_RDONLY : O_RDWR);
  if ((flags & SQLITE_OPEN_CREATE) != 0) probe_flags |= O_CREAT;
  const int probe = ::openat(owner->directory_descriptor, base.c_str(),
                             probe_flags, kAuthorityFileMode);
  if (probe < 0) {
    // ELOOP is a planted symlink; ENOENT without CREATE is an honest miss.
    if (errno == ENOENT && (flags & SQLITE_OPEN_CREATE) == 0) {
      return SQLITE_CANTOPEN;
    }
    owner->count_rejected_identity();
    return SQLITE_CANTOPEN;
  }
  struct stat before {};
  if (::fstat(probe, &before) != 0 ||
      !safe_authority_inode(before, owner->expected_owner_uid)) {
    (void)::close(probe);
    owner->count_rejected_identity();
    return SQLITE_CANTOPEN;
  }
  (void)::close(probe);

  AuthorityFile* self = std::construct_at(authority_file(file));
  self->owner = owner;
  self->guard_shared_memory = (flags & SQLITE_OPEN_MAIN_DB) != 0;

  sqlite3_file* parent = parent_file(file);
  parent->pMethods = nullptr;
  const int result =
      owner->parent->xOpen(owner->parent, pathname, parent, flags, out_flags);
  if (result != SQLITE_OK) {
    std::destroy_at(self);
    return result;
  }

  // The parent resolved the same pathname with a plain open. If the name was
  // substituted in that window the parent now holds an impostor inode, so the
  // handle is closed and refused before SQLite reads or writes a byte.
  struct stat after {};
  if (::fstatat(owner->directory_descriptor, base.c_str(), &after,
                AT_SYMLINK_NOFOLLOW) != 0 ||
      !safe_authority_inode(after, owner->expected_owner_uid) ||
      identity_of(after) != identity_of(before)) {
    (void)parent->pMethods->xClose(parent);
    std::destroy_at(self);
    owner->count_rejected_substitution();
    return SQLITE_IOERR;
  }

  file->pMethods = methods_for(parent->pMethods->iVersion);
  owner->count_open();
  return SQLITE_OK;
}

int authority_delete(sqlite3_vfs* vfs, const char* pathname, int sync_directory) {
  Implementation* owner = owner_of(vfs);
  std::string base;
  if (!owner->resolve(pathname, &base)) {
    owner->count_rejected_name();
    return SQLITE_IOERR_DELETE;
  }
  if (::unlinkat(owner->directory_descriptor, base.c_str(), 0) != 0) {
    if (errno == ENOENT) return SQLITE_IOERR_DELETE_NOENT;
    return SQLITE_IOERR_DELETE;
  }
  owner->count_delete();
  if (sync_directory != 0 && ::fsync(owner->directory_descriptor) != 0) {
    return SQLITE_IOERR_DIR_FSYNC;
  }
  return SQLITE_OK;
}

int authority_access(sqlite3_vfs* vfs, const char* pathname, int flags,
                     int* result) {
  Implementation* owner = owner_of(vfs);
  *result = 0;
  std::string base;
  if (!owner->resolve(pathname, &base)) {
    owner->count_rejected_name();
    return SQLITE_OK;
  }
  struct stat status {};
  if (::fstatat(owner->directory_descriptor, base.c_str(), &status,
                AT_SYMLINK_NOFOLLOW) != 0) {
    return SQLITE_OK;
  }
  if (flags == SQLITE_ACCESS_EXISTS) {
    // An unsafe entry is deliberately reported as existing. Reporting absence
    // would let SQLite skip hot-journal rollback; reporting presence funnels it
    // into xOpen, which refuses it and fails the operation closed.
    *result = 1;
    return SQLITE_OK;
  }
  const bool readable = safe_authority_inode(status, owner->expected_owner_uid);
  *result = readable && (flags != SQLITE_ACCESS_READWRITE ||
                         (status.st_mode & S_IWUSR) != 0)
                ? 1
                : 0;
  return SQLITE_OK;
}

int authority_full_pathname(sqlite3_vfs* vfs, const char* pathname, int size,
                            char* out) {
  Implementation* owner = owner_of(vfs);
  std::string base;
  if (!owner->resolve(pathname, &base)) {
    owner->count_rejected_name();
    return SQLITE_CANTOPEN;
  }
  // Names are already canonical: rooted at the pinned descriptor and free of
  // any component the kernel could re-resolve.
  const std::string canonical = owner->prefix + base;
  if (size <= 0 || canonical.size() + 1U > static_cast<std::size_t>(size)) {
    return SQLITE_CANTOPEN;
  }
  std::memcpy(out, canonical.c_str(), canonical.size() + 1U);
  return SQLITE_OK;
}

void* authority_dlopen(sqlite3_vfs*, const char*) { return nullptr; }

void authority_dlerror(sqlite3_vfs*, int size, char* message) {
  if (size > 0) {
    const char text[] = "extension loading is disabled in authority databases";
    const std::size_t limit = static_cast<std::size_t>(size) - 1U;
    const std::size_t copied = sizeof(text) - 1U < limit ? sizeof(text) - 1U : limit;
    std::memcpy(message, text, copied);
    message[copied] = '\0';
  }
}

void (*authority_dlsym(sqlite3_vfs*, void*, const char*))() { return nullptr; }

void authority_dlclose(sqlite3_vfs*, void*) {}

int authority_randomness(sqlite3_vfs* vfs, int size, char* out) {
  sqlite3_vfs* parent = owner_of(vfs)->parent;
  return parent->xRandomness(parent, size, out);
}

int authority_sleep(sqlite3_vfs* vfs, int microseconds) {
  sqlite3_vfs* parent = owner_of(vfs)->parent;
  return parent->xSleep(parent, microseconds);
}

int authority_current_time(sqlite3_vfs* vfs, double* out) {
  sqlite3_vfs* parent = owner_of(vfs)->parent;
  return parent->xCurrentTime(parent, out);
}

int authority_get_last_error(sqlite3_vfs* vfs, int size, char* out) {
  sqlite3_vfs* parent = owner_of(vfs)->parent;
  return parent->xGetLastError != nullptr
             ? parent->xGetLastError(parent, size, out)
             : SQLITE_OK;
}

int authority_current_time_int64(sqlite3_vfs* vfs, sqlite3_int64* out) {
  sqlite3_vfs* parent = owner_of(vfs)->parent;
  return parent->xCurrentTimeInt64 != nullptr
             ? parent->xCurrentTimeInt64(parent, out)
             : SQLITE_ERROR;
}

std::string next_vfs_name() {
  static std::atomic<std::uint64_t> counter{0};
  return "trainvm-authority-" + std::to_string(::getpid()) + "-" +
         std::to_string(counter.fetch_add(1U, std::memory_order_relaxed));
}

}  // namespace

SqliteAuthorityVfs::SqliteAuthorityVfs(
    std::unique_ptr<Implementation> implementation)
    : implementation_(std::move(implementation)) {}

std::shared_ptr<SqliteAuthorityVfs> SqliteAuthorityVfs::create(
    int authority_directory_descriptor, std::string database_name,
    uid_t expected_owner_uid) {
  if (!plain_filename(database_name)) {
    throw SqliteAuthorityVfsError(
        "authority database name must be a plain filename");
  }
  if (authority_directory_descriptor < 0) {
    throw SqliteAuthorityVfsError(
        "authority VFS requires an open directory descriptor");
  }
  struct stat directory {};
  if (::fstat(authority_directory_descriptor, &directory) != 0 ||
      !S_ISDIR(directory.st_mode) || directory.st_uid != expected_owner_uid ||
      (directory.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw SqliteAuthorityVfsError(
        "authority VFS directory is not an owned, exclusively writable directory");
  }
  sqlite3_vfs* parent = sqlite3_vfs_find(nullptr);
  if (parent == nullptr) {
    throw SqliteAuthorityVfsError("no default SQLite VFS is registered");
  }

  auto implementation = std::make_unique<Implementation>();
  implementation->parent = parent;
  implementation->expected_owner_uid = expected_owner_uid;
  implementation->database = std::move(database_name);
  implementation->name = next_vfs_name();
  implementation->directory_descriptor =
      ::fcntl(authority_directory_descriptor, F_DUPFD_CLOEXEC, 0);
  if (implementation->directory_descriptor < 0) {
    throw SqliteAuthorityVfsError(
        "could not duplicate the authority directory descriptor");
  }
  implementation->prefix = "/proc/self/fd/" +
                           std::to_string(implementation->directory_descriptor) +
                           "/";
  implementation->path = implementation->prefix + implementation->database;

  // SQLite appends the longest auxiliary suffix it knows to this pathname
  // before handing it back through xOpen, so the namespace must leave room.
  constexpr std::size_t kLongestSuffix = 24U;
  if (parent->mxPathname <= 0 ||
      implementation->path.size() + kLongestSuffix >
          static_cast<std::size_t>(parent->mxPathname)) {
    (void)::close(implementation->directory_descriptor);
    throw SqliteAuthorityVfsError(
        "authority database pathname exceeds the VFS pathname limit");
  }

  implementation->vfs = {
      .iVersion = 2,
      .szOsFile =
          static_cast<int>(kParentOffset + static_cast<std::size_t>(parent->szOsFile)),
      .mxPathname = parent->mxPathname,
      .pNext = nullptr,
      .zName = implementation->name.c_str(),
      .pAppData = implementation.get(),
      .xOpen = &authority_open,
      .xDelete = &authority_delete,
      .xAccess = &authority_access,
      .xFullPathname = &authority_full_pathname,
      .xDlOpen = &authority_dlopen,
      .xDlError = &authority_dlerror,
      .xDlSym = &authority_dlsym,
      .xDlClose = &authority_dlclose,
      .xRandomness = &authority_randomness,
      .xSleep = &authority_sleep,
      .xCurrentTime = &authority_current_time,
      .xGetLastError = &authority_get_last_error,
      .xCurrentTimeInt64 = &authority_current_time_int64,
      .xSetSystemCall = nullptr,
      .xGetSystemCall = nullptr,
      .xNextSystemCall = nullptr,
  };
  if (sqlite3_vfs_register(&implementation->vfs, 0) != SQLITE_OK) {
    (void)::close(implementation->directory_descriptor);
    throw SqliteAuthorityVfsError("could not register the authority VFS");
  }
  implementation->registered = true;
  return std::shared_ptr<SqliteAuthorityVfs>(
      new SqliteAuthorityVfs(std::move(implementation)));
}

SqliteAuthorityVfs::~SqliteAuthorityVfs() {
  if (!implementation_) return;
  if (implementation_->registered) {
    (void)sqlite3_vfs_unregister(&implementation_->vfs);
  }
  if (implementation_->directory_descriptor >= 0) {
    (void)::close(implementation_->directory_descriptor);
  }
}

const std::string& SqliteAuthorityVfs::vfs_name() const noexcept {
  return implementation_->name;
}

const std::string& SqliteAuthorityVfs::database_path() const noexcept {
  return implementation_->path;
}

const std::string& SqliteAuthorityVfs::database_name() const noexcept {
  return implementation_->database;
}

SqliteAuthorityVfsStatistics SqliteAuthorityVfs::statistics() const noexcept {
  const std::scoped_lock guard(implementation_->mutex);
  return implementation_->statistics;
}

}  // namespace trainvm
