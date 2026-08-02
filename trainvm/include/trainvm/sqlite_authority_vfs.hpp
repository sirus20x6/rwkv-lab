#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <sys/types.h>

namespace trainvm {

inline constexpr std::string_view kSqliteAuthorityVfsApiVersion =
    "trainvm.sqlite-authority-vfs/v1";

class SqliteAuthorityVfsError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Observable evidence of what the authority VFS admitted and refused. The
// counters are the machine-readable half of the hostile-race qualification:
// a passing race test must show refusals, not merely an absence of damage.
struct SqliteAuthorityVfsStatistics final {
  std::uint64_t guarded_opens{};
  std::uint64_t guarded_deletes{};
  std::uint64_t shared_memory_guards{};
  // A pathname outside the pinned authority namespace, or a namespaced name
  // that is not a recognized SQLite auxiliary of this database.
  std::uint64_t rejected_names{};
  // The name resolved, but the inode was a symlink, a hardlinked alias, a
  // non-regular file, or owned/writable outside the authority identity.
  std::uint64_t rejected_identities{};
  // The inode observed after SQLite's own open differed from the inode this
  // VFS validated immediately before it: a live substitution race.
  std::uint64_t rejected_substitutions{};

  bool operator==(const SqliteAuthorityVfsStatistics&) const = default;
};

// A controlled SQLite VFS that confines every file SQLite opens on behalf of
// one database - the main database, `-wal`, `-shm`, `-journal`, and super
// journals - to a pinned authority directory descriptor.
//
// Stock SQLite derives auxiliary filenames by string concatenation and opens
// them with plain `open(2)`: no `O_NOFOLLOW`, no identity check. A same-UID
// process that plants a symlink or substitutes an inode at `<db>-wal` between
// two authority boundaries therefore redirects authority state through a
// pathname the authority never validated.
//
// This VFS removes pathname resolution from the attack surface:
//
//   * every name is resolved with `openat`/`unlinkat`/`fstatat` relative to a
//     directory descriptor the authority pinned once, so no component of the
//     path can be re-resolved;
//   * every open is preceded by an `O_NOFOLLOW` open and an identity check, so
//     a planted symlink or hardlink fails closed rather than being followed;
//   * every open is followed by an inode re-check, so a substitution racing
//     between this VFS's validation and SQLite's own open is detected before
//     the handle is returned and SQLite never reads or writes the impostor;
//   * names inside the authority namespace that are not recognized auxiliaries
//     of this database are refused outright.
//
// Residual, and deliberately not claimed: a same-UID process retains ptrace
// and `/proc/self/mem` access to this process and can therefore corrupt the
// authority without touching the filesystem at all. Same-UID isolation is a
// deployment property (separate service accounts), not something a VFS can
// establish. What this class does establish is that no filesystem pathname
// race can redirect or split authority state undetected.
class SqliteAuthorityVfs final {
 public:
  // `authority_directory_descriptor` must be an open directory descriptor the
  // caller already resolved and validated; it is duplicated, so the caller may
  // close its own copy. `database_name` must be a plain filename.
  static std::shared_ptr<SqliteAuthorityVfs> create(
      int authority_directory_descriptor, std::string database_name,
      uid_t expected_owner_uid);

  ~SqliteAuthorityVfs();

  SqliteAuthorityVfs(const SqliteAuthorityVfs&) = delete;
  SqliteAuthorityVfs& operator=(const SqliteAuthorityVfs&) = delete;
  SqliteAuthorityVfs(SqliteAuthorityVfs&&) = delete;
  SqliteAuthorityVfs& operator=(SqliteAuthorityVfs&&) = delete;

  // The name to pass as the final argument of `sqlite3_open_v2`.
  [[nodiscard]] const std::string& vfs_name() const noexcept;

  // The pathname to pass as the first argument of `sqlite3_open_v2`. SQLite
  // derives `-wal`, `-shm`, and `-journal` from it by concatenation, so it is
  // rooted at the pinned directory descriptor and every derived name stays
  // inside the authority namespace by construction.
  [[nodiscard]] const std::string& database_path() const noexcept;

  [[nodiscard]] const std::string& database_name() const noexcept;

  [[nodiscard]] SqliteAuthorityVfsStatistics statistics() const noexcept;

  // Defined only in the implementation unit. It is nameable so the VFS
  // callbacks, which SQLite requires to be free functions, can reach it.
  struct Implementation;

 private:
  explicit SqliteAuthorityVfs(std::unique_ptr<Implementation> implementation);

  std::unique_ptr<Implementation> implementation_;
};

}  // namespace trainvm
