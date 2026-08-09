#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

#include "trainvm/sqlite_authority_vfs.hpp"

namespace trainvm {

inline constexpr std::string_view kSqliteAuthorityApiVersion =
    "trainvm.host-ledger-authority/v1";
// No earlier release is claimed safe: this is the exact stock SQLite release
// against which auxiliary symlink refusal and exclusive-WAL SHM suppression
// were validated.
inline constexpr int kSqliteAuthorityMinimumVersionNumber = 3'053'003;

enum class SqliteAuthorityEnforcementGrade {
  strict_filesystem,
  // A host-global daemon that must retain kernel capabilities may own its
  // ledger as uid 0.  This remains an exact 0700/0600 local-filesystem
  // boundary; the separate grade prevents ordinary journal services from
  // silently acquiring the broader privilege model.
  strict_privileged_filesystem,
  cooperative_test,
};

struct SqliteAuthorityConfig final {
  std::string api_version;
  std::filesystem::path ledger_path;
  uid_t expected_owner_uid{};
  gid_t expected_owner_gid{};
  SqliteAuthorityEnforcementGrade enforcement_grade{};
};

struct SqliteFileIdentity final {
  std::uint64_t device{};
  std::uint64_t inode{};
  std::uint64_t size{};
  std::uint32_t mode{};
  std::uint32_t owner_uid{};
  std::uint32_t owner_gid{};
  std::uint64_t link_count{};

  bool operator==(const SqliteFileIdentity &) const = default;
};

struct SqliteAuthorityAttestation final {
  std::string api_version;
  std::filesystem::path ledger_path;
  SqliteAuthorityEnforcementGrade enforcement_grade{};
  bool protected_filesystem_boundary{};
  std::string filesystem_name;
  SqliteFileIdentity authority_directory;
  SqliteFileIdentity database_file;
  SqliteFileIdentity lock_file;

  bool operator==(const SqliteAuthorityAttestation &) const = default;
};

struct SqliteAuxiliaryFile final {
  std::string suffix;
  SqliteFileIdentity identity;

  bool operator==(const SqliteAuxiliaryFile &) const = default;
};

class SqliteAuthorityError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Owns the filesystem authority boundary around one stock-SQLite database.
// The protected directory, not SQLite pathname handling, closes WAL/SHM name
// races: only the configured authority owner may mutate that directory. The
// cooperative_test grade is diagnostic test isolation, never a security
// boundary against another process with the same UID.
class SqliteFilesystemAuthority final {
public:
  static SqliteFilesystemAuthority acquire(SqliteAuthorityConfig config);

  ~SqliteFilesystemAuthority();
  SqliteFilesystemAuthority(SqliteFilesystemAuthority &&) noexcept;
  SqliteFilesystemAuthority &operator=(SqliteFilesystemAuthority &&) noexcept;

  SqliteFilesystemAuthority(const SqliteFilesystemAuthority &) = delete;
  SqliteFilesystemAuthority &
  operator=(const SqliteFilesystemAuthority &) = delete;

  // Called immediately before SQLite opens the public pathname. It reopens
  // the path beneath / with openat2 and verifies every pinned identity.
  [[nodiscard]] SqliteAuthorityAttestation attest_before_open() const;

  // Called after SQLite opens/configures the database. It detects pathname or
  // parent replacement between the two boundaries.
  [[nodiscard]] SqliteAuthorityAttestation attest_after_open() const;

  // Validates any currently present SQLite -wal, -shm, and -journal files.
  // Absence is valid. Presence requires a protected regular 0600 singleton
  // inode owned by the configured authority identity.
  [[nodiscard]] std::vector<SqliteAuxiliaryFile>
  validate_auxiliary_files() const;

  [[nodiscard]] const std::filesystem::path &ledger_path() const;
  // This reports only the filesystem boundary. Host-level strict enforcement
  // additionally requires externally attested hostd identity, peer policy,
  // cgroups, and device enforcement; callers cannot claim that broader grade
  // from this object alone.
  [[nodiscard]] bool is_protected_filesystem_boundary() const;
  [[nodiscard]] int duplicate_database_fd() const;

  // The controlled VFS that confines SQLite's own opens - including the `-wal`,
  // `-shm`, and `-journal` auxiliaries it derives by pathname - to the pinned
  // authority directory. Callers must open the database through
  // `sqlite_vfs().database_path()` and `sqlite_vfs().vfs_name()` rather than
  // through `ledger_path()`, which remains the public, raceable name.
  [[nodiscard]] const SqliteAuthorityVfs &sqlite_vfs() const;

private:
  struct Implementation;
  explicit SqliteFilesystemAuthority(
      std::unique_ptr<Implementation> implementation) noexcept;

  std::unique_ptr<Implementation> implementation_;
};

// Source-compatibility aliases for host-ledger callers. The durable API string
// intentionally remains trainvm.host-ledger-authority/v1.
using HostLedgerEnforcementGrade = SqliteAuthorityEnforcementGrade;
using HostLedgerAuthorityConfig = SqliteAuthorityConfig;
using HostLedgerFileIdentity = SqliteFileIdentity;
using HostLedgerAuthorityAttestation = SqliteAuthorityAttestation;
using HostLedgerAuxiliaryFile = SqliteAuxiliaryFile;
using HostLedgerAuthorityError = SqliteAuthorityError;
using HostLedgerFilesystemAuthority = SqliteFilesystemAuthority;
inline constexpr std::string_view kHostLedgerAuthorityApiVersion =
    kSqliteAuthorityApiVersion;

} // namespace trainvm
