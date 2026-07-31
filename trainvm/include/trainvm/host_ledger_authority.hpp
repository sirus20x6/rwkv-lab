#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

namespace trainvm {

inline constexpr std::string_view kHostLedgerAuthorityApiVersion =
    "trainvm.host-ledger-authority/v1";

enum class HostLedgerEnforcementGrade {
  strict_filesystem,
  cooperative_test,
};

struct HostLedgerAuthorityConfig final {
  std::string api_version;
  std::filesystem::path ledger_path;
  uid_t expected_owner_uid{};
  gid_t expected_owner_gid{};
  HostLedgerEnforcementGrade enforcement_grade{};
};

struct HostLedgerFileIdentity final {
  std::uint64_t device{};
  std::uint64_t inode{};
  std::uint64_t size{};
  std::uint32_t mode{};
  std::uint32_t owner_uid{};
  std::uint32_t owner_gid{};
  std::uint64_t link_count{};

  bool operator==(const HostLedgerFileIdentity &) const = default;
};

struct HostLedgerAuthorityAttestation final {
  std::string api_version;
  std::filesystem::path ledger_path;
  HostLedgerEnforcementGrade enforcement_grade{};
  bool protected_filesystem_boundary{};
  std::string filesystem_name;
  HostLedgerFileIdentity authority_directory;
  HostLedgerFileIdentity database_file;
  HostLedgerFileIdentity lock_file;

  bool operator==(const HostLedgerAuthorityAttestation &) const = default;
};

struct HostLedgerAuxiliaryFile final {
  std::string suffix;
  HostLedgerFileIdentity identity;

  bool operator==(const HostLedgerAuxiliaryFile &) const = default;
};

class HostLedgerAuthorityError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Owns the filesystem authority boundary around one stock-SQLite database.
// The protected directory, not SQLite pathname handling, closes WAL/SHM name
// races: only the configured authority owner may mutate that directory. The
// cooperative_test grade is diagnostic test isolation, never a security
// boundary against another process with the same UID.
class HostLedgerFilesystemAuthority final {
public:
  static HostLedgerFilesystemAuthority
  acquire(HostLedgerAuthorityConfig config);

  ~HostLedgerFilesystemAuthority();
  HostLedgerFilesystemAuthority(HostLedgerFilesystemAuthority &&) noexcept;
  HostLedgerFilesystemAuthority &
  operator=(HostLedgerFilesystemAuthority &&) noexcept;

  HostLedgerFilesystemAuthority(const HostLedgerFilesystemAuthority &) = delete;
  HostLedgerFilesystemAuthority &
  operator=(const HostLedgerFilesystemAuthority &) = delete;

  // Called immediately before SQLite opens the public pathname. It reopens
  // the path beneath / with openat2 and verifies every pinned identity.
  [[nodiscard]] HostLedgerAuthorityAttestation attest_before_open() const;

  // Called after SQLite opens/configures the database. It detects pathname or
  // parent replacement between the two boundaries.
  [[nodiscard]] HostLedgerAuthorityAttestation attest_after_open() const;

  // Validates any currently present SQLite -wal, -shm, and -journal files.
  // Absence is valid. Presence requires a protected regular 0600 singleton
  // inode owned by the configured authority identity.
  [[nodiscard]] std::vector<HostLedgerAuxiliaryFile>
  validate_auxiliary_files() const;

  [[nodiscard]] const std::filesystem::path &ledger_path() const;
  // This reports only the filesystem boundary. Host-level strict enforcement
  // additionally requires externally attested hostd identity, peer policy,
  // cgroups, and device enforcement; callers cannot claim that broader grade
  // from this object alone.
  [[nodiscard]] bool is_protected_filesystem_boundary() const;
  [[nodiscard]] int duplicate_database_fd() const;

private:
  struct Implementation;
  explicit HostLedgerFilesystemAuthority(
      std::unique_ptr<Implementation> implementation) noexcept;

  std::unique_ptr<Implementation> implementation_;
};

} // namespace trainvm
