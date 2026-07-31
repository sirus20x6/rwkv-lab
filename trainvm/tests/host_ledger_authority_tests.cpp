#include "trainvm/host_ledger_authority.hpp"

#include <sqlite3.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using trainvm::HostLedgerAuthorityConfig;
using trainvm::HostLedgerAuthorityError;
using trainvm::HostLedgerEnforcementGrade;
using trainvm::HostLedgerFilesystemAuthority;
using trainvm::kHostLedgerAuthorityApiVersion;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Callable>
void require_authority_error(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const HostLedgerAuthorityError &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    const std::string value = "/tmp/trainvm-ledger-authority-XXXXXX";
    require(value.size() + 1U <= pattern.size(), "temporary path fits");
    std::copy(value.begin(), value.end(), pattern.begin());
    char *created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0,
            "temporary authority directory is mode 0700");
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

HostLedgerAuthorityConfig
config_for(const std::filesystem::path &ledger,
           HostLedgerEnforcementGrade grade =
               HostLedgerEnforcementGrade::cooperative_test) {
  return {.api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = ledger,
          .expected_owner_uid = ::getuid(),
          .expected_owner_gid = ::getgid(),
          .enforcement_grade = grade};
}

void create_directory_0700(const std::filesystem::path &path) {
  require(std::filesystem::create_directory(path), "create test directory");
  require(::chmod(path.c_str(), 0700) == 0, "protect test directory");
}

void create_regular_0600(const std::filesystem::path &path) {
  const int descriptor =
      ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  require(descriptor >= 0, "create test regular file");
  require(::fchmod(descriptor, 0600) == 0, "protect test regular file");
  require(::close(descriptor) == 0, "close test regular file");
}

struct stat inspect(const std::filesystem::path &path) {
  struct stat status{};
  require(::lstat(path.c_str(), &status) == 0, "inspect test path");
  return status;
}

void cooperative_acquisition_and_attestation() {
  TemporaryDirectory temporary;
  const auto ledger = temporary.path() / "ledger.db";
  auto authority = HostLedgerFilesystemAuthority::acquire(config_for(ledger));
  const auto before = authority.attest_before_open();
  const auto after = authority.attest_after_open();

  require(authority.ledger_path() == ledger, "authority retains ledger path");
  require(!authority.is_protected_filesystem_boundary() &&
              !before.protected_filesystem_boundary &&
              !after.protected_filesystem_boundary,
          "cooperative grade is explicitly not a security boundary");
  require(before.enforcement_grade ==
              HostLedgerEnforcementGrade::cooperative_test,
          "attestation reports cooperative grade");
  require(!before.filesystem_name.empty(), "attestation names filesystem");
  require(before.authority_directory.inode == after.authority_directory.inode &&
              before.database_file.inode == after.database_file.inode &&
              before.lock_file.inode == after.lock_file.inode,
          "before and after attestations retain pinned inodes");

  const auto database_status = inspect(ledger);
  const auto lock_status = inspect(ledger.string() + ".authority.lock");
  for (const auto *status : {&database_status, &lock_status}) {
    require(S_ISREG(status->st_mode) && status->st_nlink == 1 &&
                status->st_uid == ::getuid() && status->st_gid == ::getgid() &&
                (status->st_mode & 07777) == 0600,
            "authority files are owned 0600 singleton regular files");
  }

  const int duplicate = authority.duplicate_database_fd();
  require(duplicate >= 0, "database descriptor duplicates");
  struct stat duplicate_status{};
  require(::fstat(duplicate, &duplicate_status) == 0 &&
              duplicate_status.st_ino == database_status.st_ino,
          "duplicated descriptor remains pinned to the database inode");
  require(::close(duplicate) == 0, "close duplicated descriptor");
}

void singleton_flock_is_host_global() {
  TemporaryDirectory temporary;
  const auto ledger = temporary.path() / "ledger.db";
  {
    auto first = HostLedgerFilesystemAuthority::acquire(config_for(ledger));
    require_authority_error(
        [&] {
          auto second =
              HostLedgerFilesystemAuthority::acquire(config_for(ledger));
          (void)second;
        },
        "second live authority must fail its nonblocking flock");
  }
  auto reacquired = HostLedgerFilesystemAuthority::acquire(config_for(ledger));
  require(reacquired.attest_before_open().database_file.inode != 0U,
          "flock releases when authority lifetime ends");
}

void rejects_noncanonical_and_unprotected_paths() {
  require_authority_error(
      [] {
        auto authority = HostLedgerFilesystemAuthority::acquire(
            config_for(std::filesystem::path("relative") / "ledger.db"));
        (void)authority;
      },
      "relative ledger path must fail");
  require_authority_error(
      [] {
        auto authority = HostLedgerFilesystemAuthority::acquire(
            config_for("//tmp/ledger.db"));
        (void)authority;
      },
      "implementation-defined double-slash root must fail");
  require_authority_error(
      [] {
        auto config = config_for("/tmp/ledger.db");
        config.enforcement_grade = static_cast<HostLedgerEnforcementGrade>(255);
        auto authority = HostLedgerFilesystemAuthority::acquire(config);
        (void)authority;
      },
      "unknown enforcement grade must fail closed");

  TemporaryDirectory temporary;
  const auto ledger = temporary.path() / "ledger.db";
  require_authority_error(
      [&] {
        auto authority = HostLedgerFilesystemAuthority::acquire(
            config_for(ledger,
                       HostLedgerEnforcementGrade::strict_filesystem));
        (void)authority;
      },
      "strict authority beneath sticky /tmp must fail");

  require(::chmod(temporary.path().c_str(), 0777) == 0,
          "make final directory world writable");
  require_authority_error(
      [&] {
        auto authority =
            HostLedgerFilesystemAuthority::acquire(config_for(ledger));
        (void)authority;
      },
      "world-writable authority directory must fail");
}

void rejects_wrong_owner_identity() {
  TemporaryDirectory temporary;
  auto config = config_for(temporary.path() / "ledger.db");
  config.expected_owner_uid =
      static_cast<uid_t>(static_cast<std::uint64_t>(::getuid()) + 1U);
  require_authority_error(
      [&] {
        auto authority = HostLedgerFilesystemAuthority::acquire(config);
        (void)authority;
      },
      "explicit wrong authority owner must fail");
}

void rejects_symlinks_hardlinks_fifos_and_modes() {
  {
    TemporaryDirectory temporary;
    const auto target = temporary.path() / "target.db";
    const auto ledger = temporary.path() / "ledger.db";
    create_regular_0600(target);
    require(::symlink(target.filename().c_str(), ledger.c_str()) == 0,
            "create ledger symlink");
    require_authority_error(
        [&] {
          auto authority =
              HostLedgerFilesystemAuthority::acquire(config_for(ledger));
          (void)authority;
        },
        "ledger symlink must fail");
  }
  {
    TemporaryDirectory temporary;
    const auto real = temporary.path() / "real";
    const auto linked = temporary.path() / "linked";
    create_directory_0700(real);
    require(::symlink(real.filename().c_str(), linked.c_str()) == 0,
            "create ancestry symlink");
    require_authority_error(
        [&] {
          auto authority = HostLedgerFilesystemAuthority::acquire(
              config_for(linked / "ledger.db"));
          (void)authority;
        },
        "ancestry symlink must fail");
  }
  {
    TemporaryDirectory temporary;
    const auto source = temporary.path() / "source.db";
    const auto ledger = temporary.path() / "ledger.db";
    create_regular_0600(source);
    require(::link(source.c_str(), ledger.c_str()) == 0,
            "create hardlinked ledger");
    require_authority_error(
        [&] {
          auto authority =
              HostLedgerFilesystemAuthority::acquire(config_for(ledger));
          (void)authority;
        },
        "hardlinked ledger must fail");
  }
  {
    TemporaryDirectory temporary;
    const auto ledger = temporary.path() / "ledger.db";
    require(::mkfifo(ledger.c_str(), 0600) == 0, "create ledger FIFO");
    require_authority_error(
        [&] {
          auto authority =
              HostLedgerFilesystemAuthority::acquire(config_for(ledger));
          (void)authority;
        },
        "ledger FIFO must fail without blocking");
  }
  {
    TemporaryDirectory temporary;
    const auto ledger = temporary.path() / "ledger.db";
    create_regular_0600(ledger);
    require(::chmod(ledger.c_str(), 0644) == 0, "weaken ledger mode");
    require_authority_error(
        [&] {
          auto authority =
              HostLedgerFilesystemAuthority::acquire(config_for(ledger));
          (void)authority;
        },
        "ledger with permissive mode must fail");
  }
}

void detects_database_and_parent_inode_replacement() {
  {
    TemporaryDirectory temporary;
    const auto ledger = temporary.path() / "ledger.db";
    auto authority = HostLedgerFilesystemAuthority::acquire(config_for(ledger));
    const auto displaced = temporary.path() / "displaced.db";
    require(::rename(ledger.c_str(), displaced.c_str()) == 0,
            "displace database inode");
    create_regular_0600(ledger);
    require_authority_error(
        [&] { (void)authority.attest_after_open(); },
        "database pathname replacement must fail attestation");
  }
  {
    TemporaryDirectory temporary;
    const auto directory = temporary.path() / "authority";
    const auto displaced = temporary.path() / "displaced";
    create_directory_0700(directory);
    auto authority = HostLedgerFilesystemAuthority::acquire(
        config_for(directory / "ledger.db"));
    require(::rename(directory.c_str(), displaced.c_str()) == 0,
            "displace authority directory inode");
    create_directory_0700(directory);
    require_authority_error(
        [&] { (void)authority.attest_after_open(); },
        "authority directory replacement must fail attestation");
  }
}

void rejects_malformed_sqlite_auxiliary_files() {
  TemporaryDirectory temporary;
  const auto ledger = temporary.path() / "ledger.db";
  auto authority = HostLedgerFilesystemAuthority::acquire(config_for(ledger));
  require(authority.validate_auxiliary_files().empty(),
          "absent SQLite auxiliary files are valid");

  const auto wal = std::filesystem::path(ledger.string() + "-wal");
  const auto target = temporary.path() / "target";
  create_regular_0600(target);
  require(::symlink(target.filename().c_str(), wal.c_str()) == 0,
          "create WAL symlink");
  require_authority_error([&] { (void)authority.validate_auxiliary_files(); },
                          "WAL symlink must fail");
  require(std::filesystem::remove(wal), "remove WAL symlink");

  create_regular_0600(wal);
  require(::chmod(wal.c_str(), 0644) == 0, "weaken WAL mode");
  require_authority_error([&] { (void)authority.validate_auxiliary_files(); },
                          "permissive WAL must fail");
  require(std::filesystem::remove(wal), "remove permissive WAL");

  create_regular_0600(wal);
  const auto alias = temporary.path() / "wal-alias";
  require(::link(wal.c_str(), alias.c_str()) == 0, "hardlink WAL");
  require_authority_error([&] { (void)authority.validate_auxiliary_files(); },
                          "hardlinked WAL must fail");
  require(std::filesystem::remove(alias) && std::filesystem::remove(wal),
          "remove hardlinked WAL");

  const auto shared_memory = std::filesystem::path(ledger.string() + "-shm");
  require(::mkfifo(shared_memory.c_str(), 0600) == 0, "create SHM FIFO");
  require_authority_error([&] { (void)authority.validate_auxiliary_files(); },
                          "SHM FIFO must fail without blocking");
  require(std::filesystem::remove(shared_memory), "remove SHM FIFO");

  create_regular_0600(wal);
  create_regular_0600(shared_memory);
  const auto valid = authority.validate_auxiliary_files();
  require(valid.size() == 2U && valid[0].suffix == "-wal" &&
              valid[1].suffix == "-shm",
          "owned 0600 singleton SQLite auxiliary files validate");
}

void execute_sql(sqlite3 *database, const char *sql) {
  char *error = nullptr;
  const int result = sqlite3_exec(database, sql, nullptr, nullptr, &error);
  const std::string detail = error == nullptr ? "" : error;
  sqlite3_free(error);
  if (result != SQLITE_OK) {
    throw std::runtime_error("SQLite execution failed: " + detail);
  }
}

void stock_sqlite_wal_lifecycle_is_attested() {
  TemporaryDirectory temporary;
  const auto ledger = temporary.path() / "ledger.db";
  auto authority = HostLedgerFilesystemAuthority::acquire(config_for(ledger));
  (void)authority.attest_before_open();

  sqlite3 *database = nullptr;
  require(sqlite3_open(ledger.c_str(), &database) == SQLITE_OK,
          "stock SQLite opens protected ledger pathname");
  try {
    execute_sql(database, "PRAGMA journal_mode=WAL;");
    execute_sql(database, "CREATE TABLE authority_test(value INTEGER);");
    execute_sql(database, "INSERT INTO authority_test VALUES(1);");
    const auto auxiliary = authority.validate_auxiliary_files();
    require(
        auxiliary.size() == 2U,
        "stock SQLite WAL and SHM files satisfy protected-directory policy");
    const auto after = authority.attest_after_open();
    require(after.database_file.inode != 0U,
            "post-SQLite attestation preserves database inode");
  } catch (...) {
    (void)sqlite3_close(database);
    throw;
  }
  require(sqlite3_close(database) == SQLITE_OK, "close stock SQLite database");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string_view, void (*)()>> tests{
      {"cooperative acquisition and attestation",
       cooperative_acquisition_and_attestation},
      {"singleton flock", singleton_flock_is_host_global},
      {"path policy", rejects_noncanonical_and_unprotected_paths},
      {"owner identity", rejects_wrong_owner_identity},
      {"special files", rejects_symlinks_hardlinks_fifos_and_modes},
      {"inode replacement", detects_database_and_parent_inode_replacement},
      {"SQLite auxiliary policy", rejects_malformed_sqlite_auxiliary_files},
      {"stock SQLite WAL", stock_sqlite_wal_lifecycle_is_attested},
  };
  try {
    for (const auto &[name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
  return 0;
}
