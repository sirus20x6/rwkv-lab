#include "trainvm/journal.hpp"
#include "trainvm/sqlite_filesystem_authority.hpp"

#include <sqlite3.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using trainvm::HostGrantEnforcement;
using trainvm::Journal;
using trainvm::JournalFileIdentity;
using trainvm::kSqliteAuthorityApiVersion;
using trainvm::kSqliteAuthorityMinimumVersionNumber;
using trainvm::SqliteAuthorityConfig;
using trainvm::SqliteAuthorityEnforcementGrade;
using trainvm::SqliteAuthorityError;
using trainvm::SqliteFilesystemAuthority;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Callable>
void require_authority_error(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const SqliteAuthorityError &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    const std::string value = "/tmp/trainvm-sqlite-race-XXXXXX";
    require(value.size() + 1U <= pattern.size(), "temporary path fits");
    std::copy(value.begin(), value.end(), pattern.begin());
    const char *created = ::mkdtemp(pattern.data());
    require(created != nullptr, "create temporary directory");
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0, "protect temporary directory");
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

class Database final {
public:
  explicit Database(const std::filesystem::path &path) {
    require(sqlite3_open_v2(path.c_str(), &value_,
                            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                SQLITE_OPEN_NOFOLLOW,
                            nullptr) == SQLITE_OK,
            "open SQLite test database");
  }

  ~Database() {
    if (value_ != nullptr) {
      (void)sqlite3_close(value_);
    }
  }

  Database(const Database &) = delete;
  Database &operator=(const Database &) = delete;

  [[nodiscard]] sqlite3 *get() const { return value_; }

  void close() {
    require(value_ != nullptr && sqlite3_close(value_) == SQLITE_OK,
            "close SQLite test database");
    value_ = nullptr;
  }

private:
  sqlite3 *value_{};
};

int execute(sqlite3 *database, std::string_view sql) {
  char *error = nullptr;
  const int result = sqlite3_exec(database, std::string(sql).c_str(), nullptr,
                                  nullptr, &error);
  sqlite3_free(error);
  return result;
}

void execute_ok(sqlite3 *database, std::string_view sql,
                std::string_view message) {
  require(execute(database, sql) == SQLITE_OK, message);
}

void create_regular_0600(const std::filesystem::path &path,
                         std::string_view contents = {}) {
  const int descriptor =
      ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  require(descriptor >= 0, "create protected regular file");
  if (!contents.empty()) {
    const ssize_t written =
        ::write(descriptor, contents.data(), contents.size());
    require(written >= 0 &&
                static_cast<std::size_t>(written) == contents.size(),
            "write protected regular file");
  }
  require(::fchmod(descriptor, 0600) == 0 && ::close(descriptor) == 0,
          "finish protected regular file");
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

SqliteAuthorityConfig
config_for(const std::filesystem::path &path,
           SqliteAuthorityEnforcementGrade grade =
               SqliteAuthorityEnforcementGrade::cooperative_test) {
  return {.api_version = std::string(kSqliteAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade = grade};
}

JournalFileIdentity
journal_identity(const std::filesystem::path &path,
                 const trainvm::SqliteAuthorityAttestation &attestation) {
  return {.directory_path = path.parent_path().string(),
          .journal_name = path.filename().string(),
          .authority_name = path.filename().string() + ".authority.lock",
          .directory_device = attestation.authority_directory.device,
          .directory_inode = attestation.authority_directory.inode,
          .device = attestation.database_file.device,
          .inode = attestation.database_file.inode,
          .authority_device = attestation.lock_file.device,
          .authority_inode = attestation.lock_file.inode,
          .owner_uid = attestation.authority_directory.owner_uid};
}

void stock_sqlite_refuses_symlinked_auxiliaries() {
  require(sqlite3_libversion_number() >= kSqliteAuthorityMinimumVersionNumber,
          "linked SQLite satisfies the validated-at version floor");
  for (const std::string_view suffix :
       {std::string_view{"-wal"}, std::string_view{"-journal"}}) {
    TemporaryDirectory temporary;
    const auto database_path = temporary.path() / "journal.db";
    {
      Database database(database_path);
      execute_ok(database.get(), "CREATE TABLE evidence(value TEXT NOT NULL);",
                 "initialize auxiliary symlink fixture");
    }
    const auto victim = temporary.path() / "victim";
    create_regular_0600(victim, "untouched-victim");
    const auto auxiliary =
        std::filesystem::path(database_path.string() + std::string(suffix));
    require(::symlink(victim.filename().c_str(), auxiliary.c_str()) == 0,
            "plant SQLite auxiliary symlink");

    Database database(database_path);
    const std::string sql = suffix == "-wal"
                                ? "PRAGMA journal_mode=WAL; "
                                  "INSERT INTO evidence VALUES('redirect');"
                                : "INSERT INTO evidence VALUES('redirect');";
    const int result = execute(database.get(), sql);
    require(result != SQLITE_OK && (sqlite3_extended_errcode(database.get()) &
                                    0xff) == SQLITE_CANTOPEN,
            "stock SQLite must fail closed on a symlinked auxiliary");
    require(read_file(victim) == "untouched-victim",
            "SQLite must not write through an auxiliary symlink");
  }
}

void authority_rejects_auxiliary_policy_violations() {
  TemporaryDirectory temporary;
  const auto directory = temporary.path() / "authority";
  require(std::filesystem::create_directory(directory),
          "create authority directory");
  require(::chmod(directory.c_str(), 0700) == 0, "protect authority directory");
  const auto database_path = directory / "journal.db";
  auto authority =
      SqliteFilesystemAuthority::acquire(config_for(database_path));

  const auto wal = std::filesystem::path(database_path.string() + "-wal");
  const auto outside = temporary.path() / "outside-wal";
  create_regular_0600(outside);
  require(::link(outside.c_str(), wal.c_str()) == 0,
          "plant hardlinked WAL from outside authority directory");
  require_authority_error(
      [&] { (void)authority.validate_auxiliary_files(); },
      "authority must reject WAL link count greater than one");
  require(std::filesystem::remove(wal), "remove hardlinked WAL");

  create_regular_0600(wal);
  require(::chmod(wal.c_str(), 0644) == 0, "weaken WAL mode");
  require_authority_error([&] { (void)authority.validate_auxiliary_files(); },
                          "authority must reject permissive WAL mode");
  require(std::filesystem::remove(wal), "remove permissive WAL");

  auto wrong_owner = config_for(temporary.path() / "wrong-owner.db");
  wrong_owner.expected_owner_uid =
      static_cast<uid_t>(static_cast<std::uint64_t>(::geteuid()) + 1U);
  require_authority_error(
      [&] {
        auto rejected = SqliteFilesystemAuthority::acquire(wrong_owner);
        (void)rejected;
      },
      "authority must reject a configured owner mismatch");
}

void exclusive_wal_never_creates_shm() {
  TemporaryDirectory temporary;
  const auto database_path = temporary.path() / "exclusive.db";
  const auto shared_memory =
      std::filesystem::path(database_path.string() + "-shm");
  Database database(database_path);
  execute_ok(database.get(), "PRAGMA locking_mode=EXCLUSIVE;",
             "select exclusive SQLite locking");
  execute_ok(database.get(), "PRAGMA journal_mode=WAL;",
             "enable exclusive SQLite WAL");
  execute_ok(database.get(),
             "CREATE TABLE evidence(value INTEGER);"
             "INSERT INTO evidence VALUES(1);"
             "BEGIN IMMEDIATE; INSERT INTO evidence VALUES(2); COMMIT;",
             "write through exclusive SQLite WAL");
  require(!std::filesystem::exists(shared_memory),
          "exclusive SQLite WAL must not create SHM while open");
  database.close();
  require(!std::filesystem::exists(shared_memory),
          "exclusive SQLite WAL must not leave SHM after close");
}

void strict_directory_and_inode_replacement_fail_closed() {
  TemporaryDirectory temporary;
  const auto permissive = temporary.path() / "permissive";
  require(std::filesystem::create_directory(permissive),
          "create permissive authority directory");
  require(::chmod(permissive.c_str(), 0755) == 0,
          "set permissive authority mode");
  require_authority_error(
      [&] {
        auto authority = SqliteFilesystemAuthority::acquire(
            config_for(permissive / "journal.db",
                       SqliteAuthorityEnforcementGrade::strict_filesystem));
        (void)authority;
      },
      "strict authority must reject a directory that is not mode 0700");

  const auto database_path = temporary.path() / "replace.db";
  auto authority =
      SqliteFilesystemAuthority::acquire(config_for(database_path));
  (void)authority.attest_before_open();
  const auto displaced = temporary.path() / "replace.displaced";
  require(::rename(database_path.c_str(), displaced.c_str()) == 0,
          "replace database after pre-open attestation");
  create_regular_0600(database_path);
  require_authority_error(
      [&] { (void)authority.attest_after_open(); },
      "post-open attestation must detect inode replacement");
}

void hostile_same_uid_acquisition_race_fails_closed() {
  TemporaryDirectory temporary;
  const auto authority_directory = temporary.path() / "acquire-authority";
  const auto displaced_directory = temporary.path() / "acquire-displaced";
  require(std::filesystem::create_directory(authority_directory),
          "create acquisition-race authority directory");
  require(::chmod(authority_directory.c_str(), 0700) == 0,
          "protect acquisition-race authority directory");
  const auto database_path = authority_directory / "journal.db";
  const auto victim = temporary.path() / "acquire-victim";
  create_regular_0600(victim, "acquisition-victim");

  std::array<int, 2U> command{};
  std::array<int, 2U> acknowledgement{};
  require(::pipe(command.data()) == 0 && ::pipe(acknowledgement.data()) == 0,
          "create acquisition-race pipes");
  const pid_t child = ::fork();
  require(child >= 0, "fork same-UID acquisition helper");
  if (child == 0) {
    (void)::close(command[1]);
    (void)::close(acknowledgement[0]);
    char token = 0;
    if (::read(command[0], &token, 1U) != 1 ||
        ::link(victim.c_str(), database_path.c_str()) != 0 ||
        ::write(acknowledgement[1], "x", 1U) != 1 ||
        ::read(command[0], &token, 1U) != 1 ||
        ::unlink(database_path.c_str()) != 0 ||
        ::rename(authority_directory.c_str(), displaced_directory.c_str()) !=
            0 ||
        ::symlink(displaced_directory.filename().c_str(),
                  authority_directory.c_str()) != 0 ||
        ::write(acknowledgement[1], "x", 1U) != 1 ||
        ::read(command[0], &token, 1U) != 1 ||
        ::unlink(authority_directory.c_str()) != 0 ||
        ::rename(displaced_directory.c_str(), authority_directory.c_str()) !=
            0) {
      ::_exit(2);
    }
    ::_exit(0);
  }

  (void)::close(command[0]);
  (void)::close(acknowledgement[1]);
  char token = 0;
  require(::write(command[1], "x", 1U) == 1 &&
              ::read(acknowledgement[0], &token, 1U) == 1,
          "helper plants hardlinked database during acquisition");
  require_authority_error(
      [&] {
        auto authority =
            SqliteFilesystemAuthority::acquire(config_for(database_path));
        (void)authority;
      },
      "authority acquisition must reject a helper-planted hardlink");
  require(::write(command[1], "x", 1U) == 1 &&
              ::read(acknowledgement[0], &token, 1U) == 1,
          "helper swaps authority directory for a symlink");
  require_authority_error(
      [&] {
        auto authority =
            SqliteFilesystemAuthority::acquire(config_for(database_path));
        (void)authority;
      },
      "authority acquisition must reject a helper-swapped directory");
  require(::write(command[1], "x", 1U) == 1,
          "allow acquisition helper to restore namespace");
  int status = 0;
  require(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0,
          "same-UID acquisition helper completed");
  (void)::close(command[1]);
  (void)::close(acknowledgement[0]);
  require(read_file(victim) == "acquisition-victim",
          "failed acquisition never writes through a hardlink victim");
}

void hostile_same_uid_boundary_race_poison_is_latched() {
  TemporaryDirectory temporary;
  const auto authority_directory = temporary.path() / "authority";
  const auto displaced_directory = temporary.path() / "authority-displaced";
  require(std::filesystem::create_directory(authority_directory),
          "create raced authority directory");
  require(::chmod(authority_directory.c_str(), 0700) == 0,
          "protect raced authority directory");
  const auto database_path = authority_directory / "journal.db";
  auto authority = std::make_shared<SqliteFilesystemAuthority>(
      SqliteFilesystemAuthority::acquire(config_for(database_path)));
  const auto identity =
      journal_identity(database_path, authority->attest_before_open());
  const auto victim = temporary.path() / "race-victim";
  create_regular_0600(victim, "same-uid-victim");

  std::array<int, 2U> command{};
  std::array<int, 2U> acknowledgement{};
  require(::pipe(command.data()) == 0 && ::pipe(acknowledgement.data()) == 0,
          "create race synchronization pipes");

  bool first_boundary_failed = false;
  bool poison_stayed_latched = false;
  std::uint64_t before = 0U;
  {
    Journal journal(database_path, identity,
                    HostGrantEnforcement::legacy_process_free_test,
                    std::nullopt, authority, true);
    before = journal.event_count();
    require(!std::filesystem::exists(database_path.string() + "-shm"),
            "authority-bound Journal suppresses SHM");

    const pid_t child = ::fork();
    require(child >= 0, "fork hostile same-UID helper");
    if (child == 0) {
      (void)::close(command[1]);
      (void)::close(acknowledgement[0]);
      char token = 0;
      if (::read(command[0], &token, 1U) != 1 ||
          ::rename(authority_directory.c_str(), displaced_directory.c_str()) !=
              0 ||
          ::mkdir(authority_directory.c_str(), 0700) != 0 ||
          ::symlink("../race-victim", database_path.c_str()) != 0 ||
          ::link(
              victim.c_str(),
              std::filesystem::path(database_path.string() + "-wal").c_str()) !=
              0 ||
          ::write(acknowledgement[1], "x", 1U) != 1) {
        ::_exit(2);
      }
      if (::read(command[0], &token, 1U) != 1) {
        ::_exit(3);
      }
      (void)::unlink(
          std::filesystem::path(database_path.string() + "-wal").c_str());
      (void)::unlink(database_path.c_str());
      if (::rmdir(authority_directory.c_str()) != 0 ||
          ::rename(displaced_directory.c_str(), authority_directory.c_str()) !=
              0) {
        ::_exit(4);
      }
      ::_exit(0);
    }

    (void)::close(command[0]);
    (void)::close(acknowledgement[1]);
    require(::write(command[1], "x", 1U) == 1,
            "start hostile namespace mutation");
    char token = 0;
    require(::read(acknowledgement[0], &token, 1U) == 1,
            "observe hostile namespace mutation");
    const trainvm::AuthorityTimeSample now{
        .wall = {.nanoseconds = 1'000},
        .boot = {.nanoseconds = 1'000},
        .boot_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"};
    try {
      (void)journal.acquire_lease("race-resource", "race-run", "race-lease",
                                  now, 10'000);
    } catch (const std::runtime_error &) {
      first_boundary_failed = true;
    }
    require(::write(command[1], "x", 1U) == 1,
            "allow hostile helper to restore namespace");
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0,
            "hostile helper completed its same-UID mutations");
    try {
      (void)journal.acquire_lease("race-resource", "race-run", "race-lease",
                                  now, 10'000);
    } catch (const std::runtime_error &) {
      poison_stayed_latched = true;
    }
    (void)::close(command[1]);
    (void)::close(acknowledgement[0]);
  }
  authority.reset();

  require(
      first_boundary_failed && poison_stayed_latched,
      "authority poisoning must trigger and stay latched after restoration");
  require(read_file(victim) == "same-uid-victim",
          "hostile auxiliary aliases must not redirect a Journal write");

  Database database(database_path);
  sqlite3_stmt *statement = nullptr;
  require(sqlite3_prepare_v2(database.get(),
                             "SELECT COUNT(*) FROM resource_leases", -1,
                             &statement, nullptr) == SQLITE_OK,
          "inspect raced journal after closing poisoned connection");
  require(sqlite3_step(statement) == SQLITE_ROW &&
              sqlite3_column_int64(statement, 0) == 0,
          "no partial lease commit is observable after poisoning");
  require(sqlite3_finalize(statement) == SQLITE_OK,
          "finalize raced journal inspection");
  statement = nullptr;
  require(sqlite3_prepare_v2(database.get(), "SELECT COUNT(*) FROM events", -1,
                             &statement, nullptr) == SQLITE_OK,
          "inspect raced journal event count");
  require(sqlite3_step(statement) == SQLITE_ROW &&
              static_cast<std::uint64_t>(sqlite3_column_int64(statement, 0)) ==
                  before,
          "no partial event commit is observable after poisoning");
  require(sqlite3_finalize(statement) == SQLITE_OK,
          "finalize raced journal event inspection");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string_view, void (*)()>> tests{
      {"stock SQLite symlink refusal",
       stock_sqlite_refuses_symlinked_auxiliaries},
      {"authority auxiliary policy",
       authority_rejects_auxiliary_policy_violations},
      {"exclusive WAL SHM suppression", exclusive_wal_never_creates_shm},
      {"strict directory and inode replacement",
       strict_directory_and_inode_replacement_fail_closed},
      {"hostile same-UID acquisition race",
       hostile_same_uid_acquisition_race_fails_closed},
      {"hostile same-UID race",
       hostile_same_uid_boundary_race_poison_is_latched},
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
