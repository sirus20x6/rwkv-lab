#include "trainvm/sqlite_authority_vfs.hpp"

#include <sqlite3.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using trainvm::SqliteAuthorityVfs;
using trainvm::SqliteAuthorityVfsError;


void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    const std::string value = "/tmp/trainvm-authority-vfs-XXXXXX";
    require(value.size() + 1U <= pattern.size(), "temporary path fits");
    std::copy(value.begin(), value.end(), pattern.begin());
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) throw std::runtime_error("mkdtemp failed");
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0, "temporary directory is 0700");
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

class DirectoryDescriptor final {
 public:
  explicit DirectoryDescriptor(const std::filesystem::path& path)
      : descriptor_(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)) {
    require(descriptor_ >= 0, "open authority directory");
  }
  ~DirectoryDescriptor() {
    if (descriptor_ >= 0) (void)::close(descriptor_);
  }

  DirectoryDescriptor(const DirectoryDescriptor&) = delete;
  DirectoryDescriptor& operator=(const DirectoryDescriptor&) = delete;

  [[nodiscard]] int get() const { return descriptor_; }

 private:
  int descriptor_{-1};
};

class Connection final {
 public:
  Connection(const SqliteAuthorityVfs& vfs, int flags) {
    const int result =
        sqlite3_open_v2(vfs.database_path().c_str(), &database_, flags,
                        vfs.vfs_name().c_str());
    if (result != SQLITE_OK) {
      const std::string message =
          database_ != nullptr ? sqlite3_errmsg(database_) : "unknown";
      if (database_ != nullptr) (void)sqlite3_close(database_);
      database_ = nullptr;
      throw std::runtime_error("sqlite3_open_v2 failed: " + message);
    }
  }

  ~Connection() {
    if (database_ != nullptr) (void)sqlite3_close(database_);
  }

  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;

  [[nodiscard]] sqlite3* get() const { return database_; }

  // Returns SQLITE_OK or the first failing result code. Never throws, so
  // hostile-race cases can assert on the code instead of on an exception type.
  [[nodiscard]] int try_execute(const char* sql) const {
    char* error = nullptr;
    const int result = sqlite3_exec(database_, sql, nullptr, nullptr, &error);
    if (error != nullptr) sqlite3_free(error);
    return result;
  }

  void execute(const char* sql) const {
    char* error = nullptr;
    if (sqlite3_exec(database_, sql, nullptr, nullptr, &error) != SQLITE_OK) {
      const std::string message = error != nullptr ? error : "unknown";
      if (error != nullptr) sqlite3_free(error);
      throw std::runtime_error(std::string("sql failed: ") + sql + ": " +
                               message);
    }
  }

  [[nodiscard]] std::int64_t scalar(const char* sql) const {
    sqlite3_stmt* statement = nullptr;
    require(sqlite3_prepare_v2(database_, sql, -1, &statement, nullptr) ==
                SQLITE_OK,
            "prepare scalar query");
    require(sqlite3_step(statement) == SQLITE_ROW, "scalar query has a row");
    const std::int64_t value = sqlite3_column_int64(statement, 0);
    (void)sqlite3_finalize(statement);
    return value;
  }

  [[nodiscard]] std::string text(const char* sql) const {
    sqlite3_stmt* statement = nullptr;
    require(sqlite3_prepare_v2(database_, sql, -1, &statement, nullptr) ==
                SQLITE_OK,
            "prepare text query");
    require(sqlite3_step(statement) == SQLITE_ROW, "text query has a row");
    const auto* raw = sqlite3_column_text(statement, 0);
    const std::string value =
        raw != nullptr ? reinterpret_cast<const char*>(raw) : std::string{};
    (void)sqlite3_finalize(statement);
    return value;
  }

 private:
  sqlite3* database_{nullptr};
};

constexpr int kReadWrite = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
constexpr std::string_view kDatabaseName = "authority.db";

std::shared_ptr<SqliteAuthorityVfs> authority_for(const DirectoryDescriptor&
                                                      directory) {
  return SqliteAuthorityVfs::create(directory.get(),
                                    std::string(kDatabaseName), ::getuid());
}

void write_file(const std::filesystem::path& path, std::string_view content) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  require(stream.good(), "open decoy file");
  stream.write(content.data(), static_cast<std::streamsize>(content.size()));
  require(stream.good(), "write decoy file");
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  require(stream.good(), "open file for reading");
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------

void seed_wal_database(const SqliteAuthorityVfs& vfs) {
  const Connection connection(vfs, kReadWrite);
  connection.execute("PRAGMA journal_mode=WAL;");
  connection.execute(
      "CREATE TABLE event(sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL);");
  connection.execute("INSERT INTO event(payload) VALUES('alpha'),('beta');");
}

void wal_database_round_trips() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);

  seed_wal_database(*vfs);

  {
    const Connection connection(*vfs, kReadWrite);
    require(connection.text("PRAGMA journal_mode;") == "wal",
            "journal mode survives reopen through the authority VFS");
    require(connection.scalar("SELECT COUNT(*) FROM event;") == 2,
            "rows survive reopen");
    require(connection.text("PRAGMA integrity_check;") == "ok",
            "database is intact");
    connection.execute("INSERT INTO event(payload) VALUES('gamma');");
    require(connection.scalar("SELECT COUNT(*) FROM event;") == 3,
            "writes commit through the authority VFS");
  }

  // The database and its auxiliaries must live in the pinned directory, not
  // somewhere a derived pathname wandered off to.
  require(std::filesystem::exists(directory.path() / kDatabaseName),
          "database is inside the authority directory");
  const auto statistics = vfs->statistics();
  require(statistics.guarded_opens >= 2U, "opens were guarded");
  require(statistics.shared_memory_guards >= 1U, "wal-index was guarded");
  require(statistics.rejected_names == 0U && statistics.rejected_identities == 0U &&
              statistics.rejected_substitutions == 0U,
          "honest traffic is not rejected");
}

void rollback_journal_round_trips() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);
  {
    const Connection connection(*vfs, kReadWrite);
    connection.execute("PRAGMA journal_mode=DELETE;");
    connection.execute("CREATE TABLE t(a INTEGER);");
    connection.execute("BEGIN; INSERT INTO t VALUES(1); INSERT INTO t VALUES(2); COMMIT;");
    require(connection.scalar("SELECT COUNT(*) FROM t;") == 2, "rollback-mode commit");
  }
  // xDelete runs through unlinkat on the pinned descriptor, so the rollback
  // journal must actually be gone rather than merely unreferenced.
  require(!std::filesystem::exists(
              directory.path() / (std::string(kDatabaseName) + "-journal")),
          "rollback journal is deleted through the pinned descriptor");
  require(vfs->statistics().guarded_deletes >= 1U, "deletes were guarded");
}

// A planted symlink at an auxiliary name must fail closed: the operation must
// report an error and the symlink target must never receive a byte. Stock
// SQLite also refuses these, so this pins that the authority VFS does not
// regress a protection the platform already provides - and records it as an
// explicit rejection rather than an accident.
void planted_symlink_is_refused(std::string_view suffix, bool wal_mode) {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);

  const auto decoy = directory.path() / "decoy.outside";
  write_file(decoy, "untouched");
  const auto planted =
      directory.path() / (std::string(kDatabaseName) + std::string(suffix));

  {
    const Connection connection(*vfs, kReadWrite);
    connection.execute("CREATE TABLE t(a INTEGER);");
    if (wal_mode) connection.execute("PRAGMA journal_mode=WAL;");
  }
  std::filesystem::create_symlink(decoy, planted);

  const Connection connection(*vfs, kReadWrite);
  int outcome = SQLITE_OK;
  if (wal_mode) {
    outcome = connection.try_execute("PRAGMA journal_mode=WAL;");
    if (outcome == SQLITE_OK) {
      outcome = connection.try_execute("INSERT INTO t VALUES(1);");
    }
  } else {
    outcome = connection.try_execute(
        "BEGIN IMMEDIATE; INSERT INTO t VALUES(1); COMMIT;");
  }
  require(outcome != SQLITE_OK,
          std::string("planted ") + std::string(suffix) + " symlink is refused");
  require(read_file(decoy) == "untouched",
          std::string("planted ") + std::string(suffix) +
              " symlink target is never written");
  const auto statistics = vfs->statistics();
  require(statistics.rejected_identities >= 1U,
          "a refused symlink is recorded as an identity rejection");
}

void planted_hardlink_alias_is_refused() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);
  {
    const Connection connection(*vfs, kReadWrite);
    connection.execute("CREATE TABLE t(a INTEGER);");
  }

  const auto alias = directory.path() / "alias.copy";
  const auto planted =
      directory.path() / (std::string(kDatabaseName) + "-journal");
  write_file(alias, "");
  require(::link(alias.c_str(), planted.c_str()) == 0, "create hardlink alias");

  const Connection connection(*vfs, kReadWrite);
  require(connection.try_execute(
              "BEGIN IMMEDIATE; INSERT INTO t VALUES(1); COMMIT;") != SQLITE_OK,
          "a hardlinked rollback journal is refused");
  require(vfs->statistics().rejected_identities >= 1U,
          "a hardlink alias is recorded as an identity rejection");
}

void names_outside_the_namespace_are_refused() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);
  seed_wal_database(*vfs);

  sqlite3_vfs* registered = sqlite3_vfs_find(vfs->vfs_name().c_str());
  require(registered != nullptr, "the authority VFS is registered");

  const auto refuses = [&](const std::string& pathname) {
    std::vector<char> storage(
        static_cast<std::size_t>(registered->szOsFile) + 64U, '\0');
    auto* file = reinterpret_cast<sqlite3_file*>(storage.data());
    const int result = registered->xOpen(registered, pathname.c_str(), file,
                                         SQLITE_OPEN_READWRITE, nullptr);
    if (result == SQLITE_OK && file->pMethods != nullptr) {
      (void)file->pMethods->xClose(file);
    }
    return result != SQLITE_OK;
  };

  // A sibling of the database inside the pinned directory that is not one of
  // this database's declared auxiliaries.
  require(refuses(vfs->database_path() + "-evil"), "unknown suffix is refused");
  require(refuses(std::string("/proc/self/fd/") + "0/" +
                  std::string(kDatabaseName)),
          "a different directory descriptor is refused");
  require(refuses("/etc/passwd"), "an absolute path outside the namespace is refused");
  require(refuses(vfs->database_path() + "/../" + std::string(kDatabaseName)),
          "a traversal component is refused");
  require(vfs->statistics().rejected_names >= 4U,
          "every refused name is recorded");
}

void unusable_configuration_is_refused() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto expect_error = [](auto&& callable, std::string_view message) {
    try {
      callable();
    } catch (const SqliteAuthorityVfsError&) {
      return;
    }
    throw std::runtime_error(std::string(message));
  };
  expect_error(
      [&] {
        (void)SqliteAuthorityVfs::create(descriptor.get(), "sub/dir.db",
                                         ::getuid());
      },
      "a database name with a separator is refused");
  expect_error(
      [&] { (void)SqliteAuthorityVfs::create(-1, "a.db", ::getuid()); },
      "a closed directory descriptor is refused");
  expect_error(
      [&] {
        (void)SqliteAuthorityVfs::create(descriptor.get(), "a.db",
                                         ::getuid() + 1U);
      },
      "a directory owned by another identity is refused");
}

constexpr std::string_view kAuthoritySecret = "TOPSECRET-AUTHORITY-PAYLOAD";

// The negative control, and the concrete defect this card closes.
//
// Stock SQLite refuses a *symlinked* auxiliary (it opens journals with
// O_NOFOLLOW), so that vector is already closed upstream. It performs no
// ownership, permission, or link-count check at all, however. A same-UID
// process that pre-creates `<db>-wal` as a hardlink to a file it retains gets
// a live view of every byte the authority writes to its write-ahead log - and
// stock SQLite reports complete success while doing it.
//
// This test asserts the vulnerability still reproduces on the default VFS. If
// a future SQLite closes it, this test fails loudly rather than letting the
// guard below quietly become untested.
void default_vfs_accepts_a_hardlinked_auxiliary() {
  const TemporaryDirectory directory;
  const auto database = directory.path() / kDatabaseName;
  const auto alias = directory.path() / "attacker_view";
  const auto planted =
      directory.path() / (std::string(kDatabaseName) + "-wal");

  const auto run = [&](const char* sql) {
    sqlite3* raw = nullptr;
    require(sqlite3_open_v2(database.c_str(), &raw, kReadWrite, nullptr) ==
                SQLITE_OK,
            "open through the default VFS");
    char* error = nullptr;
    const int result = sqlite3_exec(raw, sql, nullptr, nullptr, &error);
    if (error != nullptr) sqlite3_free(error);
    (void)sqlite3_close(raw);
    return result;
  };
  require(run("PRAGMA journal_mode=WAL; CREATE TABLE t(secret TEXT);") ==
              SQLITE_OK,
          "seed a WAL database on the default VFS");
  std::filesystem::remove(planted);
  write_file(alias, "");
  require(::link(alias.c_str(), planted.c_str()) == 0, "plant the alias");

  const std::string statement = "PRAGMA journal_mode=WAL; INSERT INTO t VALUES('" +
                                std::string(kAuthoritySecret) + "');";
  require(run(statement.c_str()) == SQLITE_OK,
          "the default VFS accepts a hardlinked write-ahead log without complaint");
  require(read_file(alias).find(kAuthoritySecret) != std::string::npos,
          "the default VFS leaked authority payload into the attacker's alias");
}

// The same attack, through the authority VFS: refused, and not one byte of
// authority payload reaches the alias the attacker retained.
void authority_vfs_refuses_a_hardlinked_auxiliary() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);
  const auto alias = directory.path() / "attacker_view";
  const auto planted =
      directory.path() / (std::string(kDatabaseName) + "-wal");

  {
    const Connection connection(*vfs, kReadWrite);
    connection.execute("PRAGMA journal_mode=WAL;");
    connection.execute("CREATE TABLE t(secret TEXT);");
  }
  std::filesystem::remove(planted);
  write_file(alias, "");
  require(::link(alias.c_str(), planted.c_str()) == 0, "plant the alias");

  const Connection connection(*vfs, kReadWrite);
  const std::string statement = "INSERT INTO t VALUES('" +
                                std::string(kAuthoritySecret) + "');";
  require(connection.try_execute(statement.c_str()) != SQLITE_OK,
          "the authority VFS refuses a hardlinked write-ahead log");
  require(read_file(alias).find(kAuthoritySecret) == std::string::npos,
          "no authority payload reaches the attacker's alias");
  require(vfs->statistics().rejected_identities >= 1U,
          "the hardlinked alias is recorded as an identity rejection");
}

// The card's qualification gate: a same-UID process racing auxiliary creation
// must not be able to split authority or redirect a byte of it.
void hostile_same_uid_race_cannot_redirect() {
  const TemporaryDirectory directory;
  const DirectoryDescriptor descriptor(directory.path());
  const auto vfs = authority_for(descriptor);
  seed_wal_database(*vfs);

  const auto decoy = directory.path() / "decoy.outside";
  write_file(decoy, "untouched");

  const std::string wal = std::string(kDatabaseName) + "-wal";
  const std::string shm = std::string(kDatabaseName) + "-shm";
  const std::string rollback = std::string(kDatabaseName) + "-journal";

  const ::pid_t child = ::fork();
  require(child >= 0, "fork the hostile racer");
  if (child == 0) {
    // Same UID, no cooperation: unlink each auxiliary and put a symlink to the
    // decoy in its place, as fast as the kernel allows.
    const int racer_directory =
        ::open(directory.path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (racer_directory >= 0) {
      for (int iteration = 0; iteration < 40000; ++iteration) {
        for (const std::string& name : {wal, shm, rollback}) {
          (void)::unlinkat(racer_directory, name.c_str(), 0);
          (void)::symlinkat(decoy.c_str(), racer_directory, name.c_str());
          (void)::unlinkat(racer_directory, name.c_str(), 0);
        }
      }
      (void)::close(racer_directory);
    }
    ::_exit(0);
  }

  std::uint64_t committed = 0;
  std::uint64_t refused = 0;
  for (int iteration = 0; iteration < 600; ++iteration) {
    Connection connection(*vfs, kReadWrite);
    (void)connection.try_execute("PRAGMA busy_timeout=250;");
    if (connection.try_execute("INSERT INTO event(payload) VALUES('raced');") ==
        SQLITE_OK) {
      ++committed;
    } else {
      ++refused;
    }
  }

  int status = 0;
  (void)::kill(child, SIGKILL);
  (void)::waitpid(child, &status, 0);

  // The decisive assertion: no authority byte was ever redirected into the
  // attacker-controlled target, whatever the outcome of any individual write.
  require(read_file(decoy) == "untouched",
          "no authority write reached the racer's target");

  require(vfs->statistics().rejected_identities +
                  vfs->statistics().rejected_substitutions >=
              1U,
          "the race was actually refused, not merely survived");
  require(committed + refused == 600U, "every attempt was accounted for");

  // The racer can be killed mid-plant, leaving a symlink in place. The
  // authority then stays fail-closed until that artifact is removed, which is
  // the intended behavior, so removing it is part of the recovery path rather
  // than a defect to assert away.
  for (const std::string& name : {wal, shm, rollback}) {
    struct stat residue {};
    if (::fstatat(descriptor.get(), name.c_str(), &residue,
                  AT_SYMLINK_NOFOLLOW) == 0 &&
        S_ISLNK(residue.st_mode)) {
      require(::unlinkat(descriptor.get(), name.c_str(), 0) == 0,
              "remove planted residue");
    }
  }

  // The database must remain a coherent authority: never torn, never forked,
  // and never holding a row that no write reported as committed.
  //
  // The converse is deliberately not asserted. A same-UID process that can
  // unlink the write-ahead log destroys the durability of the frames it
  // removes, and no VFS can prevent that. What must hold is that nothing was
  // corrupted and nothing was invented.
  const Connection verify(*vfs, kReadWrite);
  require(verify.text("PRAGMA integrity_check;") == "ok",
          "the database survives the race intact");
  const auto surviving = static_cast<std::uint64_t>(
      verify.scalar("SELECT COUNT(*) FROM event WHERE payload='raced';"));
  require(surviving <= committed,
          "no row survives that was never reported as committed");

  // Fail-closed must be recoverable, not poisoning: once the racer is gone the
  // authority accepts writes again through the same pinned namespace.
  require(verify.try_execute("INSERT INTO event(payload) VALUES('after');") ==
              SQLITE_OK,
          "writes resume once the race stops");
  require(verify.scalar("SELECT COUNT(*) FROM event WHERE payload='after';") == 1,
          "the resumed write is durable");

  std::cout << "  race: committed=" << committed << " refused=" << refused
            << " surviving=" << surviving << '\n';
}

struct Test final {
  std::string_view name;
  void (*run)();
};

const Test kTests[] = {
    {"wal_database_round_trips", &wal_database_round_trips},
    {"rollback_journal_round_trips", &rollback_journal_round_trips},
    {"planted_wal_symlink_is_refused",
     [] { planted_symlink_is_refused("-wal", true); }},
    {"planted_shm_symlink_is_refused",
     [] { planted_symlink_is_refused("-shm", true); }},
    {"planted_rollback_symlink_is_refused",
     [] { planted_symlink_is_refused("-journal", false); }},
    {"planted_hardlink_alias_is_refused", &planted_hardlink_alias_is_refused},
    {"names_outside_the_namespace_are_refused",
     &names_outside_the_namespace_are_refused},
    {"unusable_configuration_is_refused", &unusable_configuration_is_refused},
    {"default_vfs_accepts_a_hardlinked_auxiliary",
     &default_vfs_accepts_a_hardlinked_auxiliary},
    {"authority_vfs_refuses_a_hardlinked_auxiliary",
     &authority_vfs_refuses_a_hardlinked_auxiliary},
    {"hostile_same_uid_race_cannot_redirect",
     &hostile_same_uid_race_cannot_redirect},
};

}  // namespace

int main() {
  int failures = 0;
  for (const Test& test : kTests) {
    try {
      test.run();
      std::cout << "ok   " << test.name << '\n';
    } catch (const std::exception& error) {
      std::cout << "FAIL " << test.name << ": " << error.what() << '\n';
      ++failures;
    }
  }
  if (failures != 0) {
    std::cout << failures << " sqlite authority VFS test(s) failed\n";
    return 1;
  }
  std::cout << "all sqlite authority VFS tests passed\n";
  return 0;
}
