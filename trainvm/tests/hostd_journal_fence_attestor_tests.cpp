#include "trainvm/hostd_journal_fence_attestor.hpp"

#include <fcntl.h>
#include <sqlite3.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace trainvm;

constexpr std::string_view kBootId =
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

AuthorityTimeSample authority_time(std::int64_t boottime_ns) {
  return {.wall = {.nanoseconds = boottime_ns},
          .boot = {.nanoseconds = boottime_ns},
          .boot_id = std::string(kBootId)};
}

class FakeTimeSource final : public IHostdSessionChallengeTimeSource {
 public:
  HostdSessionChallengeTime now() override {
    if (fail)
      throw std::runtime_error("injected time failure");
    const HostdSessionChallengeTime observed = value;
    ++calls;
    if (callback && calls == callback_on_call) {
      auto invoked = std::move(callback);
      callback = {};
      invoked();
    }
    if (next_boottime_ns) {
      value.boottime_ns = *next_boottime_ns;
      next_boottime_ns.reset();
    }
    return observed;
  }

  HostdSessionChallengeTime value{.boot_id = std::string(kBootId),
                                  .boottime_ns = 1'100'000'000LL};
  std::optional<std::int64_t> next_boottime_ns;
  std::function<void()> callback;
  std::size_t calls{};
  std::size_t callback_on_call{};
  bool fail{};
};

class PinnedAuthority final {
 public:
  PinnedAuthority() {
    std::array<char, 64U> pattern{};
    const std::string value = "/tmp/trainvm-journal-attestor-XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    char* made = ::mkdtemp(pattern.data());
    if (made == nullptr)
      throw std::runtime_error("could not create attestor test directory");
    directory_ = made;
    database_ = directory_ / "journal.db";
    authority_ = directory_ / "journal.db.authority.lock";
    database_descriptor_ = open_retained(database_);
    authority_descriptor_ = open_retained(authority_);
    if (::flock(database_descriptor_, LOCK_EX | LOCK_NB) != 0 ||
        ::flock(authority_descriptor_, LOCK_EX | LOCK_NB) != 0) {
      throw std::runtime_error("could not retain attestor test authority");
    }
    struct stat directory_status {};
    struct stat database_status {};
    struct stat authority_status {};
    if (::stat(directory_.c_str(), &directory_status) != 0 ||
        ::fstat(database_descriptor_, &database_status) != 0 ||
        ::fstat(authority_descriptor_, &authority_status) != 0) {
      throw std::runtime_error("could not stat attestor test authority");
    }
    identity_ = {
        .directory_path = directory_.string(),
        .journal_name = database_.filename().string(),
        .authority_name = authority_.filename().string(),
        .directory_device =
            static_cast<std::uint64_t>(directory_status.st_dev),
        .directory_inode =
            static_cast<std::uint64_t>(directory_status.st_ino),
        .device = static_cast<std::uint64_t>(database_status.st_dev),
        .inode = static_cast<std::uint64_t>(database_status.st_ino),
        .authority_device =
            static_cast<std::uint64_t>(authority_status.st_dev),
        .authority_inode =
            static_cast<std::uint64_t>(authority_status.st_ino),
        .owner_uid = static_cast<std::uint64_t>(::geteuid())};
  }

  ~PinnedAuthority() {
    if (authority_descriptor_ >= 0)
      (void)::close(authority_descriptor_);
    if (database_descriptor_ >= 0)
      (void)::close(database_descriptor_);
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  PinnedAuthority(const PinnedAuthority&) = delete;
  PinnedAuthority& operator=(const PinnedAuthority&) = delete;

  [[nodiscard]] const std::filesystem::path& database() const noexcept {
    return database_;
  }
  [[nodiscard]] const JournalFileIdentity& identity() const noexcept {
    return identity_;
  }

 private:
  static int open_retained(const std::filesystem::path& path) {
    const int descriptor =
        ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW,
               S_IRUSR | S_IWUSR);
    if (descriptor < 0)
      throw std::runtime_error("could not create attestor authority file");
    return descriptor;
  }

  std::filesystem::path directory_;
  std::filesystem::path database_;
  std::filesystem::path authority_;
  int database_descriptor_{-1};
  int authority_descriptor_{-1};
  JournalFileIdentity identity_;
};

struct Fixture final {
  Fixture()
      : time(std::make_shared<FakeTimeSource>()),
        journal(std::make_unique<Journal>(
            authority.database(), authority.identity(),
            HostGrantEnforcement::required,
            HostIdentity{.host_id = "host-001",
                         .boot_id = std::string(kBootId)})) {
    const LeaseAcquireResult acquired = journal->acquire_lease(
        "gpu:0", "run-001", "lease-001", authority_time(1'000'000'000LL),
        10'000'000'000LL);
    require(acquired.status == LeaseAcquireStatus::acquired,
            "fixture acquires one boot-scoped logical fence");
    controller = {.run_id = "run-001",
                  .concurrency_key = "gpu:0",
                  .controller_id = "controller-001",
                  .controller_generation = 7U,
                  .logical_lease_id = acquired.lease.lease_id,
                  .logical_fencing_token = acquired.lease.fencing_token};
    rebuild_attestor(controller);
  }

  void rebuild_attestor(HostdJournalControllerFence value) {
    controller = std::move(value);
    attestor = std::make_shared<HostdJournalFenceAttestor>(
        *journal,
        HostdJournalFenceAttestorConfig{.api_version = std::string(
                                            kHostdJournalFenceAttestorApiVersion),
                                        .broker_epoch = "broker-001",
                                        .controller = controller},
        time);
  }

  [[nodiscard]] HostdJournalFenceQuery query(
      const HostdSessionChallengeClaim* override_claim = nullptr) const {
    return {.api_version = std::string(kHostdJournalFenceQueryApiVersion),
            .challenge_id = "hostd-challenge-" + std::string(64U, 'a'),
            .session_nonce = "hostd-session-nonce-" + std::string(64U, 'b'),
            .host_id = "host-001",
            .boot_id = std::string(kBootId),
            .broker_epoch = "broker-001",
            .claim = override_claim == nullptr ? attestor->claim()
                                               : *override_claim,
            .issued_boottime_ns = 1'000'000'000LL,
            .expires_boottime_ns = 3'000'000'000LL};
  }

  PinnedAuthority authority;
  std::shared_ptr<FakeTimeSource> time;
  std::unique_ptr<Journal> journal;
  HostdJournalControllerFence controller;
  std::shared_ptr<HostdJournalFenceAttestor> attestor;
};

void execute_external(const std::filesystem::path& path,
                      std::string_view statement) {
  sqlite3* database = nullptr;
  if (sqlite3_open_v2(path.c_str(), &database, SQLITE_OPEN_READWRITE,
                      nullptr) != SQLITE_OK) {
    if (database != nullptr)
      sqlite3_close(database);
    throw std::runtime_error("could not open corruption connection");
  }
  char* message = nullptr;
  const std::string owned(statement);
  const int status =
      sqlite3_exec(database, owned.c_str(), nullptr, nullptr, &message);
  const std::string detail =
      message == nullptr ? sqlite3_errmsg(database) : message;
  sqlite3_free(message);
  sqlite3_close(database);
  if (status != SQLITE_OK)
    throw std::runtime_error("corruption statement failed: " + detail);
}

std::string query_external_text(const std::filesystem::path& path,
                                std::string_view statement) {
  sqlite3* database = nullptr;
  if (sqlite3_open_v2(path.c_str(), &database, SQLITE_OPEN_READONLY,
                      nullptr) != SQLITE_OK) {
    if (database != nullptr)
      sqlite3_close(database);
    throw std::runtime_error("could not open inspection connection");
  }
  sqlite3_stmt* query = nullptr;
  const std::string owned(statement);
  if (sqlite3_prepare_v2(database, owned.c_str(), -1, &query, nullptr) !=
          SQLITE_OK ||
      sqlite3_step(query) != SQLITE_ROW ||
      sqlite3_column_type(query, 0) != SQLITE_TEXT) {
    if (query != nullptr)
      sqlite3_finalize(query);
    sqlite3_close(database);
    throw std::runtime_error("could not read inspection value");
  }
  const auto* bytes = sqlite3_column_text(query, 0);
  const std::string result =
      bytes == nullptr ? std::string{} : reinterpret_cast<const char*>(bytes);
  if (sqlite3_step(query) != SQLITE_DONE) {
    sqlite3_finalize(query);
    sqlite3_close(database);
    throw std::runtime_error("inspection value is ambiguous");
  }
  sqlite3_finalize(query);
  sqlite3_close(database);
  return result;
}

void successful_attestation_is_exact_and_observational() {
  Fixture fixture;
  const HostdJournalFenceEvidence evidence =
      fixture.attestor->attest(fixture.query());
  require(evidence.live && evidence.host_id == "host-001" &&
              evidence.boot_id == kBootId &&
              evidence.broker_epoch == "broker-001" &&
              evidence.controller == fixture.controller &&
              evidence.journal == fixture.attestor->claim().journal &&
              evidence.observed_boottime_ns == 1'100'000'000LL &&
              !evidence.evidence_digest.empty(),
          "attestation binds authority, epoch, controller, lease, and time");
  require(fixture.journal->active_lease("gpu:0", authority_time(
                                                   1'100'000'000LL))
              .has_value(),
          "read-only attestation does not release or advance the lease");
}

void controller_run_and_generation_mismatch_fail_closed() {
  Fixture fixture;
  for (std::size_t field = 0U; field < 4U; ++field) {
    HostdSessionChallengeClaim changed = fixture.attestor->claim();
    if (field == 0U)
      changed.controller.run_id = "run-other";
    else if (field == 1U)
      changed.controller.concurrency_key = "gpu:1";
    else if (field == 2U)
      changed.controller.controller_id = "controller-other";
    else
      ++changed.controller.controller_generation;
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query(&changed)); },
        "run, controller, and generation are exact retained bindings");
  }
  {
    auto wrong_host = fixture.query();
    wrong_host.host_id = "host-other";
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(wrong_host); },
        "query host must match the Journal retained host authority");
  }
  {
    Journal unpinned(fixture.authority.database());
    require_throws<HostdJournalFenceAttestorError>(
        [&] {
          HostdJournalFenceAttestor rejected(
              unpinned,
              {.api_version =
                   std::string(kHostdJournalFenceAttestorApiVersion),
               .broker_epoch = "broker-001",
               .controller = fixture.controller},
              fixture.time);
          (void)rejected;
        },
        "an arbitrary-path Journal without retained authority cannot attest");
  }
}

void controller_generation_is_durable_monotonic_and_unique() {
  Fixture fixture;
  const auto prior_attestor = fixture.attestor;
  const auto prior_query = fixture.query();
  fixture.attestor.reset();
  const auto construct = [&](const HostdJournalControllerFence& controller) {
    return std::make_shared<HostdJournalFenceAttestor>(
        *fixture.journal,
        HostdJournalFenceAttestorConfig{
            .api_version =
                std::string(kHostdJournalFenceAttestorApiVersion),
            .broker_epoch = "broker-001",
            .controller = controller},
        fixture.time);
  };
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)construct(fixture.controller); },
      "an already registered generation cannot be reused");
  HostdJournalControllerFence decreased = fixture.controller;
  decreased.controller_id = "controller-decreased";
  --decreased.controller_generation;
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)construct(decreased); },
      "a lower durable controller generation is rejected");
  HostdJournalControllerFence reused_id = fixture.controller;
  ++reused_id.controller_generation;
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)construct(reused_id); },
      "a controller ID cannot be reused at a newer generation");
  HostdJournalControllerFence next = reused_id;
  next.controller_id = "controller-002";
  const auto accepted = construct(next);
  require(accepted->claim().controller == next,
          "failed registrations leave the exact next generation available");
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)prior_attestor->attest(prior_query); },
      "a newer durable controller generation supersedes the old attestor");
}

void controller_generations_are_independent_per_exact_scope() {
  Fixture fixture;
  const auto first_attestor = fixture.attestor;
  const auto first_query = fixture.query();
  const LeaseAcquireResult second_lease = fixture.journal->acquire_lease(
      "gpu:1", "run-002", "lease-002", authority_time(1'000'000'000LL),
      10'000'000'000LL);
  require(second_lease.status == LeaseAcquireStatus::acquired,
          "second scope acquires an independent logical fence");
  const HostdJournalControllerFence second_controller{
      .run_id = "run-002",
      .concurrency_key = "gpu:1",
      // A process controller identity may legitimately control more than one
      // resource scope; its durable reuse authority is therefore per scope.
      .controller_id = fixture.controller.controller_id,
      .controller_generation = fixture.controller.controller_generation,
      .logical_lease_id = second_lease.lease.lease_id,
      .logical_fencing_token = second_lease.lease.fencing_token};
  const auto construct = [&](const HostdJournalControllerFence& controller) {
    return std::make_shared<HostdJournalFenceAttestor>(
        *fixture.journal,
        HostdJournalFenceAttestorConfig{
            .api_version =
                std::string(kHostdJournalFenceAttestorApiVersion),
            .broker_epoch = "broker-001",
            .controller = controller},
        fixture.time);
  };
  const auto second_attestor = construct(second_controller);
  const auto query_for = [&](const auto& attestor, char challenge,
                             char nonce) {
    return HostdJournalFenceQuery{
        .api_version = std::string(kHostdJournalFenceQueryApiVersion),
        .challenge_id = "hostd-challenge-" + std::string(64U, challenge),
        .session_nonce =
            "hostd-session-nonce-" + std::string(64U, nonce),
        .host_id = "host-001",
        .boot_id = std::string(kBootId),
        .broker_epoch = "broker-001",
        .claim = attestor->claim(),
        .issued_boottime_ns = 1'000'000'000LL,
        .expires_boottime_ns = 3'000'000'000LL};
  };
  const auto second_query = query_for(second_attestor, 'c', 'd');
  require(first_attestor->attest(first_query).controller ==
              fixture.controller &&
              second_attestor->attest(second_query).controller ==
                  second_controller,
          "equal generations and controller IDs remain live in two scopes");

  HostdJournalControllerFence advanced_first = fixture.controller;
  advanced_first.controller_id = "controller-002";
  ++advanced_first.controller_generation;
  const auto advanced_first_attestor = construct(advanced_first);
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)first_attestor->attest(first_query); },
      "a newer controller invalidates the old controller in the same scope");
  require(second_attestor->attest(second_query).controller ==
              second_controller,
          "advancing one scope does not invalidate another scope");
  require(advanced_first_attestor
              ->attest(query_for(advanced_first_attestor, 'e', 'f'))
              .controller == advanced_first,
          "the advanced same-scope controller remains live");
}

void controller_scope_aliases_and_legacy_global_heads_fail_closed() {
  {
    Fixture fixture;
    const LeaseAcquireResult second_lease = fixture.journal->acquire_lease(
        "gpu:1", "run-002", "lease-002", authority_time(1'000'000'000LL),
        10'000'000'000LL);
    require(second_lease.status == LeaseAcquireStatus::acquired,
            "alias fixture acquires a second logical scope");
    const HostdJournalControllerFence second_controller{
        .run_id = "run-002",
        .concurrency_key = "gpu:1",
        .controller_id = "controller-scope-1",
        .controller_generation = fixture.controller.controller_generation,
        .logical_lease_id = second_lease.lease.lease_id,
        .logical_fencing_token = second_lease.lease.fencing_token};
    auto second_attestor = std::make_shared<HostdJournalFenceAttestor>(
        *fixture.journal,
        HostdJournalFenceAttestorConfig{
            .api_version =
                std::string(kHostdJournalFenceAttestorApiVersion),
            .broker_epoch = "broker-001",
            .controller = second_controller},
        fixture.time);
    execute_external(
        fixture.authority.database(),
        "UPDATE journal_meta AS target SET value=(SELECT value FROM "
        "journal_meta WHERE key LIKE 'hostd_controller_head:%' AND "
        "value LIKE '%\"concurrency_key\":\"gpu:0\"%') WHERE "
        "target.key LIKE 'hostd_controller_head:%' AND target.value LIKE "
        "'%\"concurrency_key\":\"gpu:1\"%'");
    require_throws<OperationPreconditionError>(
        [&] { (void)fixture.journal->journal_authority_snapshot(); },
        "generic authority verification immediately detects a copied "
        "cross-scope head");
    require_throws<OperationPreconditionError>(
        [&] { (void)fixture.journal->journal_authority_snapshot(); },
        "a cross-scope authority alias permanently poisons the Journal");
    second_attestor.reset();
    fixture.attestor.reset();
    fixture.journal.reset();
    require_throws<std::runtime_error>(
        [&] {
          fixture.journal = std::make_unique<Journal>(
              fixture.authority.database(), fixture.authority.identity(),
              HostGrantEnforcement::required,
              HostIdentity{.host_id = "host-001",
                           .boot_id = std::string(kBootId)});
        },
        "startup rejects a scoped head whose key hashes a different scope");
  }
  {
    Fixture fixture;
    execute_external(
        fixture.authority.database(),
        "UPDATE journal_meta SET key='hostd_controller_head' WHERE key LIKE "
        "'hostd_controller_head:%'");
    fixture.attestor.reset();
    fixture.journal.reset();
    require_throws<std::runtime_error>(
        [&] {
          fixture.journal = std::make_unique<Journal>(
              fixture.authority.database(), fixture.authority.identity(),
              HostGrantEnforcement::required,
              HostIdentity{.host_id = "host-001",
                           .boot_id = std::string(kBootId)});
        },
        "startup preserves but fails closed on a legacy global controller "
        "head");
  }
}

void release_expiry_and_superseding_fence_fail_closed() {
  {
    Fixture fixture;
    const auto query = fixture.query();
    fixture.time->callback_on_call = fixture.time->calls + 2U;
    fixture.time->callback = [&] {
      require(fixture.journal->release_lease(
                  "gpu:0", "run-001", "lease-001",
                  fixture.controller.logical_fencing_token,
                  authority_time(1'200'000'000LL)),
              "mid-attestation callback releases the exact lease");
    };
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(query); },
        "release between the first snapshot and final revision reread rejects");
    require(!fixture.journal->journal_authority_snapshot().journal_id.empty(),
            "an ordinary concurrent release does not poison journal authority");
  }
  {
    Fixture fixture;
    require(fixture.journal->release_lease(
                "gpu:0", "run-001", "lease-001",
                fixture.controller.logical_fencing_token,
                authority_time(1'200'000'000LL)),
            "fixture releases its logical fence");
    fixture.time->value.boottime_ns = 1'300'000'000LL;
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "durably released lease cannot attest live");
  }
  {
    Fixture fixture;
    fixture.time->value.boottime_ns = 10'000'000'000LL;
    fixture.time->next_boottime_ns = 12'000'000'000LL;
    auto slow_query = fixture.query();
    slow_query.issued_boottime_ns = 9'000'000'000LL;
    slow_query.expires_boottime_ns = 20'000'000'000LL;
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(slow_query); },
        "snapshot callback crossing logical lease expiry fails closed");
  }
  {
    Fixture fixture;
    fixture.time->value.boottime_ns = 12'000'000'000LL;
    auto stale_query = fixture.query();
    stale_query.issued_boottime_ns = 11'000'000'000LL;
    stale_query.expires_boottime_ns = 20'000'000'000LL;
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(stale_query); },
        "expired logical lease cannot attest live");

    const LeaseAcquireResult successor = fixture.journal->acquire_lease(
        "gpu:0", "run-002", "lease-002",
        authority_time(12'000'000'000LL), 10'000'000'000LL);
    require(successor.status == LeaseAcquireStatus::acquired &&
                successor.lease.fencing_token >
                    fixture.controller.logical_fencing_token,
            "successor lease advances the durable fence");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(stale_query); },
        "challenge racing a superseding fence fails closed");
  }
}

void torn_projection_and_namespace_replacement_poison_fail_closed() {
  {
    Fixture fixture;
    const ResourceLease before = *fixture.journal->active_lease(
        "gpu:0", authority_time(1'200'000'000LL));
    const LeaseRenewalResult renewal = fixture.journal->renew_lease_exact(
        before, authority_time(1'200'000'000LL), 12'000'000'000LL);
    require(renewal.status == LeaseRenewalStatus::renewed,
            "fixture records a renewal lineage");
    execute_external(
        fixture.authority.database(),
        "UPDATE resource_leases SET expires_boottime_ns="
        "expires_boottime_ns+1 WHERE concurrency_key='gpu:0'");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "current lease divergence from renewal receipts fails closed");
    execute_external(
        fixture.authority.database(),
        "UPDATE resource_leases SET expires_boottime_ns="
        "expires_boottime_ns-1 WHERE concurrency_key='gpu:0'");
    require_throws<OperationPreconditionError>(
        [&] { (void)fixture.journal->journal_authority_snapshot(); },
        "verified projection corruption permanently poisons the Journal");
  }
  {
    Fixture fixture;
    require(fixture.journal->release_lease(
                "gpu:0", "run-001", "lease-001",
                fixture.controller.logical_fencing_token,
                authority_time(1'200'000'000LL)),
            "fixture records authenticated release closure");
    execute_external(fixture.authority.database(),
                     "DELETE FROM resource_lease_releases WHERE "
                     "concurrency_key='gpu:0'");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "deleting a release projection cannot revive a released lease");
    require_throws<OperationPreconditionError>(
        [&] { (void)fixture.journal->journal_authority_snapshot(); },
        "torn authenticated release closure permanently poisons authority");
  }
  {
    Fixture fixture;
    execute_external(
        fixture.authority.database(),
        "UPDATE events SET payload_json='{}' WHERE "
        "event_type='authority.resource_lease_acquired'");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "acquisition-root corruption is detected by the journal hash chain");
  }
  {
    Fixture fixture;
    const auto moved = fixture.authority.database().string() + ".moved";
    std::filesystem::rename(fixture.authority.database(), moved);
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "missing or replaced authority inode fails closed");
    std::filesystem::rename(moved, fixture.authority.database());
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "authority replacement permanently poisons the live Journal");
  }
}

void authority_head_rollback_cannot_reenable_immutable_history() {
  {
    Fixture fixture;
    const std::string prior_head = query_external_text(
        fixture.authority.database(),
        "SELECT value FROM journal_meta WHERE key LIKE "
        "'hostd_controller_head:%'");
    HostdJournalControllerFence next = fixture.controller;
    next.controller_id = "controller-rollback-next";
    ++next.controller_generation;
    auto next_attestor = std::make_shared<HostdJournalFenceAttestor>(
        *fixture.journal,
        HostdJournalFenceAttestorConfig{
            .api_version =
                std::string(kHostdJournalFenceAttestorApiVersion),
            .broker_epoch = "broker-001",
            .controller = next},
        fixture.time);
    execute_external(
        fixture.authority.database(),
        "UPDATE journal_meta SET value='" + prior_head +
            "' WHERE key LIKE 'hostd_controller_head:%'");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "rolled-back controller head cannot hide the immutable next epoch");
    next_attestor.reset();
    fixture.attestor.reset();
    fixture.journal.reset();
    fixture.journal = std::make_unique<Journal>(
        fixture.authority.database(), fixture.authority.identity(),
        HostGrantEnforcement::required,
        HostIdentity{.host_id = "host-001", .boot_id = std::string(kBootId)});
    require_throws<OperationPreconditionError>(
        [&] {
          fixture.journal->require_current_hostd_controller_fence(
              {.broker_epoch = "broker-001",
               .run_id = fixture.controller.run_id,
               .concurrency_key = fixture.controller.concurrency_key,
               .controller_id = fixture.controller.controller_id,
               .controller_generation =
                   fixture.controller.controller_generation,
               .logical_lease_id = fixture.controller.logical_lease_id,
               .logical_fencing_token =
                   fixture.controller.logical_fencing_token});
        },
        "controller rollback remains detectable after reopening the Journal");
    require_throws<OperationPreconditionError>(
        [&] { (void)fixture.journal->journal_authority_snapshot(); },
        "reopened controller rollback permanently poisons authority");
  }
  {
    Fixture fixture;
    const ResourceLease before = *fixture.journal->active_lease(
        "gpu:0", authority_time(1'100'000'000LL));
    const std::string prior_head = query_external_text(
        fixture.authority.database(),
        "SELECT value FROM journal_meta WHERE key LIKE "
        "'lease_authority_head:%'");
    const LeaseRenewalResult renewed = fixture.journal->renew_lease_exact(
        before, authority_time(1'200'000'000LL), 12'000'000'000LL);
    require(renewed.status == LeaseRenewalStatus::renewed,
            "rollback fixture records an immutable renewal event");
    execute_external(
        fixture.authority.database(),
        "BEGIN IMMEDIATE; UPDATE journal_meta SET value='" + prior_head +
            "' WHERE key LIKE 'lease_authority_head:%'; COMMIT;");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "rolled-back renewal head cannot hide the immutable next revision");
    fixture.attestor.reset();
    fixture.journal.reset();
    fixture.journal = std::make_unique<Journal>(
        fixture.authority.database(), fixture.authority.identity(),
        HostGrantEnforcement::required,
        HostIdentity{.host_id = "host-001", .boot_id = std::string(kBootId)});
    require_throws<OperationPreconditionError>(
        [&] {
          (void)fixture.journal->journal_logical_fence_snapshot(
              "gpu:0", "run-001", "lease-001",
              fixture.controller.logical_fencing_token,
              authority_time(1'300'000'000LL));
        },
        "renewal rollback remains detectable after reopening the Journal");
  }
  {
    Fixture fixture;
    const std::string prior_head = query_external_text(
        fixture.authority.database(),
        "SELECT value FROM journal_meta WHERE key LIKE "
        "'lease_authority_head:%'");
    require(fixture.journal->release_lease(
                "gpu:0", "run-001", "lease-001",
                fixture.controller.logical_fencing_token,
                authority_time(1'200'000'000LL)),
            "rollback fixture records an immutable release event");
    execute_external(
        fixture.authority.database(),
        "BEGIN IMMEDIATE; UPDATE journal_meta SET value='" + prior_head +
            "' WHERE key LIKE 'lease_authority_head:%'; "
            "UPDATE resource_leases SET released_wall_time_ns=NULL "
            "WHERE concurrency_key='gpu:0'; "
            "DELETE FROM resource_lease_releases WHERE "
            "concurrency_key='gpu:0'; COMMIT;");
    require_throws<HostdJournalFenceAttestorError>(
        [&] { (void)fixture.attestor->attest(fixture.query()); },
        "rolled-back release head and mutable rows cannot revive the lease");
    fixture.attestor.reset();
    fixture.journal.reset();
    fixture.journal = std::make_unique<Journal>(
        fixture.authority.database(), fixture.authority.identity(),
        HostGrantEnforcement::required,
        HostIdentity{.host_id = "host-001", .boot_id = std::string(kBootId)});
    require_throws<OperationPreconditionError>(
        [&] {
          (void)fixture.journal->journal_logical_fence_snapshot(
              "gpu:0", "run-001", "lease-001",
              fixture.controller.logical_fencing_token,
              authority_time(1'300'000'000LL));
        },
        "release rollback remains detectable after reopening the Journal");
  }
}

void long_lived_renewal_head_has_no_lifetime_scan_cap() {
  Fixture fixture;
  ResourceLease current = *fixture.journal->active_lease(
      "gpu:0", authority_time(1'100'000'000LL));
  constexpr std::size_t renewals = 4097U;
  for (std::size_t index = 0U; index < renewals; ++index) {
    const std::int64_t renewed_at =
        1'200'000'000LL + static_cast<std::int64_t>(index) * 1'000'000LL;
    const LeaseRenewalResult result = fixture.journal->renew_lease_exact(
        current, authority_time(renewed_at), 10'000'000'000LL);
    require(result.status == LeaseRenewalStatus::renewed && result.receipt,
            "long-lived fixture advances an exact renewal authority head");
    current.expires_boottime_ns = result.receipt->new_expires_boottime_ns;
    current.expires_wall_time_ns = result.receipt->new_expires_wall_time_ns;
  }
  fixture.time->value.boottime_ns = 6'000'000'000LL;
  auto query = fixture.query();
  query.issued_boottime_ns = 5'000'000'000LL;
  query.expires_boottime_ns = 10'000'000'000LL;
  const auto evidence = fixture.attestor->attest(query);
  require(evidence.live,
          "authenticated O(1) renewal head supports more than 4096 renewals");
}

void prior_boot_fence_is_stale_without_poisoning_journal() {
  Fixture fixture;
  fixture.attestor.reset();
  fixture.journal.reset();
  constexpr std::string_view next_boot =
      "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
  fixture.journal = std::make_unique<Journal>(
      fixture.authority.database(), fixture.authority.identity(),
      HostGrantEnforcement::required,
      HostIdentity{.host_id = "host-001", .boot_id = std::string(next_boot)});
  AuthorityTimeSample after_reboot = authority_time(1'300'000'000LL);
  after_reboot.boot_id = std::string(next_boot);
  require_throws<OperationPreconditionError>(
      [&] {
        (void)fixture.journal->journal_logical_fence_snapshot(
            "gpu:0", "run-001", "lease-001",
            fixture.controller.logical_fencing_token, after_reboot);
      },
      "a prior-boot logical fence is rejected as stale");
  require(!fixture.journal->journal_id().empty(),
          "a stale prior-boot fence does not poison unrelated journal reads");
}

// The card this qualifies ("Prevent post-reboot journal authority poisoning")
// asks for sustained dashboard reads with hostd reconciliation active, not one
// read after one rejected fence. The reported symptom was latching: the
// controller served ListRuns correctly at first and became permanently
// unreadable "within seconds", because `corrupt()` sets `authority_poisoned_`,
// which the SQLite authorizer consults on every prepare from then on. A
// single-read assertion therefore passes against the broken build too -- the
// first read is the one that always worked. This drives hostd's real
// reconciliation entry point repeatedly against a journal whose retained lease
// row carries a prior boot id, and reads the dashboard's own ListRuns surface
// between every attempt, for a window wider than the one the symptom appeared
// in.
void sustained_dashboard_reads_survive_prior_boot_reconciliation() {
  Fixture fixture;
  constexpr std::string_view next_boot =
      "cccccccc-cccc-cccc-cccc-cccccccccccc";

  // What the dashboard's ListRuns handler actually touches: the journal
  // identity it stamps on the response, and the bounded run projection scan.
  // Both are ordinary prepares, so both are denied once the journal is
  // poisoned.
  struct DashboardRead final {
    std::string journal_id;
    std::size_t runs{};
    std::size_t authority_events{};
    bool operator==(const DashboardRead&) const = default;
  };
  const auto list_runs = [](const Journal& journal) -> DashboardRead {
    const std::vector<RunProjection> projections =
        journal.run_projections({.observed_states = {},
                                 .labels = {},
                                 .after = std::nullopt,
                                 .limit = 50U});
    return {.journal_id = journal.journal_id(),
            .runs = projections.size(),
            .authority_events = journal.events_for_run("run-001").size()};
  };

  const DashboardRead before_reboot = list_runs(*fixture.journal);
  require(!before_reboot.journal_id.empty(),
          "the fixture journal is readable before the reboot");

  // Fake the reboot the way the host does: the retained lease row keeps the
  // boot id it was acquired under, and everything attested at startup -- the
  // host grant identity and the authority time samples hostd reconciles with
  // -- carries the new one. Leaving the time source on the old boot id would
  // instead trip the earlier, already non-poisoning "fence time disagrees with
  // retained host boot authority" guard, and this test would pass against the
  // broken build without ever reaching the branch it is about.
  fixture.attestor.reset();
  fixture.journal.reset();
  fixture.journal = std::make_unique<Journal>(
      fixture.authority.database(), fixture.authority.identity(),
      HostGrantEnforcement::required,
      HostIdentity{.host_id = "host-001", .boot_id = std::string(next_boot)});
  fixture.time->value.boot_id = std::string(next_boot);

  const DashboardRead first_after_reboot = list_runs(*fixture.journal);
  require(first_after_reboot == before_reboot,
          "the first post-reboot dashboard read still serves the same runs");

  // hostd reconciliation: constructing the fence attestor is the startup
  // handshake, and it registers the controller fence, which takes the logical
  // fence snapshot over the retained prior-boot lease row. That is the call
  // that used to poison the journal.
  const auto reconcile = [&fixture]() -> std::string {
    try {
      fixture.rebuild_attestor(fixture.controller);
    } catch (const std::exception& error) {
      return error.what();
    }
    return {};
  };

  constexpr auto kWindow = std::chrono::seconds(3);
  constexpr std::size_t kMinimumRounds = 250U;
  const auto started = std::chrono::steady_clock::now();
  std::size_t rounds = 0U;
  std::size_t reconciliations_that_succeeded = 0U;
  while (rounds < kMinimumRounds ||
         std::chrono::steady_clock::now() - started < kWindow) {
    ++rounds;
    if (reconcile().empty()) ++reconciliations_that_succeeded;
    DashboardRead observed;
    try {
      observed = list_runs(*fixture.journal);
    } catch (const std::exception& error) {
      throw std::runtime_error(
          "dashboard run reads stopped working after " +
          std::to_string(rounds) +
          " hostd reconciliation attempts against a prior-boot fence: " +
          error.what());
    }
    if (observed != before_reboot) {
      throw std::runtime_error(
          "dashboard run reads changed under hostd reconciliation at round " +
          std::to_string(rounds));
    }
  }
  require(reconciliations_that_succeeded == 0U,
          "a prior-boot fence is never accepted as live reconciliation");
  require(rounds >= kMinimumRounds,
          "the sustained window ran the intended number of rounds");
  require(list_runs(*fixture.journal) == before_reboot,
          "dashboard run reads still work after the sustained window");
}

void restart_rotates_controller_generation_without_losing_fence() {
  Fixture fixture;
  const HostdSessionChallengeClaim old_claim = fixture.attestor->claim();
  fixture.attestor.reset();
  fixture.journal.reset();
  fixture.journal = std::make_unique<Journal>(
      fixture.authority.database(), fixture.authority.identity(),
      HostGrantEnforcement::required,
      HostIdentity{.host_id = "host-001", .boot_id = std::string(kBootId)});
  HostdJournalControllerFence restarted = fixture.controller;
  restarted.controller_id = "controller-002";
  ++restarted.controller_generation;
  fixture.rebuild_attestor(restarted);
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)fixture.attestor->attest(fixture.query(&old_claim)); },
      "pre-restart controller generation cannot cross the new attestor");
  const auto evidence = fixture.attestor->attest(fixture.query());
  require(evidence.controller == restarted,
          "restart retains the exact live lease under a new controller generation");
}

void dynamic_attestor_tracks_current_scoped_controller_read_only() {
  Fixture fixture;
  HostdDynamicJournalFenceAttestor dynamic(
      *fixture.journal,
      {.api_version = std::string(kHostdJournalFenceAttestorApiVersion),
       .host_id = "host-001",
       .boot_id = std::string(kBootId),
       .broker_epoch = "broker-001"},
      fixture.time);
  const HostdJournalFenceQuery old_query = fixture.query();
  const auto first = dynamic.attest(old_query);
  require(first.live && first.controller == fixture.controller &&
              first.journal == old_query.claim.journal,
          "dynamic attestor accepts the exact current journal controller");

  HostdJournalControllerFence next = fixture.controller;
  next.controller_id = "controller-dynamic-next";
  ++next.controller_generation;
  fixture.rebuild_attestor(next);
  require_throws<HostdJournalFenceAttestorError>(
      [&] { (void)dynamic.attest(old_query); },
      "dynamic attestor rejects a controller superseded after construction");
  const auto current = dynamic.attest(fixture.query());
  require(current.live && current.controller == next,
          "dynamic attestor follows a newly durable current scope without mutation");
}

}  // namespace

int main() {
  try {
    successful_attestation_is_exact_and_observational();
    std::cout << "PASS exact-read-only\n";
    controller_run_and_generation_mismatch_fail_closed();
    std::cout << "PASS controller-binding\n";
    controller_generation_is_durable_monotonic_and_unique();
    std::cout << "PASS controller-authority\n";
    controller_generations_are_independent_per_exact_scope();
    std::cout << "PASS controller-scoped-authority\n";
    controller_scope_aliases_and_legacy_global_heads_fail_closed();
    std::cout << "PASS controller-scope-alias-legacy\n";
    release_expiry_and_superseding_fence_fail_closed();
    std::cout << "PASS release-expiry-fence-race\n";
    torn_projection_and_namespace_replacement_poison_fail_closed();
    std::cout << "PASS corruption-namespace-poison\n";
    authority_head_rollback_cannot_reenable_immutable_history();
    std::cout << "PASS authority-head-rollback\n";
    long_lived_renewal_head_has_no_lifetime_scan_cap();
    std::cout << "PASS long-lived-renewal-head\n";
    prior_boot_fence_is_stale_without_poisoning_journal();
    std::cout << "PASS prior-boot-stale-fence\n";
    sustained_dashboard_reads_survive_prior_boot_reconciliation();
    std::cout << "PASS prior-boot-sustained-dashboard-reads\n";
    restart_rotates_controller_generation_without_losing_fence();
    std::cout << "PASS restart-generation\n";
    dynamic_attestor_tracks_current_scoped_controller_read_only();
    std::cout << "PASS dynamic-controller\n";
    std::cout << "hostd journal fence attestor tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd journal fence attestor test failure: " << error.what()
              << '\n';
    return 1;
  }
}
