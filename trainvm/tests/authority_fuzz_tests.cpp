// Generated adversarial coverage for the durable authority boundary.
//
// This is a property suite, not a table of hand-written cases: every hostile
// document is derived structurally from a canonical one, so a decoder that
// grows a field is fuzzed on that field without anyone remembering to add a
// case. Two properties are asserted for every codec:
//
//   1. canonical round-trip is exact;
//   2. every derived mutation is REJECTED.
//
// A third property is asserted at the ledger: feeding hostile documents and
// hostile request content through the mutating paths must leave record count,
// chain head, occupancy, and every resource generation byte-identical. That is
// the card's actual requirement — rejection alone is not enough if a rejected
// request can still fork durable history.
//
// The generator is a deterministic LCG. TRAINVM_FUZZ_SEED and TRAINVM_FUZZ_
// ROUNDS widen the sweep in CI without making the default run nondeterministic.

#include <sqlite3.h>

#include <unistd.h>

#include <cstdint>
#include <limits>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/host_ledger.hpp"
#include "trainvm/host_ledger_authority.hpp"
#include "trainvm/host_resources.hpp"
#include "trainvm/host_startup_audit.hpp"

namespace {

using namespace trainvm;

std::size_t g_checks = 0U;
std::size_t g_mutations = 0U;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
  ++g_checks;
}

// Deterministic by construction. A failure reproduces from the printed seed.
class Rng final {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed | 1U) {}
  std::uint64_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return state_ >> 17U;
  }
  std::size_t below(std::size_t bound) {
    return bound == 0U ? 0U : static_cast<std::size_t>(next() % bound);
  }

 private:
  std::uint64_t state_;
};

std::uint64_t env_number(const char* name, std::uint64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  try {
    return std::stoull(value);
  } catch (const std::exception&) {
    return fallback;
  }
}

// Every JSON pointer in a document, so mutation reaches nested objects and
// array elements rather than only the top level.
void collect_paths(const nlohmann::json& value, nlohmann::json::json_pointer at,
                   std::vector<nlohmann::json::json_pointer>& out) {
  out.push_back(at);
  if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      collect_paths(child, at / key, out);
    }
  } else if (value.is_array()) {
    for (std::size_t index = 0U; index < value.size(); ++index) {
      collect_paths(value[index], at / std::to_string(index), out);
    }
  }
}

std::vector<nlohmann::json::json_pointer> document_paths(
    const nlohmann::json& document) {
  std::vector<nlohmann::json::json_pointer> paths;
  collect_paths(document, nlohmann::json::json_pointer{}, paths);
  return paths;
}

// A value of a deliberately different shape from the one already there, so a
// decoder that accepts on type coercion is caught.
nlohmann::json foreign_value(const nlohmann::json& original, Rng& rng) {
  const std::vector<nlohmann::json> pool = {
      nlohmann::json(nullptr),
      nlohmann::json(true),
      nlohmann::json(-1),
      nlohmann::json(0),
      nlohmann::json(std::numeric_limits<std::int64_t>::min()),
      nlohmann::json(std::numeric_limits<std::uint64_t>::max()),
      nlohmann::json(1.5),
      nlohmann::json(""),
      nlohmann::json(std::string(4096U, 'x')),
      nlohmann::json(std::string("\xef\xbf\xbd")),
      nlohmann::json::array(),
      nlohmann::json::object(),
      nlohmann::json::array({1, 2, 3}),
      nlohmann::json({{"unexpected", 1}}),
  };
  for (std::size_t attempt = 0U; attempt < pool.size(); ++attempt) {
    const nlohmann::json& candidate = pool[(rng.below(pool.size()) + attempt) %
                                           pool.size()];
    if (candidate != original) return candidate;
  }
  return nlohmann::json(nullptr);
}

enum class Mutation {
  replace_value,
  remove_member,
  add_unknown_member,
  duplicate_member_text,
  truncate_array,
  extend_array,
};

// Returns the hostile document text. Duplicate keys cannot be produced through
// the DOM (nlohmann collapses them), so that case is applied to the serialized
// text instead — which is exactly how a hostile peer would send it.
std::string mutate(const nlohmann::json& canonical, Mutation mutation,
                   const nlohmann::json::json_pointer& at, Rng& rng) {
  if (mutation == Mutation::duplicate_member_text) {
    const nlohmann::json& target = canonical.at(at);
    if (!target.is_object() || target.empty()) return {};
    const std::string key = target.begin().key();
    const std::string dumped = canonical.dump();
    const std::string needle = "\"" + key + "\":";
    const std::size_t found = dumped.find(needle);
    if (found == std::string::npos) return {};
    // Same key twice, second copy carrying different content.
    const std::string injected =
        needle + foreign_value(target.begin().value(), rng).dump() + ",";
    return dumped.substr(0U, found) + injected + dumped.substr(found);
  }

  nlohmann::json document = canonical;
  switch (mutation) {
    case Mutation::replace_value: {
      document[at] = foreign_value(canonical.at(at), rng);
      break;
    }
    case Mutation::remove_member: {
      if (at.empty()) return {};
      auto parent = at.parent_pointer();
      const std::string leaf = at.back();
      if (!document.at(parent).is_object()) return {};
      document.at(parent).erase(leaf);
      break;
    }
    case Mutation::add_unknown_member: {
      if (!document.at(at).is_object()) return {};
      document.at(at)["trainvm_fuzz_unknown"] = 1;
      break;
    }
    case Mutation::truncate_array: {
      if (!document.at(at).is_array() || document.at(at).empty()) return {};
      document.at(at).erase(document.at(at).size() - 1U);
      break;
    }
    case Mutation::extend_array: {
      if (!document.at(at).is_array()) return {};
      document.at(at).push_back(
          document.at(at).empty() ? nlohmann::json(0) : document.at(at).back());
      break;
    }
    case Mutation::duplicate_member_text:
      break;
  }
  return document.dump();
}

struct Codec final {
  std::string name;
  nlohmann::json canonical;
  // Must throw on anything that is not exactly the canonical document.
  std::function<nlohmann::json(const nlohmann::json&)> decode_and_reencode;
};

// A hostile document must be refused. Anything that decodes is a finding: the
// decoder accepted a document that is not the canonical one.
void assert_rejects_every_mutation(const Codec& codec, Rng& rng,
                                   std::size_t rounds) {
  const auto paths = document_paths(codec.canonical);
  const Mutation mutations[] = {
      Mutation::replace_value,     Mutation::remove_member,
      Mutation::add_unknown_member, Mutation::duplicate_member_text,
      Mutation::truncate_array,    Mutation::extend_array,
  };
  for (std::size_t round = 0U; round < rounds; ++round) {
    for (const auto& at : paths) {
      for (const Mutation mutation : mutations) {
        const std::string hostile = mutate(codec.canonical, mutation, at, rng);
        if (hostile.empty()) continue;
        nlohmann::json parsed;
        try {
          parsed = nlohmann::json::parse(hostile);
        } catch (const std::exception&) {
          continue;  // Not valid JSON; the parser is not under test here.
        }
        if (parsed == codec.canonical) continue;  // Mutation was a no-op.
        ++g_mutations;
        bool accepted = false;
        try {
          (void)codec.decode_and_reencode(parsed);
          accepted = true;
        } catch (const std::exception&) {
          accepted = false;
        }
        if (accepted) {
          throw std::runtime_error(
              codec.name + " accepted mutation kind " +
              std::to_string(static_cast<int>(mutation)) + " at " +
              at.to_string() + ": " + hostile.substr(0U, 200U));
        }
      }
    }
  }
}

void assert_canonical_round_trip(const Codec& codec) {
  const nlohmann::json again = codec.decode_and_reencode(codec.canonical);
  require(again == codec.canonical,
          codec.name + " canonical round-trip is not exact");
}

// ---------------------------------------------------------------------------
// Canonical fixtures
// ---------------------------------------------------------------------------

std::string digest_of(char digit) {
  return "sha256:" + std::string(64U, digit);
}

HostResourceId mutex_id(const std::string& id) {
  return {.kind = HostResourceKind::host_mutex,
          .vendor = std::nullopt,
          .stable_id = "host-mutex:" + id,
          .parent_id = std::nullopt};
}

HostInventoryReceipt inventory() {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-fuzz",
      .boot_id = "boot-fuzz",
      .broker_epoch = "broker-fuzz",
      .begin_revision = "revision-1",
      .end_revision = "revision-1",
      .probes = {},
      .resources = {{
          .id = mutex_id("fuzz"),
          .disposition = ResourceObservationDisposition::audited_eligible,
          .compute_contexts = ResourceContextDisposition::absent,
          .graphics_contexts = ResourceContextDisposition::absent,
          .pci_bdf = std::nullopt,
          .device_major = std::nullopt,
          .device_minor = std::nullopt,
          .device_nodes = {},
          .numa_node = std::nullopt,
          .pcie_root_id = std::nullopt,
          .fabric_clique_id = std::nullopt,
          .total_memory_bytes = 0U,
          .labels = {{"scope", "fuzz"}},
      }},
  };
  FakeHostKernel kernel({{.snapshot = std::move(snapshot),
                          .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResourceBundleRequest bundle_request(std::string id) {
  return seal_resource_request({
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = std::move(id),
      .journal_id = "journal-fuzz",
      .run_id = "run-fuzz",
      .logical_lease_id = "lease-fuzz",
      .logical_fencing_token = 3,
      .count = 1U,
      .access_mode = ResourceAccessMode::mutex_exclusive,
      .topology = TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
}

ResourceReleaseRequest release_request(const ResourceBundleGrant& grant,
                                       std::string id) {
  return seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id = std::move(id),
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .canonical_request_digest = {},
  });
}

HostProcessLaunchRequest launch_request(const ResourceBundleGrant& grant,
                                        std::string launch_id) {
  return seal_host_process_launch_request({
      .api_version = std::string(kHostProcessLaunchRequestApiVersion),
      .launch_id = std::move(launch_id),
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .resolved_launch_digest = digest_of('1'),
      .executable_path = "/usr/bin/trainvm-fuzz-worker",
      .executable_digest = digest_of('2'),
      .cgroup_path = "/sys/fs/cgroup/trainvm/fuzz",
      .cgroup_device = 71,
      .cgroup_inode = 72,
      .worker_credentials = std::nullopt,
      .device_policy = std::nullopt,
      .process_policy = std::nullopt,
      .canonical_request_digest = {},
  });
}

HostProcessSpawnRequest spawn_request(const HostProcessLaunchIntent& intent) {
  return seal_host_process_spawn_request({
      .api_version = std::string(kHostProcessSpawnRequestApiVersion),
      .launch_id = intent.request.launch_id,
      .launch_intent_digest = intent.receipt_digest,
      .host_pid = 5150,
      .process_starttime_ticks = 8675309U,
      .boot_id = intent.boot_id,
      .cgroup_path = intent.request.cgroup_path,
      .cgroup_device = intent.request.cgroup_device,
      .cgroup_inode = intent.request.cgroup_inode,
      .executable_digest = intent.request.executable_digest,
      .worker_credentials = std::nullopt,
      .device_policy = std::nullopt,
      .process_policy = std::nullopt,
      .canonical_request_digest = {},
  });
}

HostProcessExitRequest exit_request(const HostProcessSpawnReceipt& spawn) {
  return seal_host_process_exit_request({
      .api_version = std::string(kHostProcessExitRequestApiVersion),
      .exit_request_id = "exit-fuzz",
      .launch_id = spawn.request.launch_id,
      .spawn_receipt_digest = spawn.receipt_digest,
      .host_pid = spawn.request.host_pid,
      .process_starttime_ticks = spawn.request.process_starttime_ticks,
      .wait_code = 1,
      .wait_status = 0,
      .cgroup_path = spawn.request.cgroup_path,
      .cgroup_device = spawn.request.cgroup_device,
      .cgroup_inode = spawn.request.cgroup_inode,
      .cgroup_empty = true,
      .accelerator_contexts_empty = true,
      .context_audit_digest = digest_of('4'),
      .canonical_request_digest = {},
  });
}

HostProcessRecoveryExitRequest recovery_exit_request(
    const HostProcessSpawnReceipt& spawn) {
  return seal_host_process_recovery_exit_request({
      .api_version = std::string(kHostProcessRecoveryExitRequestApiVersion),
      .recovery_exit_request_id = "recovery-exit-fuzz",
      .launch_id = spawn.request.launch_id,
      .spawn_receipt_digest = spawn.receipt_digest,
      .host_pid = spawn.request.host_pid,
      .process_starttime_ticks = spawn.request.process_starttime_ticks,
      .observation = HostProcessRecoveryExitObservation::pid_absent,
      .observation_digest = digest_of('5'),
      .cgroup_path = spawn.request.cgroup_path,
      .cgroup_device = spawn.request.cgroup_device,
      .cgroup_inode = spawn.request.cgroup_inode,
      .cgroup_empty = true,
      .accelerator_contexts_empty = true,
      .context_audit_digest = digest_of('6'),
      .canonical_request_digest = {},
  });
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "trainvm-fuzz-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = ::mkdtemp(writable.data());
    if (created == nullptr) throw std::runtime_error("mkdtemp failed");
    path_ = created;
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

std::shared_ptr<HostLedgerFilesystemAuthority> authority_for(
    const std::filesystem::path& path) {
  return std::make_shared<HostLedgerFilesystemAuthority>(
      HostLedgerFilesystemAuthority::acquire({
          .api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade = HostLedgerEnforcementGrade::cooperative_test,
      }));
}

// A byte-exact fingerprint of everything durable the ledger exposes. Any
// hostile input that changes one field of this has forked durable history.
struct DurableFingerprint final {
  std::uint64_t record_count{};
  std::string chain_hash;
  std::uint64_t ledger_sequence{};
  std::string occupancy_digest;
  std::uint64_t generation{};
  std::size_t active_fences{};
  std::size_t recovery_records{};
  std::size_t terminal_records{};

  bool operator==(const DurableFingerprint&) const = default;
};

DurableFingerprint fingerprint(SQLiteHostLedger& ledger) {
  const auto head = ledger.chain_head();
  const auto occupancy = ledger.occupancy();
  return {
      .record_count = ledger.record_count(),
      .chain_hash = head.chain_hash,
      .ledger_sequence = head.ledger_sequence,
      .occupancy_digest = occupancy.occupancy_digest,
      .generation = ledger.generation(mutex_id("fuzz")),
      .active_fences = occupancy.active_fences.size(),
      .recovery_records = ledger.active_process_recovery_records().size(),
      .terminal_records = ledger.active_terminal_process_release_records().size(),
  };
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

std::vector<Codec> build_codecs(const ResourceBundleGrant& grant,
                                const HostProcessLaunchIntent& intent,
                                const HostProcessSpawnReceipt& spawn,
                                const HostProcessExitReceipt& exited) {
  std::vector<Codec> codecs;
  const auto add = [&codecs](std::string name, nlohmann::json canonical,
                             std::function<nlohmann::json(const nlohmann::json&)>
                                 decode) {
    codecs.push_back({std::move(name), std::move(canonical), std::move(decode)});
  };

  add("resource_request", resource_request_json(bundle_request("fuzz-request")),
      [](const nlohmann::json& value) {
        return resource_request_json(resource_request_from_json(value));
      });
  add("resource_bundle_grant", resource_bundle_grant_json(grant),
      [](const nlohmann::json& value) {
        return resource_bundle_grant_json(resource_bundle_grant_from_json(value));
      });
  add("resource_release_request",
      resource_release_request_json(release_request(grant, "fuzz-release")),
      [](const nlohmann::json& value) {
        return resource_release_request_json(
            resource_release_request_from_json(value));
      });
  add("host_inventory", host_inventory_json(inventory()),
      [](const nlohmann::json& value) {
        return host_inventory_json(host_inventory_from_json(value));
      });
  add("host_process_launch_request",
      host_process_launch_request_json(intent.request),
      [](const nlohmann::json& value) {
        return host_process_launch_request_json(
            host_process_launch_request_from_json(value));
      });
  add("host_process_launch_intent", host_process_launch_intent_json(intent),
      [](const nlohmann::json& value) {
        return host_process_launch_intent_json(
            host_process_launch_intent_from_json(value));
      });
  add("host_process_spawn_request",
      host_process_spawn_request_json(spawn.request),
      [](const nlohmann::json& value) {
        return host_process_spawn_request_json(
            host_process_spawn_request_from_json(value));
      });
  add("host_process_spawn_receipt", host_process_spawn_receipt_json(spawn),
      [](const nlohmann::json& value) {
        return host_process_spawn_receipt_json(
            host_process_spawn_receipt_from_json(value));
      });
  add("host_process_exit_request",
      host_process_exit_request_json(exited.request),
      [](const nlohmann::json& value) {
        return host_process_exit_request_json(
            host_process_exit_request_from_json(value));
      });
  add("host_process_exit_receipt", host_process_exit_receipt_json(exited),
      [](const nlohmann::json& value) {
        return host_process_exit_receipt_json(
            host_process_exit_receipt_from_json(value));
      });
  return codecs;
}

void canonical_codecs_round_trip_and_reject_every_mutation(Rng& rng,
                                                           std::size_t rounds) {
  TemporaryDirectory directory;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority_for(directory.path() / "host.db"), inventory());
  const auto granted =
      ledger->request_bundle(bundle_request("fuzz-grant"), {10, 20});
  require(granted.grant.has_value(), "fuzz fixture obtains a grant");
  const auto intent = ledger->commit_process_launch_intent(
      launch_request(*granted.grant, "fuzz-launch"), {30, 40});
  const auto spawn =
      ledger->commit_process_spawn(spawn_request(intent.intent), {50, 60});
  const auto exited =
      ledger->commit_process_exit(exit_request(spawn.receipt), {70, 80});

  const auto codecs =
      build_codecs(*granted.grant, intent.intent, spawn.receipt, exited.receipt);
  require(codecs.size() >= 10U, "the codec table covers the durable surface");
  for (const Codec& codec : codecs) {
    assert_canonical_round_trip(codec);
    assert_rejects_every_mutation(codec, rng, rounds);
  }
}

// The card's real bar: a refused document must also leave nothing behind.
void hostile_input_cannot_fork_durable_history(Rng& rng, std::size_t rounds) {
  TemporaryDirectory directory;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority_for(directory.path() / "host.db"), inventory());
  const auto granted =
      ledger->request_bundle(bundle_request("durable-grant"), {10, 20});
  require(granted.grant.has_value(), "durable fixture obtains a grant");
  const auto intent = ledger->commit_process_launch_intent(
      launch_request(*granted.grant, "durable-launch"), {30, 40});
  const auto spawn =
      ledger->commit_process_spawn(spawn_request(intent.intent), {50, 60});

  const DurableFingerprint before = fingerprint(*ledger);

  // Every mutating entry point, fed content derived from a valid request but
  // altered. Each call must either throw or replay; none may append a record.
  const auto launch_document = host_process_launch_request_json(intent.intent.request);
  const auto spawn_document = host_process_spawn_request_json(spawn.receipt.request);
  const auto exit_document = host_process_exit_request_json(
      exit_request(spawn.receipt));
  const auto recovery_document = host_process_recovery_exit_request_json(
      recovery_exit_request(spawn.receipt));
  const auto release_document =
      resource_release_request_json(release_request(*granted.grant, "hostile"));

  std::vector<std::pair<std::string, nlohmann::json>> documents;
  documents.emplace_back("launch", launch_document);
  documents.emplace_back("spawn", spawn_document);
  documents.emplace_back("exit", exit_document);
  documents.emplace_back("recovery_exit", recovery_document);
  documents.emplace_back("release", release_document);

  for (const auto& [name, canonical] : documents) {
    const auto paths = document_paths(canonical);
    for (std::size_t round = 0U; round < rounds; ++round) {
      for (const auto& at : paths) {
        const std::string hostile =
            mutate(canonical, Mutation::replace_value, at, rng);
        if (hostile.empty()) continue;
        nlohmann::json parsed;
        try {
          parsed = nlohmann::json::parse(hostile);
        } catch (const std::exception&) {
          continue;
        }
        if (parsed == canonical) continue;
        ++g_mutations;
        try {
          // Decode first. A decoder that refuses never reaches the ledger,
          // which is the intended shape; the point is that when a mutation
          // IS decodable, the ledger still refuses to act on it.
          if (name == "launch") {
            (void)ledger->commit_process_launch_intent(
                host_process_launch_request_from_json(parsed), {90, 100});
          } else if (name == "spawn") {
            (void)ledger->commit_process_spawn(
                host_process_spawn_request_from_json(parsed), {90, 100});
          } else if (name == "exit") {
            (void)ledger->commit_process_exit(
                host_process_exit_request_from_json(parsed), {90, 100});
          } else if (name == "recovery_exit") {
            (void)ledger->commit_process_recovery_exit(
                host_process_recovery_exit_request_from_json(parsed),
                {90, 100});
          } else {
            (void)ledger->release_bundle(
                resource_release_request_from_json(parsed), {90, 100});
          }
        } catch (const std::exception&) {
          // Refused, which is the expected outcome for a hostile document.
        }
        const DurableFingerprint after = fingerprint(*ledger);
        if (!(after == before)) {
          throw std::runtime_error(
              "hostile " + name + " document at /" + at.to_string() +
              " mutated durable history: " + hostile.substr(0U, 300U));
        }
        std::string reason;
        require(ledger->verify(&reason),
                "ledger chain broke after a hostile " + name +
                    " document: " + reason);
      }
    }
  }
}

// Replay must be exact and order-independent for lost replies: re-submitting
// the same durable request any number of times, interleaved arbitrarily, may
// never append a record or change an outcome.
void exact_replay_is_idempotent_under_arbitrary_order(Rng& rng,
                                                      std::size_t rounds) {
  TemporaryDirectory directory;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority_for(directory.path() / "host.db"), inventory());
  const auto request = bundle_request("replay-grant");
  const auto granted = ledger->request_bundle(request, {10, 20});
  require(granted.grant.has_value(), "replay fixture obtains a grant");
  const auto launch = launch_request(*granted.grant, "replay-launch");
  const auto intent = ledger->commit_process_launch_intent(launch, {30, 40});
  const auto spawn_input = spawn_request(intent.intent);
  const auto spawn = ledger->commit_process_spawn(spawn_input, {50, 60});

  const DurableFingerprint settled = fingerprint(*ledger);
  const std::vector<std::function<void()>> replays = {
      [&] {
        const auto again = ledger->request_bundle(request, {91, 101});
        require(again.replayed && again.grant == granted.grant,
                "bundle request replay is not exact");
      },
      [&] {
        const auto again =
            ledger->commit_process_launch_intent(launch, {92, 102});
        require(again.replayed && again.intent == intent.intent,
                "launch intent replay is not exact");
      },
      [&] {
        const auto again = ledger->commit_process_spawn(spawn_input, {93, 103});
        require(again.replayed && again.receipt == spawn.receipt,
                "spawn receipt replay is not exact");
      },
      [&] {
        const auto observed = ledger->reconcile_bundle_outcome(request);
        require(observed && observed->grant == granted.grant,
                "inspection-only reconciliation is not exact");
      },
  };

  for (std::size_t round = 0U; round < rounds * 4U; ++round) {
    replays[rng.below(replays.size())]();
    require(fingerprint(*ledger) == settled,
            "an exact replay appended durable history");
  }
  std::string reason;
  require(ledger->verify(&reason), "replay broke the ledger chain: " + reason);
}

// Regression for the hole this suite found: nlohmann narrows silently on
// get<T>(), so before the reflected decoder range-checked, any 32-bit field
// accepted a 64-bit wire value and truncated it. wait_status is the field the
// generator happened to hit; worker uid is the one that matters, because
// 4294967296 truncates to 0, which is root.
void oversized_integers_cannot_truncate_into_a_narrow_field() {
  TemporaryDirectory directory;
  auto ledger = std::make_shared<SQLiteHostLedger>(
      authority_for(directory.path() / "host.db"), inventory());
  const auto granted =
      ledger->request_bundle(bundle_request("narrowing-grant"), {10, 20});
  require(granted.grant.has_value(), "narrowing fixture obtains a grant");
  const auto intent = ledger->commit_process_launch_intent(
      launch_request(*granted.grant, "narrowing-launch"), {30, 40});
  const auto spawn =
      ledger->commit_process_spawn(spawn_request(intent.intent), {50, 60});
  const auto exited =
      ledger->commit_process_exit(exit_request(spawn.receipt), {70, 80});

  const auto reject = [](const std::string& what,
                         const std::function<void()>& call) {
    bool refused = false;
    try {
      call();
    } catch (const std::exception&) {
      refused = true;
    }
    require(refused, what + " must be refused, not truncated");
  };

  // int32 field carrying a value only int64 can hold.
  auto oversized_exit = host_process_exit_receipt_json(exited.receipt);
  oversized_exit["request"]["wait_status"] =
      std::numeric_limits<std::int64_t>::min();
  reject("an int64 wait_status in an int32 field", [&] {
    (void)host_process_exit_receipt_from_json(oversized_exit);
  });

  // uint32 credential field: 2^32 truncates to 0, which is root.
  auto oversized_uid = host_process_launch_request_json(intent.intent.request);
  oversized_uid["worker_credentials"] = {{"uid", 4294967296LL},
                                         {"gid", 1000},
                                         {"no_new_privileges", true}};
  reject("a uint32 worker uid of 2^32", [&] {
    (void)host_process_launch_request_from_json(oversized_uid);
  });

  // Negative value in an unsigned field.
  auto negative_inode = host_process_spawn_request_json(spawn.receipt.request);
  negative_inode["cgroup_inode"] = -1;
  reject("a negative cgroup inode in a uint64 field", [&] {
    (void)host_process_spawn_request_from_json(negative_inode);
  });
}

}  // namespace

int main() {
  const std::uint64_t seed = env_number("TRAINVM_FUZZ_SEED", 0x5eed1234ULL);
  const std::size_t rounds =
      static_cast<std::size_t>(env_number("TRAINVM_FUZZ_ROUNDS", 1U));
  try {
    Rng rng(seed);
    canonical_codecs_round_trip_and_reject_every_mutation(rng, rounds);
    hostile_input_cannot_fork_durable_history(rng, rounds);
    exact_replay_is_idempotent_under_arbitrary_order(rng, rounds);
    oversized_integers_cannot_truncate_into_a_narrow_field();
    std::cout << "authority fuzz tests passed (seed=" << seed
              << " rounds=" << rounds << " checks=" << g_checks
              << " mutations=" << g_mutations << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "authority fuzz test failure (seed=" << seed
              << " rounds=" << rounds << "): " << error.what() << '\n';
    return 1;
  }
}
