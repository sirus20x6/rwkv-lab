#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include "trainvm/document.hpp"
#include "trainvm/adapter_invocation.hpp"
#include "trainvm/dispatch.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_process_protocol.hpp"
#include "trainvm/command.hpp"
#include "trainvm/lease.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

class Controller;

// A durable command was valid when issued but has lost the active run/resource
// fence required to apply it. Boundary services map this typed condition to
// FAILED_PRECONDITION; untyped runtime failures remain authority corruption.
class OperationPreconditionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Event {
  std::string event_id;
  std::string run_id;
  std::uint64_t run_revision{};
  std::uint64_t plan_revision{};
  std::string node_id;
  std::string attempt_id;
  std::uint64_t worker_sequence{};
  std::string event_type;
  std::uint32_t event_version{1};
  std::int64_t wall_time_ns{};
  std::uint64_t monotonic_time_ns{};
  std::optional<std::uint64_t> optimizer_step;
  nlohmann::json payload = nlohmann::json::object();
};

struct RunProjection {
  std::string run_id;
  std::string experiment_name;
  std::string plan_hash;
  std::string desired_state;
  std::string observed_state;
  std::string current_node_id;
  std::string current_attempt_id;
  std::uint64_t run_revision{};
  std::uint64_t optimizer_step{};
  std::int64_t last_heartbeat_ns{};
  std::uint64_t last_event_sequence{};
  std::string failure_summary;

  bool operator==(const RunProjection&) const = default;
};

struct RunProjectionCursor final {
  std::uint64_t last_event_sequence{};
  std::string run_id;

  bool operator==(const RunProjectionCursor&) const = default;
};

struct RunProjectionQuery final {
  std::set<std::string, std::less<>> observed_states;
  std::map<std::string, std::string, std::less<>> labels;
  std::optional<RunProjectionCursor> after;
  std::size_t limit{};
};

struct SequencedEvent final {
  std::uint64_t journal_sequence{};
  Event event;

  bool operator==(const SequencedEvent&) const = default;
};

struct EventScanQuery final {
  std::uint64_t after_journal_sequence{};
  std::set<std::string, std::less<>> run_ids;
  std::set<std::string, std::less<>> event_types;
  std::size_t limit{};
};

struct RunWallTimeBounds final {
  std::int64_t created_wall_time_ns{};
  std::int64_t updated_wall_time_ns{};

  bool operator==(const RunWallTimeBounds&) const = default;
};

enum class RunCreationDisposition { inserted, replayed };

class RunCreationConflict final : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

struct RunCreationResult {
  RunCreationDisposition disposition{};
  Event created_event;

  bool operator==(const RunCreationResult&) const = default;
};

// Expected Linux identity for the already-authority-locked main database.
// Service construction supplies this before Journal performs schema writes.
struct JournalFileIdentity final {
  std::string directory_path;
  std::string journal_name;
  std::string authority_name;
  std::uint64_t directory_device{};
  std::uint64_t directory_inode{};
  std::uint64_t device{};
  std::uint64_t inode{};
  std::uint64_t authority_device{};
  std::uint64_t authority_inode{};
  std::uint64_t owner_uid{};

  bool operator==(const JournalFileIdentity&) const = default;
};

// Exact read-only authority facts retained by an already pinned Journal. This
// is inspection data, not a lease or admission capability.
struct JournalAuthoritySnapshot final {
  JournalFileIdentity file;
  HostIdentity host;
  std::string journal_id;

  bool operator==(const JournalAuthoritySnapshot&) const = default;
};

// One consistent SQLite read snapshot of the retained authority and one live
// boot-scoped logical fence. It cannot be constructed from an arbitrary path.
struct JournalLogicalFenceSnapshot final {
  JournalAuthoritySnapshot authority;
  ResourceLease lease;
  std::uint64_t authority_revision{};
  std::uint64_t authority_event_sequence{};
  std::string authority_event_hash;

  bool operator==(const JournalLogicalFenceSnapshot&) const = default;
};

// Durable controller epoch registered into the journal event authority. The
// concurrency key is part of the fence identity; lease IDs are not assumed to
// be globally unique across logical resource scopes.
struct JournalControllerFence final {
  std::string broker_epoch;
  std::string run_id;
  std::string concurrency_key;
  std::string controller_id;
  std::uint64_t controller_generation{};
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};

  bool operator==(const JournalControllerFence&) const = default;
};

// Immutable logical scope behind either a host resource request ID or its
// release request ID. Mutation clients use this journal-derived identity to
// construct authority claims; serialized host requests are never authority.
struct JournalResourceMutationIdentity final {
  std::string request_id;
  std::string run_id;
  std::string concurrency_key;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};

  bool operator==(const JournalResourceMutationIdentity&) const = default;
};

struct HostGrantSagaSnapshot final {
  ResourceBundleRequest request;
  std::optional<std::string> busy_outcome_digest;
  std::optional<ResourceBundleGrant> grant;
  std::optional<ResourceReleaseRequest> release_intent;
  std::optional<ResourceReleaseReceipt> release_receipt;

  bool operator==(const HostGrantSagaSnapshot&) const = default;
};

// Journal-owned copy of the host process prepare/commit transaction. The
// replay flags carried by transport replies are deliberately normalized out:
// they describe delivery, not the immutable process identity.
struct HostProcessSagaSnapshot final {
  HostdProcessPrepareRequest prepare;
  HostdProcessPreparedResult prepared;
  std::optional<HostdProcessCommitRequest> commit;
  std::optional<HostdProcessCommittedResult> committed;
  std::optional<HostdProcessExitCommand> exit_command;
  std::optional<HostProcessExitResult> exited;

  bool operator==(const HostProcessSagaSnapshot&) const = default;
};

struct EffectiveControlSnapshot final {
  std::uint64_t revision{};
  nlohmann::json values = nlohmann::json::object();

  bool operator==(const EffectiveControlSnapshot&) const = default;
};

enum class HostGrantEnforcement {
  required,
  legacy_process_free_test,
};

class Journal {
public:
  explicit Journal(
      const std::filesystem::path& path,
      std::optional<JournalFileIdentity> expected_file = std::nullopt,
      HostGrantEnforcement host_grant_enforcement =
          HostGrantEnforcement::required,
      std::optional<HostIdentity> expected_host_grant_authority =
          std::nullopt);
  ~Journal();

  Journal(const Journal&) = delete;
  Journal& operator=(const Journal&) = delete;
  Journal(Journal&&) = delete;
  Journal& operator=(Journal&&) = delete;

  [[nodiscard]] std::optional<Event> event(const std::string& event_id) const;
  [[nodiscard]] std::vector<Event> events_for_run(const std::string& run_id) const;
  [[nodiscard]] std::uint64_t latest_worker_sequence(
      const std::string& run_id, const std::string& node_id,
      const std::string& attempt_id) const;
  [[nodiscard]] std::optional<RunProjection> projection(const std::string& run_id) const;
  // Bounded, stable pagination for the service supervisor. Only runs whose
  // desired/observed states can still be advanced are returned; completed
  // history therefore does not make daemon restart scans grow without bound.
  [[nodiscard]] std::vector<RunProjection> reconcilable_projections(
      std::string_view after_run_id, std::size_t limit) const;
  [[nodiscard]] std::vector<RunProjection> run_projections(
      const RunProjectionQuery& query) const;
  [[nodiscard]] std::vector<SequencedEvent> sequenced_events(
      const EventScanQuery& query) const;
  [[nodiscard]] std::optional<RunWallTimeBounds> run_wall_time_bounds(
      const std::string& run_id) const;
  [[nodiscard]] std::optional<CompiledPlan> compiled_plan(const std::string& plan_hash) const;
  [[nodiscard]] std::optional<Dispatch> dispatch(const std::string& dispatch_id) const;
  [[nodiscard]] std::optional<ResolvedLaunchSpec> launch_binding(
      const std::string& launch_event_id) const;
  [[nodiscard]] std::optional<ControlCommand> control_command(
      const std::string& command_id) const;
  [[nodiscard]] std::uint64_t control_command_sequence(
      const std::string& command_id) const;
  [[nodiscard]] std::vector<ControlCommand> pending_control_commands(
      const std::string& run_id,
      std::uint64_t after_control_revision) const;
  [[nodiscard]] std::vector<ControlCommand> control_commands(
      const std::string& run_id, std::size_t limit) const;
  [[nodiscard]] std::uint64_t latest_control_revision(const std::string& run_id) const;
  [[nodiscard]] std::uint64_t latest_effective_control_revision(
      const std::string& run_id) const;
  [[nodiscard]] EffectiveControlSnapshot effective_controls(
      const std::string& run_id) const;
  [[nodiscard]] std::optional<CheckpointCommand> checkpoint_command(
      const std::string& command_id) const;
  [[nodiscard]] std::vector<CheckpointCommand> pending_checkpoint_commands(
      const std::string& run_id,
      std::uint64_t after_controller_sequence) const;
  [[nodiscard]] std::optional<LifecycleCommand> lifecycle_command(
      const std::string& command_id) const;
  [[nodiscard]] std::vector<LifecycleCommand> pending_lifecycle_commands(
      const std::string& run_id,
      std::uint64_t after_controller_sequence) const;
  [[nodiscard]] std::optional<WorkerInvocationSpec> worker_invocation(
      const std::string& dispatch_id) const;
  LeaseAcquireResult acquire_lease(const std::string& concurrency_key,
                                   const std::string& owner_run_id,
                                   const std::string& lease_id,
                                   const AuthorityTimeSample& now,
                                   std::int64_t timeout_ns);
  bool renew_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                   const std::string& lease_id, std::uint64_t fencing_token,
                   const AuthorityTimeSample& now, std::int64_t timeout_ns);
  // Replay identity includes the exact acquisition and expected expiry,
  // authority-time sample, and timeout because those values define the
  // immutable receipt bytes. A later sample against stale expected state is a
  // conflict. After restart, resume from active_lease() rather than retrying
  // stale expected state.
  LeaseRenewalResult renew_lease_exact(const ResourceLease& expected,
                                       const AuthorityTimeSample& now,
                                       std::int64_t timeout_ns);
  bool release_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                     const std::string& lease_id, std::uint64_t fencing_token,
                     const AuthorityTimeSample& now);
  [[nodiscard]] std::optional<ResourceLease> active_lease(
      const std::string& concurrency_key,
      const AuthorityTimeSample& now) const;
  [[nodiscard]] std::uint64_t event_count() const;
  [[nodiscard]] std::string journal_id() const;
  [[nodiscard]] JournalAuthoritySnapshot journal_authority_snapshot() const;
  [[nodiscard]] JournalLogicalFenceSnapshot journal_logical_fence_snapshot(
      const std::string& concurrency_key, const std::string& owner_run_id,
      const std::string& lease_id, std::uint64_t fencing_token,
      const AuthorityTimeSample& now) const;
  // Explicit authority mutation. Registration appends a hash-chained durable
  // controller epoch and accepts only a fresh generation monotonically
  // increasing within the exact concurrency-key scope and bound to its
  // currently live logical fence.
  [[nodiscard]] JournalControllerFence register_hostd_controller_fence(
      const JournalControllerFence& requested,
      const AuthorityTimeSample& now);
  [[nodiscard]] std::optional<JournalControllerFence>
  current_hostd_controller_fence(
      const std::string& concurrency_key) const;
  // Read-only validation that the requested controller is still the exact
  // current durable controller epoch. A newer valid epoch supersedes it;
  // malformed or torn durable authority permanently poisons this Journal.
  void require_current_hostd_controller_fence(
      const JournalControllerFence& requested) const;
  [[nodiscard]] bool verify_chain(std::string* reason = nullptr) const;
  std::uint64_t rebuild_projections();
  // Journal-side half of the host-grant saga. These methods copy exact host
  // receipts; they never allocate or advance physical resource generations.
  [[nodiscard]] HostGrantSagaSnapshot record_host_resource_request(
      const ResourceBundleRequest& request, const AuthorityTimeSample& now);
  [[nodiscard]] HostGrantSagaSnapshot record_host_grant_receipt(
      const ResourceBundleGrant& grant);
  [[nodiscard]] HostGrantSagaSnapshot record_host_busy_outcome(
      const std::string& request_id, const std::string& outcome_digest,
      const AuthorityTimeSample& now);
  [[nodiscard]] HostGrantSagaSnapshot record_host_release_intent(
      const std::string& request_id, const ResourceReleaseRequest& release,
      const AuthorityTimeSample& now);
  [[nodiscard]] HostGrantSagaSnapshot record_host_release_receipt(
      const std::string& request_id, const ResourceReleaseReceipt& receipt);
  [[nodiscard]] std::optional<HostGrantSagaSnapshot> host_grant_saga(
      const std::string& request_id) const;
  [[nodiscard]] std::optional<JournalResourceMutationIdentity>
  host_resource_mutation_identity(
      const std::string& request_or_release_id) const;
  // The prepare receipt must be durable before hostd may release the stopped
  // child to exec. Exact retries return the same normalized snapshot.
  [[nodiscard]] HostProcessSagaSnapshot record_host_process_prepared(
      const HostdProcessPrepareRequest& request,
      const HostdProcessPreparedResult& result,
      const AuthorityTimeSample& now);
  [[nodiscard]] HostProcessSagaSnapshot record_host_process_committed(
      const HostdProcessCommitRequest& request,
      const HostdProcessCommittedResult& result,
      const AuthorityTimeSample& now);
  [[nodiscard]] HostProcessSagaSnapshot record_host_process_exited(
      const HostdProcessExitCommand& request,
      const HostProcessExitResult& result,
      const AuthorityTimeSample& now);
  [[nodiscard]] std::optional<HostProcessSagaSnapshot> host_process_saga(
      const std::string& launch_id) const;
  [[nodiscard]] std::optional<HostLaunchGrantClaim> host_launch_grant_claim(
      const std::string& run_id, const std::string& concurrency_key,
      const std::string& lease_id, std::uint64_t fencing_token,
      const AuthorityTimeSample& now) const;
  // Throws OperationPreconditionError unless the claim exactly matches the
  // durable host receipt and its logical lease is still live under the same
  // boot-scoped fencing token.
  void require_host_launch_eligible(const HostLaunchGrantClaim& claim,
                                    const AuthorityTimeSample& now) const;

 private:
  friend class Controller;

  class ReadSnapshot {
   public:
    ReadSnapshot(ReadSnapshot&& other) noexcept;
    ReadSnapshot& operator=(ReadSnapshot&&) = delete;
    ~ReadSnapshot();
    ReadSnapshot(const ReadSnapshot&) = delete;
    ReadSnapshot& operator=(const ReadSnapshot&) = delete;

   private:
    friend class Journal;
    explicit ReadSnapshot(sqlite3* database);

    sqlite3* database_{};
  };

  sqlite3* database_{};
  std::optional<JournalFileIdentity> expected_file_;
  HostGrantEnforcement host_grant_enforcement_;
  std::optional<HostIdentity> expected_host_grant_authority_;
  mutable std::atomic<bool> authority_poisoned_{false};

  [[nodiscard]] ReadSnapshot read_snapshot() const;
  void require_live_host_grant_claim(
      const std::optional<HostLaunchGrantClaim>& claim,
      const std::string& run_id, const std::string& concurrency_key,
      const std::string& lease_id, std::uint64_t fencing_token) const;
  std::uint64_t append(const Event& event);
  std::vector<std::uint64_t> append_batch(const std::vector<Event>& events);
  Dispatch prepare_dispatch(const Dispatch& dispatch,
                            const Event& prepared_event);
  void complete_dispatch(const std::string& dispatch_id,
                         const std::string& result_event_id,
                         const std::vector<Event>& events);
  void initialize();
  void require_file_identity(const JournalFileIdentity& expected) const;
  void require_namespace_identity(const JournalFileIdentity& expected) const;
  void require_attested_authority() const;
  [[nodiscard]] bool validate_authority_boundary() const noexcept;
  [[nodiscard]] bool verify_event_chain(std::string* reason = nullptr) const;
  static int authorize_database_operation(void* context, int action,
                                          const char*, const char*,
                                          const char*, const char*) noexcept;
  static int authorize_commit(void* context) noexcept;
  [[nodiscard]] std::pair<std::string, std::uint64_t>
  append_authority_event_uncommitted(
      const Event& event);
  void record_lease_authority_acquisition_uncommitted(
      const ResourceLease& lease);
  void record_lease_authority_renewal_uncommitted(
      const LeaseRenewalReceipt& renewal);
  void record_lease_authority_release_uncommitted(
      const ResourceLease& lease, std::int64_t released_wall_time_ns);
  std::uint64_t append_uncommitted(const Event& event,
                                   bool allow_host_saga = false);
  RunCreationResult create_run(const CompiledPlan& plan, const std::vector<Event>& events);
  LeaseAcquireResult acquire_lease_with_events(
      const std::string& concurrency_key, const std::string& owner_run_id,
      const std::string& lease_id, const AuthorityTimeSample& now,
      std::int64_t timeout_ns,
      const std::vector<Event>& events);
  bool complete_builtin_admission(const ResourceLease& lease,
                                  const AuthorityTimeSample& now,
                                  const std::vector<Event>& events);
  bool prepare_worker_launch(const WorkerLaunchTicket& launch,
                             const AuthorityTimeSample& now,
                             const Event& event);
  bool bind_worker_launch(const ResolvedLaunchSpec& binding,
                          const AuthorityTimeSample& now, const Event& event);
  bool bind_worker_invocation(const WorkerInvocationSpec& invocation,
                              const WorkerSessionIdentity& identity,
                              const AuthorityTimeSample& now,
                              const Event& event);
  WorkerReadinessDisposition accept_worker_ready(
      const WorkerLaunchTicket& launch, const WorkerHelloEvidence& hello,
      const AuthorityTimeSample& now, const std::vector<Event>& events);
  Dispatch prepare_fenced_dispatch(const Dispatch& dispatch,
                                   const Event& prepared_event,
                                   const WorkerLaunchTicket& launch,
                                   const AuthorityTimeSample& now);
  Dispatch prepare_dispatch_impl(
      const Dispatch& dispatch, const Event& prepared_event,
      const std::optional<WorkerLaunchTicket>& launch,
      std::optional<AuthorityTimeSample> now);
  void complete_fenced_dispatch(const std::string& dispatch_id,
                                const std::string& result_event_id,
                                const std::vector<Event>& events,
                                const WorkerSessionIdentity& identity,
                                const AuthorityTimeSample& now);
  void append_fenced_worker_observation(
      const Event& event, const WorkerSessionIdentity& identity,
      const AuthorityTimeSample& now);
  void complete_dispatch_impl(
      const std::string& dispatch_id, const std::string& result_event_id,
      const std::vector<Event>& events,
      const std::optional<WorkerSessionIdentity>& identity,
      std::optional<AuthorityTimeSample> now);
  void complete_managed_builtin_dispatch(
      const Dispatch& dispatch, const ResourceLease& lease,
      const AuthorityTimeSample& now, bool release_lease,
      const std::vector<Event>& events);
  [[nodiscard]] bool has_lease_release_receipt(
      const std::string& concurrency_key, const std::string& owner_run_id,
      const std::string& lease_id, std::uint64_t fencing_token,
      std::string_view clock_domain, std::string_view boot_id,
      std::int64_t released_wall_time_ns) const;
  ControlSubmission submit_control_command(ControlCommand command);
  ControlCommand acknowledge_control_command(const std::string& run_id,
                                              const std::string& command_id,
                                              const ControlAcknowledgementIdentity& identity,
                                              ControlCommandStatus status,
                                              std::optional<std::uint64_t> effective_step,
                                              nlohmann::json effective_values,
                                              nlohmann::json diagnostics,
                                              const AuthorityTimeSample& now);
  CheckpointSubmission submit_checkpoint_command(CheckpointCommand command);
  CheckpointCommand acknowledge_checkpoint_command(
      const std::string& run_id, const std::string& command_id,
      const ControlAcknowledgementIdentity& identity,
      CheckpointCommandStatus status,
      std::optional<std::uint64_t> optimizer_step,
      std::string artifact_id, nlohmann::json diagnostics,
      const AuthorityTimeSample& now);
  LifecycleSubmission submit_lifecycle_command(LifecycleCommand command);
  LifecycleCommand acknowledge_lifecycle_command(
      const std::string& run_id, const std::string& command_id,
      const ControlAcknowledgementIdentity& identity,
      LifecycleCommandStatus status,
      std::optional<std::uint64_t> optimizer_step,
      std::string artifact_id, nlohmann::json diagnostics,
      const AuthorityTimeSample& now);
  void complete_cancellation(const std::string& run_id,
                             const std::string& command_id,
                             const AuthorityTimeSample& now);
};

nlohmann::json event_json(const Event& event);
Event event_from_json(const nlohmann::json& input);
nlohmann::json projection_json(const RunProjection& projection);

}  // namespace trainvm
