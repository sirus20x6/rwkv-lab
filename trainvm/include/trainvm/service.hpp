#pragma once

#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "trainvm/journal.hpp"
#include "trainvm/sqlite_filesystem_authority.hpp"
#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/hostd_client_bootstrap.hpp"
#include "trainvm/host_process_saga.hpp"
#include "trainvm/resource_request_builder.hpp"
#include "trainvm/reconciler.hpp"
#include "trainvm/run_authoring.hpp"
#include "trainvm/lease_renewal.hpp"
#include "trainvm/training_component_registry.hpp"
#include "trainvm/worker_runtime_evidence.hpp"
#include "trainvm/v1/trainvm.grpc.pb.h"

namespace trainvm {

class AuthorityLock {
 public:
  explicit AuthorityLock(
      const std::filesystem::path& journal_path,
      SqliteAuthorityEnforcementGrade enforcement_grade =
          SqliteAuthorityEnforcementGrade::cooperative_test);
  ~AuthorityLock();

  AuthorityLock(const AuthorityLock&) = delete;
  AuthorityLock& operator=(const AuthorityLock&) = delete;

  [[nodiscard]] const std::filesystem::path& journal_path() const noexcept;
  [[nodiscard]] const JournalFileIdentity& journal_identity() const noexcept;
  [[nodiscard]] const std::shared_ptr<SqliteFilesystemAuthority>&
  filesystem_authority() const noexcept;

 private:
  int kernel_namespace_descriptor_{-1};
  std::shared_ptr<SqliteFilesystemAuthority> filesystem_authority_;
  std::filesystem::path journal_path_;
  JournalFileIdentity journal_identity_;
};

// Why the reconciliation supervisor is not doing work for a run right now.
// This is the diagnostic that was missing when a controller burned a core for
// eight hours: `top` could see the CPU, but nothing could say which run the
// loop was re-reconciling or what it was waiting for.
struct ReconciliationSupervisorMetrics final {
  // Supervisor loop iterations, split by what ended the condition wait.
  std::uint64_t wakes{};
  std::uint64_t explicit_wakes{};
  std::uint64_t cadence_wakes{};
  std::uint64_t scans{};
  // Calls to reconcile_until_quiescent, and the reconcile_once steps inside
  // them. `steps` growing while the journal does not is the exact signature of
  // the spin this instrumentation exists to catch.
  std::uint64_t reconcile_passes{};
  std::uint64_t reconcile_steps{};
  // Runs the scan found but deliberately did not reconcile because neither
  // their journal state nor their backoff deadline had moved.
  std::uint64_t skipped_idle_runs{};
  // Passes that exhausted the per-wake step budget and were requeued.
  std::uint64_t budget_requeues{};
  std::uint64_t failures{};
  std::uint64_t tracked_runs{};

  bool operator==(const ReconciliationSupervisorMetrics&) const = default;
};

struct ReconciliationRunWait final {
  std::string run_id;
  std::string wait_reason;
  std::uint64_t idle_passes{};
  std::uint64_t retries{};
  std::int64_t backoff_ns{};
  std::int64_t next_due_ns{};
  std::uint64_t last_event_sequence{};

  bool operator==(const ReconciliationRunWait&) const = default;
};

class TrainVMService final : public v1::TrainVM::Service,
                             public v1::WorkerControl::Service {
 public:
  explicit TrainVMService(
      const std::filesystem::path& journal_path,
      AdapterRegistry adapter_registry,
      HostLaunchRegistry host_launch_registry,
      std::function<AuthorityTimeSample()> authority_clock = {},
      TrainingComponentRegistry training_components =
          TrainingComponentRegistry({}),
      std::optional<HostdClientConfiguration> hostd_configuration =
          std::nullopt,
      std::string controller_target = {},
      // Optional authority seam for qualify_cache nodes. A plan containing one
      // fails closed when this is absent rather than assuming a verdict.
      ICacheQualificationEvidenceResolver* cache_qualification = nullptr,
      SqliteAuthorityEnforcementGrade filesystem_enforcement_grade =
          SqliteAuthorityEnforcementGrade::cooperative_test,
      std::shared_ptr<ITrainingPreflightEvidenceProvider>
          preflight_evidence = {},
      std::filesystem::path recipe_registry_path =
          std::filesystem::path(std::string(kInstalledRecipeProfilePath)),
      // Optional authority seam for worker-measured runtime evidence.
      // Deployments that configure no immutable receipt root hold none, and a
      // WorkerRuntimeEvidence message is then refused rather than accepted
      // and dropped.
      IWorkerRuntimeEvidenceAuthority* worker_runtime_evidence = nullptr,
      // The deployment's immutable cache evidence receipt root. Supplying it
      // is what makes the receipt root a property of the deployment rather
      // than of a test, and it is attested at construction: a root that is
      // missing, on another device, not owned by the effective uid, writable
      // by group or other, or missing its `runtime/` or `qualification/`
      // subdirectory fails the daemon at startup rather than at the first
      // worker message.
      std::optional<std::filesystem::path> cache_evidence_root = std::nullopt);
  ~TrainVMService() override;

  // The attested receipt root configuration, or nullopt when the deployment
  // configured none. `authority_uid` is always the effective uid: the
  // publisher refuses any other value, so it is not separately configurable.
  [[nodiscard]] const std::optional<LinuxCacheEvidenceConfig>&
  cache_evidence_configuration() const {
    return cache_evidence_;
  }

  grpc::Status SubmitExperiment(grpc::ServerContext* context,
                                const v1::SubmitExperimentRequest* request,
                                v1::SubmitExperimentResponse* response) override;

  grpc::Status AuthorRun(
      grpc::ServerContext* context, const v1::AuthorRunRequest* request,
      grpc::ServerWriter<v1::AuthorRunUpdate>* writer) override;

  grpc::Status DiffPlan(grpc::ServerContext* context,
                        const v1::PlanDiffRequest* request,
                        v1::PlanDiffResponse* response) override;

  grpc::Status CommandRun(grpc::ServerContext* context,
                          const v1::RunCommandRequest* request,
                          v1::RunCommandResponse* response) override;

  grpc::Status GetRun(grpc::ServerContext* context,
                      const v1::GetRunRequest* request,
                      v1::RunSummary* response) override;

  grpc::Status GetCompiledPlan(
      grpc::ServerContext* context,
      const v1::GetCompiledPlanRequest* request,
      v1::GetCompiledPlanResponse* response) override;

  grpc::Status ListRuns(grpc::ServerContext* context,
                        const v1::ListRunsRequest* request,
                        v1::ListRunsResponse* response) override;

  grpc::Status WatchEvents(
      grpc::ServerContext* context,
      const v1::WatchEventsRequest* request,
      grpc::ServerWriter<v1::EventEnvelope>* writer) override;

  grpc::Status GetControlView(
      grpc::ServerContext* context,
      const v1::GetControlViewRequest* request,
      v1::GetControlViewResponse* response) override;

  grpc::Status GetDescriptor(
      grpc::ServerContext* context,
      const v1::DescriptorRequest* request,
      v1::DescriptorResponse* response) override;

  grpc::Status GetHostAuthorityStatus(
      grpc::ServerContext* context,
      const v1::GetHostAuthorityStatusRequest* request,
      v1::GetHostAuthorityStatusResponse* response) override;

  grpc::Status GetReconciliationStatus(
      grpc::ServerContext* context,
      const v1::GetReconciliationStatusRequest* request,
      v1::GetReconciliationStatusResponse* response) override;

  grpc::Status Connect(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<v1::ControllerToWorker,
                               v1::WorkerToController>* stream) override;

  // Embedded servers may explicitly bracket the same supervisor lifecycle
  // that serve() owns. Destruction is an idempotent stop boundary.
  void start_reconciliation_supervisor();
  void stop_reconciliation_supervisor() noexcept;

  // Supervisor telemetry. Safe to call from any thread, including while the
  // supervisor is running.
  static constexpr std::size_t kMaximumReportedWaits = 256U;
  [[nodiscard]] ReconciliationSupervisorMetrics reconciliation_metrics() const;
  [[nodiscard]] std::vector<ReconciliationRunWait> reconciliation_waits(
      std::size_t limit = kMaximumReportedWaits) const;

 private:
  struct AuthorRunAuthority final {
    std::string request_digest;
    TrainingPreflightReceipt receipt;
    std::optional<InputContentMeasurementReceipt> content_measurement_receipt;
  };

  grpc::Status submit_experiment(
      grpc::ServerContext* context,
      const v1::SubmitExperimentRequest* request,
      v1::SubmitExperimentResponse* response,
      const std::optional<AuthorRunAuthority>& authoring);
  struct WorkerConnection {
    WorkerSessionIdentity identity;
    Dispatch dispatch;
    nlohmann::json publishes = nlohmann::json::object();
    std::uint64_t attempt_baseline_optimizer_step{};
    v1::WorkerWelcome welcome;
    std::optional<v1::WorkerReceipt> completed_receipt;
  };

  grpc::Status open_worker_connection(const v1::WorkerHello& hello,
                                      WorkerConnection& connection);
  grpc::Status complete_worker_connection(
      const v1::EventEnvelope& envelope,
      const WorkerConnection& connection, v1::WorkerReceipt& receipt);
  grpc::Status record_worker_heartbeat(
      const v1::WorkerHeartbeat& heartbeat,
      const WorkerConnection& connection, std::uint64_t& acknowledged);
  grpc::Status record_worker_metric(
      const v1::MetricSample& metric, const WorkerConnection& connection,
      std::uint64_t& acknowledged);
  grpc::Status record_worker_artifact(
      const v1::ArtifactManifest& artifact,
      const WorkerConnection& connection, std::uint64_t& acknowledged);
  grpc::Status record_worker_execution_phase_receipt(
      const v1::WorkerExecutionPhaseReceipt& receipt,
      const WorkerConnection& connection, std::uint64_t& acknowledged);
  // No `acknowledged` out-parameter, and that is deliberate: runtime evidence
  // carries no worker_sequence, because the transport is exactly the report
  // struct. Its durable record is the immutable receipt this publishes, and a
  // refusal reaches the worker as the stream's terminal status.
  grpc::Status record_worker_runtime_evidence(
      const v1::WorkerRuntimeEvidence& evidence,
      const WorkerConnection& connection);
  grpc::Status acknowledge_worker_control(
      const v1::ControlPatchAcknowledgement& acknowledgement,
      const WorkerConnection& connection, std::uint64_t& acknowledged);
  grpc::Status acknowledge_worker_checkpoint(
      const v1::CheckpointAcknowledgement& acknowledgement,
      const WorkerConnection& connection, std::uint64_t& acknowledged);
  grpc::Status acknowledge_worker_lifecycle(
      const v1::LifecycleAcknowledgement& acknowledgement,
      const WorkerConnection& connection, std::uint64_t& acknowledged);
  grpc::Status commit_worker_observation(
      Event event, const WorkerConnection& connection,
      std::uint64_t& acknowledged);
  bool claim_worker_attempt(const std::string& key);
  void release_worker_attempt(const std::string& key);
  [[nodiscard]] AuthorityTimeSample authority_now() const;
  ReconcileResult reconcile_once(const std::string& run_id);
  void notify_reconciliation(const std::string& run_id);
  void reconciliation_loop(std::stop_token stop);
  // Returns the disposition the pass settled on, or nullopt when the pass
  // exhausted its step budget without reaching a quiescent disposition.
  std::optional<ReconcileDisposition> reconcile_until_quiescent(
      const std::string& run_id);
  void synchronize_lease_renewal(const std::string& run_id);
  [[nodiscard]] std::optional<std::string> reconciliation_failure(
      const std::string& run_id) const;
  void record_reconciliation_failure(std::string run_id,
                                     std::string message);
  // Process-free supervisor boundary. The ticket must already be durable.
  // Successful bindings retain their sealed descriptor bundle for a later
  // launcher boundary; this method never forks or executes a process.
  ResolvedLaunchSpec bind_worker_launch(const WorkerLaunchTicket& launch);
  HostProcessSagaSnapshot launch_worker_process(
      const std::string& launch_id);
  [[nodiscard]] std::optional<ReconcileDisposition> reconcile_host_grant(
      const std::string& run_id);
  [[nodiscard]] std::optional<ReconcileDisposition> reconcile_host_release(
      const std::string& run_id);
  [[nodiscard]] bool reconcile_external_profiler_artifact(
      const RunProjection& projection, const CompiledPlan& plan,
      const HostProcessSagaSnapshot& process,
      const ResolvedLaunchSpec& binding);
  // Returns host authority a run kept past its own terminal state. The four
  // handlers beside this one each require a specific non-terminal state pair,
  // and Reconciler::step reports no_action once the controller is not running,
  // so without this a rediscovered orphan reconciles to nothing forever.
  [[nodiscard]] std::optional<ReconcileDisposition>
  reconcile_terminal_host_drain(const std::string& run_id);
  [[nodiscard]] std::optional<ReconcileDisposition> reconcile_cancellation(
      const std::string& run_id);
  [[nodiscard]] std::optional<ReconcileDisposition>
  reconcile_resource_releasing_pause(const std::string& run_id);
  void configure_hostd(const HostdClientConfiguration& configuration,
                       std::string controller_target);

  // Deterministic host-identity injection is restricted to focused authority
  // tests. Production construction always captures the local Linux identity.
  TrainVMService(const std::filesystem::path& journal_path,
                 AdapterRegistry adapter_registry,
                 HostLaunchRegistry host_launch_registry,
                 HostIdentity authority_host,
                 std::function<AuthorityTimeSample()> authority_clock,
                 HostGrantEnforcement host_grant_enforcement =
                     HostGrantEnforcement::required,
                 TrainingComponentRegistry training_components =
                     TrainingComponentRegistry({}),
                 std::shared_ptr<IHostGrantClient> host_grant_client = {},
                 std::shared_ptr<IHostProcessClient> host_process_client = {},
                 std::string controller_target = {},
                 ICacheQualificationEvidenceResolver* cache_qualification =
                     nullptr,
                 SqliteAuthorityEnforcementGrade filesystem_enforcement_grade =
                     SqliteAuthorityEnforcementGrade::cooperative_test,
                 std::shared_ptr<ITrainingPreflightEvidenceProvider>
                     preflight_evidence = {},
                 std::filesystem::path recipe_registry_path =
                     std::filesystem::path(
                         std::string(kInstalledRecipeProfilePath)),
                 IWorkerRuntimeEvidenceAuthority* worker_runtime_evidence =
                     nullptr,
                 std::optional<std::filesystem::path> cache_evidence_root =
                     std::nullopt);

  static constexpr std::size_t kMaximumRetainedLaunches = 32U;
  static constexpr std::uint64_t kMaximumRetainedLaunchBytes = 2ULL << 30U;
  void prune_retained_launches(const AuthorityTimeSample& now);
  void require_retained_launch_capacity(const ResolvedLaunchSpec& candidate) const;

  std::unique_ptr<AuthorityLock> authority_lock_;
  Journal journal_;
  InputContentMeasurementCache input_content_measurement_cache_;
  std::mutex command_mutex_;
  std::shared_ptr<AuthorityClock> authority_clock_;
  LeaseRenewalCoordinator lease_renewals_;
  const AdapterRegistry adapter_registry_;
  const HostLaunchRegistry host_launch_registry_;
  const TrainingComponentRegistry training_components_;
  std::shared_ptr<ITrainingPreflightEvidenceProvider> preflight_evidence_;
  const std::filesystem::path recipe_registry_path_;
  const HostIdentity authority_host_;
  HostLaunchResolver host_launch_resolver_;
  std::shared_ptr<JournalHostdMutationClaimProvider>
      hostd_claim_provider_;
  std::shared_ptr<IHostGrantClient> host_grant_client_;
  std::unique_ptr<HostGrantSagaReconciler> host_grant_saga_;
  std::shared_ptr<IHostProcessClient> host_process_client_;
  std::optional<HostdStatusClientConfig> hostd_status_client_;
  std::int64_t hostd_status_timeout_ns_{};
  std::atomic<std::uint64_t> hostd_status_correlation_{1U};
  std::string controller_target_;
  std::unique_ptr<HostProcessSagaReconciler> host_process_saga_;
  Reconciler reconciler_;
  static constexpr std::size_t kReconciliationPageSize = 128U;
  static constexpr std::size_t kMaximumImmediateReconcileSteps = 32U;
  static constexpr std::size_t kMaximumSupervisorWakeRuns = 4'096U;
  static constexpr std::size_t kMaximumSupervisorFailures = 4'096U;
  static constexpr std::size_t kMaximumSupervisorFailureBytes = 4'096U;

  // Idle scheduling. The supervisor still ticks at kSupervisorCadenceNs, but a
  // tick now costs one indexed projection scan rather than a full controller
  // recovery per reconcilable run: a run is only reconciled when its journal
  // state moved, an explicit wake named it, or its backoff deadline expired.
  //
  // Two caps, because the two kinds of wait have different failure modes. A
  // run parked on worker evidence cannot change without a journal write, so its
  // re-check is a safety net and may be slow. A run parked on lease or host
  // grant contention *can* change on the clock alone, with no journal write to
  // wake it, so its re-check must stay well inside any lease timeout.
  static constexpr std::int64_t kSupervisorCadenceNs = 250'000'000LL;
  static constexpr std::int64_t kIdleBackoffCeilingNs = 30'000'000'000LL;
  static constexpr std::int64_t kContendedBackoffCeilingNs = 2'000'000'000LL;
  static constexpr std::size_t kMaximumSupervisorSchedules = 4'096U;

  struct SupervisorRunSchedule final {
    std::uint64_t last_event_sequence{};
    std::int64_t next_due_ns{};
    std::int64_t backoff_ns{};
    std::uint64_t idle_passes{};
    std::uint64_t retries{};
    std::string wait_reason;
  };

  [[nodiscard]] bool reconciliation_due(const RunProjection& projection,
                                        std::int64_t now_ns) const;
  void record_reconciliation_outcome(
      const std::string& run_id,
      std::optional<ReconcileDisposition> disposition,
      std::uint64_t observed_event_sequence, std::int64_t now_ns);
  void record_reconciliation_retry(const std::string& run_id,
                                   std::uint64_t observed_event_sequence,
                                   std::int64_t now_ns);
  // Caller holds reconciliation_mutex_.
  SupervisorRunSchedule& reconciliation_schedule(const std::string& run_id);

  mutable std::mutex reconciliation_mutex_;
  std::condition_variable reconciliation_condition_;
  std::set<std::string, std::less<>> reconciliation_wake_runs_;
  std::map<std::string, std::string, std::less<>> reconciliation_failures_;
  std::map<std::string, SupervisorRunSchedule, std::less<>>
      reconciliation_schedules_;
  ReconciliationSupervisorMetrics reconciliation_metrics_;
  std::string reconciliation_scan_cursor_;
  bool reconciliation_started_{};
  std::jthread reconciliation_thread_;
  std::map<std::string, ResolvedLaunch> resolved_launches_;
  // Which host inventory this launch was granted against, read from the
  // grant its own sealed claim names. Answering it from the durable grant
  // rather than from a live inventory is what makes the answer stable: the
  // current inventory differs from the granted one exactly when the host
  // changed between grant and report, so a live read would name a receipt the
  // namespace derivation never looks for, and would do so only sometimes.
  //
  // Absent -- not an error -- when the launch holds no host grant at all, or
  // holds one recorded before grants carried a projection.
  [[nodiscard]] std::optional<GrantInventoryProjection>
  granted_inventory_projection(const ResolvedLaunchSpec& launch) const;

  IWorkerRuntimeEvidenceAuthority* worker_runtime_evidence_{};
  const std::optional<LinuxCacheEvidenceConfig> cache_evidence_;
  // The authority a rooted deployment holds. It is owned here rather than
  // handed in by `serve()` because its inventory lookup reads this service's
  // own journal, which does not exist until the service does. An explicitly
  // injected authority wins, so a test can still substitute one.
  std::unique_ptr<LinuxWorkerRuntimeEvidenceAuthority>
      owned_worker_runtime_evidence_;
  std::mutex worker_sessions_mutex_;
  std::mutex author_run_mutex_;
  std::set<std::string> active_worker_attempts_;
};

// Blocks until SIGINT or SIGTERM after binding a permission-restricted Unix
// socket. Throws if another authority owns the journal lock.
int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path,
          AdapterRegistry adapter_registry,
          HostLaunchRegistry host_launch_registry,
          TrainingComponentRegistry training_components =
              TrainingComponentRegistry({}),
          std::optional<HostdClientConfiguration> hostd_configuration =
              std::nullopt,
          std::optional<std::uint32_t> worker_socket_gid = std::nullopt,
          std::filesystem::path recipe_registry_path =
              std::filesystem::path(std::string(kInstalledRecipeProfilePath)),
          std::optional<std::filesystem::path> cache_evidence_root =
              std::nullopt);

}  // namespace trainvm
