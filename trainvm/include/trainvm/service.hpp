#pragma once

#include <cstdint>
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

#include <grpcpp/grpcpp.h>

#include "trainvm/journal.hpp"
#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/hostd_client_bootstrap.hpp"
#include "trainvm/host_process_saga.hpp"
#include "trainvm/resource_request_builder.hpp"
#include "trainvm/reconciler.hpp"
#include "trainvm/lease_renewal.hpp"
#include "trainvm/training_component_registry.hpp"
#include "trainvm/v1/trainvm.grpc.pb.h"

namespace trainvm {

class AuthorityLock {
 public:
  explicit AuthorityLock(const std::filesystem::path& journal_path);
  ~AuthorityLock();

  AuthorityLock(const AuthorityLock&) = delete;
  AuthorityLock& operator=(const AuthorityLock&) = delete;

  [[nodiscard]] const std::filesystem::path& journal_path() const noexcept;
  [[nodiscard]] const JournalFileIdentity& journal_identity() const noexcept;

 private:
  int kernel_namespace_descriptor_{-1};
  int descriptor_{-1};
  int journal_descriptor_{-1};
  int directory_descriptor_{-1};
  std::filesystem::path stable_journal_path_;
  JournalFileIdentity journal_identity_;
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
      std::string controller_target = {});
  ~TrainVMService() override;

  grpc::Status SubmitExperiment(grpc::ServerContext* context,
                                const v1::SubmitExperimentRequest* request,
                                v1::SubmitExperimentResponse* response) override;

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

  grpc::Status Connect(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<v1::ControllerToWorker,
                               v1::WorkerToController>* stream) override;

  // Embedded servers may explicitly bracket the same supervisor lifecycle
  // that serve() owns. Destruction is an idempotent stop boundary.
  void start_reconciliation_supervisor();
  void stop_reconciliation_supervisor() noexcept;

 private:
  struct WorkerConnection {
    WorkerSessionIdentity identity;
    Dispatch dispatch;
    nlohmann::json publishes = nlohmann::json::object();
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
  void reconcile_until_quiescent(const std::string& run_id);
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
                 std::string controller_target = {});

  static constexpr std::size_t kMaximumRetainedLaunches = 32U;
  static constexpr std::uint64_t kMaximumRetainedLaunchBytes = 2ULL << 30U;
  void prune_retained_launches(const AuthorityTimeSample& now);
  void require_retained_launch_capacity(const ResolvedLaunchSpec& candidate) const;

  std::unique_ptr<AuthorityLock> authority_lock_;
  Journal journal_;
  std::mutex command_mutex_;
  std::shared_ptr<AuthorityClock> authority_clock_;
  LeaseRenewalCoordinator lease_renewals_;
  const AdapterRegistry adapter_registry_;
  const HostLaunchRegistry host_launch_registry_;
  const TrainingComponentRegistry training_components_;
  const HostIdentity authority_host_;
  HostLaunchResolver host_launch_resolver_;
  std::shared_ptr<JournalHostdMutationClaimProvider>
      hostd_claim_provider_;
  std::shared_ptr<IHostGrantClient> host_grant_client_;
  std::unique_ptr<HostGrantSagaReconciler> host_grant_saga_;
  std::shared_ptr<IHostProcessClient> host_process_client_;
  std::string controller_target_;
  std::unique_ptr<HostProcessSagaReconciler> host_process_saga_;
  Reconciler reconciler_;
  static constexpr std::size_t kReconciliationPageSize = 128U;
  static constexpr std::size_t kMaximumImmediateReconcileSteps = 32U;
  static constexpr std::size_t kMaximumSupervisorWakeRuns = 4'096U;
  static constexpr std::size_t kMaximumSupervisorFailures = 4'096U;
  static constexpr std::size_t kMaximumSupervisorFailureBytes = 4'096U;
  mutable std::mutex reconciliation_mutex_;
  std::condition_variable reconciliation_condition_;
  std::set<std::string, std::less<>> reconciliation_wake_runs_;
  std::map<std::string, std::string, std::less<>> reconciliation_failures_;
  std::string reconciliation_scan_cursor_;
  bool reconciliation_started_{};
  std::jthread reconciliation_thread_;
  std::map<std::string, ResolvedLaunch> resolved_launches_;
  std::mutex worker_sessions_mutex_;
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
              std::nullopt);

}  // namespace trainvm
