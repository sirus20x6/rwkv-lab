#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>

#include <grpcpp/grpcpp.h>

#include "trainvm/journal.hpp"
#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/reconciler.hpp"
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
      std::function<AuthorityTimeSample()> authority_clock = {});
  ~TrainVMService() override;

  grpc::Status SubmitExperiment(grpc::ServerContext* context,
                                const v1::SubmitExperimentRequest* request,
                                v1::SubmitExperimentResponse* response) override;

  grpc::Status CommandRun(grpc::ServerContext* context,
                          const v1::RunCommandRequest* request,
                          v1::RunCommandResponse* response) override;

  grpc::Status Connect(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<v1::ControllerToWorker,
                               v1::WorkerToController>* stream) override;

 private:
  struct WorkerConnection {
    WorkerSessionIdentity identity;
    Dispatch dispatch;
    v1::WorkerWelcome welcome;
    std::optional<v1::WorkerReceipt> completed_receipt;
  };

  grpc::Status open_worker_connection(const v1::WorkerHello& hello,
                                      WorkerConnection& connection);
  grpc::Status complete_worker_connection(
      const v1::EventEnvelope& envelope,
      const WorkerConnection& connection, v1::WorkerReceipt& receipt);
  bool claim_worker_attempt(const std::string& key);
  void release_worker_attempt(const std::string& key);
  [[nodiscard]] AuthorityTimeSample authority_now() const;
  ReconcileResult reconcile_once(const std::string& run_id);
  // Process-free supervisor boundary. The ticket must already be durable.
  // Successful bindings retain their sealed descriptor bundle for a later
  // launcher boundary; this method never forks or executes a process.
  ResolvedLaunchSpec bind_worker_launch(const WorkerLaunchTicket& launch);

  // Deterministic host-identity injection is restricted to focused authority
  // tests. Production construction always captures the local Linux identity.
  TrainVMService(const std::filesystem::path& journal_path,
                 AdapterRegistry adapter_registry,
                 HostLaunchRegistry host_launch_registry,
                 HostIdentity authority_host,
                 std::function<AuthorityTimeSample()> authority_clock,
                 HostGrantEnforcement host_grant_enforcement =
                     HostGrantEnforcement::required);

  static constexpr std::size_t kMaximumRetainedLaunches = 32U;
  static constexpr std::uint64_t kMaximumRetainedLaunchBytes = 2ULL << 30U;
  void prune_retained_launches(const AuthorityTimeSample& now);
  void require_retained_launch_capacity(const ResolvedLaunchSpec& candidate) const;

  std::unique_ptr<AuthorityLock> authority_lock_;
  Journal journal_;
  std::mutex command_mutex_;
  std::shared_ptr<AuthorityClock> authority_clock_;
  const AdapterRegistry adapter_registry_;
  const HostLaunchRegistry host_launch_registry_;
  const HostIdentity authority_host_;
  HostLaunchResolver host_launch_resolver_;
  Reconciler reconciler_;
  std::map<std::string, ResolvedLaunch> resolved_launches_;
  std::mutex worker_sessions_mutex_;
  std::set<std::string> active_worker_attempts_;
};

// Blocks until SIGINT or SIGTERM after binding a permission-restricted Unix
// socket. Throws if another authority owns the journal lock.
int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path,
          AdapterRegistry adapter_registry,
          HostLaunchRegistry host_launch_registry);

}  // namespace trainvm
