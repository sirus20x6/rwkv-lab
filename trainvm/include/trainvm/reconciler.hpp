#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/authority_time.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/training_component_registry.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

class TrainVMService;

enum class ReconcileDisposition {
  lease_acquired,
  lease_busy,
  launch_prepared,
  launch_replayed,
  awaiting_worker,
  builtin_completed,
  input_required,
  no_action,
};

struct ReconcileResult {
  ReconcileDisposition disposition{ReconcileDisposition::no_action};
  std::string run_id;
  std::optional<WorkerLaunchTicket> launch;
};

enum class HostGrantSagaFaultPoint {
  journal_before_host,
  host_before_journal,
  replay_boundary,
};

class IHostGrantSagaFaultInjector {
 public:
  virtual ~IHostGrantSagaFaultInjector() = default;
  virtual void hit(HostGrantSagaFaultPoint point) = 0;
};

class IHostGrantClient {
 public:
  virtual ~IHostGrantClient() = default;
  [[nodiscard]] virtual BundleRequestResult request_bundle(
      const ResourceBundleRequest& request) = 0;
  [[nodiscard]] virtual BundleReleaseResult release_bundle(
      const ResourceReleaseRequest& request) = 0;
};

// This coordinator is intentionally not constructed by TrainVMService yet.
// Production wiring must give it the same authority mutex used by Reconciler
// and WorkerControl before a real host RPC client is admitted; otherwise a
// grant/release can race launch, hello, dispatch, or result authority checks.
class HostGrantSagaReconciler final {
 public:
  HostGrantSagaReconciler(Journal& journal, IHostGrantClient& host,
                          IHostGrantSagaFaultInjector* faults = nullptr);

  [[nodiscard]] HostGrantSagaSnapshot reconcile_request(
      const ResourceBundleRequest& request, const AuthorityTimeSample& now);
  [[nodiscard]] HostGrantSagaSnapshot reconcile_release(
      const std::string& request_id, const ResourceReleaseRequest& release,
      const AuthorityTimeSample& now);

 private:
  void fault(HostGrantSagaFaultPoint point) const;

  Journal& journal_;
  IHostGrantClient& host_;
  IHostGrantSagaFaultInjector* faults_{};
};

// This is launch authorization only: it may persist worker.launch_requested as
// a protocol intent, but it MUST NOT spawn or signal an OS process. A later
// supervisor must require a host-bound resolved launch-spec digest before exec.
//
// Performs one controller reconciliation command per step. Resource admission
// is an existing idempotent composite boundary and may span its lease and
// builtin-admission transactions. Production construction is restricted to
// TrainVMService so reconciliation shares its command/WorkerControl gate.
class Reconciler {
 public:
  ReconcileResult step(const std::string& run_id);

 private:
  friend class TrainVMService;

  Reconciler(Journal& journal, const AdapterRegistry& registry,
             std::mutex& authority_mutex,
             std::function<AuthorityTimeSample()> authority_clock);
  Reconciler(Journal& journal, const AdapterRegistry& registry,
             const TrainingComponentRegistry& training_components,
             std::mutex& authority_mutex,
             std::function<AuthorityTimeSample()> authority_clock);

  Journal& journal_;
  const AdapterRegistry& registry_;
  const TrainingComponentRegistry& training_components_;
  std::mutex& authority_mutex_;
  std::shared_ptr<AuthorityClock> authority_clock_;
};

}  // namespace trainvm
