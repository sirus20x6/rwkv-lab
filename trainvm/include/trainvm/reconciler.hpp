#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/authority_time.hpp"
#include "trainvm/cache_artifact_authority.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/training_component_registry.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

class TrainVMService;

enum class ReconcileDisposition {
  lease_acquired,
  lease_busy,
  host_grant_acquired,
  host_grant_busy,
  host_process_exited,
  host_grant_released,
  launch_prepared,
  launch_replayed,
  awaiting_worker,
  builtin_completed,
  qualification_completed,
  qualification_rejected,
  qualification_evidence_required,
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

// TrainVMService constructs this only with an explicitly configured typed
// hostd client and serializes every call with the same authority mutex used by
// Reconciler and WorkerControl. Grant/release therefore cannot race launch,
// hello, dispatch, or result authority checks.
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

// The authority seam for a qualify_cache node. Production resolves evidence
// from immutable, worker-published qualification artifacts bound to the node's
// own run/attempt; tests inject an exact fake. An experiment document is never
// an evidence source, and the supervisor never accepts a bare verdict: it
// always re-runs the implemented gate over whatever this returns.
class ICacheQualificationEvidenceResolver {
 public:
  virtual ~ICacheQualificationEvidenceResolver() = default;
  // Returns the evidence published for this run/node/attempt, or nullopt when
  // the worker has not published any yet. Returning nullopt leaves the node
  // pending instead of failing the run, so a slow publisher is a wait rather
  // than a rejection.
  [[nodiscard]] virtual std::optional<CacheQualificationEvidence>
  resolve(const std::string& run_id, const std::string& node_id,
          const std::string& attempt_id) = 0;
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
             std::function<AuthorityTimeSample()> authority_clock,
             ICacheQualificationEvidenceResolver* qualification = nullptr);

  Journal& journal_;
  const AdapterRegistry& registry_;
  const TrainingComponentRegistry& training_components_;
  std::mutex& authority_mutex_;
  std::shared_ptr<AuthorityClock> authority_clock_;
  ICacheQualificationEvidenceResolver* qualification_{};
};

}  // namespace trainvm
