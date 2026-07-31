#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/journal.hpp"
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
             std::function<std::int64_t()> authority_clock);

  Journal& journal_;
  const AdapterRegistry& registry_;
  std::mutex& authority_mutex_;
  std::function<std::int64_t()> authority_clock_;
};

}  // namespace trainvm
