#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "trainvm/adapter_registry.hpp"

namespace trainvm {

inline constexpr std::string_view kWorkerInvocationApiVersion =
    "trainvm.worker-invocation/v1";
inline constexpr std::size_t kMaximumWorkerInvocationBytes = 48U * 1024U;

struct WorkerInvocationContext final {
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string dispatch_id;
  std::uint64_t plan_revision{};
  std::string host_id;
  // Complete, already-authorized artifact manifests keyed by logical artifact
  // name. Only inputs referenced by the selected node are retained.
  std::map<std::string, nlohmann::json, std::less<>> artifacts;
  // Applied control overrides only; defaults come from the immutable plan.
  nlohmann::json effective_controls = nlohmann::json::object();
  std::uint64_t effective_control_revision{};
};

// Immutable, content-addressed operation context delivered in WorkerWelcome.
// It contains public resolved inputs and authority metadata, never raw secret
// values or executable paths supplied by an experiment document.
struct WorkerInvocationSpec final {
  std::string api_version;
  std::string run_id;
  std::string host_id;
  std::string plan_hash;
  std::uint64_t plan_revision{};
  std::string node_id;
  std::string attempt_id;
  std::string dispatch_id;
  AdapterKey adapter;
  nlohmann::json workspace;
  nlohmann::json resources;
  nlohmann::json inputs;
  nlohmann::json controls;
  std::uint64_t effective_control_revision{};
  nlohmann::json publishes;
  nlohmann::json observability;
  nlohmann::json execution;
  std::string invocation_digest;

  bool operator==(const WorkerInvocationSpec&) const = default;
};

[[nodiscard]] WorkerInvocationSpec build_worker_invocation(
    const CompiledPlan& plan, const WorkerInvocationContext& context);
[[nodiscard]] std::string worker_invocation_canonical_json(
    const WorkerInvocationSpec& value);
[[nodiscard]] WorkerInvocationSpec worker_invocation_from_canonical_json(
    std::string_view value);

}  // namespace trainvm
