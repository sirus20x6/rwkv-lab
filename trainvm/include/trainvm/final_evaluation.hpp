#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/adapter_registry.hpp"

namespace trainvm {

struct Event;
struct WorkerInvocationSpec;

inline constexpr std::size_t kMaximumFinalEvaluationManifestBytes =
    32U * 1024U * 1024U;

// Finalization is deliberately separate from execution. A successful trainer
// return is only a proposal to complete; these policies describe the durable
// evidence that must already exist before that proposal is accepted.
enum class FinalEvidenceKind {
  checkpoint,
  scalar,
  examples,
  test,
  audit,
  closure,
};

enum class FinalCoveragePolicy {
  durable_nonempty,
  full_membership,
};

enum class FinalErrorPolicy {
  not_applicable,
  zero_unresolved_errors,
};

struct FinalOutputPolicy final {
  std::string output_name;
  FinalEvidenceKind evidence_kind{};
  bool required{};
  // Optional outputs become finalization requirements whenever the immutable
  // invocation declares them. This covers conditional galleries without
  // silently weakening adapters that do declare one.
  bool required_when_declared{};
  bool exact_optimizer_step{};
  bool checkpoint_bound{};
  FinalCoveragePolicy coverage{};
  FinalErrorPolicy errors{};
  std::optional<std::string> artifact_schema;

  auto operator<=>(const FinalOutputPolicy &) const = default;
};

struct OperationFinalizationPolicy final {
  AdapterKey key;
  std::vector<FinalOutputPolicy> outputs;
  bool eval_only_recovery{};
  std::optional<std::string> closure_output_name;
  bool closure_required{};
  bool migration_pending{};

  auto operator<=>(const OperationFinalizationPolicy &) const = default;
};

// Builds a closed policy inventory from the registered operation descriptors.
// Every stateful profile must resolve exactly once; adding a profile or output
// without a known semantic classification fails construction.
class FinalizationPolicyRegistry final {
public:
  explicit FinalizationPolicyRegistry(
      const std::vector<AdapterProfile> &profiles);

  [[nodiscard]] const OperationFinalizationPolicy &
  resolve(const AdapterKey &key) const;
  [[nodiscard]] const std::map<AdapterKey, OperationFinalizationPolicy> &
  policies() const;
  [[nodiscard]] nlohmann::json inventory_json() const;

private:
  std::map<AdapterKey, OperationFinalizationPolicy> policies_;
};

enum class FinalMemberDisposition { error, success };

struct FinalMemberRecord final {
  std::string member_id;
  std::string context_digest;
  std::uint64_t attempt{};
  FinalMemberDisposition disposition{};
  // Required for a success and absent for an error. The digest binds the
  // nonempty result bytes without copying model output into the journal.
  std::optional<std::string> result_digest;
  std::optional<std::string> error_code;

  auto operator<=>(const FinalMemberRecord &) const = default;
};

struct FinalOutputReceipt final {
  std::string output_name;
  std::string artifact_id;
  std::string artifact_fingerprint;

  auto operator<=>(const FinalOutputReceipt &) const = default;
};

struct FinalMemberContext final {
  std::string member_id;
  std::string context_digest;

  auto operator<=>(const FinalMemberContext &) const = default;
};

struct FinalScalarRequirement final {
  std::string metric_name;
  std::string step_domain;

  auto operator<=>(const FinalScalarRequirement &) const = default;
};

// A controller-owned pointer to one durable terminal-step metric event.  The
// worker manifest declares which scalar semantics it expects, but it cannot
// assert that the corresponding journal event exists.
struct FinalScalarObservation final {
  std::string metric_name;
  std::string step_domain;
  std::uint64_t optimizer_step{};
  std::string event_id;

  auto operator<=>(const FinalScalarObservation &) const = default;
};

struct EvalOnlyRecoveryReceipt final {
  std::vector<std::string> requested_members;
  std::string optimizer_state_before;
  std::string optimizer_state_after;

  auto operator<=>(const EvalOnlyRecoveryReceipt &) const = default;
};

// Worker-authored semantic bytes. This is deliberately separate from the
// controller receipt below so strict reflection can round-trip the artifact
// without making controller-owned durability fields worker assertions.
struct FinalEvaluationManifest final {
  std::string api_version;
  std::string policy_digest;
  std::uint64_t optimizer_step{};
  std::string checkpoint_artifact_id;
  std::string checkpoint_fingerprint;
  std::string membership_digest;
  std::uint64_t membership_count{};
  std::uint64_t resolved_member_count{};
  std::uint64_t failed_member_count{};
  std::vector<std::string> required_members;
  std::vector<FinalOutputReceipt> output_receipts;
  // These are expectations, not worker assertions. The controller resolves
  // them against durable metric.sampled events at the exact terminal step.
  std::vector<FinalScalarRequirement> required_scalars;
  std::vector<FinalMemberRecord> records;
  std::optional<EvalOnlyRecoveryReceipt> recovery;

  auto operator<=>(const FinalEvaluationManifest &) const = default;
};

// Controller-owned durable envelope around one strictly decoded manifest.
// Later receipts may append retry records but may never rewrite or delete
// history from an earlier receipt.
struct FinalEvaluationReceipt final {
  std::string artifact_id;
  std::string artifact_fingerprint;
  std::uint64_t durable_sequence{};
  std::vector<FinalOutputReceipt> durable_output_receipts;
  std::vector<FinalScalarObservation> durable_scalar_observations;
  FinalEvaluationManifest manifest;

  auto operator<=>(const FinalEvaluationReceipt &) const = default;
};

// Immutable controller authority derived from the invocation, terminal
// checkpoint, metric composition, and frozen evaluation membership. None of
// these values may be learned from a worker closure.
struct FinalEvaluationExpectation final {
  std::string output_name;
  std::vector<std::string> required_output_names;
  std::string policy_digest;
  std::uint64_t optimizer_step{};
  std::string checkpoint_artifact_id;
  std::string checkpoint_fingerprint;
  std::vector<std::string> required_members;
  std::vector<FinalMemberContext> member_contexts;
  std::string membership_digest;
  std::uint64_t membership_count{};
  std::vector<FinalScalarRequirement> required_scalars;
  // Present only when the controller has an independently verified optimizer
  // state identity and the operation explicitly supports eval-only recovery.
  std::optional<std::string> terminal_optimizer_fingerprint;

  auto operator<=>(const FinalEvaluationExpectation &) const = default;
};

enum class FinalizationDisposition { complete, pending, failed };

struct FinalizationVerdict final {
  FinalizationDisposition disposition{};
  std::string cause;
  std::vector<std::string> unresolved_members;
  std::optional<std::string> selected_artifact_id;
  std::optional<std::string> selected_artifact_fingerprint;
};

// Reduces append-only evaluation history at one exact terminal checkpoint.
// Invalid/stale/context-poisoned evidence fails; valid but incomplete evidence
// stays pending so an eval-only attempt can retry only unresolved members.
[[nodiscard]] FinalizationVerdict
reduce_final_evaluation(const OperationFinalizationPolicy &operation_policy,
                        const FinalEvaluationExpectation &expectation,
                        const std::vector<FinalEvaluationReceipt> &history);

[[nodiscard]] std::string
finalization_policy_digest(const OperationFinalizationPolicy &policy);

[[nodiscard]] std::string
final_membership_digest(const std::vector<std::string> &canonical_members);

[[nodiscard]] nlohmann::json
finalization_verdict_json(const FinalizationVerdict &verdict);

// Emits worker-owned semantic fields only. Artifact identity, fingerprint,
// durable sequence, and durable parent checks are controller-derived inputs to
// the reducer and are intentionally absent from these canonical bytes.
[[nodiscard]] nlohmann::json
final_evaluation_manifest_json(const FinalEvaluationManifest &manifest);

// The sole authority entry point for untrusted closure bytes. The size check
// occurs before JSON allocation, then strict reflection and an exact canonical
// byte round-trip prevent duplicate keys, aliases, or alternate encodings.
[[nodiscard]] FinalEvaluationManifest
decode_final_evaluation_manifest(std::string_view bytes);

// First live-family authority adapter. It freezes held-out membership from the
// admitted content root and resolved component graph, then resolves worker
// closures only through exact durable parent/scalar events.
[[nodiscard]] FinalEvaluationExpectation
derive_hf_final_evaluation_expectation(
    const OperationFinalizationPolicy &policy,
    const WorkerInvocationSpec &invocation, std::uint64_t terminal_step,
    const std::vector<Event> &durable_events);

[[nodiscard]] std::vector<FinalEvaluationReceipt>
resolve_final_evaluation_receipts(
    const WorkerInvocationSpec &invocation,
    const FinalEvaluationExpectation &expectation,
    const std::vector<Event> &durable_events);

} // namespace trainvm
