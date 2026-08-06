#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "trainvm/json.hpp"

namespace trainvm {

struct Event;

inline constexpr std::string_view kEvalExamplesSchema =
    "rwkv-lab.eval-examples.v1";

struct EvalExamplesPart final {
  std::string kind;
  std::optional<std::string> text;
  std::optional<std::string> path;
  std::optional<std::string> media_type;
  std::optional<std::string> sha256;
  std::optional<std::uint64_t> size_bytes;
  std::optional<std::string> schema;
  std::optional<nlohmann::json> value;
};

struct EvalExample final {
  std::string example_id;
  std::string heldout_item_id;
  std::string heldout_item_digest;
  std::vector<EvalExamplesPart> input;
  std::vector<EvalExamplesPart> target;
  std::vector<EvalExamplesPart> prediction;
};

struct EvalExamplesHeldout final {
  std::string identity_field;
  std::string identities_digest;
  std::string selector_digest;
};

struct EvalExamplesEvaluator final {
  std::string component_digest;
  std::vector<std::string> metric_names;
};

struct EvalExamplesCheckpoint final {
  std::string artifact_id;
  std::string manifest_digest;
};

struct EvalExamplesManifest final {
  std::string api_version;
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::uint64_t optimizer_step{};
  std::string step_domain;
  std::string series_id;
  EvalExamplesHeldout heldout;
  EvalExamplesEvaluator evaluator;
  EvalExamplesCheckpoint checkpoint;
  std::string policy_digest;
  std::vector<EvalExample> examples;
  std::string canonical_manifest_digest;
};

// Strict reflected decode plus semantic validation. Throws invalid_argument on
// malformed, noncanonical, empty, or internally inconsistent evidence.
EvalExamplesManifest
validate_eval_examples_manifest(const nlohmann::json &document);

// Pins the manifest and every referenced media object beneath its immutable
// revision directory, rejects aliases/traversal/symlinks/special files, and
// cross-checks bytes, size, and SHA-256 before gate authority is granted.
void validate_eval_examples_payload(const EvalExamplesManifest &manifest,
                                    std::string_view manifest_uri,
                                    std::string_view canonical_manifest_bytes,
                                    std::string_view manifest_fingerprint,
                                    std::string_view run_directory,
                                    std::string_view artifact_id);

// Binds the semantic manifest to the resolved evaluator and to prior durable
// scalar/checkpoint events. This is evaluated before artifact publication.
void validate_eval_examples_gate_provenance(
    const EvalExamplesManifest &manifest,
    const nlohmann::json &resolved_training,
    const std::vector<Event> &prior_events);

bool invocation_requires_step_zero_eval_gate(const nlohmann::json &publishes);
// Gate evidence is intentionally attempt-local. A resumed attempt fails closed
// until card-986f974e defines checkpoint-carried gate lineage.
bool durable_step_zero_eval_gate_satisfied(const std::vector<Event> &events,
                                           std::string_view run_id,
                                           std::string_view node_id,
                                           std::string_view attempt_id);

} // namespace trainvm
