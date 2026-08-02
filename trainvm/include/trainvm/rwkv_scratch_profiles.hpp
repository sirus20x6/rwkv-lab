#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/model.hpp"

namespace trainvm {

inline constexpr std::string_view kRwkvScratchProfilesApiVersion =
    "trainvm.rwkv-scratch-profiles/v1";

class RwkvScratchProfileError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// The research topologies the scratch RWKV trainer supports, each a closed
// subset of rwkv_pretrain rather than a passthrough of its 100+ switches. A
// topology that is not listed here cannot be selected by an experiment
// document, and a switch that is not declared by its topology cannot be set.
enum class RwkvScratchTopology {
  baseline,
  loop,
  latent_lookahead,
  engram,
  routing_free_moe,
  online_memory,
  byte_aware_tokenizer,
  seed_chain,
  deep_embed,
  distributed,
};

enum class RwkvScratchParameterType {
  integer,
  number,
  boolean,
  enumeration,
};

// One declared knob. Bounds are part of the contract, not advice: an
// experiment document outside them is rejected before any trainer runs.
struct RwkvScratchParameter final {
  std::string name;
  // The exact rwkv_pretrain long option this lowers to, without the leading
  // dashes. Parity tests assert every one of these exists in the trainer.
  std::string trainer_flag;
  RwkvScratchParameterType type{};
  std::optional<std::int64_t> minimum;
  std::optional<std::int64_t> maximum;
  std::vector<std::string> allowed;
  std::string default_value;
  std::string notes;

  bool operator==(const RwkvScratchParameter&) const = default;
};

// What the worker must report and persist for this topology. A topology that
// adds state without adding it here would resume silently wrong, so the
// contract names it explicitly.
struct RwkvScratchStateContract final {
  // Metric series the worker emits for this topology, beyond the baseline.
  std::vector<std::string> metrics;
  // Checkpoint keys this topology adds. Empty means it introduces no state
  // beyond the baseline model and optimizer.
  std::vector<std::string> checkpoint_keys;

  bool operator==(const RwkvScratchStateContract&) const = default;
};

struct RwkvScratchTopologyProfile final {
  RwkvScratchTopology topology{};
  std::string version;
  // The adapter contract identity this topology resolves to.
  std::string contract;
  // Component slots this topology requires on top of the baseline.
  std::map<std::string, TrainingComponentCategory> additional_slots;
  std::vector<RwkvScratchParameter> parameters;
  RwkvScratchStateContract state;
  // Honest, not aspirational. The scratch trainer writes a terminal
  // checkpoint and cannot resume mid-run, so no topology may claim better
  // than the baseline it is built on.
  ResumeGrade resume_grade{ResumeGrade::terminal_checkpoint};
  std::string summary;

  bool operator==(const RwkvScratchTopologyProfile&) const = default;
};

struct RwkvScratchProfileDocument final {
  std::string api_version{std::string(kRwkvScratchProfilesApiVersion)};
  std::vector<RwkvScratchTopologyProfile> profiles;
  std::string document_digest;

  bool operator==(const RwkvScratchProfileDocument&) const = default;
};

[[nodiscard]] std::string_view rwkv_scratch_topology_name(
    RwkvScratchTopology topology);
[[nodiscard]] std::optional<RwkvScratchTopology> rwkv_scratch_topology_from_name(
    std::string_view name);

// The closed registry. Every call returns the same sealed document.
[[nodiscard]] const RwkvScratchProfileDocument& rwkv_scratch_profiles();
[[nodiscard]] const RwkvScratchTopologyProfile& rwkv_scratch_profile(
    RwkvScratchTopology topology);

// Rejects an unknown topology, an undeclared switch, a value outside its
// declared bound or enumeration, and a wrong-typed value. This is the gate
// that keeps arbitrary legacy switches out of a declarative experiment.
void validate_rwkv_scratch_selection(
    RwkvScratchTopology topology,
    const std::map<std::string, nlohmann::json>& assignments);

[[nodiscard]] nlohmann::json rwkv_scratch_profiles_json(
    const RwkvScratchProfileDocument& document);
[[nodiscard]] RwkvScratchProfileDocument rwkv_scratch_profiles_from_json(
    const nlohmann::json& source);
[[nodiscard]] std::string rwkv_scratch_profiles_digest(
    const RwkvScratchProfileDocument& document);

// A lowered, canonical training block for a selected set of topologies.
// Topologies compose: the trainer's switches are independent, so a run may
// attach e.g. engram and loop together. Pairs the trainer cannot combine are
// declared below and refused here rather than discovered on a GPU.
struct RwkvScratchSelection final {
  RwkvScratchTopology topology{};
  std::map<std::string, nlohmann::json> assignments;

  bool operator==(const RwkvScratchSelection&) const = default;
};

// Declared incompatible pairs, refused by the lowering.
[[nodiscard]] std::vector<std::pair<RwkvScratchTopology, RwkvScratchTopology>>
rwkv_scratch_incompatible_topologies();

// Validates every selection, refuses duplicates and declared-incompatible
// pairs, and lowers to a canonical block. A parameter left at its declared
// default is omitted, so the block stays minimal and two runs that differ only
// in an explicit default are byte-identical.
[[nodiscard]] nlohmann::json rwkv_scratch_training_block(
    const std::vector<RwkvScratchSelection>& selections);

// Every trainer flag the registry claims, deduplicated and sorted. The parity
// test asserts this is a subset of rwkv_pretrain's actual argument surface.
[[nodiscard]] std::vector<std::string> rwkv_scratch_declared_trainer_flags();

}  // namespace trainvm
