#include "trainvm/rwkv_scratch_profiles.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <memory>
#include <ranges>
#include <set>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

RwkvScratchParameter integer_parameter(std::string name, std::string flag,
                                       std::int64_t minimum,
                                       std::int64_t maximum,
                                       std::string default_value,
                                       std::string notes) {
  return {.name = std::move(name),
          .trainer_flag = std::move(flag),
          .type = RwkvScratchParameterType::integer,
          .minimum = minimum,
          .maximum = maximum,
          .allowed = {},
          .default_value = std::move(default_value),
          .notes = std::move(notes)};
}

RwkvScratchParameter number_parameter(std::string name, std::string flag,
                                      std::string default_value,
                                      std::string notes) {
  return {.name = std::move(name),
          .trainer_flag = std::move(flag),
          .type = RwkvScratchParameterType::number,
          .minimum = std::nullopt,
          .maximum = std::nullopt,
          .allowed = {},
          .default_value = std::move(default_value),
          .notes = std::move(notes)};
}

RwkvScratchParameter enum_parameter(std::string name, std::string flag,
                                    std::vector<std::string> allowed,
                                    std::string default_value,
                                    std::string notes) {
  return {.name = std::move(name),
          .trainer_flag = std::move(flag),
          .type = RwkvScratchParameterType::enumeration,
          .minimum = std::nullopt,
          .maximum = std::nullopt,
          .allowed = std::move(allowed),
          .default_value = std::move(default_value),
          .notes = std::move(notes)};
}

std::string hex_digest(std::string_view material) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), material.data(), material.size()) != 1)
    throw RwkvScratchProfileError("scratch profile digest failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> value{};
  unsigned int size = 0U;
  if (EVP_DigestFinal_ex(context.get(), value.data(), &size) != 1 || size != 32U)
    throw RwkvScratchProfileError("scratch profile digest final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  for (unsigned int index = 0U; index < size; ++index) {
    result.push_back(digits[value[index] >> 4U]);
    result.push_back(digits[value[index] & 0x0fU]);
  }
  return result;
}

std::vector<RwkvScratchTopologyProfile> build_profiles() {
  std::vector<RwkvScratchTopologyProfile> profiles;

  profiles.push_back(
      {.topology = RwkvScratchTopology::baseline,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.v1.Train",
       .additional_slots = {},
       .parameters = {},
       .state = {.metrics = {"loss", "tokens_per_second"},
                 .checkpoint_keys = {}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary = "Dense from-scratch RWKV with no research topology attached."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::loop,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.loop.v1.Train",
       .additional_slots = {{"loop_gate_gradient_clipping",
                             TrainingComponentCategory::gradient_clipping}},
       .parameters =
           {integer_parameter("count", "loop-count", 1, 16, "1",
                              "Weight-shared iterations per looped block."),
            enum_parameter("gate", "loop-gate",
                           {"scalar", "head", "channel", "factored"}, "scalar",
                           "Per-iteration gate granularity."),
            number_parameter("gate_cap", "loop-gate-cap", "0.0",
                             "Upper bound on the gate; 0 disables the cap."),
            integer_parameter("deq_window", "loop-deq-window", 1, 64, "1",
                              "Deep-equilibrium averaging window."),
            integer_parameter("hyper", "loop-hyper", 0, 1, "0",
                              "Hyperconnections across loop iterations.")},
       .state = {.metrics = {"loop_gate_mean", "loop_iterations"},
                 .checkpoint_keys = {"loop_gate"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary =
           "Weight-shared iterated blocks with a trained per-iteration gate."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::latent_lookahead,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.latent.v1.Train",
       .additional_slots = {{"auxiliary_objective",
                             TrainingComponentCategory::objective}},
       .parameters =
           {number_parameter("nextlat_weight", "nextlat-weight", "0.0",
                             "Next-latent prediction auxiliary weight."),
            number_parameter("top_weight", "top-weight", "0.0",
                             "Token-order-prediction auxiliary weight."),
            number_parameter("lmtp_weight", "lmtp-weight", "0.0",
                             "Multi-token-prediction auxiliary weight."),
            number_parameter("bst_weight", "bst-weight", "0.0",
                             "Belief-state auxiliary weight."),
            number_parameter("jtp_weight", "jtp-weight", "0.0",
                             "Joint-token-prediction auxiliary weight."),
            number_parameter("lmtp_cooldown_fraction", "lmtp-cooldown-fraction",
                             "0.0",
                             "Fraction of training over which the multi-token "
                             "auxiliary decays to zero.")},
       .state = {.metrics = {"aux_loss", "lm_loss"},
                 .checkpoint_keys = {"lookahead_heads"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary =
           "Auxiliary prediction heads on the final hidden state; the language "
           "-model head is unchanged."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::engram,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.engram.v1.Train",
       .additional_slots = {},
       .parameters =
           {integer_parameter("enabled", "engram", 0, 1, "0",
                              "Attach the Engram lexical memory bank."),
            enum_parameter("sites", "engram-sites", {"auto"}, "auto",
                           "Layer placement; auto selects shallow and mid "
                           "depths. Explicit indices are a separate version."),
            integer_parameter("row_width", "engram-drow", 1, 4096, "64",
                              "Learned-table row width."),
            integer_parameter("rows", "engram-rows", 1, 1'048'576, "4096",
                              "Hashed table rows, capped at the vocabulary."),
            integer_parameter("warmup_steps", "engram-warmup", 0, 1'000'000,
                              "1000", "Steps to ramp injection from 0 to 1."),
            integer_parameter("boundary_id", "engram-boundary-id", -1, 1'048'575,
                              "-1",
                              "End-of-document token segmenting recall; -1 "
                              "disables segmentation.")},
       .state = {.metrics = {"engram_hit_rate", "engram_injection_scale"},
                 .checkpoint_keys = {"engram_table", "engram_gate"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary =
           "Token-suffix-automaton recall into a learned table with gated "
           "injection and a copy head."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::routing_free_moe,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.moe.v1.Train",
       .additional_slots = {{"expert_router",
                             TrainingComponentCategory::parameter_router}},
       .parameters =
           {integer_parameter("enabled", "routing-free-moe", 0, 1, "0",
                              "Attach routing-free mixture-of-experts."),
            integer_parameter("experts", "routing-free-experts", 1, 1024, "8",
                              "Expert count."),
            integer_parameter("rank", "routing-free-rank", 1, 4096, "8",
                              "Per-expert low-rank width."),
            number_parameter("threshold", "routing-free-threshold", "0.0",
                             "Activation threshold below which an expert is "
                             "skipped."),
            number_parameter("balance", "routing-free-balance", "0.0",
                             "Load-balance strength."),
            number_parameter("aux_weight", "routing-free-aux-weight", "0.0",
                             "Auxiliary balancing loss weight.")},
       .state = {.metrics = {"expert_utilization", "moe_aux_loss"},
                 .checkpoint_keys = {"expert_weights"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary = "Routing-free mixture of experts with a balancing auxiliary."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::online_memory,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.online-memory.v1.Train",
       .additional_slots = {},
       .parameters =
           {integer_parameter("enabled", "online-memory", 0, 1, "0",
                              "Attach the online-memory module."),
            integer_parameter("dim", "online-memory-dim", 1, 16384, "0",
                              "Memory width; 0 follows the model width."),
            enum_parameter("kernel", "online-memory-kernel",
                           {"linear", "elu", "relu"}, "linear",
                           "Feature map for the memory read."),
            number_parameter("learning_rate", "online-memory-lr", "0.0",
                             "Inner-loop update rate."),
            enum_parameter("mode", "online-memory-mode", {"read", "write",
                                                          "read_write"},
                           "read_write", "Memory access mode."),
            number_parameter("retention", "online-memory-retention", "0.0",
                             "Retention factor per step."),
            integer_parameter("window", "online-memory-window", 0, 1'048'576,
                              "0", "Bounded lookback; 0 is unbounded.")},
       .state = {.metrics = {"memory_read_norm", "memory_write_norm"},
                 .checkpoint_keys = {"online_memory"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary = "Test-time inner-loop memory read and write."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::byte_aware_tokenizer,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.byte-aware.v1.Train",
       .additional_slots = {},
       .parameters =
           {integer_parameter("enabled", "byte-aware-vocab", 0, 1, "0",
                              "Attach byte-aware vocabulary embeddings."),
            integer_parameter("dim", "byte-aware-dim", 1, 16384, "0",
                              "Byte-embedding width; 0 follows the model."),
            integer_parameter("maximum_bytes", "byte-aware-max-bytes", 1, 64,
                              "16", "Bytes retained per token.")},
       .state = {.metrics = {"byte_coverage"},
                 .checkpoint_keys = {"byte_embeddings"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary =
           "Byte-aware vocabulary embeddings alongside the token embedding."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::seed_chain,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.seed-chain.v1.Train",
       .additional_slots = {},
       .parameters = {integer_parameter(
           "enabled", "seed-chain", 0, 1, "0",
           "Seed layer L's scan with layer L-1's final state.")},
       .state = {.metrics = {"seed_state_norm"}, .checkpoint_keys = {}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary =
           "Future-Seed cross-layer state chaining. Incompatible with loops."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::deep_embed,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.deep-embed.v1.Train",
       .additional_slots = {},
       .parameters =
           {integer_parameter("enabled", "deepembed", 0, 1, "0",
                              "Per-layer per-token feed-forward output gate."),
            integer_parameter("dim", "de-dim", 0, 16384, "0",
                              "Low-rank width; 0 uses the full model width."),
            enum_parameter("mode", "de-mode", {"out", "hidden"}, "out",
                           "Gate the feed-forward output or its hidden state."),
            integer_parameter("shift", "de-shift", 0, 1, "0",
                              "Apply the DeepEmbed token shift."),
            integer_parameter("embedding_residual", "de-emb-res", 0, 1, "0",
                              "Add the DeepEmbed embedding residual.")},
       .state = {.metrics = {"deepembed_gate_mean"},
                 .checkpoint_keys = {"deepembed"}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary = "RWKV-8 DeepEmbed per-token feed-forward gate."});

  profiles.push_back(
      {.topology = RwkvScratchTopology::distributed,
       .version = "1.0.0",
       .contract = "rwkv_lab.rwkv_scratch.distributed.v1.Train",
       .additional_slots = {},
       .parameters =
           {integer_parameter("enabled", "distributed", 0, 1, "0",
                              "Run the fully-sharded data-parallel topology."),
            integer_parameter("prefetch_depth", "fsdp-prefetch-depth", 0, 16,
                              "0", "Forward prefetch depth."),
            integer_parameter("sparse_embeddings", "fsdp-sparse-embeddings", 0,
                              1, "0", "Keep embeddings sparse when sharding.")},
       .state = {.metrics = {"shard_wait_seconds"}, .checkpoint_keys = {}},
       .resume_grade = ResumeGrade::terminal_checkpoint,
       .summary =
           "Fully-sharded data parallelism. Rank topology is host authority, "
           "not an experiment switch."});

  return profiles;
}

const RwkvScratchProfileDocument& sealed_document() {
  static const RwkvScratchProfileDocument value = [] {
    RwkvScratchProfileDocument document;
    document.profiles = build_profiles();
    // Closed-set invariants, enforced at construction so a malformed profile
    // cannot reach a caller.
    std::set<std::string> contracts;
    std::set<int> topologies;
    for (const RwkvScratchTopologyProfile& profile : document.profiles) {
      if (!contracts.insert(profile.contract).second ||
          !topologies.insert(static_cast<int>(profile.topology)).second)
        throw RwkvScratchProfileError(
            "scratch topology profiles must be unique per topology and "
            "contract");
      if (profile.version.empty() || profile.summary.empty())
        throw RwkvScratchProfileError(
            "scratch topology profile is missing its version or summary");
      if (profile.resume_grade != ResumeGrade::terminal_checkpoint)
        throw RwkvScratchProfileError(
            "the scratch trainer writes a terminal checkpoint; no topology "
            "may claim a better resume grade than its baseline");
      std::set<std::string> names;
      std::set<std::string> flags;
      for (const RwkvScratchParameter& parameter : profile.parameters) {
        if (parameter.name.empty() || parameter.trainer_flag.empty() ||
            parameter.notes.empty())
          throw RwkvScratchProfileError(
              "scratch topology parameter is missing its name, flag, or note");
        if (!names.insert(parameter.name).second ||
            !flags.insert(parameter.trainer_flag).second)
          throw RwkvScratchProfileError(
              "scratch topology parameter names and flags must be unique");
        if (parameter.type == RwkvScratchParameterType::enumeration &&
            parameter.allowed.empty())
          throw RwkvScratchProfileError(
              "an enumeration parameter must declare its allowed values");
        if (parameter.type != RwkvScratchParameterType::enumeration &&
            !parameter.allowed.empty())
          throw RwkvScratchProfileError(
              "only an enumeration parameter may declare allowed values");
        if (parameter.minimum && parameter.maximum &&
            *parameter.minimum > *parameter.maximum)
          throw RwkvScratchProfileError(
              "scratch topology parameter bound is inverted");
      }
    }
    document.document_digest = rwkv_scratch_profiles_digest(document);
    return document;
  }();
  return value;
}

}  // namespace

std::string_view rwkv_scratch_topology_name(RwkvScratchTopology topology) {
  switch (topology) {
    case RwkvScratchTopology::baseline:
      return "baseline";
    case RwkvScratchTopology::loop:
      return "loop";
    case RwkvScratchTopology::latent_lookahead:
      return "latent_lookahead";
    case RwkvScratchTopology::engram:
      return "engram";
    case RwkvScratchTopology::routing_free_moe:
      return "routing_free_moe";
    case RwkvScratchTopology::online_memory:
      return "online_memory";
    case RwkvScratchTopology::byte_aware_tokenizer:
      return "byte_aware_tokenizer";
    case RwkvScratchTopology::seed_chain:
      return "seed_chain";
    case RwkvScratchTopology::deep_embed:
      return "deep_embed";
    case RwkvScratchTopology::distributed:
      return "distributed";
  }
  throw RwkvScratchProfileError("unknown scratch topology");
}

std::optional<RwkvScratchTopology> rwkv_scratch_topology_from_name(
    std::string_view name) {
  for (const RwkvScratchTopologyProfile& profile : sealed_document().profiles) {
    if (rwkv_scratch_topology_name(profile.topology) == name)
      return profile.topology;
  }
  return std::nullopt;
}

const RwkvScratchProfileDocument& rwkv_scratch_profiles() {
  return sealed_document();
}

const RwkvScratchTopologyProfile& rwkv_scratch_profile(
    RwkvScratchTopology topology) {
  for (const RwkvScratchTopologyProfile& profile : sealed_document().profiles) {
    if (profile.topology == topology) return profile;
  }
  throw RwkvScratchProfileError("scratch topology is not registered");
}

void validate_rwkv_scratch_selection(
    RwkvScratchTopology topology,
    const std::map<std::string, nlohmann::json>& assignments) {
  const RwkvScratchTopologyProfile& profile = rwkv_scratch_profile(topology);
  for (const auto& [name, value] : assignments) {
    const auto found = std::ranges::find_if(
        profile.parameters, [&name](const RwkvScratchParameter& parameter) {
          return parameter.name == name;
        });
    if (found == profile.parameters.end()) {
      throw RwkvScratchProfileError(
          std::string("topology ") +
          std::string(rwkv_scratch_topology_name(topology)) +
          " does not declare a parameter named " + name);
    }
    switch (found->type) {
      case RwkvScratchParameterType::integer: {
        if (!value.is_number_integer())
          throw RwkvScratchProfileError(name + " expects an integer");
        const std::int64_t number = value.get<std::int64_t>();
        if ((found->minimum && number < *found->minimum) ||
            (found->maximum && number > *found->maximum))
          throw RwkvScratchProfileError(name + " is outside its declared bound");
        break;
      }
      case RwkvScratchParameterType::number:
        if (!value.is_number())
          throw RwkvScratchProfileError(name + " expects a number");
        break;
      case RwkvScratchParameterType::boolean:
        if (!value.is_boolean())
          throw RwkvScratchProfileError(name + " expects a boolean");
        break;
      case RwkvScratchParameterType::enumeration: {
        if (!value.is_string())
          throw RwkvScratchProfileError(name + " expects a string");
        const std::string chosen = value.get<std::string>();
        if (std::ranges::find(found->allowed, chosen) == found->allowed.end())
          throw RwkvScratchProfileError(name +
                                        " is not a declared value for this "
                                        "topology");
        break;
      }
    }
  }
}

nlohmann::json rwkv_scratch_profiles_json(
    const RwkvScratchProfileDocument& document) {
  return encode_json(document);
}

RwkvScratchProfileDocument rwkv_scratch_profiles_from_json(
    const nlohmann::json& source) {
  RwkvScratchProfileDocument document;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, document, "", diagnostics) ||
      !diagnostics.empty() ||
      document.api_version != kRwkvScratchProfilesApiVersion)
    throw RwkvScratchProfileError("scratch profile document is not canonical");
  if (document.document_digest != rwkv_scratch_profiles_digest(document))
    throw RwkvScratchProfileError(
        "scratch profile document digest does not bind its content");
  return document;
}

std::string rwkv_scratch_profiles_digest(
    const RwkvScratchProfileDocument& document) {
  RwkvScratchProfileDocument material = document;
  material.document_digest.clear();
  return hex_digest(rwkv_scratch_profiles_json(material).dump());
}

std::vector<std::string> rwkv_scratch_declared_trainer_flags() {
  std::set<std::string> flags;
  for (const RwkvScratchTopologyProfile& profile : sealed_document().profiles) {
    for (const RwkvScratchParameter& parameter : profile.parameters)
      flags.insert(parameter.trainer_flag);
  }
  return {flags.begin(), flags.end()};
}

}  // namespace trainvm
