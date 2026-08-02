#include "trainvm/rwkv_scratch_profiles.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const RwkvScratchProfileError&) {
    return;
  }
  throw std::runtime_error(message);
}

// Every topology the card names must be registered, and nothing may be
// registered that the card does not name. A closed set is the point.
void the_declared_topology_set_is_closed() {
  const auto& document = rwkv_scratch_profiles();
  const std::set<std::string> expected{
      "baseline",           "loop",
      "latent_lookahead",   "engram",
      "routing_free_moe",   "online_memory",
      "byte_aware_tokenizer", "seed_chain",
      "deep_embed",         "distributed",
  };
  std::set<std::string> actual;
  for (const auto& profile : document.profiles)
    actual.insert(std::string(rwkv_scratch_topology_name(profile.topology)));
  require(actual == expected,
          "the registered topology set drifted from the declared contract");
  require(document.api_version == kRwkvScratchProfilesApiVersion,
          "the profile document carries its api version");
  require(!document.document_digest.empty(),
          "the profile document is sealed with a digest");

  for (const auto& name : expected) {
    const auto topology = rwkv_scratch_topology_from_name(name);
    require(topology.has_value(), name + " resolves from its stable name");
    require(rwkv_scratch_topology_name(rwkv_scratch_profile(*topology).topology) ==
                name,
            name + " round-trips through the registry");
  }
  require(!rwkv_scratch_topology_from_name("loop_v2").has_value(),
          "an unregistered topology name does not resolve");
}

// The scratch trainer writes a terminal checkpoint and cannot resume mid-run.
// No topology may present itself as better than that.
void no_topology_overclaims_its_resume_grade() {
  for (const auto& profile : rwkv_scratch_profiles().profiles) {
    require(profile.resume_grade == ResumeGrade::terminal_checkpoint,
            std::string(rwkv_scratch_topology_name(profile.topology)) +
                " claims a resume grade its baseline cannot support");
    // A topology that adds durable state must say so, or a resumed run would
    // silently drop it.
    if (!profile.state.checkpoint_keys.empty())
      require(!profile.state.metrics.empty(),
              std::string(rwkv_scratch_topology_name(profile.topology)) +
                  " persists state but reports no metric for it");
  }
}

// The gate that keeps arbitrary legacy switches out of a declarative document.
void undeclared_and_out_of_bound_switches_are_refused() {
  const auto loop = RwkvScratchTopology::loop;
  validate_rwkv_scratch_selection(loop, {{"count", 4}, {"gate", "factored"}});

  require_throws(
      [&] {
        // A real rwkv_pretrain switch, but not one this topology declares.
        validate_rwkv_scratch_selection(loop, {{"engram_rows", 4096}});
      },
      "a switch from another topology must be refused");
  require_throws(
      [&] { validate_rwkv_scratch_selection(loop, {{"count", 0}}); },
      "an integer below its declared minimum must be refused");
  require_throws(
      [&] { validate_rwkv_scratch_selection(loop, {{"count", 17}}); },
      "an integer above its declared maximum must be refused");
  require_throws(
      [&] { validate_rwkv_scratch_selection(loop, {{"gate", "quantum"}}); },
      "a value outside the declared enumeration must be refused");
  require_throws(
      [&] { validate_rwkv_scratch_selection(loop, {{"count", "four"}}); },
      "a wrong-typed value must be refused");
  require_throws(
      [&] {
        validate_rwkv_scratch_selection(RwkvScratchTopology::baseline,
                                        {{"count", 2}});
      },
      "the baseline topology declares no switches at all");

  // Every declared parameter of every topology must accept a value at each end
  // of its declared bound, or the bound is decorative.
  for (const auto& profile : rwkv_scratch_profiles().profiles) {
    for (const auto& parameter : profile.parameters) {
      if (parameter.type == RwkvScratchParameterType::integer &&
          parameter.minimum && parameter.maximum) {
        validate_rwkv_scratch_selection(profile.topology,
                                        {{parameter.name, *parameter.minimum}});
        validate_rwkv_scratch_selection(profile.topology,
                                        {{parameter.name, *parameter.maximum}});
      }
      if (parameter.type == RwkvScratchParameterType::enumeration) {
        for (const auto& allowed : parameter.allowed)
          validate_rwkv_scratch_selection(profile.topology,
                                          {{parameter.name, allowed}});
      }
    }
  }
}

void the_profile_document_round_trips_and_binds_its_digest() {
  const auto& document = rwkv_scratch_profiles();
  const auto encoded = rwkv_scratch_profiles_json(document);
  const auto decoded = rwkv_scratch_profiles_from_json(encoded);
  require(decoded == document, "the profile document round-trip is not exact");

  auto forged = encoded;
  forged["profiles"][0]["summary"] = "tampered";
  require_throws([&] { (void)rwkv_scratch_profiles_from_json(forged); },
                 "a tampered profile document must fail its digest");

  auto unknown = encoded;
  unknown["trainvm_unknown"] = 1;
  require_throws([&] { (void)rwkv_scratch_profiles_from_json(unknown); },
                 "an unknown member must be refused");
}

// The contract names exact rwkv_pretrain long options. If the trainer renames
// or drops one, this catches it here rather than at launch time on a GPU.
void every_declared_flag_exists_in_the_trainer() {
  const std::string path =
      std::string(TRAINVM_SOURCE_ROOT) + "/src/rwkv_lab/rwkv_pretrain.py";
  std::ifstream input(path);
  require(input.good(), "the scratch trainer source is readable at " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string source = buffer.str();
  require(source.size() > 1024U, "the scratch trainer source is non-trivial");

  // Flags are declared either as a literal "--name" or built from a list of
  // bare names in a loop; accept both forms.
  const auto declared_in_trainer = [&source](const std::string& flag) {
    // Literal form: ap.add_argument("--loop-count", ...)
    if (source.find("\"--" + flag + "\"") != std::string::npos) return true;
    // Loop form: for f in ["nextlat-weight", ...]: add_argument(f"--{f}")
    return source.find("\"" + flag + "\"") != std::string::npos;
  };

  std::vector<std::string> missing;
  for (const auto& flag : rwkv_scratch_declared_trainer_flags()) {
    if (!declared_in_trainer(flag)) missing.push_back(flag);
  }
  if (!missing.empty()) {
    std::string detail;
    for (const auto& flag : missing) detail += " --" + flag;
    throw std::runtime_error(
        "the profile registry declares trainer flags that rwkv_pretrain does "
        "not accept:" + detail);
  }
  require(rwkv_scratch_declared_trainer_flags().size() >= 30U,
          "the registry declares a meaningful share of the trainer surface");
}

// Lowering: a validated selection becomes a canonical training block that a
// worker and the dashboard can both read without trainer knowledge.
void selections_lower_to_a_canonical_training_block() {
  const auto block = rwkv_scratch_training_block(
      {{.topology = RwkvScratchTopology::loop,
        .assignments = {{"count", 4}, {"gate", "factored"}}},
       {.topology = RwkvScratchTopology::engram,
        .assignments = {{"enabled", 1}, {"rows", 8192}}}});

  require(block.at("api_version") == kRwkvScratchProfilesApiVersion,
          "the training block carries its api version");
  require(block.at("topologies").size() == 2U,
          "the training block carries both selected topologies");
  // Declaration order in the enum, not caller order.
  require(block.at("topologies")[0].at("topology") == "loop" &&
              block.at("topologies")[1].at("topology") == "engram",
          "the training block orders topologies deterministically");
  require(block.at("topologies")[0].at("parameters").at("count").at(
              "trainer_flag") == "loop-count",
          "each parameter carries the exact trainer flag it lowers to");
  require(block.at("topologies")[0].at("parameters").at("count").at("value") ==
              4,
          "each parameter carries its selected value");

  // Merged, deduplicated, and sorted across the selection.
  const auto metrics = block.at("metrics");
  require(metrics.end() != std::find(metrics.begin(), metrics.end(), nlohmann::json("loop_gate_mean")) &&
              metrics.end() != std::find(metrics.begin(), metrics.end(), nlohmann::json("engram_hit_rate")),
          "the block merges the declared metrics of every selected topology");
  const auto keys = block.at("checkpoint_keys");
  require(keys.end() != std::find(keys.begin(), keys.end(), nlohmann::json("loop_gate")) &&
              keys.end() != std::find(keys.begin(), keys.end(), nlohmann::json("engram_table")),
          "the block merges the declared checkpoint keys");

  // Caller order must not change the bytes.
  const auto reordered = rwkv_scratch_training_block(
      {{.topology = RwkvScratchTopology::engram,
        .assignments = {{"enabled", 1}, {"rows", 8192}}},
       {.topology = RwkvScratchTopology::loop,
        .assignments = {{"count", 4}, {"gate", "factored"}}}});
  require(reordered == block,
          "the lowered block is independent of caller ordering");
}

// A value equal to the declared default carries no information. Omitting it
// keeps two otherwise-identical runs byte-identical.
void declared_defaults_are_omitted_from_the_block() {
  const auto explicit_default = rwkv_scratch_training_block(
      {{.topology = RwkvScratchTopology::loop, .assignments = {{"count", 1}}}});
  const auto omitted = rwkv_scratch_training_block(
      {{.topology = RwkvScratchTopology::loop, .assignments = {}}});
  require(explicit_default == omitted,
          "spelling out a declared default must not change the lowered block");
  require(explicit_default.at("topologies")[0].at("parameters").empty(),
          "a default-valued parameter is omitted entirely");

  const auto non_default = rwkv_scratch_training_block(
      {{.topology = RwkvScratchTopology::loop, .assignments = {{"count", 2}}}});
  require(non_default != omitted,
          "a non-default value must appear in the lowered block");
}

// The lowering is the last gate before a worker sees the selection.
void the_lowering_refuses_invalid_and_incompatible_selections() {
  require_throws(
      [] {
        (void)rwkv_scratch_training_block(
            {{.topology = RwkvScratchTopology::loop,
              .assignments = {{"engram_rows", 4096}}}});
      },
      "the lowering refuses a switch the topology does not declare");
  require_throws(
      [] {
        (void)rwkv_scratch_training_block(
            {{.topology = RwkvScratchTopology::loop,
              .assignments = {{"count", 99}}}});
      },
      "the lowering refuses a value outside its declared bound");
  require_throws(
      [] {
        (void)rwkv_scratch_training_block(
            {{.topology = RwkvScratchTopology::loop, .assignments = {}},
             {.topology = RwkvScratchTopology::loop, .assignments = {}}});
      },
      "the lowering refuses a topology selected twice");

  // Declared incompatibility, refused here rather than on a GPU.
  require(!rwkv_scratch_incompatible_topologies().empty(),
          "at least one incompatible pair is declared");
  for (const auto& [left, right] : rwkv_scratch_incompatible_topologies()) {
    require_throws(
        [&] {
          (void)rwkv_scratch_training_block(
              {{.topology = left, .assignments = {}},
               {.topology = right, .assignments = {}}});
        },
        "a declared-incompatible topology pair must be refused");
    // Either alone must still be accepted, or the pair rule is too broad.
    (void)rwkv_scratch_training_block({{.topology = left, .assignments = {}}});
    (void)rwkv_scratch_training_block({{.topology = right, .assignments = {}}});
  }
}

// Every topology must lower on its own, so none is unreachable in practice.
void every_topology_lowers_on_its_own() {
  for (const auto& profile : rwkv_scratch_profiles().profiles) {
    const auto block =
        rwkv_scratch_training_block({{.topology = profile.topology,
                                      .assignments = {}}});
    require(block.at("topologies").size() == 1U,
            std::string(rwkv_scratch_topology_name(profile.topology)) +
                " does not lower on its own");
    require(block.at("topologies")[0].at("contract") == profile.contract,
            "the lowered block names the topology's declared contract");
    for (const auto& key : profile.state.checkpoint_keys) {
      const auto keys = block.at("checkpoint_keys");
      require(keys.end() != std::find(keys.begin(), keys.end(), nlohmann::json(key)),
              std::string(rwkv_scratch_topology_name(profile.topology)) +
                  " drops a declared checkpoint key when lowered");
    }
  }
}

}  // namespace

int main() {
  try {
    the_declared_topology_set_is_closed();
    no_topology_overclaims_its_resume_grade();
    undeclared_and_out_of_bound_switches_are_refused();
    the_profile_document_round_trips_and_binds_its_digest();
    every_declared_flag_exists_in_the_trainer();
    selections_lower_to_a_canonical_training_block();
    declared_defaults_are_omitted_from_the_block();
    the_lowering_refuses_invalid_and_incompatible_selections();
    every_topology_lowers_on_its_own();
    std::cout << "rwkv scratch profile tests passed ("
              << rwkv_scratch_profiles().profiles.size() << " topologies, "
              << rwkv_scratch_declared_trainer_flags().size()
              << " declared trainer flags)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rwkv scratch profile test failure: " << error.what() << '\n';
    return 1;
  }
}
