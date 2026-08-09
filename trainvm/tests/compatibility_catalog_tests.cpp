#include "trainvm/adapter_registry.hpp"
#include "trainvm/compatibility_catalog.hpp"
#include "trainvm/reflection_json.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/json.hpp"

namespace {

static_assert(!std::same_as<trainvm::CompatibilityResumeEvidence,
                            trainvm::ResumeGrade>);

int failures = 0;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    constexpr std::string_view prefix =
        "/tmp/trainvm-compatibility-catalog-XXXXXX";
    std::ranges::copy(prefix, pattern.begin());
    const auto created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("could not create temporary test directory");
    }
    path_ = created;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
bool throws_invalid_argument(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

nlohmann::json load_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open " + path.string());
  nlohmann::json value;
  input >> value;
  return value;
}

void write_json(const std::filesystem::path& path,
                const nlohmann::json& value) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("could not write " + path.string());
  output << value.dump(2) << '\n';
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

// Extracts every mutating route literal registered by the legacy Go dashboard
// router. This is deliberately a byte scan of the checked-in registration
// table: the catalog must not be able to drift away from the routes the
// dashboard actually serves, and no Go toolchain is required to notice.
std::set<std::string> registered_mutation_routes(const std::string& router) {
  static constexpr std::string_view kRegistration = "HandleFunc(\"POST ";
  std::set<std::string> routes;
  for (std::size_t cursor = router.find(kRegistration);
       cursor != std::string::npos;
       cursor = router.find(kRegistration, cursor + 1U)) {
    const std::size_t begin = cursor + kRegistration.size();
    const std::size_t end = router.find('"', begin);
    if (end == std::string::npos) {
      throw std::runtime_error("unterminated dashboard route registration");
    }
    routes.insert("POST " + router.substr(begin, end - begin));
  }
  return routes;
}

void restore_source(const std::filesystem::path& source,
                    const std::filesystem::path& destination) {
  std::error_code error;
  std::filesystem::remove_all(destination, error);
  error.clear();
  std::filesystem::create_directories(destination.parent_path());
  std::filesystem::copy_file(
      source, destination, std::filesystem::copy_options::overwrite_existing);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path root = fs::canonical(TRAINVM_SOURCE_ROOT);
  const fs::path fixture =
      root / "docs/experiment-vm/compatibility-workflows.v1.json";

  const auto catalog = trainvm::CompatibilityCatalog::load_file(fixture, root);
  check(catalog.authority() ==
            trainvm::CompatibilityAuthority::compatibility_evidence_only,
        "catalog is explicitly compatibility evidence only");
  check(catalog.entries().size() == 156U,
        "checked-in catalog matches the compiled reviewed v1 inventory");
  check(catalog.catalog_digest() ==
            trainvm::CompatibilityCatalog::reviewed_catalog_digest(),
        "catalog has the exact compiled reviewed canonical digest");
  check(catalog.source_tree_digest().starts_with("sha256:") &&
            catalog.source_tree_digest().size() == 71U,
        "catalog binds the referenced source tree");
  check(catalog.repository_root_identity_display().contains(root.string()) &&
            catalog.repository_root_identity_display().contains("ino="),
        "validation reports the opened repository root identity");

  std::set<trainvm::WorkflowFamily> families;
  std::set<trainvm::ObservedInvocationKind> invocation_kinds;
  std::set<std::string> identifiers;
  std::set<std::string> reviewed_sources;
  std::set<std::string> required_additions = {
      "conversion.drive-isolation",
      "conversion.gdn-sweep-supervisor",
      "conversion.rel-sweep-supervisor",
      "conversion.gate-ab-campaign",
      "conversion.assemble-looped",
      "conversion.memory-target-cache",
      "rlvr.recursive-improve",
      "posttraining.adapter-recursive",
      "acquisition.kimi-teacher",
      "export.legacy-mutable-bundle",
      "rwkv.legacy-sweep-campaigns",
      "rwkv.config-campaign",
      "transformer.engram-staged-supervisor",
      "transformer.qwen-ao3-audit",
      "transformer.qwen-ao3-plan",
      "data.ao3-tokenize",
      "data.ao3-pack",
      "data.ao3-rewrite-eos",
      "data.engram-frequency",
      "vision.representation-ab",
      "vision.teacher-student-supervisor",
      "vision.continuation-watchdog",
      "vision.radio1d-launch-profiles",
      "vision.v4h-launch-profiles",
      "vision.moonvit-continuation-launch-profiles",
      "vision.native-head-launch-profile",
      "vision.raw-pixel-student-launch-profile",
      "vision.teacher-compressor-launch-profile",
      "mageflow.full-backbone-plan",
      "mageflow.expert-encoder-cache",
      "mageflow.terminal-cache-span-prepare",
      "mageflow.terminal-expert-migration",
      "mageflow.tread-loop-conversion",
      "mageflow.adaptation-benchmark-spec",
      "mageflow.adaptation-domain-prepare",
      "mageflow.adaptation-domain-audit",
      "acquisition.civitai-balanced",
      "data.gelbooru-trainer-snapshot",
      "data.reddit-trainer-snapshot",
      "scoring.i1-deepghs-classification",
      "data.midjourney-v6-caption-routing",
      "data.midjourney-v6-expert-stage",
      "data.midjourney-v6-continuation",
      "oracle.engram-lmb",
      "oracle.mla-training-components",
      "oracle.rosa-sam",
      "oracle.rosa-soft-layer",
      "evaluation.rwkv-generation",
      "evaluation.loop-probe",
      "export.megakernel-aot",
      "export.production-kernels-aot",
      "oracle.lossless-gdn-map",
      "qualification.posttraining-kernels",
      "qualification.production-kernels",
      "qualification.lossless-gdn-map",
      "external.ltx23-plan",
      "external.ltx23-prepare",
      "external.ltx23-run",
      "control.manual-training-launch",
      "control.gpu-launch-queue",
      "control.sample-launch",
  };
  std::set<std::string> sealed_compatible = {
      "vision.frozen-adapter-train",
  };
  bool saw_restart_only_review = false;
  bool saw_mutable_legacy_export = false;
  bool saw_mutable_frozen_export = false;
  bool saw_lossless_qualification = false;
  bool saw_split_production_qualification = false;
  bool saw_ltx_training_only = false;
  bool saw_lossless_library_oracle = false;
  bool saw_production_aot_export = false;
  bool saw_http_control_handler = false;
  bool saw_retired_legacy = false;
  const std::set<std::string> smoke_only_library_sources = {
      "src/rwkv_lab/engram_lmb.py",
      "src/rwkv_lab/layer_swap.py",
      "src/rwkv_lab/mla_module.py",
      "src/rwkv_lab/rosa_sam.py",
      "src/rwkv_lab/rosa_soft_layer.py",
      "src/rwkv_lab/svd_init.py",
  };
  auto unseen_smoke_only_library_sources = smoke_only_library_sources;
  bool smoke_only_source_claimed_invocable = false;
  for (const auto& entry : catalog.entries()) {
    families.insert(entry.family);
    invocation_kinds.insert(entry.observed_invocation);
    identifiers.insert(entry.stable_id);
    reviewed_sources.insert(entry.source_paths.begin(), entry.source_paths.end());
    required_additions.erase(entry.stable_id);
    if (sealed_compatible.contains(entry.stable_id) && entry.stateful &&
        entry.resume_evidence ==
            trainvm::CompatibilityResumeEvidence::compatible) {
      sealed_compatible.erase(entry.stable_id);
    }
    saw_restart_only_review = saw_restart_only_review ||
        (entry.stable_id == "review.dedupe-cutoff" && entry.stateful &&
         entry.resume_evidence ==
             trainvm::CompatibilityResumeEvidence::restart_only);
    saw_mutable_legacy_export = saw_mutable_legacy_export ||
        (entry.stable_id == "export.legacy-mutable-bundle" &&
         entry.notes.contains("recursively") &&
         entry.notes.contains("shallow"));
    saw_mutable_frozen_export = saw_mutable_frozen_export ||
        (entry.stable_id == "export.frozen-vision-compressor" &&
         entry.notes.contains("overwriteable") &&
         entry.notes.contains("no self-bound content hash"));
    saw_lossless_qualification = saw_lossless_qualification ||
        (entry.stable_id == "qualification.lossless-gdn-map" &&
         entry.operation_role ==
             trainvm::CompatibilityOperationRole::qualification &&
         entry.notes.contains("publishes no converted model"));
    saw_split_production_qualification =
        saw_split_production_qualification ||
        (entry.stable_id == "qualification.production-kernels" &&
         entry.operation_role ==
             trainvm::CompatibilityOperationRole::qualification &&
         entry.source_paths.size() == 1U &&
         entry.source_paths.front() ==
             "src/rwkv_lab/production_kernels.py");
    saw_ltx_training_only = saw_ltx_training_only ||
        (entry.stable_id == "external.ltx23-lora" &&
         entry.operation_role ==
             trainvm::CompatibilityOperationRole::training &&
         entry.legacy_invocation_display &&
         entry.legacy_invocation_display->contains(" train "));
    saw_lossless_library_oracle = saw_lossless_library_oracle ||
        (entry.stable_id == "oracle.lossless-gdn-map" &&
         entry.observed_invocation ==
             trainvm::ObservedInvocationKind::library_only &&
         entry.operation_role ==
             trainvm::CompatibilityOperationRole::library_oracle);
    saw_production_aot_export = saw_production_aot_export ||
        (entry.stable_id == "export.production-kernels-aot" &&
         entry.operation_role ==
             trainvm::CompatibilityOperationRole::export_artifact &&
         entry.source_paths.size() == 1U &&
         entry.source_paths.front() ==
             "src/rwkv_lab/production_kernels.py");
    saw_http_control_handler = saw_http_control_handler ||
        (entry.family == trainvm::WorkflowFamily::control_plane &&
         entry.observed_invocation ==
             trainvm::ObservedInvocationKind::http_control_handler);
    saw_retired_legacy = saw_retired_legacy ||
        (entry.family == trainvm::WorkflowFamily::control_plane &&
         entry.observed_invocation ==
             trainvm::ObservedInvocationKind::retired_legacy);
    for (const auto& source : entry.source_paths) {
      if (smoke_only_library_sources.contains(source)) {
        smoke_only_source_claimed_invocable =
            smoke_only_source_claimed_invocable ||
            entry.observed_invocation !=
                trainvm::ObservedInvocationKind::library_only;
        if (entry.observed_invocation ==
            trainvm::ObservedInvocationKind::library_only) {
          unseen_smoke_only_library_sources.erase(source);
        }
      }
    }
  }
  check(families.size() == 11U && invocation_kinds.size() == 7U &&
            identifiers.size() == catalog.entries().size(),
        "catalog contains every family and observed invocation kind");
  check(required_additions.empty(),
        "catalog retains the expanded audited workflow inventory");
  check(reviewed_sources.size() == 155U,
        "catalog binds the complete reviewed source inventory");
  check(sealed_compatible.empty() && saw_restart_only_review,
        "sealed compatible operations and restart-only review are classified narrowly");
  check(saw_mutable_legacy_export && saw_mutable_frozen_export,
        "export notes disclose overwrite and verification limitations");
  check(saw_lossless_qualification && saw_split_production_qualification &&
            saw_ltx_training_only && saw_lossless_library_oracle &&
            saw_production_aot_export,
        "effectfully distinct qualification, export, and external phases stay split");
  check(saw_http_control_handler,
        "dashboard HTTP controls are not mislabeled as host scripts");
  check(saw_retired_legacy,
        "retired dashboard mutations remain explicit migration evidence");

  // Scope gate: every legacy dashboard mutation route must be classified, and
  // every classified route must still be served. TrainVM's own namespace is
  // excluded because it is declarative authority, not legacy evidence.
  static constexpr std::string_view kRouterSource =
      "dashboard/internal/server/server.go";
  static constexpr std::string_view kDeclarativePrefix = "POST /api/trainvm/";
  const auto registered = registered_mutation_routes(
      read_text(root / kRouterSource));
  std::set<std::string> legacy_routes;
  std::set<std::string> declarative_routes;
  for (const auto& route : registered) {
    (route.starts_with(kDeclarativePrefix) ? declarative_routes : legacy_routes)
        .insert(route);
  }
  check(!legacy_routes.empty() && !declarative_routes.empty(),
        "router source exposes both legacy and declarative mutation routes");
  std::set<std::string> classified_routes;
  std::set<std::string> retired_legacy_routes;
  bool router_bound_by_every_control_record = true;
  bool duplicate_route_classification = false;
  for (const auto& entry : catalog.entries()) {
    if (entry.observed_invocation ==
        trainvm::ObservedInvocationKind::retired_legacy) {
      check(entry.legacy_invocation_display.has_value(),
            "every retired legacy record displays its historical route");
      if (entry.legacy_invocation_display) {
        retired_legacy_routes.insert(*entry.legacy_invocation_display);
      }
      continue;
    }
    if (entry.observed_invocation !=
        trainvm::ObservedInvocationKind::http_control_handler) {
      continue;
    }
    router_bound_by_every_control_record =
        router_bound_by_every_control_record &&
        std::ranges::find(entry.source_paths, kRouterSource) !=
            entry.source_paths.end();
    check(entry.legacy_invocation_display.has_value(),
          "every HTTP control record displays its observed route");
    if (!entry.legacy_invocation_display) continue;
    duplicate_route_classification = duplicate_route_classification ||
        !classified_routes.insert(*entry.legacy_invocation_display).second;
  }
  check(router_bound_by_every_control_record,
        "every HTTP control record binds the router registration bytes");
  check(!duplicate_route_classification,
        "no dashboard mutation route is classified twice");
  std::set<std::string> unclassified_routes;
  std::ranges::set_difference(
      legacy_routes, classified_routes,
      std::inserter(unclassified_routes, unclassified_routes.end()));
  for (const auto& route : unclassified_routes) {
    std::cerr << "unclassified legacy dashboard mutation route: " << route
              << '\n';
  }
  check(unclassified_routes.empty(),
        "every legacy dashboard mutation route is classified in the catalog");
  std::set<std::string> stale_active_routes;
  std::ranges::set_difference(
      classified_routes, legacy_routes,
      std::inserter(stale_active_routes, stale_active_routes.end()));
  for (const auto& route : stale_active_routes) {
    std::cerr << "catalog actively classifies an unserved route: " << route << '\n';
  }
  check(stale_active_routes.empty(),
        "active control records name only routes the dashboard still serves");
  std::set<std::string> accidentally_resurrected_routes;
  std::ranges::set_intersection(
      retired_legacy_routes, legacy_routes,
      std::inserter(accidentally_resurrected_routes,
                    accidentally_resurrected_routes.end()));
  check(accidentally_resurrected_routes.empty(),
        "retired legacy routes cannot silently regain dashboard authority");
  check(retired_legacy_routes.contains("POST /api/runs/{name}/control") &&
            retired_legacy_routes.contains("POST /api/autostop") &&
            retired_legacy_routes.contains("POST /api/queue/auto"),
        "retired live-tuning and background arming paths remain explicit evidence");

  // The route scan must never degrade into an empty set, which would make the
  // classification gate silently vacuous.
  const auto sample_routes = registered_mutation_routes(
      "\ts.mux.HandleFunc(\"GET /api/runs\", s.handleRuns)\n"
      "\ts.mux.HandleFunc(\"POST /api/runs/{name}/stop\", s.handleStop)\n"
      "\ts.mux.HandleFunc(\"POST /api/trainvm/compile\", s.handleCompile)\n");
  check(sample_routes ==
            std::set<std::string>{"POST /api/runs/{name}/stop",
                                  "POST /api/trainvm/compile"},
        "route extraction reads exactly the registered mutation literals");
  check(registered_mutation_routes("no registrations here").empty(),
        "route extraction reports nothing for a router without registrations");
  check(unseen_smoke_only_library_sources.empty() &&
            !smoke_only_source_claimed_invocable,
        "essential smoke-only modules remain explicit nonlaunchable libraries");
  check(!trainvm::enum_from_string<trainvm::CompatibilityResumeEvidence>(
             "exact") &&
            !trainvm::enum_from_string<trainvm::ObservedInvocationKind>(
                "module") &&
            trainvm::enum_from_string<trainvm::ObservedInvocationKind>(
                "python_module") ==
                trainvm::ObservedInvocationKind::python_module,
        "catalog vocabulary cannot decode AdapterRegistry exact or legacy module aliases");

  TemporaryDirectory temporary;
  const auto original = load_json(fixture);
  std::size_t case_number = 0;
  const auto rejects = [&](const nlohmann::json& value) {
    const fs::path path = temporary.path() /
        ("invalid-" + std::to_string(case_number++) + ".json");
    write_json(path, value);
    return throws_invalid_argument([&] {
      (void)trainvm::CompatibilityCatalog::load_file(path, root);
    });
  };

  auto duplicate_id = original;
  duplicate_id["entries"].push_back(duplicate_id["entries"].front());

  auto family_preserving_prune = original;
  family_preserving_prune["entries"].erase(
      family_preserving_prune["entries"].begin());

  auto unreviewed_addition = original;
  auto addition = unreviewed_addition["entries"].front();
  addition["stable_id"] = "rwkv.unreviewed-addition";
  unreviewed_addition["entries"].push_back(std::move(addition));

  auto traversal = original;
  traversal["entries"][0]["source_paths"][0] = "../outside.py";

  auto absolute = original;
  absolute["entries"][0]["source_paths"][0] = "/etc/passwd";

  auto duplicate_source = original;
  duplicate_source["entries"][0]["source_paths"].push_back(
      duplicate_source["entries"][0]["source_paths"][0]);

  auto empty_notes = original;
  empty_notes["entries"][0]["notes"] = "   ";

  auto library_resume = original;
  for (auto& entry : library_resume["entries"]) {
    if (entry.at("observed_invocation") == "library_only") {
      entry["stateful"] = true;
      entry["resume_evidence"] = "compatible";
      break;
    }
  }

  auto design_resume = original;
  for (auto& entry : design_resume["entries"]) {
    if (entry.at("observed_invocation") == "design_only") {
      entry["resume_evidence"] = "terminal_checkpoint";
      break;
    }
  }

  auto role_contradiction = original;
  for (auto& entry : role_contradiction["entries"]) {
    if (entry.at("observed_invocation") == "library_only") {
      entry["operation_role"] = "training";
      break;
    }
  }

  auto missing_display = original;
  for (auto& entry : missing_display["entries"]) {
    if (entry.at("observed_invocation") == "python_module") {
      entry.erase("legacy_invocation_display");
      break;
    }
  }

  auto library_display = original;
  for (auto& entry : library_display["entries"]) {
    if (entry.at("observed_invocation") == "library_only") {
      entry["legacy_invocation_display"] = "python accidental.py";
      break;
    }
  }

  auto unknown_invocation = original;
  unknown_invocation["entries"][0]["observed_invocation"] = "executable";

  auto authority_escalation = original;
  authority_escalation["authority"] = "execution_authority";

  auto unknown_field = original;
  unknown_field["entries"][0]["adapter"] = "must-not-be-accepted";

  auto adapter_exact = original;
  for (auto& entry : adapter_exact["entries"]) {
    if (entry.at("resume_evidence") == "exact_candidate") {
      entry["resume_evidence"] = "exact";
      break;
    }
  }

  check(rejects(duplicate_id), "validator rejects duplicate stable IDs");
  check(rejects(family_preserving_prune) && rejects(unreviewed_addition),
        "compiled v1 inventory rejects family-preserving pruning and additions");
  check(rejects(traversal) && rejects(absolute) &&
            rejects(duplicate_source),
        "validator rejects unsafe and duplicate source paths");
  check(rejects(empty_notes),
        "validator requires nonempty bounded explanatory fields");
  check(rejects(library_resume) && rejects(design_resume) &&
            rejects(role_contradiction),
        "library and design evidence cannot claim state or command roles");
  check(rejects(missing_display) && rejects(library_display),
        "the exact reviewed mapping is pinned even for optional display metadata");
  check(rejects(unknown_invocation) && rejects(adapter_exact),
        "unknown invocation and AdapterRegistry exact vocabularies are rejected");
  check(rejects(authority_escalation) && rejects(unknown_field),
        "catalog cannot grow adapter or execution authority fields");

  const fs::path duplicate_key_path = temporary.path() / "duplicate-key.json";
  {
    std::ofstream output(duplicate_key_path);
    output << R"({"api_version":"trainvm.compatibility-workflows/v1","api_version":"duplicate","authority":"compatibility_evidence_only","entries":[]})";
  }
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(duplicate_key_path,
                                                         root);
        }),
        "parser rejects duplicate JSON object keys");

  const fs::path catalog_link = temporary.path() / "catalog-link.json";
  fs::create_symlink(fixture, catalog_link);
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(catalog_link, root);
        }),
        "catalog open rejects symlinks");
  const fs::path catalog_directory = temporary.path() / "catalog-directory";
  fs::create_directory(catalog_directory);
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(catalog_directory,
                                                         root);
        }),
        "catalog open rejects nonregular files");
  const fs::path oversized = temporary.path() / "oversized.json";
  {
    std::ofstream output(oversized, std::ios::binary);
    output << std::string((1U << 20U) + 1U, 'x');
  }
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(oversized, root);
        }),
        "catalog open rejects oversized files");

  const fs::path cloned_root = temporary.path() / "cloned-root";
  fs::create_directory(cloned_root);
  std::set<std::string> source_paths;
  for (const auto& entry : original.at("entries")) {
    for (const auto& source : entry.at("source_paths")) {
      source_paths.insert(source.get<std::string>());
    }
  }
  for (const auto& relative : source_paths) {
    restore_source(root / relative, cloned_root / relative);
  }
  check(!throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(fixture, cloned_root);
        }),
        "identical cloned sources satisfy the content digest");

  const std::string victim = *source_paths.begin();
  const fs::path cloned_victim = cloned_root / victim;
  {
    std::ofstream output(cloned_victim, std::ios::app | std::ios::binary);
    output << '\n';
  }
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(fixture, cloned_root);
        }),
        "source-byte mutation invalidates the source tree digest");

  restore_source(root / victim, cloned_victim);
  fs::remove(cloned_victim);
  fs::create_symlink(root / victim, cloned_victim);
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(fixture, cloned_root);
        }),
        "descriptor-relative source open rejects symlinks");

  restore_source(root / victim, cloned_victim);
  fs::remove(cloned_victim);
  fs::create_directory(cloned_victim);
  check(throws_invalid_argument([&] {
          (void)trainvm::CompatibilityCatalog::load_file(fixture, cloned_root);
        }),
        "descriptor-relative source open rejects nonregular files");

  // ---------------------------------------------------------------------
  // The classification surface. The point of this whole mechanism is that the
  // reviewed digest moves when a source's CLASSIFICATION could have changed and
  // stays put when only its internals did. Both halves are asserted; a gate
  // that cannot fail is worse than none, and one that always fails gets cleared
  // without the review it exists to force.
  // ---------------------------------------------------------------------
  const auto surface = [](std::string_view path, std::string_view bytes) {
    return trainvm::CompatibilityCatalog::classification_surface_for_testing(
        path, bytes);
  };
  const auto read_file = [](const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  };

  const std::string module_source =
      "\"\"\"A docstring.\"\"\"\n"
      "import argparse\n"
      "\n"
      "def helper(x):\n"
      "    return x * 2  # internal arithmetic\n"
      "\n"
      "def main():\n"
      "    parser = argparse.ArgumentParser()\n"
      "    parser.add_argument('--steps')\n"
      "    torch.save(state, path)\n"
      "\n"
      "if __name__ == '__main__':\n"
      "    main()\n";
  const auto module_surface = surface("src/example.py", module_source);

  check(!module_surface.empty(),
        "a Python module with an entrypoint has a nonempty surface");
  check(module_surface.find("def helper") == std::string::npos &&
            module_surface.find("A docstring") == std::string::npos,
        "the surface excludes internals and prose");
  check(module_surface.find("def main():") != std::string::npos &&
            module_surface.find("add_argument") != std::string::npos,
        "the surface includes the entrypoint and the argument surface");

  // A comment, a docstring, whitespace, and a changed internal computation.
  // None of these can alter family, operation_role, statefulness or
  // resume_evidence, so none may move the digest.
  std::string edited_internals = module_source;
  edited_internals.replace(edited_internals.find("return x * 2"),
                           std::string("return x * 2").size(),
                           "return x * 3");
  edited_internals += "\n# a trailing comment, added later\n";
  check(surface("src/example.py", edited_internals) == module_surface,
        "editing internals, comments and trailing lines leaves the surface put");

  std::string renamed_entrypoint = module_source;
  renamed_entrypoint.replace(renamed_entrypoint.find("def main():"),
                             std::string("def main():").size(),
                             "def run():");
  check(surface("src/example.py", renamed_entrypoint) != module_surface,
        "renaming the entrypoint moves the surface");

  std::string added_argument = module_source;
  added_argument.replace(added_argument.find("torch.save"),
                         std::string("torch.save").size(),
                         "parser.add_argument('--resume'); torch.save");
  check(surface("src/example.py", added_argument) != module_surface,
        "adding an argument moves the surface");

  std::string nested_entrypoint = module_source;
  nested_entrypoint.replace(nested_entrypoint.find("def main():"),
                            std::string("def main():").size(),
                            "    def main():");
  check(surface("src/example.py", nested_entrypoint) != module_surface,
        "indenting the entrypoint moves the surface: scope is classification");

  // A library module genuinely has no entrypoint, so an empty surface is the
  // honest answer -- but it must stop being empty the moment one appears, or
  // the file would be permanently invisible to review.
  const std::string library_source =
      "def transform(batch):\n    return batch.mean()\n";
  check(surface("src/library.py", library_source).empty(),
        "a library module with no entrypoint has an empty surface");
  check(!surface("src/library.py",
                 library_source + "\ndef main():\n    transform(None)\n")
             .empty(),
        "gaining an entrypoint makes a previously empty surface fire");

  // Only Python is understood. Anything else keeps its full bytes, so no
  // language silently loses byte-level binding it used to have.
  const std::string shell_source = "#!/bin/sh\n# a comment\nexec python -m x\n";
  check(surface("scripts/run.sh", shell_source) == shell_source,
        "a non-Python source falls back to its whole bytes");
  check(surface("scripts/run.sh", shell_source + "# another comment\n") !=
            shell_source,
        "a non-Python source is still bound byte for byte");

  // The headline property, end to end against the real catalog: an edit that
  // cannot change classification must NOT require touching the compiled
  // reviewed digest. Only the byte digest is re-pinned, which the generator
  // now produces, and the catalog still loads.
  std::string python_victim;
  for (const auto& candidate : source_paths) {
    if (candidate.ends_with(".py") &&
        read_file(root / candidate).find("\ndef main(") != std::string::npos) {
      python_victim = candidate;
      break;
    }
  }
  check(!python_victim.empty(),
        "the catalog references a Python module with an entrypoint to mutate");

  if (!python_victim.empty()) {
    // The symlink and directory checks above leave the cloned tree with their
    // victim replaced by a directory. Put it back, or every digest computed
    // below fails on that file rather than on what is being tested.
    fs::remove(cloned_victim);
    restore_source(root / victim, cloned_victim);

    const auto write_fixture = [&](const nlohmann::json& document,
                                   std::string_view name) {
      const fs::path path = temporary.path() / name;
      std::ofstream output(path);
      output << document.dump(2);
      return path;
    };

    const fs::path victim_path = cloned_root / python_victim;
    restore_source(root / python_victim, victim_path);
    {
      std::ofstream output(victim_path, std::ios::app | std::ios::binary);
      output << "\n# an explanatory comment that changes no classification\n";
    }
    auto recomputed = trainvm::CompatibilityCatalog::compute_digests(
        fixture, cloned_root);
    const auto encode_source_digests =
        [](const std::vector<trainvm::CompatibilitySourceDigest>& pinned) {
          nlohmann::json encoded = nlohmann::json::array();
          for (const auto& one : pinned) {
            encoded.push_back({{"source_path", one.source_path},
                               {"source_sha256", one.source_sha256}});
          }
          return encoded;
        };
    nlohmann::json repinned = original;
    repinned["source_digests"] = encode_source_digests(recomputed.source_digests);
    check(recomputed.classification_surface_digest ==
              original.at("classification_surface_digest").get<std::string>(),
          "a comment leaves the classification surface digest untouched");
    check(!throws_invalid_argument([&] {
            (void)trainvm::CompatibilityCatalog::load_file(
                write_fixture(repinned, "repinned.json"), cloned_root);
          }),
          "a classification-irrelevant edit needs no reviewed-digest bump");

    // The converse. Renaming the entrypoint IS a classification change, so even
    // with both digests honestly re-pinned the compiled reviewed digest must
    // refuse it. This is what fails closed.
    restore_source(root / python_victim, victim_path);
    {
      const auto text = read_file(victim_path);
      const auto position = text.find("\ndef main(");
      std::string mutated = text;
      mutated.replace(position, std::string("\ndef main(").size(),
                      "\ndef entry_point(");
      std::ofstream output(victim_path, std::ios::trunc | std::ios::binary);
      output << mutated;
    }
    recomputed = trainvm::CompatibilityCatalog::compute_digests(fixture,
                                                                cloned_root);
    check(recomputed.classification_surface_digest !=
              original.at("classification_surface_digest").get<std::string>(),
          "renaming an entrypoint moves the classification surface digest");
    nlohmann::json honestly_repinned = original;
    honestly_repinned["source_digests"] =
        encode_source_digests(recomputed.source_digests);
    honestly_repinned["classification_surface_digest"] =
        recomputed.classification_surface_digest;
    check(throws_invalid_argument([&] {
            (void)trainvm::CompatibilityCatalog::load_file(
                write_fixture(honestly_repinned, "honest.json"), cloned_root);
          }),
          "a classification change still fails against the compiled digest");
    restore_source(root / python_victim, victim_path);

    // The point of the whole change: an edit to one referenced source rewrites
    // only that source's pin, so two independent refreshes touch two different
    // objects and git can merge them. While the tree digest was stored, both
    // rewrote the same line and conflicted by construction.
    {
      restore_source(root / python_victim, victim_path);
      const auto baseline = trainvm::CompatibilityCatalog::compute_digests(
          fixture, cloned_root);
      std::ofstream output(victim_path, std::ios::app | std::ios::binary);
      output << "\n# a second classification-irrelevant comment\n";
      output.close();
      const auto after = trainvm::CompatibilityCatalog::compute_digests(
          fixture, cloned_root);
      std::size_t moved = 0;
      for (std::size_t index = 0; index < after.source_digests.size(); ++index) {
        if (after.source_digests[index].source_sha256 !=
            baseline.source_digests[index].source_sha256) {
          ++moved;
          check(after.source_digests[index].source_path == python_victim,
                "the pin that moved is the edited source's own");
        }
      }
      check(moved == 1,
            "editing one referenced source moves exactly one per-source pin");
      restore_source(root / python_victim, victim_path);
    }

    // A stale whole-tree digest left behind by a hand-resolved rebase is
    // refused by name rather than as an anonymous unknown field, because the
    // unknown-field message tells the reader to rebuild a current binary.
    {
      nlohmann::json stale = original;
      stale["source_tree_digest"] = recomputed.source_tree_digest;
      check(throws_invalid_argument([&] {
              (void)trainvm::CompatibilityCatalog::load_file(
                  write_fixture(stale, "stale-tree-digest.json"), cloned_root);
            }),
            "a catalog that still stores source_tree_digest is refused");
    }

    // The per-path pins have to cover the referenced set exactly, or a source
    // could drop out of the byte binding without anything noticing -- which is
    // the one way this change could quietly weaken the gate it replaces.
    {
      nlohmann::json short_pins = original;
      short_pins["source_digests"].erase(0);
      check(throws_invalid_argument([&] {
              (void)trainvm::CompatibilityCatalog::load_file(
                  write_fixture(short_pins, "short-pins.json"), cloned_root);
            }),
            "a referenced source with no pin is refused");

      nlohmann::json extra_pins = original;
      extra_pins["source_digests"].push_back(
          {{"source_path", "zzz/not-referenced.py"},
           {"source_sha256", std::string("sha256:") + std::string(64U, 'a')}});
      check(throws_invalid_argument([&] {
              (void)trainvm::CompatibilityCatalog::load_file(
                  write_fixture(extra_pins, "extra-pins.json"), cloned_root);
            }),
            "a pin for a source no entry references is refused");

      nlohmann::json unsorted = original;
      std::swap(unsorted["source_digests"][0], unsorted["source_digests"][1]);
      check(throws_invalid_argument([&] {
              (void)trainvm::CompatibilityCatalog::load_file(
                  write_fixture(unsorted, "unsorted-pins.json"), cloned_root);
            }),
            "unsorted source_digests are refused");
    }
  }

  // The fold, pinned directly. The catalog used to store the whole-tree digest,
  // and that stored constant incidentally held this computation still: change a
  // separator and the shipped catalog stopped loading. Deriving the digest
  // removes that accident, so the fold is pinned here instead -- over fixed
  // pairs, so ordinary source edits never move it. The expected value is the
  // same construction computed independently: leaf =
  // "trainvm.compatibility-source-leaf/v1" NUL path NUL hex, tree =
  // "trainvm.compatibility-source-tree/v1" then NUL + sha256(leaf) each.
  {
    const std::vector<trainvm::CompatibilitySourceDigest> pairs = {
        {"scripts/alpha.py", "sha256:" + std::string(32U, '1') +
                                 std::string(32U, '1')},
        {"src/beta.py",
         "sha256:" + std::string(32U, '2') + std::string(32U, '2')},
    };
    check(trainvm::CompatibilityCatalog::source_tree_digest_for_testing(pairs) ==
              "sha256:bdd08803045d7349f5c85ca27e7fae14f2cc4b7a1aaf3b2448da9e77"
              "3c3ed021",
          "the source tree fold matches its independently computed value");
    auto reordered = pairs;
    std::swap(reordered[0], reordered[1]);
    check(trainvm::CompatibilityCatalog::source_tree_digest_for_testing(
              reordered) !=
              trainvm::CompatibilityCatalog::source_tree_digest_for_testing(
                  pairs),
          "the fold is order sensitive, so it binds the set and its order");
  }

  if (failures != 0) {
    std::cerr << failures << " compatibility catalog test(s) failed\n";
    return 1;
  }
  std::cout << "compatibility catalog tests passed\n";
  return 0;
}
