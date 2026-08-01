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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

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
  check(catalog.entries().size() == 140U,
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
  std::set<std::string> exact_candidates = {
      "vision.frozen-adapter-train",
      "vision.teacher-compressor",
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
    if (exact_candidates.contains(entry.stable_id) && entry.stateful &&
        entry.resume_evidence ==
            trainvm::CompatibilityResumeEvidence::exact_candidate) {
      exact_candidates.erase(entry.stable_id);
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
  check(families.size() == 11U && invocation_kinds.size() == 6U &&
            identifiers.size() == catalog.entries().size(),
        "catalog contains every family and observed invocation kind");
  check(required_additions.empty(),
        "catalog retains the expanded audited workflow inventory");
  check(reviewed_sources.size() == 141U,
        "catalog binds the complete reviewed source inventory");
  check(exact_candidates.empty() && saw_restart_only_review,
        "legacy exact candidates and restart-only review are classified narrowly");
  check(saw_mutable_legacy_export && saw_mutable_frozen_export,
        "export notes disclose overwrite and verification limitations");
  check(saw_lossless_qualification && saw_split_production_qualification &&
            saw_ltx_training_only && saw_lossless_library_oracle &&
            saw_production_aot_export,
        "effectfully distinct qualification, export, and external phases stay split");
  check(saw_http_control_handler,
        "dashboard HTTP controls are not mislabeled as host scripts");
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

  if (failures != 0) {
    std::cerr << failures << " compatibility catalog test(s) failed\n";
    return 1;
  }
  std::cout << "compatibility catalog tests passed\n";
  return 0;
}
