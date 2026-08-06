#include "trainvm/training_preflight.hpp"

#include "trainvm/recipe_profile.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace {

using namespace trainvm;

constexpr std::uint64_t kGib = 1ULL << 30U;
int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

bool has_code(const TrainingPreflightReceipt &receipt, std::string_view code) {
  return std::ranges::any_of(receipt.diagnostics, [&](const auto &diagnostic) {
    return diagnostic.code == code;
  });
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "trainvm-preflight-XXXXXX")
            .string();
    char *created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("could not create temporary directory");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

nlohmann::json load_fixture() {
  const auto path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                    "docs/experiment-vm/examples/mageflow-cache-resume.json";
  std::ifstream input(path);
  nlohmann::json fixture;
  input >> fixture;
  if (!input)
    throw std::runtime_error("could not load experiment fixture");
  return fixture;
}

CompiledPlan compiled_fixture(const TemporaryDirectory &temporary,
                              bool create_run_directory = true,
                              bool require_matching_label = false) {
  const auto root = temporary.path();
  const auto input_root = root / "input";
  const auto run_directory = root / "run";
  (void)::chmod(root.c_str(), 0755);
  std::filesystem::create_directory(input_root);
  if (create_run_directory)
    std::filesystem::create_directory(run_directory);
  std::ofstream(input_root / "config.json") << "{}\n";
  (void)::chmod(input_root.c_str(), 0755);
  if (create_run_directory)
    (void)::chmod(run_directory.c_str(), 0770);

  auto source = load_fixture();
  auto &spec = source["spec"];
  spec.erase("execution");
  spec["workspace"] = {
      {"root", root.string()},
      {"run_directory", run_directory.string()},
      {"concurrency_key", "preflight-test"},
      {"allowed_read_roots", nlohmann::json::array({input_root.string()})},
      {"allowed_write_roots", nlohmann::json::array({run_directory.string()})},
      {"input_content_roots",
       nlohmann::json::array(
           {{{"api_version", "trainvm.input-content-root/v1"},
             {"path", input_root.string()},
             {"kind", "directory"},
             {"file_count", 1},
             {"total_bytes", 3},
             {"tree_sha256", "sha256:" + std::string(64U, '1')}}})},
  };
  spec["resources"]["accelerators"] = {
      {"vendor", "nvidia"},
      {"count", 1},
      {"minimum_memory_gib", 90.0},
      {"exclusive", false},
  };
  if (require_matching_label)
    spec["resources"]["accelerators"]["selector"] = {{"pool", "training"}};
  spec["parameters"]["source_config"]["value"] =
      (input_root / "config.json").string();

  auto train = spec["workflow"]["nodes"]["train_to_boundary"];
  train["invoke"]["training"] = {
      {"model_family", "mageflow"},
      {"components",
       {{"activation",
         {{"key",
           {{"category", "activation"},
            {"name", "silu"},
            {"version", "1.0.0"}}},
          {"configuration", nlohmann::json::object()}}}}}};
  train["transitions"] = nlohmann::json::array(
      {{{"on", "worker.completed"}, {"target", "release_gpu"}},
       {{"on", "operation.failed"}, {"target", "$failed"}}});
  auto acquire = spec["workflow"]["nodes"]["acquire_gpu"];
  acquire["transitions"] = nlohmann::json::array(
      {{{"on", "resource.acquired"}, {"target", "train_to_boundary"}},
       {{"on", "operation.failed"}, {"target", "$failed"}}});
  auto release = spec["workflow"]["nodes"]["release_gpu"];
  spec["workflow"]["nodes"] = {
      {"acquire_gpu", std::move(acquire)},
      {"train_to_boundary", std::move(train)},
      {"release_gpu", std::move(release)},
  };
  spec["workflow"]["entrypoint"] = "acquire_gpu";
  spec["recovery"]["exact_resume"] = false;
  spec["recovery"].erase("checkpoint_artifact");

  auto compiled = compile_document(source);
  if (!compiled.valid() || !compiled.plan) {
    throw std::runtime_error("preflight fixture failed to compile: " +
                             diagnostics_json(compiled.diagnostics).dump());
  }
  return *compiled.plan;
}

std::vector<TrainingPreflightCheckEvidence> passing_checks() {
  std::vector<TrainingPreflightCheckEvidence> result;
  for (const auto kind : {TrainingPreflightCheckKind::model_configuration,
                          TrainingPreflightCheckKind::tokenizer,
                          TrainingPreflightCheckKind::processor,
                          TrainingPreflightCheckKind::dataset_schema,
                          TrainingPreflightCheckKind::dataset_sample_decode,
                          TrainingPreflightCheckKind::parameter_selection,
                          TrainingPreflightCheckKind::kernel_runtime,
                          TrainingPreflightCheckKind::checkpoint_compatibility,
                          TrainingPreflightCheckKind::step_zero_evaluator,
                          TrainingPreflightCheckKind::dashboard_artifacts}) {
    result.push_back({.kind = kind,
                      .disposition = TrainingPreflightCheckDisposition::passed,
                      .evidence_digest = "sha256:" + std::string(64U, '2'),
                      .detail = std::nullopt});
  }
  return result;
}

TrainingPreflightEnvironment environment(const CompiledPlan &plan) {
  const std::uint32_t uid = static_cast<std::uint32_t>(::geteuid());
  const std::uint32_t gid = static_cast<std::uint32_t>(::getegid());
  return {
      .api_version = std::string(kTrainingPreflightEnvironmentApiVersion),
      .host_id = "sha256:" + std::string(64U, '3'),
      .boot_id = "33333333-3333-3333-3333-333333333333",
      .snapshot_digest = "sha256:" + std::string(64U, '4'),
      .snapshot_observed_monotonic_ns = 1'000U,
      .snapshot_valid_until_monotonic_ns = 2'000U,
      .evaluation_monotonic_ns = 1'100U,
      .worker_uid = uid == 0U ? 1000U : uid,
      .worker_gid = gid == 0U ? 1000U : gid,
      .supplementary_gids = {},
      .worker_principal_digest = "sha256:" + std::string(64U, '5'),
      .total_host_memory_bytes = 256U * kGib,
      .available_host_memory_bytes = 192U * kGib,
      .logical_cpu_count = 64U,
      .accelerators = {{
          .vendor = AcceleratorVendor::nvidia,
          .stable_id = "GPU-passive-test",
          .total_memory_bytes = 96U * kGib,
          .free_memory_bytes = 90U * kGib,
          .selector_labels = {},
          .observation_digest = "sha256:" + std::string(64U, '6'),
      }},
      .training_nodes = {{
          .node_id = "train_to_boundary",
          .node_input_digest =
              training_preflight_node_input_digest(plan, "train_to_boundary"),
          .checks = passing_checks(),
          .minimum_free_memory_gib = 88.0,
          .runtime_profile_digest = "sha256:" + std::string(64U, '7'),
          .required_capabilities = {"flash_attention_2"},
          .provided_capabilities = {"flash_attention_2"},
      }},
      .gpu_qualification = std::nullopt,
      .recipe_provenance = std::nullopt,
  };
}

void passing_preflight_is_passive_and_deterministic() {
  TemporaryDirectory temporary;
  const auto plan = compiled_fixture(temporary);
  const auto evidence = environment(plan);
  const auto first = run_training_preflight(plan, evidence);
  const auto second = run_training_preflight(plan, evidence);
  check(first.passed && first.accelerator_passive && first.cacheable,
        "valid evidence passes without requesting a GPU qualification");
  check(first == second && first.receipt_digest.starts_with("sha256:") &&
            first.cache_key.starts_with("sha256:"),
        "same plan/input/environment identities produce a deterministic "
        "cacheable receipt");

  auto changed = evidence;
  changed.snapshot_digest = "sha256:" + std::string(64U, '7');
  const auto refreshed = run_training_preflight(plan, changed);
  check(refreshed.environment_digest != first.environment_digest &&
            refreshed.cache_key != first.cache_key &&
            refreshed.receipt_digest != first.receipt_digest,
        "relevant passive environment changes invalidate cached evidence");

  changed = evidence;
  const ExpandedRecipe expanded{
      .recipe = {.name = "test", .version = "1"},
      .run_identity = "preflight-test",
      .registry_digest = "sha256:" + std::string(64U, 'b'),
      .profile_digest = "sha256:" + std::string(64U, 'c'),
      .instance_digest = "sha256:" + std::string(64U, 'd'),
      .expanded_plan_digest = "sha256:" + plan.plan_hash,
      .effective_overrides = {},
      .provenance = {},
      .plan = plan,
  };
  changed.recipe_provenance = training_preflight_recipe_provenance(expanded);
  const auto recipe_bound = run_training_preflight(plan, changed);
  check(recipe_bound.passed && recipe_bound.cache_key != first.cache_key,
        "optional recipe provenance is bound without changing the CompiledPlan "
        "interface");

  changed.recipe_provenance->expanded_plan_digest =
      "sha256:" + std::string(64U, 'e');
  const auto recipe_mismatch = run_training_preflight(plan, changed);
  check(!recipe_mismatch.passed &&
            has_code(recipe_mismatch, "preflight.recipe_plan_mismatch"),
        "recipe provenance cannot be replayed against another compiled plan");

  const auto environment_path = temporary.path() / "environment.json";
  std::ofstream(environment_path) << encode_json(evidence).dump();
  check(load_training_preflight_environment(environment_path) == evidence,
        "the closed machine-readable environment schema round-trips exactly");
}

void qwen_total_and_free_vram_policies_are_distinct() {
  TemporaryDirectory temporary;
  const auto plan = compiled_fixture(temporary);
  auto evidence = environment(plan);
  evidence.accelerators.front().free_memory_bytes = 87U * kGib;
  evidence.accelerators.front().observation_digest =
      "sha256:" + std::string(64U, '8');
  const auto receipt = run_training_preflight(plan, evidence);
  check(!receipt.passed &&
            has_code(receipt, "resource.free_vram_insufficient") &&
            !has_code(receipt, "resource.total_vram_insufficient"),
        "Qwen-style total-capacity success cannot hide a free-VRAM policy "
        "failure");
}

void passive_memory_obeys_the_declared_selector() {
  TemporaryDirectory temporary;
  const auto plan = compiled_fixture(temporary, true, true);
  auto evidence = environment(plan);
  evidence.accelerators.front().selector_labels = {{"pool", "display"}};
  const auto rejected = run_training_preflight(plan, evidence);
  check(!rejected.passed &&
            has_code(rejected, "resource.total_vram_insufficient"),
        "same-VRAM GPU with mismatched labels is rejected before a lease");
  evidence.accelerators.front().selector_labels = {{"pool", "training"}};
  check(run_training_preflight(plan, evidence).passed,
        "matching passive selector labels satisfy the same plan");
}

void worker_permission_failure_precedes_run_creation() {
  TemporaryDirectory temporary;
  const auto plan = compiled_fixture(temporary);
  const auto run_directory =
      std::filesystem::path(plan.experiment.spec.workspace.run_directory);
  (void)::chmod(run_directory.c_str(), 0700);
  auto evidence = environment(plan);
  evidence.worker_uid = evidence.worker_uid == 42U ? 43U : 42U;
  evidence.worker_gid = evidence.worker_gid == 43U ? 44U : 43U;
  evidence.worker_principal_digest = "sha256:" + std::string(64U, '9');
  const auto receipt = run_training_preflight(plan, evidence);
  check(!receipt.passed && has_code(receipt, "output.worker_permission"),
        "run-directory ownership failure is actionable for the exact worker "
        "credentials");

  (void)::chmod(run_directory.c_str(), 01777);
  const auto sticky_receipt = run_training_preflight(plan, evidence);
  check(!sticky_receipt.passed &&
            has_code(sticky_receipt, "output.sticky_rename_policy"),
        "sticky-directory replacement semantics are checked for the exact "
        "worker owner");

  TemporaryDirectory missing;
  const auto missing_plan = compiled_fixture(missing, false);
  const auto absent = std::filesystem::path(
      missing_plan.experiment.spec.workspace.run_directory);
  auto missing_evidence = environment(missing_plan);
  missing_evidence.worker_uid = missing_evidence.worker_uid == 52U ? 53U : 52U;
  missing_evidence.worker_gid = missing_evidence.worker_gid == 53U ? 54U : 53U;
  missing_evidence.worker_principal_digest = "sha256:" + std::string(64U, 'a');
  const auto missing_receipt =
      run_training_preflight(missing_plan, missing_evidence);
  check(!missing_receipt.passed &&
            has_code(missing_receipt, "output.worker_permission") &&
            !std::filesystem::exists(absent),
        "failed preflight never creates an output directory, run, lease, or "
        "worker side effect");
}

void missing_adapter_evidence_fails_closed() {
  TemporaryDirectory temporary;
  const auto plan = compiled_fixture(temporary);
  auto evidence = environment(plan);
  evidence.training_nodes.clear();
  const auto receipt = run_training_preflight(plan, evidence);
  check(!receipt.passed && has_code(receipt, "probe.missing"),
        "missing registered adapter evidence never falls back to native "
        "inference");
}

void credential_sets_and_gpu_qualification_are_bounded() {
  TemporaryDirectory temporary;
  const auto plan = compiled_fixture(temporary);
  auto groups = environment(plan);
  groups.supplementary_gids = {100U, 99U};
  check(
      has_code(run_training_preflight(plan, groups), "preflight.worker_groups"),
      "noncanonical effective group sets are rejected");

  auto stale = environment(plan);
  stale.snapshot_valid_until_monotonic_ns = 60'000'001'001ULL;
  check(has_code(run_training_preflight(plan, stale),
                 "preflight.snapshot_freshness"),
        "volatile free-memory evidence cannot receive an unbounded cache "
        "lifetime");

  const auto run_directory =
      std::filesystem::path(plan.experiment.spec.workspace.run_directory);
  (void)::chmod(run_directory.c_str(), 0070);
  auto supplementary = environment(plan);
  supplementary.worker_uid = supplementary.worker_uid == 72U ? 73U : 72U;
  supplementary.worker_gid = supplementary.worker_gid == 73U ? 74U : 73U;
  supplementary.supplementary_gids = {static_cast<std::uint32_t>(::getegid())};
  supplementary.worker_principal_digest = "sha256:" + std::string(64U, 'f');
  check(run_training_preflight(plan, supplementary).passed,
        "the complete effective supplementary group set participates in "
        "POSIX output authority");

  (void)::chmod(run_directory.c_str(), 0770);
  auto qualified = environment(plan);
  qualified.gpu_qualification = BoundedGpuQualificationEvidence{
      .maximum_duration_milliseconds = 1'000U,
      .passed = true,
      .receipt_digest = "sha256:" + std::string(64U, 'b'),
  };
  const auto receipt = run_training_preflight(plan, qualified);
  check(
      receipt.passed && !receipt.accelerator_passive,
      "an explicit bounded qualification receipt is the only non-passive mode");

  qualified.gpu_qualification->maximum_duration_milliseconds = 60'001U;
  check(has_code(run_training_preflight(plan, qualified),
                 "gpu_qualification.unbounded"),
        "unbounded GPU qualification fails before submission");
}

} // namespace

int main() {
  try {
    passing_preflight_is_passive_and_deterministic();
    qwen_total_and_free_vram_policies_are_distinct();
    passive_memory_obeys_the_declared_selector();
    worker_permission_failure_precedes_run_creation();
    missing_adapter_evidence_fails_closed();
    credential_sets_and_gpu_qualification_are_bounded();
  } catch (const std::exception &error) {
    std::cerr << "training preflight test failure: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " training preflight assertion(s) failed\n";
    return 1;
  }
  std::cout << "training preflight tests passed\n";
  return 0;
}
