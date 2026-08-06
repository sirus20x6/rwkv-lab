#include "trainvm/run_authoring.hpp"
#include "trainvm/run_authoring_cli.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace {

using namespace trainvm;
int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "trainvm-authoring-XXXXXX")
            .string();
    char *created = ::mkdtemp(pattern.data());
    if (created == nullptr)
      throw std::runtime_error("could not create temporary directory");
    path_ = created;
    (void)::chmod(path_.c_str(), 0755);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

nlohmann::json load_fixture() {
  std::ifstream input(std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                      "docs/experiment-vm/examples/mageflow-cache-resume.json");
  nlohmann::json value;
  input >> value;
  if (!input)
    throw std::runtime_error("could not read authoring fixture");
  return value;
}

nlohmann::json authorable_fixture(const TemporaryDirectory &temporary) {
  const auto input_root = temporary.path() / "input";
  const auto run_directory = temporary.path() / "run";
  std::filesystem::create_directory(input_root);
  std::ofstream(input_root / "config.json") << "{}\n";
  (void)::chmod(input_root.c_str(), 0755);
  auto source = load_fixture();
  auto &spec = source["spec"];
  spec.erase("execution");
  spec["workspace"] = {
      {"root", temporary.path().string()},
      {"run_directory", run_directory.string()},
      {"concurrency_key", "authoring-test"},
      {"allowed_read_roots", nlohmann::json::array({input_root.string()})},
      {"allowed_write_roots", nlohmann::json::array({run_directory.string()})},
  };
  spec["parameters"]["source_config"]["value"] =
      (input_root / "config.json").string();
  spec["resources"]["accelerators"]["count"] = 0;
  spec["resources"]["accelerators"].erase("minimum_memory_gib");
  return source;
}

AuthorRunDocument author_document(const TemporaryDirectory &temporary) {
  const nlohmann::json document{
      {"api_version", kAuthorRunApiVersion},
      {"source", {{"experiment", authorable_fixture(temporary)}}},
      {"input_content",
       {{"api_version", kInputContentRootSetApiVersion},
        {"paths",
         nlohmann::json::array({(temporary.path() / "input").string()})}}},
      {"author", "authoring-test"},
      {"reason", "exercise deterministic one-command preparation"},
  };
  return decode_author_run_document(document.dump(), "json");
}

void rejects_duplicate_keys_recursively() {
  for (const auto &[format, source] : {
           std::pair<std::string_view, std::string_view>{
               "json", R"({"api_version":"trainvm.author-run/v1","source":{"experiment":{},"experiment":{}},"author":"a","reason":"r"})"},
           {"yaml", "api_version: trainvm.author-run/v1\nsource:\n  experiment: {}\n  experiment: {}\nauthor: a\nreason: r\n"},
       }) {
    bool rejected = false;
    try {
      (void)decode_author_run_document(source, format);
    } catch (const RunAuthoringError &) {
      rejected = true;
    }
    check(rejected, "duplicate nested keys are rejected for JSON and YAML");
  }
}

void passive_lora_selector_matches_python_module_semantics() {
  const std::vector<std::string> keys{
      "model.language_model.layers.3.self_attn.q_proj.weight",
      "model.language_model.layers.3.self_attn.q_proj.bias",
      "model.language_model.layers.3.self_attn.k_proj.weight",
  };
  check(hf_lora_selectors_match_parameter_index(
            {"model.language_model.layers.*.self_attn.?_proj"}, keys),
        "passive LoRA parity strips parameter suffixes and applies fnmatchcase "
        "to Qwen named-module identities");
  check(!hf_lora_selectors_match_parameter_index(
            {"model.language_model.layers.*.mlp.*_proj"}, keys),
        "passive LoRA selector fails closed when Python named_modules would "
        "match nothing");
}

void resolves_locks_and_reuses_exact_lock() {
  TemporaryDirectory temporary;
  const ResolvedAuthorRun first =
      resolve_and_lock_author_run(author_document(temporary));
  check(first.plan.experiment.spec.workspace.input_content_roots.has_value(),
        "authoring measures and injects immutable input roots");
  check(!first.content_lock_reused &&
            first.request_digest.starts_with("sha256:"),
        "newly measured authoring identifies the exact request");

  AuthorRunDocument retry{
      .api_version = std::string(kAuthorRunApiVersion),
      .source = {.experiment =
                     std::optional<nlohmann::json>(first.plan.canonical_plan),
                 .recipe = std::nullopt},
      .input_content = std::nullopt,
      .author = "authoring-test",
      .reason = "reuse exact immutable lock",
  };
  const ResolvedAuthorRun second = resolve_and_lock_author_run(retry);
  check(second.content_lock_reused,
        "an exact existing content lock is recognized as reusable");
  check(second.plan.plan_hash == first.plan.plan_hash,
        "reusing an exact content lock preserves the compiled plan");
}

TrainingPreflightEnvironment worker_environment(std::uint32_t uid,
                                                std::uint32_t gid) {
  return {.api_version = std::string(kTrainingPreflightEnvironmentApiVersion),
          .host_id = {},
          .boot_id = {},
          .snapshot_digest = {},
          .snapshot_observed_monotonic_ns = 0U,
          .snapshot_valid_until_monotonic_ns = 0U,
          .evaluation_monotonic_ns = 0U,
          .worker_uid = uid,
          .worker_gid = gid,
          .supplementary_gids = {},
          .worker_principal_digest =
              "sha256:" + std::string(64U, 'a'),
          .total_host_memory_bytes = 0U,
          .available_host_memory_bytes = 0U,
          .logical_cpu_count = 0U,
          .accelerators = {},
          .training_nodes = {},
          .gpu_qualification = std::nullopt,
          .recipe_provenance = std::nullopt};
}

void provisioning_rolls_back_then_retries() {
  TemporaryDirectory temporary;
  const ResolvedAuthorRun resolved =
      resolve_and_lock_author_run(author_document(temporary));
  const auto run = temporary.path() / "run";
  const auto environment = worker_environment(
      static_cast<std::uint32_t>(::geteuid()),
      static_cast<std::uint32_t>(::getegid()));
  bool rejected = false;
  try {
    (void)provision_authorized_run_directory(
        resolved.plan, environment, resolved.request_digest,
        [] { throw std::runtime_error("deterministic marker fault"); });
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected && !std::filesystem::exists(run),
        "failed post-mkdir provisioning removes only its exact new target");
  const auto seam_displaced = temporary.path() / "seam-displaced-run";
  rejected = false;
  try {
    (void)provision_authorized_run_directory(
        resolved.plan, environment, resolved.request_digest, [&] {
          std::filesystem::rename(run, seam_displaced);
          std::filesystem::create_directory(run);
          throw std::runtime_error("substitute target during marker seam");
        });
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected && std::filesystem::is_directory(run) &&
            std::filesystem::is_directory(seam_displaced),
        "marker-seam rollback never deletes a substituted pathname");
  std::filesystem::remove_all(run);
  std::filesystem::remove_all(seam_displaced);
  {
    auto pending = provision_authorized_run_directory(
        resolved.plan, environment, resolved.request_digest);
    check(pending.created(),
          "fresh provision reports authority ownership until submission");
    // Simulates cancellation/preview/create failure before a queued run is
    // durable: scope exit must remove the exact owned marker and directory.
  }
  check(!std::filesystem::exists(run),
        "non-durable provision rolls back on submit/cancellation scope exit");
  const auto displaced = temporary.path() / "displaced-run";
  {
    auto pending = provision_authorized_run_directory(
        resolved.plan, environment, resolved.request_digest);
    std::filesystem::rename(run, displaced);
    std::filesystem::create_directory(run);
    (void)::chmod(run.c_str(), 0770);
  }
  check(std::filesystem::is_directory(run) &&
            std::filesystem::is_regular_file(
                displaced / ".trainvm-authoring.json"),
        "rollback refuses a pathname substituted after directory pinning");
  std::filesystem::remove_all(run);
  std::filesystem::remove_all(displaced);
  auto first = provision_authorized_run_directory(
      resolved.plan, environment, resolved.request_digest);
  first.mark_durable();
  auto retry = provision_authorized_run_directory(
      resolved.plan, environment, resolved.request_digest);
  retry.mark_durable();
  check(std::filesystem::is_regular_file(run / ".trainvm-authoring.json"),
        "successful provisioning is idempotent and marker-bound");
}

void direct_training_rejects_forged_content_lock() {
  TemporaryDirectory temporary;
  const ResolvedAuthorRun locked =
      resolve_and_lock_author_run(author_document(temporary));
  auto forged = locked.plan.canonical_plan;
  forged["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
        ["training"] = {
      {"model_family", "rwkv"},
      {"components",
       {{"activation",
         {{"key", {{"category", "activation"},
                    {"name", "silu"},
                    {"version", "1.0.0"}}},
          {"configuration", nlohmann::json::object()}}}}},
  };
  AuthorRunDocument document{
      .api_version = std::string(kAuthorRunApiVersion),
      .source = {.experiment = std::optional<nlohmann::json>(std::move(forged)),
                 .recipe = std::nullopt},
      .input_content = std::nullopt,
      .author = "forger",
      .reason = "attempt to trust document-supplied lock",
  };
  bool rejected = false;
  try {
    (void)resolve_and_lock_author_run(document);
  } catch (const RunAuthoringError &) {
    rejected = true;
  }
  check(rejected,
        "direct training cannot forge authority provenance with embedded locks");

  auto same_path_forgery = locked.plan.canonical_plan;
  same_path_forgery["spec"]["workflow"]["nodes"]["train_to_boundary"]
                   ["invoke"]["training"] = {
      {"model_family", "rwkv"},
      {"components",
       {{"activation",
         {{"key", {{"category", "activation"},
                    {"name", "silu"},
                    {"version", "1.0.0"}}},
          {"configuration", nlohmann::json::object()}}}}},
  };
  same_path_forgery["spec"]["workspace"]["input_content_roots"][0]
                   ["tree_sha256"] = "sha256:" + std::string(64U, 'f');
  AuthorRunDocument remeasure{
      .api_version = std::string(kAuthorRunApiVersion),
      .source = {.experiment =
                     std::optional<nlohmann::json>(
                         std::move(same_path_forgery)),
                 .recipe = std::nullopt},
      .input_content = InputContentRootSet{
          .api_version = std::string(kInputContentRootSetApiVersion),
          .paths = {(temporary.path() / "input").string()}},
      .author = "forger",
      .reason = "same paths cannot make forged identities reusable",
  };
  const auto corrected = resolve_and_lock_author_run(remeasure);
  check(!corrected.content_lock_reused &&
            corrected.plan.canonical_plan["spec"]["workspace"]
                                         ["input_content_roots"][0]
                                         ["tree_sha256"] !=
                "sha256:" + std::string(64U, 'f'),
        "direct input_content always overwrites same-path forged locks");
}

void cli_preserves_authority_preview_evidence() {
  v1::AuthorRunUpdate update;
  update.set_stage(v1::AUTHOR_RUN_STAGE_COMPLETE);
  update.set_terminal(true);
  update.set_dry_run(true);
  update.set_canonical_plan_json(R"({"plan":"frozen"})");
  update.set_preflight_receipt_json(R"({"passed":true})");
  const auto output = author_run_update_json(update, "http://dashboard");
  check(output.at("canonical_plan").at("plan") == "frozen" &&
            output.at("preflight_receipt").at("passed") == true,
        "CLI NDJSON retains inspectable canonical plan and preflight receipt");
  update.set_canonical_plan_json("[]");
  bool rejected = false;
  try {
    (void)author_run_update_json(update, "http://dashboard");
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "CLI rejects malformed authority preview output");
}

v1::AuthorRunUpdate successful_authority_terminal(bool dry_run,
                                                   std::string plan_hash) {
  v1::AuthorRunUpdate update;
  update.set_stage(v1::AUTHOR_RUN_STAGE_COMPLETE);
  update.set_terminal(true);
  update.set_dry_run(dry_run);
  update.set_plan_hash(plan_hash);
  update.set_canonical_plan_json(R"({"plan":"frozen"})");
  update.set_preflight_receipt_json(
      nlohmann::json{{"passed", true}, {"plan_hash", plan_hash}}.dump());
  if (!dry_run) {
    update.mutable_run()->set_run_id("run-1");
    update.mutable_run()->set_plan_hash(plan_hash);
    update.set_dashboard_url("/api/trainvm/runs/run-1");
  }
  return update;
}

void observe_authority_progress(AuthorRunStreamValidator &validator,
                                bool dry_run) {
  for (const auto stage : {
           v1::AUTHOR_RUN_STAGE_VALIDATING,
           v1::AUTHOR_RUN_STAGE_RESOLVING,
           v1::AUTHOR_RUN_STAGE_LOCKING_INPUTS,
           v1::AUTHOR_RUN_STAGE_PREFLIGHT,
           v1::AUTHOR_RUN_STAGE_PROVISIONING,
           v1::AUTHOR_RUN_STAGE_SUBMITTING,
       }) {
    if (dry_run && stage == v1::AUTHOR_RUN_STAGE_PROVISIONING)
      break;
    v1::AuthorRunUpdate update;
    update.set_stage(stage);
    update.set_dry_run(dry_run);
    validator.observe(update);
  }
}

void cli_enforces_preview_launch_fence() {
  const std::string plan_hash = sha256_hex(R"({"plan":"frozen"})");
  AuthorRunStreamValidator preview(true);
  observe_authority_progress(preview, true);
  preview.observe(successful_authority_terminal(true, plan_hash));
  const auto frozen = preview.finish();
  check(!frozen.failed && frozen.plan_hash == plan_hash,
        "CLI captures a complete passing dry-run plan fence");

  AuthorRunStreamValidator launch(false, frozen.plan_hash);
  observe_authority_progress(launch, false);
  launch.observe(successful_authority_terminal(false, plan_hash));
  check(!launch.finish().failed,
        "CLI accepts only a launch matching its preview plan fence");

  bool rejected = false;
  try {
    AuthorRunStreamValidator drifted(false, frozen.plan_hash);
    observe_authority_progress(drifted, false);
    drifted.observe(successful_authority_terminal(false, std::string(64, 'b')));
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "CLI rejects preview-to-launch plan drift");

  rejected = false;
  try {
    AuthorRunStreamValidator trailing(true);
    observe_authority_progress(trailing, true);
    const auto terminal = successful_authority_terminal(true, plan_hash);
    trailing.observe(terminal);
    trailing.observe(terminal);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "CLI rejects authority updates after the terminal");

  v1::AuthorRunUpdate failed;
  failed.set_stage(v1::AUTHOR_RUN_STAGE_FAILED);
  failed.set_terminal(true);
  failed.set_dry_run(true);
  AuthorRunStreamValidator failed_preview(true);
  v1::AuthorRunUpdate validating;
  validating.set_stage(v1::AUTHOR_RUN_STAGE_VALIDATING);
  validating.set_dry_run(true);
  failed_preview.observe(validating);
  failed_preview.observe(failed);
  check(failed_preview.finish().failed,
        "failed preview is terminal and cannot authorize a launch");

  rejected = false;
  try {
    auto malformed = successful_authority_terminal(false, plan_hash);
    malformed.clear_dashboard_url();
    AuthorRunStreamValidator validator(false, plan_hash);
    observe_authority_progress(validator, false);
    validator.observe(malformed);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "CLI rejects a launch terminal without dashboard identity");

  rejected = false;
  try {
    AuthorRunStreamValidator regressed(true);
    v1::AuthorRunUpdate update;
    update.set_dry_run(true);
    update.set_stage(v1::AUTHOR_RUN_STAGE_VALIDATING);
    regressed.observe(update);
    update.set_stage(v1::AUTHOR_RUN_STAGE_RESOLVING);
    regressed.observe(update);
    update.set_stage(v1::AUTHOR_RUN_STAGE_VALIDATING);
    regressed.observe(update);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "CLI rejects regressing authority FSM updates");
}

} // namespace

int main() {
  try {
    rejects_duplicate_keys_recursively();
    passive_lora_selector_matches_python_module_semantics();
    resolves_locks_and_reuses_exact_lock();
    provisioning_rolls_back_then_retries();
    direct_training_rejects_forged_content_lock();
    cli_preserves_authority_preview_evidence();
    cli_enforces_preview_launch_fence();
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
    ++failures;
  }
  if (failures == 0)
    std::cout << "run authoring tests passed\n";
  return failures == 0 ? 0 : 1;
}
