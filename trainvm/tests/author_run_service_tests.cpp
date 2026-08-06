#include "trainvm/service.hpp"

#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using namespace trainvm;
constexpr std::uint64_t kGib = 1ULL << 30U;

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "trainvm-author-rpc-XXXXXX")
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
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

std::uint64_t boottime_ns() {
  timespec value{};
  if (::clock_gettime(CLOCK_BOOTTIME, &value) != 0)
    throw std::runtime_error("CLOCK_BOOTTIME unavailable");
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

OperationPortDescriptor port(OperationPortType type, bool required,
                             std::optional<ArtifactType> artifact = std::nullopt,
                             std::optional<std::string> schema = std::nullopt) {
  return {.type = type,
          .required = required,
          .artifact_type = artifact,
          .artifact_schema = std::move(schema),
          .description = std::nullopt};
}

std::vector<AdapterProfile> profiles() {
  const auto key = [](std::string adapter, ComponentRuntime runtime,
                      std::string operation, std::string contract) {
    return AdapterKey{.adapter = std::move(adapter),
                      .version = "1.0.0",
                      .runtime = runtime,
                      .operation = std::move(operation),
                      .contract = std::move(contract)};
  };
  const auto core = [&](std::string operation, std::string contract,
                        Idempotency idempotency) {
    return AdapterProfile{
        .key = key("trainvm.core", ComponentRuntime::builtin,
                   std::move(operation), std::move(contract)),
        .effect = Effect::resource,
        .idempotency = idempotency,
        .code_fingerprint = {},
        .required_capabilities = {},
        .lifecycle = {},
        .training_composition = std::nullopt,
        .authoring = OperationAuthoringDeclaration{
            .inputs = {{"concurrency_key", port(OperationPortType::string,
                                                 true)}},
            .outputs = {}}};
  };
  AdapterProfile hf{
      .key = key("rwkv-lab.hf-multimodal-sft",
                 ComponentRuntime::python_worker, "train",
                 "rwkv_lab.hf_multimodal_sft.v1.Train"),
      .effect = Effect::process,
      .idempotency = Idempotency::receipt_required,
      .code_fingerprint = "sha256:" + std::string(64U, 'b'),
      .required_capabilities = {},
      .lifecycle = {.stateful = true,
                    .graceful_stop = true,
                    .checkpoint_now = true,
                    .pause_keep_resources = true,
                    .pause_release_resources = true,
                    .compile = false,
                    .warmup = false,
                    .qualify = false,
                    .profile = false,
                    .resume_grade = ResumeGrade::exact},
      .training_composition = TrainingCompositionContract{
          .model_family = "transformer",
          .slots = {{"model_loader", TrainingComponentCategory::model_loader},
                    {"trainability", TrainingComponentCategory::trainability},
                    {"data", TrainingComponentCategory::data_source},
                    {"processor", TrainingComponentCategory::sample_processor},
                    {"sample_mapping", TrainingComponentCategory::sample_mapper},
                    {"collation", TrainingComponentCategory::collator},
                    {"sampler", TrainingComponentCategory::sampler},
                    {"batching", TrainingComponentCategory::batching},
                    {"split", TrainingComponentCategory::split_selector},
                    {"evaluation_split", TrainingComponentCategory::split_selector},
                    {"objective", TrainingComponentCategory::objective},
                    {"optimizer", TrainingComponentCategory::optimizer},
                    {"precision", TrainingComponentCategory::precision},
                    {"evaluator", TrainingComponentCategory::evaluator},
                    {"evaluation_schedule", TrainingComponentCategory::evaluation_schedule},
                    {"qualitative_samples", TrainingComponentCategory::qualitative_sample},
                    {"artifact_renderer", TrainingComponentCategory::artifact_renderer},
                    {"checkpoint_policy", TrainingComponentCategory::checkpoint_policy}},
          .allowed_components = std::nullopt},
      .authoring = OperationAuthoringDeclaration{
          .inputs = {{"manifest", port(OperationPortType::string, true)},
                     {"image_column", port(OperationPortType::string, true)},
                     {"target_column", port(OperationPortType::string, true)},
                     {"maximum_steps", port(OperationPortType::integer, true)},
                     {"step_zero_examples",
                      port(OperationPortType::integer, true)},
                     {"checkpoint_every",
                      port(OperationPortType::integer, true)},
                     {"eval_every", port(OperationPortType::integer, true)},
                     {"run_directory", port(OperationPortType::string, true)}},
          .outputs =
              {{"checkpoint",
                port(OperationPortType::artifact, false,
                     ArtifactType::checkpoint, "hf.multimodal-sft.v1")},
               {"eval_gallery", port(OperationPortType::artifact, false,
                                ArtifactType::image_gallery,
                                "rwkv-lab.eval-gallery.v2")},
               {"metrics", port(OperationPortType::artifact, false,
                                ArtifactType::metrics,
                                "hf.multimodal-sft.metrics.v1")}}}};
  return {core("acquire_resources", "trainvm.v1.AcquireResources",
               Idempotency::receipt_required),
          core("release_resources", "trainvm.v1.ReleaseResources",
               Idempotency::replay_safe),
          std::move(hf)};
}

std::vector<TrainingPreflightCheckEvidence> checks() {
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
                          TrainingPreflightCheckKind::dashboard_artifacts})
    result.push_back({.kind = kind,
                      .disposition =
                          TrainingPreflightCheckDisposition::passed,
                      .evidence_digest =
                          "sha256:" + std::string(64U, 'c'),
                      .detail = std::nullopt});
  return result;
}

class FakeExactHfProbe final : public ITrainingPreflightEvidenceProvider {
public:
  TrainingPreflightEvidenceResult collect(
      const CompiledPlan &plan,
      const std::optional<TrainingPreflightRecipeProvenance> &recipe) override {
    const std::uint64_t now = boottime_ns();
    const std::uint32_t uid = ::geteuid() == 0U
                                  ? 1000U
                                  : static_cast<std::uint32_t>(::geteuid());
    const std::uint32_t gid = ::getegid() == 0U
                                  ? 1000U
                                  : static_cast<std::uint32_t>(::getegid());
    TrainingPreflightEnvironment environment{
        .api_version = std::string(kTrainingPreflightEnvironmentApiVersion),
        .host_id = "sha256:" + std::string(64U, 'd'),
        .boot_id = "11111111-2222-4333-8444-555555555555",
        .snapshot_digest = "sha256:" + std::string(64U, 'e'),
        .snapshot_observed_monotonic_ns = now,
        .snapshot_valid_until_monotonic_ns = now + 30'000'000'000ULL,
        .evaluation_monotonic_ns = now,
        .worker_uid = uid,
        .worker_gid = gid,
        .supplementary_gids = {},
        .worker_principal_digest = "sha256:" + std::string(64U, 'f'),
        .total_host_memory_bytes = 256U * kGib,
        .available_host_memory_bytes = 192U * kGib,
        .logical_cpu_count = 64U,
        .accelerators = {{.vendor = AcceleratorVendor::nvidia,
                          .stable_id = "GPU-author-run-test",
                          .total_memory_bytes = 96U * kGib,
                          .free_memory_bytes = 90U * kGib,
                          .selector_labels = {},
                          .observation_digest =
                              "sha256:" + std::string(64U, '1')}},
        .training_nodes = {{
            .node_id = "train",
            .node_input_digest =
                training_preflight_node_input_digest(plan, "train"),
            .checks = checks(),
            .minimum_free_memory_gib = 80.0,
            .runtime_profile_digest = "sha256:" + std::string(64U, '2'),
            .required_capabilities = {},
            .provided_capabilities = {},
        }},
        .gpu_qualification = std::nullopt,
        .recipe_provenance = recipe};
    return {.environment = std::move(environment), .diagnostics = {}};
  }
};

nlohmann::json prepare_registry(const TemporaryDirectory &temporary) {
  const auto source_path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                           "docs/experiment-vm/examples/"
                           "hf-multimodal-sft.recipe-profiles.v1.json";
  auto registry = nlohmann::json::parse(read_file(source_path));
  auto &spec = registry["recipes"][0]["template_document"]["spec"];
  const auto input = temporary.path() / "input";
  const auto run = temporary.path() / "runs" / "run";
  std::filesystem::create_directory(temporary.path() / "runs");
  (void)::chmod((temporary.path() / "runs").c_str(), 0777);
  std::filesystem::create_directory(input);
  std::filesystem::create_directory(input / "model");
  std::filesystem::create_directory(input / "data");
  std::ofstream(input / "data" / "manifest.jsonl")
      << "{\"image\":\"one.png\",\"caption\":\"one\"}\n";
  const std::string png_header(
      "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
      "\x00\x00\x00\x01\x00\x00\x00\x01",
      24U);
  std::ofstream image(input / "data" / "one.png", std::ios::binary);
  image.write(png_header.data(), static_cast<std::streamsize>(png_header.size()));
  std::ofstream(input / "model" / "config.json") << "{}\n";
  std::ofstream(input / "model" / "tokenizer_config.json") << "{}\n";
  std::ofstream(input / "model" / "preprocessor_config.json") << "{}\n";
  std::ofstream(input / "model" / "tokenizer.json") << "{}\n";
  std::ofstream(input / "model" / "model.safetensors.index.json")
      << nlohmann::json{
             {"weight_map",
              {{"model.language_model.layers.3.self_attn.q_proj.weight",
                "model-00001-of-00001.safetensors"}}}}
             .dump()
      << '\n';
  (void)::chmod(input.c_str(), 0755);
  (void)::chmod((input / "model").c_str(), 0755);
  (void)::chmod((input / "data").c_str(), 0755);
  auto &data = spec["workflow"]["nodes"]["train"]["invoke"]["training"]
                   ["components"]["data"]["configuration"];
  auto &loader = spec["workflow"]["nodes"]["train"]["invoke"]["training"]
                     ["components"]["model_loader"]["configuration"];
  loader["model_path"] = (input / "model").string();
  data["manifest_path"] = (input / "data" / "manifest.jsonl").string();
  data["image_root"] = (input / "data").string();
  spec["parameters"]["manifest_path"]["value"] =
      (input / "data" / "manifest.jsonl").string();
  spec["workspace"]["root"] = temporary.path().string();
  spec["workspace"]["run_directory"] = run.string();
  spec["workspace"]["allowed_read_roots"] = {input.string()};
  spec["workspace"]["allowed_write_roots"] = {run.string()};
  spec["resources"]["minimum_host_memory_gib"] = 1;
  spec["resources"]["cpu_threads"] = 16;
  return registry;
}

std::string author_document(const TemporaryDirectory &temporary,
                            const std::filesystem::path &registry_path) {
  auto instance = nlohmann::json::parse(read_file(
      std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/qwen-caption-lora-r256.recipe-instance.v1.json"));
  const auto input = temporary.path() / "input";
  instance["overrides"]["model.path"] = (input / "model").string();
  instance["overrides"]["data.manifest"] =
      (input / "data" / "manifest.jsonl").string();
  return nlohmann::json{
      {"api_version", kAuthorRunApiVersion},
      {"source", {{"recipe", {{"registry_path", registry_path.string()},
                                {"instance", std::move(instance)}}}}},
      {"author", "author-run-service-test"},
      {"reason", "prove exact-key recipe reaches queued visibility"}}
      .dump();
}

std::vector<v1::AuthorRunUpdate> invoke(v1::TrainVM::Stub &stub,
                                       const std::string &document,
                                       bool dry_run,
                                       std::string expected_hash = {}) {
  grpc::ClientContext context;
  v1::AuthorRunRequest request;
  request.set_request_document(document);
  request.set_source_format("json");
  request.set_dry_run(dry_run);
  request.set_expected_plan_hash(std::move(expected_hash));
  auto reader = stub.AuthorRun(&context, request);
  std::vector<v1::AuthorRunUpdate> updates;
  v1::AuthorRunUpdate update;
  while (reader->Read(&update))
    updates.push_back(update);
  const grpc::Status status = reader->Finish();
  if (!status.ok())
    throw std::runtime_error("AuthorRun RPC failed: " + status.error_message());
  return updates;
}

} // namespace

int main() {
  try {
    TemporaryDirectory temporary;
    const auto registry_path = temporary.path() / "recipes.json";
    std::ofstream(registry_path) << prepare_registry(temporary).dump(2) << '\n';
    const auto components = TrainingComponentRegistry::from_json(read_file(
        std::filesystem::path(TRAINVM_SOURCE_ROOT) /
        "docs/experiment-vm/examples/training-components.v1.json"));
    TrainVMService service(
        temporary.path() / "journal.db", AdapterRegistry(profiles()),
        HostLaunchRegistry({.api_version = "trainvm.host-launches/v4",
                            .trusted_roots = {},
                            .profiles = {}}),
        {}, std::move(components), std::nullopt, {}, nullptr,
        SqliteAuthorityEnforcementGrade::cooperative_test,
        std::make_shared<FakeExactHfProbe>(), registry_path);
    grpc::ServerBuilder builder;
    int port_number = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                             &port_number);
    builder.RegisterService(static_cast<v1::TrainVM::Service *>(&service));
    auto server = builder.BuildAndStart();
    if (!server || port_number <= 0)
      throw std::runtime_error("could not start AuthorRun test server");
    auto channel = grpc::CreateChannel("127.0.0.1:" +
                                           std::to_string(port_number),
                                       grpc::InsecureChannelCredentials());
    auto stub = v1::TrainVM::NewStub(channel);
    const std::string document = author_document(temporary, registry_path);

    const auto passively_resolved = resolve_and_lock_author_run(
        decode_author_run_document(document, "json"));
    const auto exact_probe = make_hf_multimodal_sft_training_node_probe(
        [](const AdapterProfile &) {
          return PassiveRuntimeProfileEvidence{
              .profile_digest = "sha256:" + std::string(64U, '9'),
              .provided_capabilities = {},
          };
        });
    const auto exact_evidence =
        exact_probe(passively_resolved.plan, "train", profiles().back());
    if (exact_evidence.checks.size() != 10U ||
        exact_evidence.node_id != "train")
      throw std::runtime_error(
          "checked-in HF recipe did not pass the production native probe");

    const auto preview = invoke(*stub, document, true);
    const auto resolving = std::ranges::find_if(
        preview, [](const v1::AuthorRunUpdate &update) {
          return update.stage() == v1::AUTHOR_RUN_STAGE_LOCKING_INPUTS &&
                 update.detail().contains(
                     "trainvm.input-content-measurement-cache/v1");
        });
    if (preview.empty() || !preview.back().terminal() ||
        preview.back().stage() != v1::AUTHOR_RUN_STAGE_COMPLETE ||
        preview.back().plan_hash().empty() ||
        resolving == preview.end() ||
        !resolving->detail().contains(
            "trainvm.input-content-measurement-cache/v1 hits=") ||
        resolving->detail().contains(" bytes_hashed=0") ||
        !resolving->detail().contains(" elapsed_nanoseconds=") ||
        resolving->content_measurement_receipt_json().empty() ||
        nlohmann::json::parse(resolving->content_measurement_receipt_json())
                .value("plan_hash", std::string{}) !=
            preview.back().plan_hash() ||
        std::filesystem::exists(temporary.path() / "runs" / "run"))
      throw std::runtime_error(
          "dry-run mutated state or omitted frozen plan: " +
          (preview.empty()
               ? std::string("no updates")
               : preview.back().detail() + " " +
                     (resolving == preview.end()
                          ? std::string("no resolving update ")
                          : resolving->detail() + " ") +
                     (preview.back().diagnostics_size() == 0
                          ? std::string{}
                          : preview.back().diagnostics(0).code() + ":" +
                                preview.back().diagnostics(0).message())));

    std::ofstream(temporary.path() / "input" / "data" / "manifest.jsonl",
                  std::ios::app)
        << "{\"image\":\"two.png\",\"caption\":\"two\"}\n";
    const auto stale = invoke(*stub, document, false,
                              preview.back().plan_hash());
    if (stale.empty() || stale.back().stage() != v1::AUTHOR_RUN_STAGE_FAILED ||
        std::filesystem::exists(temporary.path() / "runs" / "run"))
      throw std::runtime_error("stale preview crossed the provisioning fence");

    const auto refreshed_preview = invoke(*stub, document, true);
    if (refreshed_preview.empty() ||
        refreshed_preview.back().plan_hash() == preview.back().plan_hash())
      throw std::runtime_error("content change did not alter frozen preview");

    const auto launched =
        invoke(*stub, document, false, refreshed_preview.back().plan_hash());
    const auto launched_lock =
        std::ranges::find_if(launched, [](const v1::AuthorRunUpdate &update) {
          return update.stage() == v1::AUTHOR_RUN_STAGE_LOCKING_INPUTS &&
                 update.detail().contains(
                     "trainvm.input-content-measurement-cache/v1");
        });
    const std::filesystem::path launched_directory =
        nlohmann::json::parse(launched.back().canonical_plan_json())
            .at("spec")
            .at("workspace")
            .at("run_directory")
            .get<std::string>();
    const auto launched_content_receipt =
        launched.empty() || launched.back().content_measurement_receipt_json().empty()
            ? nlohmann::json::object()
            : nlohmann::json::parse(
                  launched.back().content_measurement_receipt_json());
    if (launched.empty() ||
        launched.back().stage() != v1::AUTHOR_RUN_STAGE_COMPLETE ||
        !launched.back().has_run() || launched_lock == launched.end() ||
        !launched_lock->detail().contains(" bytes_hashed=0") ||
        launched.back().content_measurement_receipt_json().empty() ||
        !launched_content_receipt.contains("roots") ||
        launched_content_receipt.at("roots").empty() ||
        launched_content_receipt.at("cache_commit")
                .value("staged_entries", std::uint64_t{1U}) != 0U ||
        !std::filesystem::exists(launched_directory))
      throw std::runtime_error(
          "exact HF recipe did not become dashboard-visible: " +
          (launched.empty()
               ? std::string("no updates")
               : launched.back().detail() + " " +
                     (launched.back().diagnostics_size() == 0
                          ? std::string{}
                          : launched.back().diagnostics(0).code() + ":" +
                                launched.back().diagnostics(0).message())));
    v1::GetRunRequest get;
    get.set_run_id(launched.back().run().run_id());
    v1::RunSummary summary;
    grpc::ClientContext get_context;
    if (!stub->GetRun(&get_context, get, &summary).ok() ||
        summary.observed_state() != v1::OBSERVED_STATE_QUEUED)
      throw std::runtime_error("submitted AuthorRun is not durably queued");
    const auto retried =
        invoke(*stub, document, false,
               refreshed_preview.back().plan_hash());
    if (retried.empty() || !retried.back().has_run() ||
        retried.back().run().run_id() != launched.back().run().run_id())
      throw std::runtime_error("AuthorRun idempotent retry changed run identity");

    auto unsupported_registry = prepare_registry(temporary);
    auto &unsupported_spec =
        unsupported_registry["recipes"][0]["template_document"]["spec"];
    unsupported_spec["components"]["trainer"]["adapter"] =
        "unsupported.hf-trainer";
    unsupported_spec["workspace"]["run_directory"] =
        (temporary.path() / "runs" / "unsupported").string();
    unsupported_spec["workspace"]["allowed_write_roots"] = {
        (temporary.path() / "runs" / "unsupported").string()};
    std::ofstream(registry_path) << unsupported_registry.dump(2) << '\n';
    const auto unsupported = invoke(*stub, document, false,
                                    std::string(64, '0'));
    if (unsupported.empty() ||
        unsupported.back().stage() != v1::AUTHOR_RUN_STAGE_FAILED ||
        std::filesystem::exists(temporary.path() / "runs" / "unsupported"))
      throw std::runtime_error(
          "unsupported adapter crossed the pre-provision authority boundary");
    v1::ListRunsRequest list;
    v1::ListRunsResponse listed;
    grpc::ClientContext list_context;
    if (!stub->ListRuns(&list_context, list, &listed).ok() ||
        listed.runs_size() != 1)
      throw std::runtime_error(
          "unsupported adapter mutated the durable run journal");
    server->Shutdown();
    std::cout << "author run service tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "author run service test failure: " << error.what() << '\n';
    return 1;
  }
}
