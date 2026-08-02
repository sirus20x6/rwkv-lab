#include "trainvm/external_profiler_artifact.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "trainvm/document.hpp"

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const ExternalProfilerArtifactError&) {
    return;
  }
  throw std::runtime_error(message);
}

nlohmann::json window_body() {
  return {
      {"api_version", kExternalProfilerWindowApiVersion},
      {"attempt_id", "train@1"},
      {"authority_digest", "sha256:" + std::string(64U, 'a')},
      {"backend", "nsys"},
      {"capture_steps", 3U},
      {"captured_step_wall_time_us", 100.0},
      {"input_stall_us", {10.0, 20.0, 0.0}},
      {"invocation_digest", "sha256:" + std::string(64U, 'b')},
      {"node_id", "train"},
      {"optimizer_steps", {41U, 42U, 43U}},
      {"run_id", "run-1"},
      {"skip_steps", 2U},
      {"warmup_steps", 1U},
  };
}

std::string sealed_window(nlohmann::json body = window_body()) {
  body["canonical_window_digest"] =
      "sha256:" + sha256_hex(body.dump());
  return body.dump();
}

void window_receipt_is_canonical_and_content_addressed() {
  const ExternalProfilerWindow window =
      external_profiler_window_from_canonical_json(sealed_window());
  require(window.backend == ProfilerBackend::nsys &&
              window.run_id == "run-1" &&
              window.optimizer_steps ==
                  std::vector<std::uint64_t>{41U, 42U, 43U} &&
              window.input_stall_us.size() == 3U &&
              window.captured_step_wall_time_us == 100.0,
          "canonical window retains exact worker evidence");

  auto tampered = nlohmann::json::parse(sealed_window());
  tampered["optimizer_steps"][1] = 99U;
  require_throws(
      [&] {
        (void)external_profiler_window_from_canonical_json(tampered.dump());
      },
      "tampered worker window must fail its digest");
  auto duplicate = window_body();
  duplicate["optimizer_steps"] = {41U, 41U, 43U};
  require_throws(
      [&] {
        (void)external_profiler_window_from_canonical_json(
            sealed_window(std::move(duplicate)));
      },
      "duplicate optimizer step identities must be refused");
}

void normalization_unions_activity_and_groups_kernels() {
  const ExternalProfilerWindow window =
      external_profiler_window_from_canonical_json(sealed_window());
  const ExternalProfilerSummary summary = normalize_external_profiler_samples(
      window,
      {{.name = "kernel-a", .start_ns = 0U, .end_ns = 50'000U},
       {.name = "kernel-a", .start_ns = 25'000U, .end_ns = 75'000U},
       {.name = "kernel-b", .start_ns = 80'000U, .end_ns = 90'000U}});
  require(summary.accelerator_launch_count == 3U &&
              summary.kernel_or_operator_count == 2U &&
              std::abs(summary.accelerator_time_us - 110.0) < 1e-9 &&
              summary.gpu_active_time_us &&
              std::abs(*summary.gpu_active_time_us - 85.0) < 1e-9 &&
              summary.gpu_active_ratio &&
              std::abs(*summary.gpu_active_ratio - 0.85) < 1e-9 &&
              summary.input_stall_time_us &&
              std::abs(*summary.input_stall_time_us - 30.0) < 1e-9 &&
              summary.top_operators.size() == 2U &&
              summary.top_operators.front().name == "kernel-a" &&
              summary.top_operators.front().calls == 2U,
          "normalizer unions overlap and emits a bounded grouped summary");
  const auto encoded = external_profiler_summary_json(summary);
  require(encoded.at("accelerator_launch_count") == 3U &&
              encoded.at("top_operators").size() == 2U,
          "normalized summary has the dashboard contract shape");

  require_throws(
      [&] {
        (void)normalize_external_profiler_samples(
            window,
            {{.name = "too-long", .start_ns = 0U, .end_ns = 200'000U}});
      },
      "activity longer than the worker wall window must be refused");
}

void nsys_sqlite_reader_resolves_string_identities() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-nsys-normalizer-" +
       std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "trace.sqlite";
  sqlite3* database = nullptr;
  require(sqlite3_open(database_path.c_str(), &database) == SQLITE_OK,
          "test opens an NSYS SQLite fixture");
  const char* schema = R"sql(
    CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
      start INTEGER NOT NULL, end INTEGER NOT NULL,
      demangledName INTEGER NOT NULL
    );
    INSERT INTO StringIds VALUES(7, 'fused_kernel');
    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1000, 6000, 7);
    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(7000, 9000, 7);
  )sql";
  require(sqlite3_exec(database, schema, nullptr, nullptr, nullptr) ==
              SQLITE_OK,
          "test creates NSYS kernel activity tables");
  sqlite3_close(database);
  const auto samples = read_nsys_profiler_samples(database_path.string());
  require(samples ==
              std::vector<ExternalProfilerKernelSample>{
                  {.name = "fused_kernel", .start_ns = 1000U, .end_ns = 6000U},
                  {.name = "fused_kernel", .start_ns = 7000U, .end_ns = 9000U}},
          "NSYS reader resolves StringIds and preserves kernel intervals");
  std::filesystem::remove_all(directory);
}

void ncu_csv_reader_preserves_aggregate_duration_without_fake_activity() {
  auto body = window_body();
  body["backend"] = "ncu";
  const ExternalProfilerWindow window =
      external_profiler_window_from_canonical_json(sealed_window(body));
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-ncu-normalizer-" +
       std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto csv_path = directory / "trace.csv";
  {
    std::ofstream target(csv_path);
    target << "==PROF== disconnected\n"
              "\"ID\",\"Kernel Name\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"\n"
              "1,\"fused,kernel\",gpu__time_duration.sum,ns,\"12,000\"\n"
              "2,second_kernel,gpu__time_duration.sum,us,8\n"
              "2,second_kernel,sm__cycles_elapsed.sum,cycle,999\n";
  }
  const ExternalProfilerSummary summary =
      read_ncu_profiler_summary(csv_path.string(), window);
  require(summary.accelerator_launch_count == 2U &&
              summary.kernel_or_operator_count == 2U &&
              std::abs(summary.accelerator_time_us - 20.0) < 1e-9 &&
              !summary.gpu_active_time_us && !summary.gpu_active_ratio &&
              summary.top_operators.size() == 2U,
          "NCU summary keeps durations but does not invent timeline overlap");
  const auto encoded = external_profiler_summary_json(summary);
  require(!encoded.contains("gpu_active_ratio") &&
              !encoded.contains("gpu_active_time_us"),
          "NCU JSON omits metrics its report cannot prove");
  std::filesystem::remove_all(directory);
}

void external_publication_is_immutable_content_addressed_and_replayable() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-external-publication-" +
       std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  const auto external =
      directory / "trainvm_artifacts" / "gpu_traces" / ".external";
  std::filesystem::create_directories(external);
  const std::string launch_id = "run-1:worker-launch:train:train@1";
  const auto prefix = external / sha256_hex(launch_id);
  {
    std::ofstream target(prefix.string() + ".window.json");
    target << sealed_window();
  }
  sqlite3* database = nullptr;
  const std::string sqlite_path = prefix.string() + ".sqlite";
  require(sqlite3_open(sqlite_path.c_str(), &database) == SQLITE_OK,
          "publication fixture opens an NSYS export");
  const char* schema = R"sql(
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
      start INTEGER NOT NULL, end INTEGER NOT NULL,
      demangledName TEXT NOT NULL
    );
    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1000, 6000, 'kernel-a');
  )sql";
  require(sqlite3_exec(database, schema, nullptr, nullptr, nullptr) ==
              SQLITE_OK,
          "publication fixture creates kernel activity");
  sqlite3_close(database);

  GpuTraceCapture capture{
      .enabled = true,
      .backend = ProfilerBackend::nsys,
      .warmup_steps = 1,
      .skip_steps = 2,
      .capture_steps = 3,
      .output_artifact = "gpu_trace",
      .activities = std::vector<ProfilerActivity>{
          ProfilerActivity::accelerator},
      .record_shapes = std::nullopt,
      .profile_memory = std::nullopt,
      .with_stack = std::nullopt,
  };
  const ExternalProfilerPublicationRequest request{
      .backend = ProfilerBackend::nsys,
      .run_id = "run-1",
      .node_id = "train",
      .attempt_id = "train@1",
      .authority_digest = "sha256:" + std::string(64U, 'a'),
      .invocation_digest = "sha256:" + std::string(64U, 'b'),
      .capture = capture,
      .raw_output_prefix = prefix,
      .run_directory = directory,
  };
  const auto first = publish_external_profiler_artifact(request);
  const auto replay = publish_external_profiler_artifact(request);
  require(first == replay && first.artifact_id.starts_with("gpu-trace-") &&
              std::filesystem::is_regular_file(first.manifest_path) &&
              std::filesystem::is_regular_file(
                  first.manifest_path.parent_path() / "trace.sqlite"),
          "external publication is content-addressed and replayable");
  const auto manifest = nlohmann::json::parse(
      std::ifstream(first.manifest_path));
  require(manifest.at("backend") == "nsys" &&
              manifest.at("trace_file_name") == "trace.sqlite" &&
              manifest.at("first_optimizer_step") == 41U &&
              manifest.at("last_optimizer_step") == 43U &&
              manifest.at("canonical_manifest_digest")
                  .get<std::string>()
                  .starts_with("sha256:"),
          "published manifest retains the exact capture identity");
  std::filesystem::permissions(
      first.manifest_path.parent_path(),
      std::filesystem::perms::owner_all,
      std::filesystem::perm_options::add);
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  try {
    window_receipt_is_canonical_and_content_addressed();
    normalization_unions_activity_and_groups_kernels();
    nsys_sqlite_reader_resolves_string_identities();
    ncu_csv_reader_preserves_aggregate_duration_without_fake_activity();
    external_publication_is_immutable_content_addressed_and_replayable();
    std::cout << "external profiler artifact tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "external profiler artifact test failure: " << error.what()
              << '\n';
    return 1;
  }
}
