#include "trainvm/external_profiler_artifact.hpp"

#include <cmath>
#include <filesystem>
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
              std::abs(summary.gpu_active_time_us - 85.0) < 1e-9 &&
              std::abs(summary.gpu_active_ratio - 0.85) < 1e-9 &&
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

}  // namespace

int main() {
  try {
    window_receipt_is_canonical_and_content_addressed();
    normalization_unions_activity_and_groups_kernels();
    nsys_sqlite_reader_resolves_string_identities();
    std::cout << "external profiler artifact tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "external profiler artifact test failure: " << error.what()
              << '\n';
    return 1;
  }
}
