#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/model.hpp"

namespace trainvm {

inline constexpr std::string_view kExternalProfilerWindowApiVersion =
    "trainvm.external-profiler-window/v1";

class ExternalProfilerArtifactError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ExternalProfilerWindow final {
  ProfilerBackend backend{};
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string authority_digest;
  std::string invocation_digest;
  std::uint64_t skip_steps{};
  std::uint64_t warmup_steps{};
  std::uint64_t capture_steps{};
  double captured_step_wall_time_us{};
  std::vector<std::uint64_t> optimizer_steps;
  std::vector<std::optional<double>> input_stall_us;
  std::string canonical_window_digest;

  bool operator==(const ExternalProfilerWindow&) const = default;
};

struct ExternalProfilerKernelSample final {
  std::string name;
  std::uint64_t start_ns{};
  std::uint64_t end_ns{};

  bool operator==(const ExternalProfilerKernelSample&) const = default;
};

struct ExternalProfilerOperatorSummary final {
  std::string name;
  std::uint64_t calls{};
  double cpu_time_us{};
  double accelerator_time_us{};

  bool operator==(const ExternalProfilerOperatorSummary&) const = default;
};

struct ExternalProfilerSummary final {
  double cpu_time_us{};
  double accelerator_time_us{};
  std::uint64_t kernel_or_operator_count{};
  std::uint64_t accelerator_launch_count{};
  double captured_step_wall_time_us{};
  std::optional<double> gpu_active_ratio;
  std::optional<double> gpu_active_time_us;
  std::optional<double> input_stall_ratio;
  std::optional<double> input_stall_time_us;
  std::vector<ExternalProfilerOperatorSummary> top_operators;

  bool operator==(const ExternalProfilerSummary&) const = default;
};

struct ExternalProfilerPublicationRequest final {
  ProfilerBackend backend{};
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string authority_digest;
  std::string invocation_digest;
  GpuTraceCapture capture;
  std::filesystem::path raw_output_prefix;
  std::filesystem::path run_directory;

  bool operator==(const ExternalProfilerPublicationRequest&) const = default;
};

struct ExternalProfilerPublishedArtifact final {
  std::string artifact_id;
  std::filesystem::path manifest_path;
  std::string manifest_fingerprint;
  std::uint64_t size_bytes{};

  bool operator==(const ExternalProfilerPublishedArtifact&) const = default;
};

[[nodiscard]] ExternalProfilerWindow
external_profiler_window_from_canonical_json(std::string_view value);

[[nodiscard]] ExternalProfilerSummary normalize_external_profiler_samples(
    const ExternalProfilerWindow& window,
    std::vector<ExternalProfilerKernelSample> samples);

[[nodiscard]] std::vector<ExternalProfilerKernelSample>
read_nsys_profiler_samples(const std::string& sqlite_path);

[[nodiscard]] ExternalProfilerSummary read_ncu_profiler_summary(
    const std::string& csv_path, const ExternalProfilerWindow& window);

[[nodiscard]] nlohmann::json external_profiler_summary_json(
    const ExternalProfilerSummary& summary);

[[nodiscard]] ExternalProfilerPublishedArtifact
publish_external_profiler_artifact(
    const ExternalProfilerPublicationRequest& request);

}  // namespace trainvm
