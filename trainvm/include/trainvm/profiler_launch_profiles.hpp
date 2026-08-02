#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"

namespace trainvm {

inline constexpr std::string_view kProfilerLaunchProfileApiVersion =
    "trainvm.profiler-launch-profile/v1";

class ProfilerLaunchProfileError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// What a produced trace artifact may be shown to.
//
// A raw Nsight capture is not neutral telemetry: it carries kernel names,
// tensor shapes, file paths, and dataset identity, which for this project can
// include private and adult corpora. A normalized summary can be publishable
// while the capture it came from is not, so the two are classified separately
// rather than inheriting one policy.
enum class TraceSensitivity {
  publishable,
  restricted,
};

// The exact command an authority may run for one backend. Every element is
// either a literal owned by this table or a bounded integer taken from an
// already-validated GpuTraceCapture. An experiment document contributes no
// string to argv and nothing at all to the environment.
struct ProfilerLaunchProfile final {
  ProfilerBackend backend{};
  std::string version;
  // Program name only. Resolution to a path is the host's job, not the
  // document's, so a document cannot select which binary runs.
  std::string program;
  std::vector<std::string> fixed_arguments;
  TraceSensitivity raw_capture_sensitivity{TraceSensitivity::restricted};
  TraceSensitivity summary_sensitivity{TraceSensitivity::publishable};
  // Instrumented timing is never usable as qualification timing, because the
  // instrumentation perturbs what it measures.
  bool timing_is_qualification_grade{};
  std::string summary;

  bool operator==(const ProfilerLaunchProfile&) const = default;
};

struct ProfilerLaunchProfileDocument final {
  std::string api_version{std::string(kProfilerLaunchProfileApiVersion)};
  std::vector<ProfilerLaunchProfile> profiles;
  std::string document_digest;

  bool operator==(const ProfilerLaunchProfileDocument&) const = default;
};

[[nodiscard]] std::string_view profiler_backend_name(ProfilerBackend backend);
[[nodiscard]] const ProfilerLaunchProfileDocument& profiler_launch_profiles();
[[nodiscard]] const ProfilerLaunchProfile& profiler_launch_profile(
    ProfilerBackend backend);

// Derives the exact argv for a capture. The output path is supplied by the
// artifact authority, never by the document, and is re-checked here so a
// hostile or careless path cannot reach a command line.
//
// Returns argv only. There is deliberately no environment overload: a capture
// that could set environment variables could redirect libraries, injection
// hooks, or credentials, and no declared use needs it.
[[nodiscard]] std::vector<std::string> profiler_capture_argv(
    const GpuTraceCapture& capture, std::string_view output_path);

// True only for a path this authority is willing to place on a command line:
// absolute, bounded, no shell metacharacters, no argument-looking prefix.
[[nodiscard]] bool profiler_output_path_is_admissible(std::string_view path);

[[nodiscard]] TraceSensitivity profiler_raw_capture_sensitivity(
    ProfilerBackend backend);

[[nodiscard]] nlohmann::json profiler_launch_profiles_json(
    const ProfilerLaunchProfileDocument& document);
[[nodiscard]] std::string profiler_launch_profiles_digest(
    const ProfilerLaunchProfileDocument& document);

}  // namespace trainvm
