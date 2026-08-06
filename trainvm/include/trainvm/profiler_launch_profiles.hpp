#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/model.hpp"

namespace trainvm {

inline constexpr std::string_view kProfilerLaunchProfileApiVersion =
    "trainvm.profiler-launch-profile/v1";
inline constexpr std::string_view kExternalProfilerAuthorityApiVersion =
    "trainvm.external-profiler-authority/v1";
inline constexpr std::size_t kMaximumExternalProfilerAuthorityBytes =
    16U * 1024U;

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

// External collectors observe kernels, not optimizer steps. The worker loop is
// therefore the only authority that can advance an exact capture window. This
// state machine is runtime-neutral: Python workers map start/stop to the CUDA
// profiler API and the fixed TrainVM NVTX range, while LibTorch workers use the
// same transitions from C++.
enum class ProfilerLifecycleAction {
  none,
  start_capture,
  stop_capture,
};

class ProfilerStepLifecycle final {
 public:
  explicit ProfilerStepLifecycle(const GpuTraceCapture& capture);

  // Called immediately before the training loop. A zero-length leading window
  // starts here so the first optimizer update is included.
  [[nodiscard]] ProfilerLifecycleAction enter();

  // Called exactly once after each successful optimizer update. Step values
  // are retained as evidence and must be strictly increasing.
  [[nodiscard]] ProfilerLifecycleAction optimizer_step(
      std::uint64_t optimizer_step);

  // A successful worker may not leave before the declared window completed.
  // An exceptional worker is allowed to stop an active collector, but its
  // incomplete window is not publication-grade.
  [[nodiscard]] ProfilerLifecycleAction finish(bool exceptional);

  [[nodiscard]] bool capture_active() const noexcept;
  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] const std::vector<std::uint64_t>& captured_steps()
      const noexcept;

 private:
  std::uint64_t capture_begin_{};
  std::uint64_t capture_end_{};
  std::uint64_t steps_seen_{};
  std::optional<std::uint64_t> last_optimizer_step_;
  std::vector<std::uint64_t> captured_steps_;
  bool entered_{};
  bool active_{};
  bool finished_{};
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

// Immutable proof inherited by the target at fd 5. It lets the training loop
// distinguish a genuinely host-wrapped process from a document that merely
// selected `nsys` or `ncu`.
struct ExternalProfilerAuthoritySpec final {
  std::string api_version;
  ProfilerBackend backend{};
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_profile_digest;
  std::string profiler_executable_digest;
  std::string authority_digest;

  bool operator==(const ExternalProfilerAuthoritySpec&) const = default;
};

[[nodiscard]] ExternalProfilerAuthoritySpec seal_external_profiler_authority(
    ExternalProfilerAuthoritySpec value);
[[nodiscard]] std::string external_profiler_authority_canonical_json(
    const ExternalProfilerAuthoritySpec& value);
[[nodiscard]] ExternalProfilerAuthoritySpec
external_profiler_authority_from_canonical_json(std::string_view value);

class SealedExternalProfilerAuthority final {
 public:
  SealedExternalProfilerAuthority(SealedExternalProfilerAuthority&&) noexcept;
  SealedExternalProfilerAuthority& operator=(
      SealedExternalProfilerAuthority&&) noexcept;
  ~SealedExternalProfilerAuthority();

  SealedExternalProfilerAuthority(const SealedExternalProfilerAuthority&) =
      delete;
  SealedExternalProfilerAuthority& operator=(
      const SealedExternalProfilerAuthority&) = delete;

  [[nodiscard]] const ExternalProfilerAuthoritySpec& spec() const;
  [[nodiscard]] int duplicate_fd() const;

 private:
  friend SealedExternalProfilerAuthority
  create_sealed_external_profiler_authority(
      ExternalProfilerAuthoritySpec value);
  SealedExternalProfilerAuthority(ExternalProfilerAuthoritySpec spec,
                                  int descriptor) noexcept;
  ExternalProfilerAuthoritySpec spec_;
  int descriptor_{-1};
};

[[nodiscard]] SealedExternalProfilerAuthority
create_sealed_external_profiler_authority(
    ExternalProfilerAuthoritySpec value);
[[nodiscard]] ExternalProfilerAuthoritySpec
external_profiler_authority_from_sealed_fd(
    int descriptor, std::string_view expected_digest = {});

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
