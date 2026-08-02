#include "trainvm/profiler_launch_profiles.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const ProfilerLaunchProfileError&) {
    return;
  }
  throw std::runtime_error(message);
}

GpuTraceCapture capture(ProfilerBackend backend) {
  GpuTraceCapture value;
  value.enabled = true;
  value.backend = backend;
  value.warmup_steps = 5;
  value.skip_steps = 10;
  value.capture_steps = 3;
  return value;
}

// The card's first property: a capture cannot inject argv. Every element of a
// derived command must be either a literal this table owns or a digit string,
// so nothing an experiment document can write reaches the command line as text.
void a_capture_cannot_inject_arguments() {
  for (const auto backend : {ProfilerBackend::nsys, ProfilerBackend::ncu}) {
    const auto argv = profiler_capture_argv(capture(backend),
                                            "/var/lib/trainvm/traces/run-1");
    require(!argv.empty(), "a derived invocation is not empty");
    require(argv.front() == std::string(profiler_backend_name(backend)),
            "the invocation names its declared program");

    const ProfilerLaunchProfile& profile = profiler_launch_profile(backend);
    for (const std::string& element : argv) {
      const bool is_program = element == profile.program;
      const bool is_fixed = std::ranges::find(profile.fixed_arguments,
                                              element) !=
                            profile.fixed_arguments.end();
      const bool is_our_option = !element.empty() && element.front() == '-';
      const bool is_digits =
          !element.empty() && std::ranges::all_of(element, [](char character) {
            return character >= '0' && character <= '9';
          });
      const bool is_output_path = element.starts_with("/var/lib/trainvm/");
      require(is_program || is_fixed || is_our_option || is_digits ||
                  is_output_path,
              "an invocation element is neither an owned literal, a bounded "
              "integer, nor the authority's output path: " + element);
    }
  }
}

// The output path is the one string that reaches argv, and it comes from the
// artifact authority. Anything that could be read as an option, escape a
// directory, or reach a shell must be refused before it gets there.
void a_hostile_output_path_never_reaches_a_command_line() {
  const std::vector<std::string> refused{
      "",
      "relative/path",
      "/var/lib/trainvm/-oh-no",
      "/var/lib/trainvm/../../etc/shadow",
      "/var/lib/trainvm/trace;rm -rf /",
      "/var/lib/trainvm/trace$(whoami)",
      "/var/lib/trainvm/trace`id`",
      "/var/lib/trainvm/trace with spaces",
      "/var/lib/trainvm/trace\nsecond-line",
      "/var/lib/trainvm/trace|tee",
      "/var/lib/trainvm/trace&",
      "/var/lib/trainvm/trace>out",
      std::string("/var/lib/trainvm/") + std::string(8192U, 'x'),
  };
  for (const std::string& path : refused) {
    require(!profiler_output_path_is_admissible(path),
            "a hostile output path was admitted: " + path);
    require_throws(
        [&] { (void)profiler_capture_argv(capture(ProfilerBackend::nsys),
                                          path); },
        "a hostile output path reached a derived invocation: " + path);
  }
  require(profiler_output_path_is_admissible("/var/lib/trainvm/traces/run-1"),
          "an ordinary authority path must still be admitted");
  require(profiler_output_path_is_admissible("/tmp/t.nsys-rep"),
          "a path with a dot and a dash must still be admitted");
}

// An out-of-range step count must be refused rather than truncated onto the
// command line, which is the same class of hole the JSON decoder had.
void out_of_range_step_counts_are_refused() {
  auto oversized = capture(ProfilerBackend::ncu);
  oversized.capture_steps = 1'000'000'001LL;
  require_throws(
      [&] { (void)profiler_capture_argv(oversized, "/tmp/trace"); },
      "an out-of-range capture_steps must be refused");

  auto negative = capture(ProfilerBackend::ncu);
  negative.skip_steps = -1;
  require_throws([&] { (void)profiler_capture_argv(negative, "/tmp/trace"); },
                 "a negative skip_steps must be refused");
}

// The card's second property. A raw capture carries kernel names, tensor
// shapes, file paths, and dataset identity, which here can include private
// corpora; only the normalized summary may be publishable.
void trace_sensitivity_is_declared_and_restrictive() {
  const auto& document = profiler_launch_profiles();
  require(document.profiles.size() == 3U,
          "every declared profiler backend has a launch profile");
  require(!document.document_digest.empty(),
          "the launch profile document is sealed with a digest");
  require(document.document_digest ==
              profiler_launch_profiles_digest(document),
          "the launch profile digest binds its content");

  for (const auto& profile : document.profiles) {
    require(profile.raw_capture_sensitivity == TraceSensitivity::restricted,
            std::string(profiler_backend_name(profile.backend)) +
                " declares a publishable raw capture");
    require(!profile.timing_is_qualification_grade,
            std::string(profiler_backend_name(profile.backend)) +
                " claims profiled timing is qualification grade");
  }
  for (const auto backend :
       {ProfilerBackend::torch, ProfilerBackend::nsys, ProfilerBackend::ncu}) {
    require(profiler_raw_capture_sensitivity(backend) ==
                TraceSensitivity::restricted,
            "a raw capture is restricted for every backend");
  }
}

// The in-process backend must have no command line at all, so there is nothing
// to inject into rather than a command that is merely well-guarded.
void the_in_process_backend_has_no_command_line() {
  const ProfilerLaunchProfile& torch =
      profiler_launch_profile(ProfilerBackend::torch);
  require(torch.program.empty() && torch.fixed_arguments.empty(),
          "the torch profiler must declare no command");
  require_throws(
      [] {
        (void)profiler_capture_argv(capture(ProfilerBackend::torch),
                                    "/tmp/trace");
      },
      "the in-process backend must refuse to produce an invocation");
}

void a_disabled_or_backendless_capture_has_no_invocation() {
  auto disabled = capture(ProfilerBackend::nsys);
  disabled.enabled = false;
  require_throws([&] { (void)profiler_capture_argv(disabled, "/tmp/t"); },
                 "a disabled capture must have no invocation");

  auto backendless = capture(ProfilerBackend::nsys);
  backendless.backend = std::nullopt;
  require_throws([&] { (void)profiler_capture_argv(backendless, "/tmp/t"); },
                 "a capture with no backend must have no invocation");
}

}  // namespace

int main() {
  try {
    a_capture_cannot_inject_arguments();
    a_hostile_output_path_never_reaches_a_command_line();
    out_of_range_step_counts_are_refused();
    trace_sensitivity_is_declared_and_restrictive();
    the_in_process_backend_has_no_command_line();
    a_disabled_or_backendless_capture_has_no_invocation();
    std::cout << "profiler launch profile tests passed ("
              << profiler_launch_profiles().profiles.size() << " backends)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "profiler launch profile test failure: " << error.what()
              << '\n';
    return 1;
  }
}
