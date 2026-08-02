#include "trainvm/profiler_launch_profiles.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumOutputPathBytes = 4096U;
constexpr std::int64_t kMaximumStep = 1'000'000'000LL;

std::string hex_digest(std::string_view material) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), material.data(), material.size()) != 1)
    throw ProfilerLaunchProfileError("profiler launch profile digest failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> value{};
  unsigned int size = 0U;
  if (EVP_DigestFinal_ex(context.get(), value.data(), &size) != 1 || size != 32U)
    throw ProfilerLaunchProfileError(
        "profiler launch profile digest final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  for (unsigned int index = 0U; index < size; ++index) {
    result.push_back(digits[value[index] >> 4U]);
    result.push_back(digits[value[index] & 0x0fU]);
  }
  return result;
}

std::vector<ProfilerLaunchProfile> build_profiles() {
  std::vector<ProfilerLaunchProfile> profiles;

  profiles.push_back(
      {.backend = ProfilerBackend::torch,
       .version = "1.0.0",
       // In-process. The worker imports the profiler; no child is spawned, so
       // there is no command line for a document to influence at all.
       .program = {},
       .fixed_arguments = {},
       .raw_capture_sensitivity = TraceSensitivity::restricted,
       .summary_sensitivity = TraceSensitivity::publishable,
       .timing_is_qualification_grade = false,
       .summary = "In-process Torch profiler; the worker owns the capture."});

  profiles.push_back(
      {.backend = ProfilerBackend::nsys,
       .version = "1.0.0",
       .program = "nsys",
       // --force-overwrite is deliberately absent: the artifact authority
       // owns a fresh output path, so a capture that would overwrite is a
       // bug to surface rather than a condition to suppress.
       .fixed_arguments = {"profile", "--sample=none", "--trace=cuda,nvtx",
                           "--cuda-memory-usage=true", "--export=sqlite"},
       .raw_capture_sensitivity = TraceSensitivity::restricted,
       .summary_sensitivity = TraceSensitivity::publishable,
       .timing_is_qualification_grade = false,
       .summary = "Nsight Systems timeline capture over a bounded step range."});

  profiles.push_back(
      {.backend = ProfilerBackend::ncu,
       .version = "1.0.0",
       .program = "ncu",
       .fixed_arguments = {"--set=full", "--target-processes=application-only"},
       .raw_capture_sensitivity = TraceSensitivity::restricted,
       .summary_sensitivity = TraceSensitivity::publishable,
       .timing_is_qualification_grade = false,
       .summary = "Nsight Compute kernel capture; accelerator activity only."});

  return profiles;
}

const ProfilerLaunchProfileDocument& sealed_document() {
  static const ProfilerLaunchProfileDocument value = [] {
    ProfilerLaunchProfileDocument document;
    document.profiles = build_profiles();
    std::vector<int> seen;
    for (const ProfilerLaunchProfile& profile : document.profiles) {
      if (std::ranges::find(seen, static_cast<int>(profile.backend)) !=
          seen.end())
        throw ProfilerLaunchProfileError(
            "a profiler backend is declared more than once");
      seen.push_back(static_cast<int>(profile.backend));
      if (profile.version.empty() || profile.summary.empty())
        throw ProfilerLaunchProfileError(
            "profiler launch profile is missing its version or summary");
      // Instrumentation perturbs latency. A profile that claimed its timing
      // was qualification-grade would let a profiled run stand in for an
      // unprofiled baseline, which is the measurement error this whole area
      // exists to prevent.
      if (profile.timing_is_qualification_grade)
        throw ProfilerLaunchProfileError(
            "profiled timing may never be qualification grade");
      if (profile.raw_capture_sensitivity != TraceSensitivity::restricted)
        throw ProfilerLaunchProfileError(
            "a raw capture may never be declared publishable");
      // Every fixed argument must be a self-contained literal, so a future
      // edit cannot smuggle an interpolated value in beside the owned ones.
      // Requiring a leading '-' would be the wrong proxy: `nsys profile`
      // needs its subcommand. What actually matters is that nothing here can
      // carry whitespace or shell syntax.
      for (const std::string& argument : profile.fixed_arguments) {
        const bool clean = !argument.empty() &&
            std::ranges::all_of(argument, [](char character) {
              const bool alphanumeric =
                  (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9');
              return alphanumeric || character == '-' || character == '=' ||
                     character == ',' || character == '.' || character == '_';
            });
        if (!clean)
          throw ProfilerLaunchProfileError(
              "profiler fixed arguments must be self-contained literals");
      }
      if (profile.backend == ProfilerBackend::torch &&
          !(profile.program.empty() && profile.fixed_arguments.empty()))
        throw ProfilerLaunchProfileError(
            "the in-process torch profiler must not declare a command");
      if (profile.backend != ProfilerBackend::torch && profile.program.empty())
        throw ProfilerLaunchProfileError(
            "an out-of-process profiler must declare its program");
    }
    document.document_digest = profiler_launch_profiles_digest(document);
    return document;
  }();
  return value;
}

}  // namespace

std::string_view profiler_backend_name(ProfilerBackend backend) {
  switch (backend) {
    case ProfilerBackend::torch:
      return "torch";
    case ProfilerBackend::nsys:
      return "nsys";
    case ProfilerBackend::ncu:
      return "ncu";
  }
  throw ProfilerLaunchProfileError("unknown profiler backend");
}

const ProfilerLaunchProfileDocument& profiler_launch_profiles() {
  return sealed_document();
}

const ProfilerLaunchProfile& profiler_launch_profile(ProfilerBackend backend) {
  for (const ProfilerLaunchProfile& profile : sealed_document().profiles) {
    if (profile.backend == backend) return profile;
  }
  throw ProfilerLaunchProfileError("profiler backend is not registered");
}

TraceSensitivity profiler_raw_capture_sensitivity(ProfilerBackend backend) {
  return profiler_launch_profile(backend).raw_capture_sensitivity;
}

bool profiler_output_path_is_admissible(std::string_view path) {
  if (path.empty() || path.size() > kMaximumOutputPathBytes) return false;
  // Absolute only: a relative path resolves against whatever directory the
  // capture happens to start in.
  if (path.front() != '/') return false;
  // A path beginning with '-' would be read as an option by the profiler.
  if (path.find("/-") != std::string_view::npos) return false;
  if (path.find("..") != std::string_view::npos) return false;
  return std::ranges::all_of(path, [](char character) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    return alphanumeric || character == '/' || character == '.' ||
           character == '_' || character == '-';
  });
}

std::vector<std::string> profiler_capture_argv(const GpuTraceCapture& capture,
                                               std::string_view output_path) {
  if (!capture.enabled)
    throw ProfilerLaunchProfileError(
        "a disabled trace capture has no invocation");
  if (!capture.backend)
    throw ProfilerLaunchProfileError("trace capture declares no backend");
  const ProfilerLaunchProfile& profile =
      profiler_launch_profile(*capture.backend);
  if (profile.program.empty())
    throw ProfilerLaunchProfileError(
        "the in-process torch profiler has no command line");
  if (!profiler_output_path_is_admissible(output_path))
    throw ProfilerLaunchProfileError(
        "trace output path is not admissible on a command line");

  const auto bounded = [](const std::optional<std::int64_t>& value,
                          std::string_view what) {
    if (!value) return std::int64_t{0};
    if (*value < 0 || *value > kMaximumStep)
      throw ProfilerLaunchProfileError(std::string(what) +
                                       " is outside its declared bound");
    return *value;
  };
  const std::int64_t warmup = bounded(capture.warmup_steps, "warmup_steps");
  const std::int64_t skip = bounded(capture.skip_steps, "skip_steps");
  const std::int64_t capture_steps =
      bounded(capture.capture_steps, "capture_steps");

  std::vector<std::string> argv;
  argv.push_back(profile.program);
  for (const std::string& argument : profile.fixed_arguments)
    argv.push_back(argument);

  // Only integers reach the command line from the document, and only after
  // the bound check above. The strings around them are ours.
  if (*capture.backend == ProfilerBackend::nsys) {
    argv.emplace_back("--output");
    argv.emplace_back(output_path);
    if (skip > 0 || warmup > 0) {
      argv.emplace_back("--capture-range=cudaProfilerApi");
      argv.emplace_back("--capture-range-end=stop");
    }
  } else {
    argv.emplace_back("--export");
    argv.emplace_back(output_path);
    argv.emplace_back("--launch-skip");
    argv.emplace_back(std::to_string(skip + warmup));
    if (capture_steps > 0) {
      argv.emplace_back("--launch-count");
      argv.emplace_back(std::to_string(capture_steps));
    }
  }
  return argv;
}

nlohmann::json profiler_launch_profiles_json(
    const ProfilerLaunchProfileDocument& document) {
  return encode_json(document);
}

std::string profiler_launch_profiles_digest(
    const ProfilerLaunchProfileDocument& document) {
  ProfilerLaunchProfileDocument material = document;
  material.document_digest.clear();
  return hex_digest(profiler_launch_profiles_json(material).dump());
}

}  // namespace trainvm
