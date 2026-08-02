#include "trainvm/profiler_launch_profiles.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/memfd.h>
#include <limits>
#include <memory>
#include <ranges>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumOutputPathBytes = 4096U;
constexpr std::int64_t kMaximumStep = 1'000'000'000LL;
constexpr int kRequiredAuthoritySeals =
    F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;

bool bounded_authority_text(std::string_view value, std::size_t maximum) {
  return !value.empty() && value.size() <= maximum && !value.contains('\0') &&
         !value.contains('\n') && !value.contains('\r');
}

bool valid_authority_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

ProfilerBackend external_backend(std::string_view value) {
  if (value == "nsys") return ProfilerBackend::nsys;
  if (value == "ncu") return ProfilerBackend::ncu;
  throw ProfilerLaunchProfileError(
      "external profiler authority backend is invalid");
}

nlohmann::json authority_body(const ExternalProfilerAuthoritySpec& value) {
  if (value.api_version != kExternalProfilerAuthorityApiVersion ||
      (value.backend != ProfilerBackend::nsys &&
       value.backend != ProfilerBackend::ncu) ||
      !bounded_authority_text(value.run_id, 1024U) ||
      !bounded_authority_text(value.node_id, 1024U) ||
      !bounded_authority_text(value.attempt_id, 1024U) ||
      !valid_authority_digest(value.launch_profile_digest) ||
      !valid_authority_digest(value.profiler_executable_digest)) {
    throw ProfilerLaunchProfileError(
        "external profiler authority semantics are invalid");
  }
  return {{"api_version", value.api_version},
          {"attempt_id", value.attempt_id},
          {"backend", std::string(profiler_backend_name(value.backend))},
          {"launch_profile_digest", value.launch_profile_digest},
          {"node_id", value.node_id},
          {"profiler_executable_digest", value.profiler_executable_digest},
          {"run_id", value.run_id}};
}

void write_authority(int descriptor, std::string_view value) {
  std::size_t offset = 0U;
  while (offset < value.size()) {
    const ssize_t count =
        ::write(descriptor, value.data() + offset, value.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0)
      throw ProfilerLaunchProfileError(
          "could not write external profiler authority");
    offset += static_cast<std::size_t>(count);
  }
}

void exact_authority_fields(const nlohmann::json& value) {
  constexpr std::array<std::string_view, 8U> fields{
      "api_version",          "attempt_id",
      "authority_digest",     "backend",
      "launch_profile_digest", "node_id",
      "profiler_executable_digest", "run_id"};
  if (!value.is_object() || value.size() != fields.size() ||
      std::ranges::any_of(fields, [&](std::string_view field) {
        return !value.contains(std::string(field));
      })) {
    throw ProfilerLaunchProfileError(
        "external profiler authority fields are inexact");
  }
}

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
       .fixed_arguments = {"--set=full", "--target-processes=application-only",
                           "--nvtx", "--nvtx-include",
                           "trainvm.profile.capture"},
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

ExternalProfilerAuthoritySpec seal_external_profiler_authority(
    ExternalProfilerAuthoritySpec value) {
  value.authority_digest.clear();
  const std::string body = authority_body(value).dump();
  if (body.size() > kMaximumExternalProfilerAuthorityBytes) {
    throw ProfilerLaunchProfileError(
        "external profiler authority exceeds its canonical size bound");
  }
  value.authority_digest = hex_digest(body);
  return value;
}

std::string external_profiler_authority_canonical_json(
    const ExternalProfilerAuthoritySpec& value) {
  const ExternalProfilerAuthoritySpec canonical =
      seal_external_profiler_authority(value);
  if (canonical.authority_digest != value.authority_digest) {
    throw ProfilerLaunchProfileError(
        "external profiler authority digest is not canonical");
  }
  nlohmann::json output = authority_body(value);
  output["authority_digest"] = value.authority_digest;
  const std::string encoded = output.dump();
  if (encoded.size() > kMaximumExternalProfilerAuthorityBytes) {
    throw ProfilerLaunchProfileError(
        "external profiler authority exceeds its wire size bound");
  }
  return encoded;
}

ExternalProfilerAuthoritySpec external_profiler_authority_from_canonical_json(
    std::string_view value) {
  if (value.empty() || value.size() > kMaximumExternalProfilerAuthorityBytes) {
    throw ProfilerLaunchProfileError(
        "external profiler authority JSON size is invalid");
  }
  try {
    const nlohmann::json parsed = nlohmann::json::parse(value);
    if (parsed.dump() != value) {
      throw ProfilerLaunchProfileError(
          "external profiler authority JSON is not canonical");
    }
    exact_authority_fields(parsed);
    ExternalProfilerAuthoritySpec result{
        .api_version = parsed.at("api_version").get<std::string>(),
        .backend = external_backend(parsed.at("backend").get<std::string>()),
        .run_id = parsed.at("run_id").get<std::string>(),
        .node_id = parsed.at("node_id").get<std::string>(),
        .attempt_id = parsed.at("attempt_id").get<std::string>(),
        .launch_profile_digest =
            parsed.at("launch_profile_digest").get<std::string>(),
        .profiler_executable_digest =
            parsed.at("profiler_executable_digest").get<std::string>(),
        .authority_digest =
            parsed.at("authority_digest").get<std::string>(),
    };
    if (!valid_authority_digest(result.authority_digest) ||
        external_profiler_authority_canonical_json(result) != value) {
      throw ProfilerLaunchProfileError(
          "external profiler authority is not content-addressed");
    }
    return result;
  } catch (const ProfilerLaunchProfileError&) {
    throw;
  } catch (...) {
    throw ProfilerLaunchProfileError(
        "external profiler authority decoding failed closed");
  }
}

SealedExternalProfilerAuthority::SealedExternalProfilerAuthority(
    ExternalProfilerAuthoritySpec spec, int descriptor) noexcept
    : spec_(std::move(spec)), descriptor_(descriptor) {}

SealedExternalProfilerAuthority::SealedExternalProfilerAuthority(
    SealedExternalProfilerAuthority&& other) noexcept
    : spec_(std::move(other.spec_)),
      descriptor_(std::exchange(other.descriptor_, -1)) {}

SealedExternalProfilerAuthority& SealedExternalProfilerAuthority::operator=(
    SealedExternalProfilerAuthority&& other) noexcept {
  if (this != &other) {
    if (descriptor_ >= 0) (void)::close(descriptor_);
    spec_ = std::move(other.spec_);
    descriptor_ = std::exchange(other.descriptor_, -1);
  }
  return *this;
}

SealedExternalProfilerAuthority::~SealedExternalProfilerAuthority() {
  if (descriptor_ >= 0) (void)::close(descriptor_);
}

const ExternalProfilerAuthoritySpec& SealedExternalProfilerAuthority::spec()
    const {
  return spec_;
}

int SealedExternalProfilerAuthority::duplicate_fd() const {
  const int duplicate = ::fcntl(descriptor_, F_DUPFD_CLOEXEC, 7);
  if (duplicate < 0) {
    throw ProfilerLaunchProfileError(
        "could not duplicate external profiler authority descriptor");
  }
  return duplicate;
}

SealedExternalProfilerAuthority create_sealed_external_profiler_authority(
    ExternalProfilerAuthoritySpec value) {
  value = seal_external_profiler_authority(std::move(value));
  const std::string encoded =
      external_profiler_authority_canonical_json(value);
  const int descriptor = static_cast<int>(::syscall(
      SYS_memfd_create, "trainvm-profiler-authority",
      MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL));
  if (descriptor < 0) {
    throw ProfilerLaunchProfileError(
        "could not create external profiler authority descriptor");
  }
  try {
    write_authority(descriptor, encoded);
    if (::fcntl(descriptor, F_ADD_SEALS, kRequiredAuthoritySeals) != 0) {
      throw ProfilerLaunchProfileError(
          "could not seal external profiler authority descriptor");
    }
    return SealedExternalProfilerAuthority(std::move(value), descriptor);
  } catch (...) {
    (void)::close(descriptor);
    throw;
  }
}

ExternalProfilerAuthoritySpec external_profiler_authority_from_sealed_fd(
    int descriptor, std::string_view expected_digest) {
  struct stat metadata {};
  const int seals = ::fcntl(descriptor, F_GET_SEALS);
  if (descriptor < 0 || ::fstat(descriptor, &metadata) != 0 ||
      !S_ISREG(metadata.st_mode) || metadata.st_size <= 0 ||
      static_cast<std::uintmax_t>(metadata.st_size) >
          kMaximumExternalProfilerAuthorityBytes ||
      seals < 0 || (seals & kRequiredAuthoritySeals) !=
                       kRequiredAuthoritySeals) {
    throw ProfilerLaunchProfileError(
        "external profiler descriptor is not sealed authority");
  }
  std::string encoded(static_cast<std::size_t>(metadata.st_size), '\0');
  std::size_t offset = 0U;
  while (offset < encoded.size()) {
    const ssize_t count = ::pread(descriptor, encoded.data() + offset,
                                  encoded.size() - offset,
                                  static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      throw ProfilerLaunchProfileError(
          "could not read external profiler authority descriptor");
    }
    offset += static_cast<std::size_t>(count);
  }
  const ExternalProfilerAuthoritySpec result =
      external_profiler_authority_from_canonical_json(encoded);
  if (!expected_digest.empty() && result.authority_digest != expected_digest) {
    throw ProfilerLaunchProfileError(
        "external profiler authority digest disagrees with request");
  }
  return result;
}

ProfilerStepLifecycle::ProfilerStepLifecycle(
    const GpuTraceCapture& capture) {
  if (!capture.enabled || !capture.backend ||
      *capture.backend == ProfilerBackend::torch || !capture.skip_steps ||
      !capture.warmup_steps || !capture.capture_steps ||
      *capture.skip_steps < 0 || *capture.skip_steps > 256 ||
      *capture.warmup_steps < 0 || *capture.warmup_steps > 256 ||
      *capture.capture_steps < 1 || *capture.capture_steps > 128 ||
      *capture.skip_steps + *capture.warmup_steps + *capture.capture_steps >
          512) {
    throw ProfilerLaunchProfileError(
        "external profiler lifecycle requires one complete bounded capture");
  }
  capture_begin_ = static_cast<std::uint64_t>(
      *capture.skip_steps + *capture.warmup_steps);
  capture_end_ = capture_begin_ +
                 static_cast<std::uint64_t>(*capture.capture_steps);
  captured_steps_.reserve(
      static_cast<std::size_t>(*capture.capture_steps));
}

ProfilerLifecycleAction ProfilerStepLifecycle::enter() {
  if (entered_ || finished_) {
    throw ProfilerLaunchProfileError(
        "external profiler lifecycle entered more than once");
  }
  entered_ = true;
  if (capture_begin_ == 0U) {
    active_ = true;
    return ProfilerLifecycleAction::start_capture;
  }
  return ProfilerLifecycleAction::none;
}

ProfilerLifecycleAction ProfilerStepLifecycle::optimizer_step(
    std::uint64_t optimizer_step) {
  if (!entered_ || finished_ || steps_seen_ ==
          std::numeric_limits<std::uint64_t>::max() ||
      (last_optimizer_step_ && optimizer_step <= *last_optimizer_step_)) {
    throw ProfilerLaunchProfileError(
        "external profiler optimizer-step sequence is invalid");
  }
  last_optimizer_step_ = optimizer_step;
  if (steps_seen_ >= capture_begin_ && steps_seen_ < capture_end_) {
    if (!active_) {
      throw ProfilerLaunchProfileError(
          "external profiler capture window is not active");
    }
    captured_steps_.push_back(optimizer_step);
  }
  ++steps_seen_;
  if (steps_seen_ == capture_end_) {
    active_ = false;
    return ProfilerLifecycleAction::stop_capture;
  }
  if (steps_seen_ == capture_begin_) {
    active_ = true;
    return ProfilerLifecycleAction::start_capture;
  }
  return ProfilerLifecycleAction::none;
}

ProfilerLifecycleAction ProfilerStepLifecycle::finish(bool exceptional) {
  if (!entered_ || finished_) {
    throw ProfilerLaunchProfileError(
        "external profiler lifecycle cannot finish from its current state");
  }
  finished_ = true;
  if (exceptional) {
    if (active_) {
      active_ = false;
      return ProfilerLifecycleAction::stop_capture;
    }
    return ProfilerLifecycleAction::none;
  }
  if (steps_seen_ < capture_end_ || active_ ||
      captured_steps_.size() != capture_end_ - capture_begin_) {
    throw ProfilerLaunchProfileError(
        "training ended before the external GPU trace window completed");
  }
  return ProfilerLifecycleAction::none;
}

bool ProfilerStepLifecycle::capture_active() const noexcept { return active_; }

bool ProfilerStepLifecycle::complete() const noexcept {
  return entered_ && !active_ && steps_seen_ >= capture_end_ &&
         captured_steps_.size() == capture_end_ - capture_begin_;
}

const std::vector<std::uint64_t>& ProfilerStepLifecycle::captured_steps()
    const noexcept {
  return captured_steps_;
}

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
  (void)bounded(capture.warmup_steps, "warmup_steps");
  (void)bounded(capture.skip_steps, "skip_steps");
  (void)bounded(capture.capture_steps, "capture_steps");

  std::vector<std::string> argv;
  argv.push_back(profile.program);
  for (const std::string& argument : profile.fixed_arguments)
    argv.push_back(argument);

  // Only integers reach the command line from the document, and only after
  // the bound check above. The strings around them are ours.
  if (*capture.backend == ProfilerBackend::nsys) {
    argv.emplace_back("--output");
    argv.emplace_back(output_path);
    // Always gate collection with the worker's exact optimizer-step
    // lifecycle. Capturing from process start when the leading window is zero
    // would include imports and initialization that are not optimizer steps.
    argv.emplace_back("--capture-range=cudaProfilerApi");
    argv.emplace_back("--capture-range-end=stop");
  } else {
    argv.emplace_back("--export");
    argv.emplace_back(output_path);
    // NCU launch counts are kernel counts, not optimizer-step counts. Filter
    // on the fixed default-domain range emitted by ProfilerStepLifecycle
    // instead of silently
    // changing the meaning of the experiment declaration.
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
