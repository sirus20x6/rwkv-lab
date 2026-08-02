#include "trainvm/host_launch_registry.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <ranges>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumProfiles = 4'096U;
constexpr std::size_t kMaximumTrustedRoots = 256U;
constexpr std::size_t kMaximumKeyBytes = 512U;
constexpr std::size_t kMaximumPathBytes = 4'096U;
constexpr std::size_t kMaximumArguments = 256U;
constexpr std::size_t kMaximumArgumentBytes = 4'096U;
constexpr std::size_t kMaximumTotalArgumentBytes = 64U << 10U;

class FileDescriptor {
 public:
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() {
    if (descriptor_ >= 0) (void)::close(descriptor_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] int get() const { return descriptor_; }

 private:
  int descriptor_;
};

bool bounded_text(std::string_view value, std::size_t maximum,
                  bool allow_empty = false) {
  return (allow_empty || !value.empty()) && value.size() <= maximum &&
         value.find('\0') == std::string_view::npos;
}

bool valid_sha256(std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.size() == prefix.size() + 64U && value.starts_with(prefix) &&
         std::ranges::all_of(value.substr(prefix.size()), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool canonical_absolute_path(std::string_view value) {
  if (!bounded_text(value, kMaximumPathBytes)) return false;
  const std::filesystem::path path(value);
  if (!path.is_absolute() || path.lexically_normal() != path) return false;
  return value == "/" || !value.ends_with('/');
}

bool path_within(const std::filesystem::path& path,
                 const std::filesystem::path& root) {
  auto path_iterator = path.begin();
  for (auto root_iterator = root.begin(); root_iterator != root.end();
       ++root_iterator, ++path_iterator) {
    if (path_iterator == path.end() || *path_iterator != *root_iterator) {
      return false;
    }
  }
  return true;
}

bool path_within_any_root(std::string_view value,
                          const std::vector<std::string>& roots) {
  const std::filesystem::path path(value);
  return std::ranges::any_of(roots, [&](const std::string& root) {
    return path_within(path, std::filesystem::path(root));
  });
}

void validate_key(const AdapterKey& key) {
  if (!bounded_text(key.adapter, kMaximumKeyBytes) ||
      !bounded_text(key.version, kMaximumKeyBytes) ||
      !bounded_text(key.operation, kMaximumKeyBytes) ||
      !bounded_text(key.contract, kMaximumKeyBytes)) {
    throw std::invalid_argument(
        "host launch profile key fields must be bounded and nonempty");
  }
  if (key.runtime != ComponentRuntime::python_worker &&
      key.runtime != ComponentRuntime::native_worker) {
    throw std::invalid_argument(
        "host launch profiles require python_worker or native_worker runtime");
  }
}

void validate_profile(HostLaunchProfile& profile,
                      const std::vector<std::string>& roots) {
  validate_key(profile.key);
  if (profile.provided_capabilities.size() > 256U ||
      std::ranges::any_of(
          profile.provided_capabilities, [](const std::string& capability) {
            return !bounded_text(capability, 256U);
          })) {
    throw std::invalid_argument(
        "host launch provided capabilities must be bounded and nonempty");
  }
  std::ranges::sort(profile.provided_capabilities);
  if (std::ranges::adjacent_find(profile.provided_capabilities) !=
      profile.provided_capabilities.end()) {
    throw std::invalid_argument(
        "host launch provided capabilities must be unique");
  }
  if (!valid_sha256(profile.code_fingerprint) ||
      !valid_sha256(profile.executable_fingerprint)) {
    throw std::invalid_argument(
        "host launch fingerprints must be canonical sha256 hex");
  }
  if (!canonical_absolute_path(profile.executable_path) ||
      !canonical_absolute_path(profile.working_directory) ||
      !path_within_any_root(profile.executable_path, roots) ||
      !path_within_any_root(profile.working_directory, roots)) {
    throw std::invalid_argument(
        "host launch executable and working directory must be canonical absolute paths beneath a trusted root");
  }

  if (profile.key.runtime == ComponentRuntime::python_worker) {
    if (!profile.code_path || !canonical_absolute_path(*profile.code_path) ||
        !path_within_any_root(*profile.code_path, roots)) {
      throw std::invalid_argument(
          "python host launch profiles require a canonical trusted code_path");
    }
    if (profile.public_arguments.empty()) {
      throw std::invalid_argument(
          "python host launch profiles require a fixed code argument slot");
    }
  } else {
    if (profile.code_path) {
      throw std::invalid_argument(
          "native host launch profiles must not declare code_path");
    }
    if (profile.code_fingerprint != profile.executable_fingerprint) {
      throw std::invalid_argument(
          "native host launch code and executable fingerprints must match");
    }
  }

  if (profile.public_arguments.size() > kMaximumArguments) {
    throw std::invalid_argument("host launch argument count exceeds the bound");
  }
  std::size_t total_bytes = 0;
  for (const std::string& argument : profile.public_arguments) {
    if (!bounded_text(argument, kMaximumArgumentBytes, false) ||
        argument.find("secret://") != std::string::npos ||
        argument.find("${") != std::string::npos ||
        argument.find("{{") != std::string::npos ||
        total_bytes > kMaximumTotalArgumentBytes - argument.size()) {
      throw std::invalid_argument(
          "host launch arguments must be bounded fixed public literals");
    }
    total_bytes += argument.size();
  }
}

std::vector<std::string> validate_roots(std::vector<std::string> roots) {
  if (roots.size() > kMaximumTrustedRoots) {
    throw std::invalid_argument(
        "host launch registry trusted_roots list exceeds the bound");
  }
  for (const std::string& root : roots) {
    if (!canonical_absolute_path(root)) {
      throw std::invalid_argument(
          "host launch trusted roots must be canonical absolute paths");
    }
  }
  std::ranges::sort(roots);
  for (std::size_t left = 0; left < roots.size(); ++left) {
    for (std::size_t right = left + 1U; right < roots.size(); ++right) {
      const std::filesystem::path left_path(roots[left]);
      const std::filesystem::path right_path(roots[right]);
      if (path_within(left_path, right_path) ||
          path_within(right_path, left_path)) {
        throw std::invalid_argument(
            "host launch trusted roots must be unique and nonoverlapping");
      }
    }
  }
  return roots;
}

}  // namespace

HostLaunchRegistry::HostLaunchRegistry(HostLaunchRegistryDocument document) {
  if (document.api_version != "trainvm.host-launches/v2") {
    throw std::invalid_argument(
        "host launch registry api_version must be trainvm.host-launches/v2");
  }
  if (document.profiles.size() > kMaximumProfiles) {
    throw std::invalid_argument(
        "host launch registry profile list exceeds the bound");
  }
  trusted_roots_ = validate_roots(std::move(document.trusted_roots));
  if (!document.profiles.empty() && trusted_roots_.empty()) {
    throw std::invalid_argument(
        "host launch profiles require at least one trusted root");
  }
  for (HostLaunchProfile& profile : document.profiles) {
    validate_profile(profile, trusted_roots_);
    const AdapterKey key = profile.key;
    if (!profiles_.emplace(key, std::move(profile)).second) {
      throw std::invalid_argument(
          "host launch registry contains a duplicate exact key");
    }
  }
  nlohmann::json canonical_profiles = nlohmann::json::array();
  for (const auto& [key, profile] : profiles_) {
    (void)key;
    canonical_profiles.push_back(encode_json(profile));
  }
  nlohmann::json canonical_registry =
      encode_json(HostLaunchRegistryDocument{
          .api_version = "trainvm.host-launches/v2",
          .trusted_roots = trusted_roots_,
          .profiles = {},
      });
  canonical_registry["profiles"] = std::move(canonical_profiles);
  registry_digest_ = "sha256:" + sha256_hex(canonical_registry.dump());
}

HostLaunchRegistry HostLaunchRegistry::load_file(
    const std::filesystem::path& path) {
  constexpr std::uintmax_t kMaximumRegistryBytes = 1U << 20U;
  if (path.empty() || !path.is_absolute()) {
    throw std::invalid_argument(
        "host launch registry path must be absolute and nonempty");
  }
  const int raw_descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (raw_descriptor < 0) {
    throw std::invalid_argument(
        "could not securely open host launch registry " + path.string() +
        ": " + std::strerror(errno));
  }
  FileDescriptor descriptor(raw_descriptor);
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) > kMaximumRegistryBytes ||
      (before.st_uid != 0U && before.st_uid != ::geteuid()) ||
      (before.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw std::invalid_argument(
        "host launch registry must be an owner/root-owned regular file that is not group/world-writable and is no larger than 1 MiB");
  }

  std::string text(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0;
  while (offset < text.size()) {
    const ssize_t count =
        ::read(descriptor.get(), text.data() + offset, text.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      throw std::invalid_argument(
          "host launch registry changed or failed while being read");
    }
    offset += static_cast<std::size_t>(count);
  }
  char extra = '\0';
  ssize_t trailing = 0;
  do {
    trailing = ::read(descriptor.get(), &extra, 1U);
  } while (trailing < 0 && errno == EINTR);
  struct stat after {};
  if (trailing != 0 || ::fstat(descriptor.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    throw std::invalid_argument(
        "host launch registry changed while it was being read");
  }

  nlohmann::json source;
  bool duplicate_key = false;
  std::vector<std::set<std::string>> object_keys;
  try {
    const nlohmann::json::parser_callback_t reject_duplicates =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (object_keys.size() <= index + 1U) {
              object_keys.resize(index + 2U);
            }
            object_keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (object_keys.size() <= index) object_keys.resize(index + 1U);
            if (!object_keys[index].insert(parsed.get<std::string>()).second) {
              duplicate_key = true;
            }
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     object_keys.size() > index + 1U) {
            object_keys[index + 1U].clear();
          }
          return true;
        };
    source = nlohmann::json::parse(text, reject_duplicates);
  } catch (const nlohmann::json::exception& exception) {
    throw std::invalid_argument(
        "host launch registry is not valid JSON: " +
        std::string(exception.what()));
  }
  if (duplicate_key) {
    throw std::invalid_argument(
        "host launch registry JSON contains a duplicate object key");
  }

  HostLaunchRegistryDocument document;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, document, "", diagnostics)) {
    throw std::invalid_argument(
        "host launch registry schema validation failed: " +
        diagnostics_json(diagnostics).dump());
  }
  return HostLaunchRegistry(std::move(document));
}

const HostLaunchProfile& HostLaunchRegistry::resolve(
    const AdapterKey& key, std::string_view code_fingerprint) const {
  const auto profile = profiles_.find(key);
  if (profile == profiles_.end()) {
    throw HostLaunchResolutionError(
        "host launch registry has no exact adapter key");
  }
  if (code_fingerprint != profile->second.code_fingerprint) {
    throw HostLaunchResolutionError(
        "host launch code fingerprint differs from the exact profile");
  }
  return profile->second;
}

const std::vector<std::string>& HostLaunchRegistry::trusted_roots() const {
  return trusted_roots_;
}

const std::string& HostLaunchRegistry::registry_digest() const {
  return registry_digest_;
}

std::string HostLaunchRegistry::profile_digest(
    const AdapterKey& key, std::string_view code_fingerprint) const {
  const HostLaunchProfile& profile = resolve(key, code_fingerprint);
  return "sha256:" +
         sha256_hex(nlohmann::json{
                        {"api_version", "trainvm.host-launch-profile/v2"},
                        {"profile", encode_json(profile)},
                    }
                        .dump());
}

}  // namespace trainvm
