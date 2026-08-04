#include "trainvm/hostd_gpu_authorization.hpp"

#include <algorithm>
#include <ranges>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/authority_document.hpp"
#include "trainvm/document.hpp"
#include "trainvm/hostd_boot_provisioning.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

[[noreturn]] void reject(std::string message) {
  throw HostdGpuAuthorizationError(std::move(message));
}

bool canonical_boot_id(std::string_view value) {
  if (value.size() != 36U) return false;
  for (std::size_t index = 0U; index < value.size(); ++index) {
    const bool hyphen =
        index == 8U || index == 13U || index == 18U || index == 23U;
    const bool hex = (value[index] >= '0' && value[index] <= '9') ||
                     (value[index] >= 'a' && value[index] <= 'f');
    if (hyphen ? value[index] != '-' : !hex) return false;
  }
  return true;
}

bool bounded_identity(std::string_view value) {
  return !value.empty() && value.size() <= 192U &&
         std::ranges::all_of(value, [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') ||
                  character == '.' || character == '_' || character == ':' ||
                  character == '/' || character == '-' || character == '@';
         });
}

bool canonical_gpu_id(std::string_view value) {
  if (!value.starts_with("GPU-") || value.size() != 40U) return false;
  const std::string_view suffix = value.substr(4U);
  for (std::size_t index = 0U; index < suffix.size(); ++index) {
    const bool hyphen =
        index == 8U || index == 13U || index == 18U || index == 23U;
    const bool hex = (suffix[index] >= '0' && suffix[index] <= '9') ||
                     (suffix[index] >= 'a' && suffix[index] <= 'f');
    if (hyphen ? suffix[index] != '-' : !hex) return false;
  }
  return true;
}

nlohmann::json authorization_body(
    const HostdGpuAuthorizationDocument& document) {
  return {{"api_version", document.api_version},
          {"host_id", document.host_id},
          {"boot_id", document.boot_id},
          {"broker_instance_id", document.broker_instance_id},
          {"driver_probe_authorized", document.driver_probe_authorized},
          {"display_policy", document.display_policy},
          {"allowed_display_gpu_ids", document.allowed_display_gpu_ids}};
}

std::string digest(const HostdGpuAuthorizationDocument& document) {
  return "sha256:" + sha256_hex(authorization_body(document).dump());
}

nlohmann::json strict_json(std::string_view source) {
  bool duplicate = false;
  std::vector<std::set<std::string>> keys;
  try {
    const nlohmann::json::parser_callback_t callback =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (keys.size() <= index + 1U) keys.resize(index + 2U);
            keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (keys.size() <= index) keys.resize(index + 1U);
            if (!keys[index].insert(parsed.get<std::string>()).second)
              duplicate = true;
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     keys.size() > index + 1U) {
            keys[index + 1U].clear();
          }
          return true;
        };
    nlohmann::json result = nlohmann::json::parse(source, callback);
    if (duplicate) reject("hostd GPU authorization contains a duplicate key");
    return result;
  } catch (const HostdGpuAuthorizationError&) {
    throw;
  } catch (const nlohmann::json::exception&) {
    reject("hostd GPU authorization is not valid JSON");
  }
}

}  // namespace

HostdGpuAuthorization::HostdGpuAuthorization(
    HostdGpuAuthorizationDocument document)
    : document_(std::move(document)) {
  if (document_.api_version != kHostdGpuAuthorizationApiVersion ||
      !bounded_identity(document_.host_id) ||
      !canonical_boot_id(document_.boot_id) ||
      !bounded_identity(document_.broker_instance_id) ||
      !document_.driver_probe_authorized ||
      document_.allowed_display_gpu_ids.size() > 16U ||
      !std::ranges::is_sorted(document_.allowed_display_gpu_ids) ||
      std::adjacent_find(document_.allowed_display_gpu_ids.begin(),
                         document_.allowed_display_gpu_ids.end()) !=
          document_.allowed_display_gpu_ids.end() ||
      !std::ranges::all_of(document_.allowed_display_gpu_ids,
                           canonical_gpu_id) ||
      document_.authorization_digest != digest(document_)) {
    reject("hostd GPU authorization identity, allowlist, or digest is invalid");
  }
  if (document_.display_policy == "deny") {
    display_policy_ = HostdDisplayGpuPolicy::deny;
    if (!document_.allowed_display_gpu_ids.empty())
      reject("deny display policy cannot carry an allowlist");
  } else if (document_.display_policy == "cooperative_allowlist") {
    display_policy_ = HostdDisplayGpuPolicy::cooperative_allowlist;
    if (document_.allowed_display_gpu_ids.empty())
      reject("cooperative display policy requires an exact GPU allowlist");
  } else {
    reject("hostd GPU authorization display policy is unsupported");
  }
}

HostdGpuAuthorization HostdGpuAuthorization::load_file(
    const std::filesystem::path& path) {
  HostdGpuAuthorizationDocument document;
  std::vector<Diagnostic> diagnostics;
  const std::string source = read_authority_document(
      path, "hostd GPU authorization", kHostdGpuAuthorizationMaximumBytes);
  if (!decode_json(strict_json(source), document, "", diagnostics)) {
    reject("hostd GPU authorization schema validation failed: " +
           diagnostics_json(diagnostics).dump());
  }
  return HostdGpuAuthorization(std::move(document));
}

const HostdGpuAuthorizationDocument& HostdGpuAuthorization::document() const
    noexcept {
  return document_;
}

HostdDisplayGpuPolicy HostdGpuAuthorization::display_policy() const noexcept {
  return display_policy_;
}

const std::vector<std::string>&
HostdGpuAuthorization::allowed_display_gpu_ids() const noexcept {
  return document_.allowed_display_gpu_ids;
}

void HostdGpuAuthorization::require_matches(
    const HostdDaemonConfiguration& daemon_configuration) const {
  const HostdDaemonConfigurationDocument& daemon =
      daemon_configuration.document();
  if (document_.host_id != daemon.host_id ||
      document_.boot_id != daemon.boot_id ||
      document_.broker_instance_id != daemon.broker_instance_id) {
    reject("hostd GPU authorization does not match daemon host, boot, and instance authority");
  }
}

HostdGpuAuthorization make_hostd_gpu_authorization(
    const HostdDaemonConfiguration& configuration_template,
    const HostdLinuxBootAuthoritySnapshot& boot_authority,
    HostdDisplayGpuPolicy display_policy,
    std::vector<std::string> allowed_display_gpu_ids) {
  const HostdDaemonConfiguration boot_configuration =
      materialize_hostd_daemon_boot(configuration_template, boot_authority);
  std::ranges::sort(allowed_display_gpu_ids);
  HostdGpuAuthorizationDocument document{
      .api_version = std::string(kHostdGpuAuthorizationApiVersion),
      .host_id = boot_configuration.document().host_id,
      .boot_id = boot_configuration.document().boot_id,
      .broker_instance_id = boot_configuration.document().broker_instance_id,
      .driver_probe_authorized = true,
      .display_policy = display_policy == HostdDisplayGpuPolicy::deny
                            ? "deny"
                            : "cooperative_allowlist",
      .allowed_display_gpu_ids = std::move(allowed_display_gpu_ids),
      .authorization_digest = {},
  };
  document.authorization_digest = digest(document);
  return HostdGpuAuthorization(std::move(document));
}

std::string hostd_gpu_authorization_json(
    const HostdGpuAuthorization& authorization) {
  return encode_json(authorization.document()).dump(2) + "\n";
}

}  // namespace trainvm
