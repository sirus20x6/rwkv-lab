#include "trainvm/resource_request_builder.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

HostAcceleratorVendor host_vendor(AcceleratorVendor vendor) {
  switch (vendor) {
    case AcceleratorVendor::nvidia:
      return HostAcceleratorVendor::nvidia;
    case AcceleratorVendor::amd:
      return HostAcceleratorVendor::amd;
    case AcceleratorVendor::intel:
      return HostAcceleratorVendor::intel;
    case AcceleratorVendor::none:
      break;
  }
  throw ResourceRequestBuildError(
      "external process has no declared accelerator resource kind");
}

std::optional<std::uint64_t> minimum_memory_bytes(
    const std::optional<double>& gibibytes) {
  if (!gibibytes) return std::nullopt;
  constexpr long double kBytesPerGibibyte = 1'073'741'824.0L;
  const long double bytes =
      std::ceil(static_cast<long double>(*gibibytes) * kBytesPerGibibyte);
  if (!std::isfinite(*gibibytes) || *gibibytes <= 0.0 || bytes < 1.0L ||
      bytes > static_cast<long double>(
                  std::numeric_limits<std::uint64_t>::max())) {
    throw ResourceRequestBuildError(
        "accelerator minimum memory cannot be represented in bytes");
  }
  return static_cast<std::uint64_t>(bytes);
}

}  // namespace

ResourceBundleRequest build_resource_bundle_request(
    const ResourceRequestBuildContext& context) {
  const auto& accelerators = context.resources.accelerators;
  if (context.journal_id.empty() || context.plan_hash.empty() ||
      context.run_id.empty() || context.lease.owner_run_id != context.run_id ||
      context.lease.lease_id.empty() || context.lease.fencing_token == 0U) {
    throw ResourceRequestBuildError(
        "resource request build context has incomplete authority identity");
  }
  if (accelerators.count <= 0 ||
      accelerators.count > static_cast<std::int64_t>(
                               HostResourceBounds::maximum_bundle_count)) {
    throw ResourceRequestBuildError(
        "external accelerator count is outside the host bundle bound");
  }
  const auto vendor = host_vendor(accelerators.vendor);
  const nlohmann::json request_identity{
      {"api_version", "trainvm.resource-request-identity/v1"},
      {"journal_id", context.journal_id},
      {"logical_fencing_token", context.lease.fencing_token},
      {"logical_lease_id", context.lease.lease_id},
      {"plan_hash", context.plan_hash},
      {"run_id", context.run_id},
  };
  ResourceBundleRequest request{
      .api_version = std::string(kHostResourceRequestApiVersion),
      .request_id = "host-request-" + sha256_hex(request_identity.dump()),
      .journal_id = context.journal_id,
      .run_id = context.run_id,
      .logical_lease_id = context.lease.lease_id,
      .logical_fencing_token = context.lease.fencing_token,
      .count = static_cast<std::uint32_t>(accelerators.count),
      .access_mode = accelerators.exclusive
                         ? ResourceAccessMode::exclusive_device
                         : ResourceAccessMode::exclusive_compute,
      .topology = TopologyPolicy::any,
      .selector = {.vendor = vendor,
                   .minimum_memory_bytes =
                       minimum_memory_bytes(accelerators.minimum_memory_gib),
                   .exact_labels = accelerators.selector.value_or(
                       std::map<std::string, std::string>{}),
                   .exact_resources = {}},
      .canonical_request_digest = {},
  };
  return seal_resource_request(std::move(request));
}

}  // namespace trainvm
