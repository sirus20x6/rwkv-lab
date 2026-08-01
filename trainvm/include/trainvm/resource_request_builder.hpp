#pragma once

#include <stdexcept>
#include <string>

#include "trainvm/host_resources.hpp"
#include "trainvm/lease.hpp"
#include "trainvm/model.hpp"

namespace trainvm {

class ResourceRequestBuildError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ResourceRequestBuildContext final {
  std::string journal_id;
  std::string plan_hash;
  std::string run_id;
  Resources resources;
  ResourceLease lease;

  bool operator==(const ResourceRequestBuildContext&) const = default;
};

// Pure deterministic lowering from the portable experiment resource contract
// into hostd's physical bundle request. It currently admits accelerator-backed
// external workers only; a typed CPU/process-slot resource must be added before
// zero-accelerator external launches can be safely enabled.
[[nodiscard]] ResourceBundleRequest build_resource_bundle_request(
    const ResourceRequestBuildContext& context);

}  // namespace trainvm
