#pragma once

#include <string>
#include <vector>

#include "trainvm/adapter_registry.hpp"

namespace trainvm {

// Deployment-facing contract for the exact rwkv_lab Python worker bundle.
// The fingerprint is supplied only after its immutable code artifact is built.
struct RwkvLabWorkerContract final {
  AdapterRegistryDocument adapter_registry;
  std::vector<std::string> provided_capabilities;
};

[[nodiscard]] RwkvLabWorkerContract rwkv_lab_worker_contract(
    std::string code_fingerprint);

}  // namespace trainvm
