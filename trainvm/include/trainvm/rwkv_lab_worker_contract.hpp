#pragma once

#include <string>
#include <vector>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/host_launch_registry.hpp"

namespace trainvm {

// Deployment-facing contract for the exact rwkv_lab Python worker bundle.
// The fingerprint is supplied only after its immutable code artifact is built.
struct RwkvLabWorkerContract final {
  AdapterRegistryDocument adapter_registry;
  std::vector<std::string> provided_capabilities;
};

[[nodiscard]] RwkvLabWorkerContract rwkv_lab_worker_contract(
    std::string code_fingerprint);

struct RwkvLabWorkerDeploymentSpec final {
  std::string code_path;
  std::string code_fingerprint;
  std::string executable_path;
  std::string executable_fingerprint;
  std::string working_directory;
  std::vector<std::string> trusted_roots;
};

struct RwkvLabWorkerDeploymentContract final {
  AdapterRegistryDocument adapter_registry;
  HostLaunchRegistryDocument host_launch_registry;
  std::vector<std::string> provided_capabilities;
};

// Lowers one exact worker artifact/interpreter deployment into the two
// registries consumed by TrainVM. Construction runs both native validators so
// emitted profiles cannot drift from the reflected worker catalog.
[[nodiscard]] RwkvLabWorkerDeploymentContract rwkv_lab_worker_deployment(
    RwkvLabWorkerDeploymentSpec spec);

}  // namespace trainvm
