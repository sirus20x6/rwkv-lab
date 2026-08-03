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

struct RwkvLabWorkerAdapterRuntimeRequirements final {
  std::string adapter;
  std::vector<std::string> root_distributions;
};

struct RwkvLabWorkerRuntimeRequirementsContract final {
  std::string api_version;
  std::vector<std::string> shared_root_distributions;
  std::vector<RwkvLabWorkerAdapterRuntimeRequirements> profiles;
};

// The native adapter catalog is the authority for the Python distributions
// that must be sealed before any adapter code is imported.
[[nodiscard]] RwkvLabWorkerRuntimeRequirementsContract
rwkv_lab_worker_runtime_requirements();

struct RwkvLabWorkerRuntimeDeploymentSpec final {
  std::string adapter;
  std::string code_path;
  std::string code_fingerprint;
  std::string bootstrap_runtime_closure_fingerprint;
  std::string executable_path;
  std::string executable_fingerprint;
  std::string working_directory;
};

struct RwkvLabWorkerDeploymentSpec final {
  std::string api_version;
  std::vector<RwkvLabWorkerRuntimeDeploymentSpec> runtimes;
  std::vector<std::string> trusted_roots;
};

struct RwkvLabWorkerDeploymentContract final {
  AdapterRegistryDocument adapter_registry;
  HostLaunchRegistryDocument host_launch_registry;
  std::vector<std::string> provided_capabilities;
};

// Lowers exact per-adapter worker artifact/interpreter deployments into the
// two registries consumed by TrainVM. Construction runs both native validators
// so emitted profiles cannot drift from the reflected worker catalog.
[[nodiscard]] RwkvLabWorkerDeploymentContract rwkv_lab_worker_deployment(
    RwkvLabWorkerDeploymentSpec spec);

}  // namespace trainvm
