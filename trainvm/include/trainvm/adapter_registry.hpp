#pragma once

#include <compare>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/model.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

struct AdapterKey {
  std::string adapter;
  std::string version;
  ComponentRuntime runtime{};
  std::string operation;
  std::string contract;

  auto operator<=>(const AdapterKey&) const = default;
};

struct AdapterProfile {
  AdapterKey key;
  Effect effect{};
  Idempotency idempotency{};
  std::string code_fingerprint;
  std::vector<std::string> required_capabilities;
};

struct AdapterRegistryDocument {
  std::string api_version;
  std::vector<AdapterProfile> profiles;
};

class AdapterResolutionError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Authority-owned registry. Experiment documents may select an exact key, but
// they never supply code identity or worker capabilities.
class AdapterRegistry {
 public:
  explicit AdapterRegistry(std::vector<AdapterProfile> profiles);

  static AdapterRegistry load_file(const std::filesystem::path& path);

  [[nodiscard]] const AdapterProfile& resolve(
      const Component& component, std::string_view operation) const;
  void validate_plan(const CompiledPlan& plan) const;
  [[nodiscard]] WorkerLaunchRequest worker_launch_request(
      const Component& component, std::string_view operation) const;
  [[nodiscard]] const std::string& registry_digest() const;
  [[nodiscard]] std::string plan_lock_manifest(const CompiledPlan& plan) const;
  [[nodiscard]] std::string plan_lock_digest(const CompiledPlan& plan) const;
  void validate_submission_lock(const CompiledPlan& plan,
                                const nlohmann::json& submission) const;

 private:
  std::map<AdapterKey, AdapterProfile> profiles_;
  std::string registry_digest_;
};

}  // namespace trainvm
