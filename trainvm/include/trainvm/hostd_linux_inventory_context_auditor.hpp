#pragma once

#include <string>

#include "trainvm/host_resources.hpp"
#include "trainvm/hostd_linux_process_authority.hpp"

namespace trainvm {

inline constexpr std::string_view kLinuxInventoryProcessContextAuditApiVersion =
    "trainvm.linux-inventory-process-context-audit/v1";

// Conservative terminal-process context evidence. The injected kernel must be
// a trusted, fresh host inventory source (the production source is the Linux
// NVIDIA/NVML collector). Only accelerators named by the durable grant are
// inspected; the guarded cgroup/device policy is responsible for confining the
// worker to those resources. Unknown vendors, missing resources, partial
// probes, and host/boot/broker drift produce incomplete evidence.
class LinuxInventoryProcessContextAuditor final
    : public ILinuxProcessContextAuditor {
 public:
  explicit LinuxInventoryProcessContextAuditor(IHostKernel& kernel);

  [[nodiscard]] LinuxProcessContextAudit audit(
      const ResourceBundleGrant& grant,
      const HostProcessSpawnReceipt& spawn) override;

 private:
  IHostKernel& kernel_;
};

}  // namespace trainvm
