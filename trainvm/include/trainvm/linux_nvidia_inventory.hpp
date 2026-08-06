#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "trainvm/host_resources.hpp"

namespace trainvm {

inline constexpr std::string_view kLinuxNvidiaInventoryApiVersion =
    "trainvm.linux-nvidia-inventory/v2";

struct LinuxNvidiaObservedContexts final {
  ResourceContextDisposition compute{ResourceContextDisposition::unknown};
  ResourceContextDisposition graphics{ResourceContextDisposition::unknown};

  bool operator==(const LinuxNvidiaObservedContexts &) const = default;
};

// PID observations are retained in the raw, point-in-time scheduling evidence
// so a later guarded hostd integration can reconcile orphaned owners.  They are
// deliberately not interpreted as durable absence or authorization here.
struct LinuxNvidiaRawProcess final {
  std::uint32_t pid{};
  std::optional<std::uint32_t> gpu_instance_id;
  std::optional<std::uint32_t> compute_instance_id;

  bool operator==(const LinuxNvidiaRawProcess &) const = default;
};

struct LinuxNvidiaRawPartition final {
  std::string uuid;
  std::uint64_t total_memory_bytes{};
  std::optional<std::uint32_t> gpu_instance_id;
  std::optional<std::uint32_t> compute_instance_id;
  LinuxNvidiaObservedContexts contexts;
  std::vector<LinuxNvidiaRawProcess> compute_processes;
  std::vector<LinuxNvidiaRawProcess> graphics_processes;
  bool evidence_complete{};

  bool operator==(const LinuxNvidiaRawPartition &) const = default;
};

struct LinuxNvidiaRawDevice final {
  std::string uuid;
  std::string pci_bdf;
  std::uint64_t total_memory_bytes{};
  std::optional<std::uint32_t> device_major;
  std::optional<std::uint32_t> device_minor;
  std::optional<std::int32_t> numa_node;
  std::optional<std::uint32_t> current_mig_mode;
  std::optional<std::uint32_t> pending_mig_mode;
  LinuxNvidiaObservedContexts contexts;
  std::vector<LinuxNvidiaRawProcess> compute_processes;
  std::vector<LinuxNvidiaRawProcess> graphics_processes;
  bool evidence_complete{};
  bool device_node_mapping_complete{};
  std::optional<bool> display_active;
  std::optional<bool> display_mode_enabled;
  bool display_evidence_complete{};
  std::vector<LinuxNvidiaRawPartition> partitions;

  bool operator==(const LinuxNvidiaRawDevice &) const = default;
};

// Display state belongs to the full physical device. It blocks selection of
// that full-device resource, but is not copied into a MIG child's per-instance
// graphics context. A caller must request/permit accelerator_partition
// resources explicitly if host policy allows partitions on a display parent.

// One bounded, point-in-time read-only capture. Implementations bind both
// revisions to the host, boot, driver, structural device set, NVML
// identity/topology, MIG state, and process-context observations.  Absence is
// never durable evidence beyond the configured freshness window. Tests inject
// this seam; the default source pins procfs/sysfs/devfs roots and dynamically
// loads and attests libnvidia-ml.so.1.
struct LinuxNvidiaRawSnapshot final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string begin_revision;
  std::string end_revision;
  bool structural_complete{};
  bool nvml_loaded{};
  bool nvml_complete{};
  bool context_details_complete{};
  bool trusted_host_namespace{};
  bool trusted_nvml_loader{};
  std::uint64_t capture_started_monotonic_ns{};
  std::uint64_t capture_finished_monotonic_ns{};
  std::vector<std::string> structural_pci_bdfs;
  std::vector<HostDeviceNodeCapability> shared_device_nodes;
  std::vector<LinuxNvidiaRawDevice> devices;
  std::vector<PassiveHostAcceleratorMemory> passive_accelerator_memory;
  std::string detail;

  bool operator==(const LinuxNvidiaRawSnapshot &) const = default;
};

class ILinuxNvidiaReadOnlyKernel {
public:
  virtual ~ILinuxNvidiaReadOnlyKernel() = default;
  [[nodiscard]] virtual LinuxNvidiaRawSnapshot capture() = 0;
  [[nodiscard]] virtual std::uint64_t monotonic_now_ns() const = 0;
};

struct LinuxNvidiaInventoryConfig final {
  std::string api_version{std::string(kLinuxNvidiaInventoryApiVersion)};
  std::string broker_epoch;
  std::size_t maximum_devices{HostResourceBounds::maximum_resources};
  std::size_t maximum_partitions_per_device{64U};
  std::size_t maximum_processes_per_device{4096U};
  std::uint64_t maximum_capture_duration_ns{5'000'000'000ULL};
  std::uint64_t maximum_snapshot_age_ns{1'000'000'000ULL};
  // This bounds completed evidence; it cannot interrupt a wedged synchronous
  // NVML call. Production use therefore requires an independently guarded
  // hostd process with an external deadline/restart policy.
  // These must only be asserted by a guarded host daemon whose mount namespace
  // and dynamic-loader policy are independently attested.  A normal process
  // must leave them false and receives observation-only/probe_unknown output.
  bool trusted_host_namespace{};
  bool trusted_nvml_loader{};

  bool operator==(const LinuxNvidiaInventoryConfig &) const = default;
};

namespace linux_nvidia_test_seam {

// ABI-v2 has only busIdLegacy plus numeric fields.  Kept as a small explicit
// seam so the fallback is testable without loading a vendor library.
[[nodiscard]] std::string pci_bdf_from_v2(std::string_view legacy,
                                          std::uint32_t domain,
                                          std::uint32_t bus,
                                          std::uint32_t device);
[[nodiscard]] std::optional<std::uint32_t>
nvidia_frontend_major(std::string_view proc_devices);
[[nodiscard]] bool pinned_pci_device_mapping(
    std::string_view pci_uevent, std::string_view proc_gpu_information,
    std::string_view uuid, std::string_view pci_bdf, std::uint32_t minor);

} // namespace linux_nvidia_test_seam

class LinuxNvidiaInventoryCollector final : public IHostKernel {
public:
  explicit LinuxNvidiaInventoryCollector(
      LinuxNvidiaInventoryConfig config,
      std::shared_ptr<ILinuxNvidiaReadOnlyKernel> kernel = nullptr);
  ~LinuxNvidiaInventoryCollector() override;

  LinuxNvidiaInventoryCollector(LinuxNvidiaInventoryCollector &&) noexcept;
  LinuxNvidiaInventoryCollector &
  operator=(LinuxNvidiaInventoryCollector &&) noexcept;
  LinuxNvidiaInventoryCollector(const LinuxNvidiaInventoryCollector &) = delete;
  LinuxNvidiaInventoryCollector &
  operator=(const LinuxNvidiaInventoryCollector &) = delete;

  [[nodiscard]] HostKernelSnapshot capture_inventory() override;
  [[nodiscard]] std::optional<PassiveHostMemorySnapshot>
  passive_memory_snapshot() const override;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace trainvm
