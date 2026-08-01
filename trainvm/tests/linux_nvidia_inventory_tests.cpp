#include "trainvm/linux_nvidia_inventory.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

constexpr std::string_view kGpuA = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
constexpr std::string_view kGpuB = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
constexpr std::string_view kMigA = "MIG-11111111-1111-1111-1111-111111111111";
constexpr std::string_view kMigB = "MIG-22222222-2222-2222-2222-222222222222";

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class FakeReadOnlyKernel final : public ILinuxNvidiaReadOnlyKernel {
public:
  explicit FakeReadOnlyKernel(LinuxNvidiaRawSnapshot snapshot)
      : snapshot_(std::move(snapshot)) {}

  LinuxNvidiaRawSnapshot capture() override {
    ++calls;
    return snapshot_;
  }

  std::uint64_t monotonic_now_ns() const override {
    return calls == 0U ? snapshot_.capture_started_monotonic_ns - 1U
                       : snapshot_.capture_finished_monotonic_ns + 1U;
  }

  std::size_t calls{};

private:
  LinuxNvidiaRawSnapshot snapshot_;
};

class ThrowingReadOnlyKernel final : public ILinuxNvidiaReadOnlyKernel {
public:
  LinuxNvidiaRawSnapshot capture() override {
    throw std::runtime_error("untrusted kernel seam failure");
  }
  std::uint64_t monotonic_now_ns() const override { return 1U; }
};

LinuxNvidiaRawDevice device(std::string uuid, std::string bdf,
                            std::uint32_t minor,
                            LinuxNvidiaObservedContexts contexts,
                            bool display = false) {
  return {.uuid = std::move(uuid),
          .pci_bdf = std::move(bdf),
          .total_memory_bytes = 24ULL << 30U,
          .device_major = 195U,
          .device_minor = minor,
          .numa_node = 0,
          .current_mig_mode = 0U,
          .pending_mig_mode = 0U,
          .contexts = contexts,
          .compute_processes = {},
          .graphics_processes = {},
          .evidence_complete = true,
          .device_node_mapping_complete = true,
          .display_active = display,
          .display_mode_enabled = display,
          .display_evidence_complete = true,
          .partitions = {}};
}

LinuxNvidiaRawSnapshot complete_snapshot() {
  LinuxNvidiaRawSnapshot snapshot{
      .api_version = std::string(kLinuxNvidiaInventoryApiVersion),
      .host_id = "host-linux-nvidia",
      .boot_id = "boot-linux-nvidia",
      .broker_epoch = "broker-linux-nvidia",
      .begin_revision = "revision-linux-nvidia-1",
      .end_revision = "revision-linux-nvidia-1",
      .structural_complete = true,
      .nvml_loaded = true,
      .nvml_complete = true,
      .context_details_complete = true,
      .trusted_host_namespace = true,
      .trusted_nvml_loader = true,
      .capture_started_monotonic_ns = 1'000'000U,
      .capture_finished_monotonic_ns = 1'000'100U,
      .structural_pci_bdfs = {"0000:01:00.0", "0000:02:00.0"},
      .devices = {device(std::string(kGpuB), "0000:02:00.0", 1U,
                         {.compute = ResourceContextDisposition::absent,
                          .graphics = ResourceContextDisposition::absent},
                         true),
                  device(std::string(kGpuA), "0000:01:00.0", 0U,
                         {.compute = ResourceContextDisposition::absent,
                          .graphics = ResourceContextDisposition::absent})},
      .detail = "deterministic-complete"};
  snapshot.devices[1].partitions.push_back(
      {.uuid = std::string(kMigA),
       .total_memory_bytes = 8ULL << 30U,
       .gpu_instance_id = 1U,
       .compute_instance_id = 2U,
       .contexts = {.compute = ResourceContextDisposition::absent,
                    .graphics = ResourceContextDisposition::absent},
       .compute_processes = {},
       .graphics_processes = {},
       .evidence_complete = true});
  snapshot.devices[1].partitions.push_back(
      {.uuid = std::string(kMigB),
       .total_memory_bytes = 8ULL << 30U,
       .gpu_instance_id = 3U,
       .compute_instance_id = 4U,
       .contexts = {.compute = ResourceContextDisposition::present,
                    .graphics = ResourceContextDisposition::absent},
       .compute_processes = {{.pid = 321U,
                              .gpu_instance_id = 3U,
                              .compute_instance_id = 4U}},
       .graphics_processes = {},
       .evidence_complete = true});
  snapshot.devices[0].current_mig_mode = 0U;
  snapshot.devices[0].pending_mig_mode = 0U;
  snapshot.devices[1].current_mig_mode = 1U;
  snapshot.devices[1].pending_mig_mode = 1U;
  return snapshot;
}

LinuxNvidiaInventoryConfig config() {
  return {.api_version = std::string(kLinuxNvidiaInventoryApiVersion),
          .broker_epoch = "broker-linux-nvidia",
          .maximum_devices = 16U,
          .maximum_partitions_per_device = 16U,
          .maximum_processes_per_device = 128U,
          .maximum_capture_duration_ns = 1'000'000U,
          .maximum_snapshot_age_ns = 1'000'000U,
          .trusted_host_namespace = true,
          .trusted_nvml_loader = true};
}

HostInventoryReceipt capture(LinuxNvidiaRawSnapshot snapshot) {
  auto fake = std::make_shared<FakeReadOnlyKernel>(std::move(snapshot));
  LinuxNvidiaInventoryCollector collector(config(), fake);
  auto receipt = capture_host_inventory(collector);
  require(fake->calls == 1U,
          "collector consumes one bounded point-in-time fake capture");
  return receipt;
}

const ObservedHostResource &resource(const HostInventoryReceipt &receipt,
                                     std::string_view stable_id) {
  const auto found = std::ranges::find_if(
      receipt.resources, [&](const ObservedHostResource &value) {
        return value.id.stable_id == stable_id;
      });
  if (found == receipt.resources.end())
    throw std::runtime_error("expected resource is absent");
  return *found;
}

void complete_capture_is_stable_and_display_is_occupied() {
  const auto first = capture(complete_snapshot());
  require(first.probes.size() == 1U &&
              first.probes.front().disposition == ProbeDisposition::complete &&
              first.probes.front().context_details_complete,
          "complete structural and NVML evidence produces a complete probe");
  require(first.resources.size() == 4U,
          "full devices and MIG partitions retain stable identities");
  const auto &partition_parent = resource(first, kGpuA);
  require(partition_parent.disposition ==
                  ResourceObservationDisposition::probe_unknown &&
              partition_parent.device_major == 195U &&
              partition_parent.device_minor == 0U,
          "MIG parent remains nonselectable rather than aggregating children");
  const auto &display_gpu = resource(first, kGpuB);
  require(display_gpu.disposition == ResourceObservationDisposition::occupied &&
              display_gpu.graphics_contexts ==
                  ResourceContextDisposition::present &&
              display_gpu.labels.at("occupancy") == "foreign-observed",
          "current display graphics activity is foreign occupied, never free");
  const auto &mig = resource(first, kMigA);
  require(mig.id.parent_id == kGpuA && !mig.device_major && !mig.device_minor &&
              mig.disposition ==
                  ResourceObservationDisposition::audited_eligible,
          "MIG identity is parent-bound without a false single-node allowlist");
  const auto &busy_mig = resource(first, kMigB);
  require(busy_mig.disposition == ResourceObservationDisposition::occupied &&
              partition_parent.compute_contexts ==
                  ResourceContextDisposition::absent,
          "sibling MIG occupancy is attributed to the child and never "
          "aggregated into the parent context");

  auto permuted = complete_snapshot();
  std::ranges::reverse(permuted.devices);
  std::ranges::reverse(permuted.structural_pci_bdfs);
  const auto second = capture(std::move(permuted));
  require(first.receipt_digest == second.receipt_digest &&
              first.resources == second.resources,
          "input enumeration order cannot alter canonical inventory identity");
}

void missing_partial_and_torn_evidence_fail_closed() {
  auto missing = complete_snapshot();
  missing.nvml_loaded = false;
  missing.nvml_complete = false;
  missing.context_details_complete = false;
  const auto unavailable = capture(std::move(missing));
  require(unavailable.probes.front().disposition ==
                  ProbeDisposition::unavailable &&
              !unavailable.probes.front().context_details_complete,
          "missing NVML is unavailable rather than guessed complete");
  require(std::ranges::all_of(
              unavailable.resources,
              [](const auto &value) {
                return value.disposition ==
                           ResourceObservationDisposition::probe_unknown &&
                       value.compute_contexts ==
                           ResourceContextDisposition::unknown &&
                       value.graphics_contexts ==
                           ResourceContextDisposition::unknown;
              }),
          "missing NVML yields no selectable resource evidence");

  auto partial = complete_snapshot();
  partial.context_details_complete = false;
  partial.devices.front().contexts.graphics =
      ResourceContextDisposition::unknown;
  const auto partial_receipt = capture(std::move(partial));
  require(partial_receipt.probes.front().disposition ==
                  ProbeDisposition::partial &&
              std::ranges::none_of(
                  partial_receipt.resources,
                  [](const auto &value) {
                    return value.disposition ==
                           ResourceObservationDisposition::audited_eligible;
                  }),
          "partial context evidence poisons selection for the whole probe");

  auto torn = complete_snapshot();
  torn.end_revision = "revision-linux-nvidia-2";
  const auto torn_receipt = capture(std::move(torn));
  require(
      torn_receipt.probes.front().disposition == ProbeDisposition::partial &&
          torn_receipt.snapshot_revision != "revision-linux-nvidia-1" &&
          std::ranges::all_of(
              torn_receipt.resources,
              [](const auto &value) {
                return value.disposition ==
                       ResourceObservationDisposition::probe_unknown;
              }),
      "torn evidence is revision-bound probe_unknown, never grant evidence");

  auto mismatch = complete_snapshot();
  mismatch.structural_pci_bdfs.pop_back();
  const auto mismatch_receipt = capture(std::move(mismatch));
  require(mismatch_receipt.probes.front().disposition ==
              ProbeDisposition::partial,
          "NVML/sysfs device-set disagreement fails closed");
}

void display_mig_pci_and_mode_regressions() {
  const auto display = capture(complete_snapshot());
  const auto &display_gpu = resource(display, kGpuB);
  require(display_gpu.compute_contexts == ResourceContextDisposition::absent &&
              display_gpu.disposition ==
                  ResourceObservationDisposition::occupied &&
              display_gpu.labels.at("display") == "active",
          "an active display with zero process PIDs is still never eligible");

  auto mode_only_snapshot = complete_snapshot();
  mode_only_snapshot.devices[0].display_active = false;
  const auto mode_only = capture(std::move(mode_only_snapshot));
  require(resource(mode_only, kGpuB).disposition ==
                  ResourceObservationDisposition::occupied &&
              resource(mode_only, kGpuB).labels.at("display") == "mode-enabled",
          "enabled display mode is independently scheduling-block evidence");

  std::string terminated_legacy = "0000:0A:1f.0";
  terminated_legacy.push_back('\0');
  require(linux_nvidia_test_seam::pci_bdf_from_v2(terminated_legacy, 0U, 0U,
                                                  0U) == "0000:0a:1f.0" &&
              linux_nvidia_test_seam::pci_bdf_from_v2("", 2U, 0x0aU, 0x1fU) ==
                  "0002:0a:1f.0" &&
              linux_nvidia_test_seam::pci_bdf_from_v2("0000:0a:1f.0", 3U, 4U,
                                                      5U) == "0003:04:05.0" &&
              linux_nvidia_test_seam::pci_bdf_from_v2("bad", 0x10000U, 0U, 0U)
                  .empty(),
          "PCI v2 uses only NUL-terminated legacy or bounded numeric evidence");

  auto display_mig = complete_snapshot();
  display_mig.devices[1].display_active = true;
  display_mig.devices[1].display_mode_enabled = true;
  const auto display_parent = capture(std::move(display_mig));
  require(
      resource(display_parent, kGpuA).labels.at("display") == "active" &&
          resource(display_parent, kMigA).graphics_contexts ==
              ResourceContextDisposition::absent &&
          resource(display_parent, kMigA).disposition ==
              ResourceObservationDisposition::audited_eligible &&
          resource(display_parent, kMigA).labels.at("parent-display-policy") ==
              "partition-requires-explicit-policy",
      "parent display blocks full-device selection without fabricating a "
      "MIG child graphics context");

  auto mode_tear = complete_snapshot();
  mode_tear.devices[1].pending_mig_mode = 0U;
  const auto torn = capture(std::move(mode_tear));
  require(torn.probes.front().disposition == ProbeDisposition::partial &&
              std::ranges::all_of(
                  torn.resources,
                  [](const auto &value) {
                    return value.disposition ==
                           ResourceObservationDisposition::probe_unknown;
                  }),
          "current/pending MIG mode disagreement poisons the capture");

  auto unattributed = complete_snapshot();
  unattributed.devices[1].partitions[0].contexts.compute =
      ResourceContextDisposition::unknown;
  unattributed.devices[1].partitions[0].evidence_complete = false;
  unattributed.context_details_complete = false;
  const auto v2_only = capture(std::move(unattributed));
  require(v2_only.probes.front().disposition == ProbeDisposition::partial &&
              resource(v2_only, kMigA).disposition ==
                  ResourceObservationDisposition::probe_unknown,
          "MIG process evidence without v3 instance attribution is "
          "nonselectable");
}

void real_driver_node_mapping_parsers_are_strict() {
  constexpr std::string_view proc_devices = "Character devices:\n"
                                            "  1 mem\n"
                                            "195 nvidia\n"
                                            "195 nvidia-modeset\n"
                                            "240 nvidia-caps\n\n"
                                            "Block devices:\n";
  require(linux_nvidia_test_seam::nvidia_frontend_major(proc_devices) == 195U &&
              linux_nvidia_test_seam::nvidia_frontend_major(
                  "Character devices:\n195 nvidia-frontend\n") == 195U &&
              !linux_nvidia_test_seam::nvidia_frontend_major(
                  "Character devices:\n195 nvidia\n196 nvidia-frontend\n") &&
              !linux_nvidia_test_seam::nvidia_frontend_major(
                  "Block devices:\n195 nvidia\n"),
          "only canonical NVIDIA frontend names with one consistent major are "
          "accepted");

  constexpr std::string_view pci_uevent = "DRIVER=nvidia\n"
                                          "PCI_CLASS=30000\n"
                                          "PCI_ID=10DE:2BB1\n"
                                          "PCI_SLOT_NAME=0000:41:00.0\n";
  constexpr std::string_view gpu_information =
      "Model: NVIDIA RTX PRO 6000 Blackwell Workstation Edition\n"
      "GPU UUID: GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n"
      "Bus Type: PCIe\n"
      "Bus Location: 0000:41:00.0\n"
      "Device Minor: 0\n";
  require(linux_nvidia_test_seam::pinned_pci_device_mapping(
              pci_uevent, gpu_information, kGpuA, "0000:41:00.0", 0U) &&
              !linux_nvidia_test_seam::pinned_pci_device_mapping(
                  pci_uevent, gpu_information, kGpuA, "0000:41:00.0", 1U) &&
              !linux_nvidia_test_seam::pinned_pci_device_mapping(
                  pci_uevent, gpu_information, kGpuB, "0000:41:00.0", 0U),
          "pinned fallback binds sysfs driver/BDF and proc UUID/BDF/minor; "
          "minor alone never suffices");
}

void device_node_trust_and_freshness_fail_closed() {
  auto wrong_major = complete_snapshot();
  wrong_major.devices[0].device_major = 511U;
  wrong_major.devices[0].device_node_mapping_complete = false;
  wrong_major.devices[0].evidence_complete = false;
  const auto wrong_node = capture(std::move(wrong_major));
  const auto &observed = resource(wrong_node, kGpuB);
  require(wrong_node.probes.front().disposition == ProbeDisposition::partial &&
              !observed.device_major && !observed.device_minor,
          "unproven or wrong registered major/path mapping is omitted");

  auto mount_scope = complete_snapshot();
  mount_scope.trusted_host_namespace = false;
  const auto untrusted_mount = capture(std::move(mount_scope));
  require(untrusted_mount.probes.front().disposition ==
                  ProbeDisposition::partial &&
              untrusted_mount.probes.front().detail.starts_with(
                  "observation-only:") &&
              std::ranges::none_of(
                  untrusted_mount.resources,
                  [](const auto &value) {
                    return value.disposition ==
                           ResourceObservationDisposition::audited_eligible;
                  }),
          "unattested mount namespace is explicitly observation-only");

  auto loader_scope = complete_snapshot();
  loader_scope.trusted_nvml_loader = false;
  const auto untrusted_loader = capture(std::move(loader_scope));
  require(untrusted_loader.probes.front().disposition ==
                  ProbeDisposition::partial &&
              resource(untrusted_loader, kMigA).disposition ==
                  ResourceObservationDisposition::probe_unknown,
          "unsafe/unattested loader environment cannot yield grant evidence");

  auto raw_claim = complete_snapshot();
  auto observation_config = config();
  observation_config.trusted_host_namespace = false;
  observation_config.trusted_nvml_loader = false;
  auto fake = std::make_shared<FakeReadOnlyKernel>(std::move(raw_claim));
  LinuxNvidiaInventoryCollector observation_only(observation_config, fake);
  const auto default_scope = capture_host_inventory(observation_only);
  require(
      default_scope.probes.front().detail.starts_with("observation-only:") &&
          std::ranges::none_of(
              default_scope.resources,
              [](const auto &value) {
                return value.disposition ==
                       ResourceObservationDisposition::audited_eligible;
              }),
      "collector policy cannot be upgraded by a raw seam trust claim");

  auto stale = complete_snapshot();
  stale.capture_finished_monotonic_ns =
      stale.capture_started_monotonic_ns + 2'000'000U;
  const auto stale_receipt = capture(std::move(stale));
  require(stale_receipt.probes.front().disposition == ProbeDisposition::partial,
          "capture duration/age policy rejects stale point-in-time absence");
}

void malformed_and_throwing_sources_are_rejected() {
  auto malformed = complete_snapshot();
  malformed.devices.front().uuid = "GPU-not-a-canonical-uuid";
  const auto receipt = capture(std::move(malformed));
  require(receipt.probes.front().disposition == ProbeDisposition::partial &&
              std::ranges::none_of(
                  receipt.resources,
                  [](const auto &value) {
                    return value.disposition ==
                           ResourceObservationDisposition::audited_eligible;
                  }),
          "malformed device identity cannot become grant evidence");

  auto duplicate = complete_snapshot();
  duplicate.devices.push_back(duplicate.devices.front());
  duplicate.structural_pci_bdfs.push_back("0000:02:00.0");
  const auto duplicate_receipt = capture(std::move(duplicate));
  require(duplicate_receipt.probes.front().disposition ==
                  ProbeDisposition::partial &&
              duplicate_receipt.resources.size() <= 4U,
          "duplicate UUID/BDF evidence degrades without emitting duplicates");

  LinuxNvidiaInventoryCollector throwing(
      config(), std::make_shared<ThrowingReadOnlyKernel>());
  require_throws<HostResourceError>(
      [&] { (void)throwing.capture_inventory(); },
      "kernel seam exceptions normalize to HostResourceError");

  auto oversized = complete_snapshot();
  oversized.detail.assign(HostResourceBounds::maximum_probe_detail_bytes + 1U,
                          'x');
  auto fake = std::make_shared<FakeReadOnlyKernel>(std::move(oversized));
  LinuxNvidiaInventoryCollector bounded(config(), fake);
  require_throws<HostResourceError>(
      [&] { (void)bounded.capture_inventory(); },
      "unbounded evidence envelope is rejected before construction");
}

bool optional_live_smoke() {
  if (std::getenv("TRAINVM_RUN_LIVE_NVIDIA_SMOKE") == nullptr)
    return false;
  auto live_config = config();
  live_config.maximum_capture_duration_ns = 5'000'000'000ULL;
  live_config.maximum_snapshot_age_ns = 1'000'000'000ULL;
  LinuxNvidiaInventoryCollector collector(std::move(live_config));
  const auto receipt = capture_host_inventory(collector);
  require(receipt.probes.size() == 1U &&
              receipt.probes.front().vendor == HostAcceleratorVendor::nvidia &&
              receipt.resources.size() <= HostResourceBounds::maximum_resources,
          "live smoke returns a bounded NVIDIA probe without host assumptions");
  const auto &probe = receipt.probes.front();
  std::cout << "LIVE nvidia probe="
            << static_cast<unsigned int>(probe.disposition) << " contexts="
            << (probe.context_details_complete ? "complete" : "unknown")
            << " resources=" << receipt.resources.size()
            << " detail=" << probe.detail << '\n';
  for (const auto &observed : receipt.resources) {
    require((observed.id.stable_id.starts_with("GPU-") ||
             observed.id.stable_id.starts_with("MIG-")) &&
                observed.pci_bdf && !observed.pci_bdf->empty() &&
                observed.device_major.has_value() ==
                    observed.device_minor.has_value() &&
                observed.labels.contains("host-namespace") &&
                observed.labels.contains("nvml-loader") &&
                observed.labels.contains("evidence-scope"),
            "live resource exposes bounded identity, trust, and node facts");
    const auto display = observed.labels.find("display");
    if (display != observed.labels.end())
      require(observed.disposition !=
                  ResourceObservationDisposition::audited_eligible,
              "live full-device display evidence cannot be eligible");
    const auto occupancy = observed.labels.find("occupancy");
    std::cout << "LIVE resource id=" << observed.id.stable_id
              << " bdf=" << *observed.pci_bdf << " node=";
    if (observed.device_major)
      std::cout << *observed.device_major << ':' << *observed.device_minor;
    else
      std::cout << '-';
    std::cout << " display="
              << (display == observed.labels.end() ? "none" : display->second)
              << " occupancy="
              << (occupancy == observed.labels.end() ? "eligible"
                                                     : occupancy->second)
              << " trust=" << observed.labels.at("host-namespace") << '/'
              << observed.labels.at("nvml-loader") << '\n';
  }
  return true;
}

} // namespace

int main() {
  try {
    complete_capture_is_stable_and_display_is_occupied();
    std::cout << "PASS complete\n";
    missing_partial_and_torn_evidence_fail_closed();
    std::cout << "PASS fail-closed\n";
    display_mig_pci_and_mode_regressions();
    std::cout << "PASS display-mig-pci-mode\n";
    real_driver_node_mapping_parsers_are_strict();
    std::cout << "PASS real-node-mapping\n";
    device_node_trust_and_freshness_fail_closed();
    std::cout << "PASS device-node-trust-freshness\n";
    malformed_and_throwing_sources_are_rejected();
    std::cout << "PASS malformed\n";
    if (optional_live_smoke())
      std::cout << "PASS optional-live-smoke\n";
    else
      std::cout << "SKIP optional-live-smoke\n";
    std::cout << "linux NVIDIA inventory tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "linux NVIDIA inventory test failure: " << error.what()
              << '\n';
    return 1;
  }
}
