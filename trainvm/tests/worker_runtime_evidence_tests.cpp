// Worker-originated cache runtime evidence: transport, binding, publication.
//
// The property under test is adversarial rather than functional. A worker is
// the only party that can measure the runtime it runs in, so its measurements
// have to reach the authority; but a worker is also the least trusted party in
// the system, and a cache hit is the most valuable thing it could forge. So
// every case below asks the same question from a different angle: can anything
// the worker writes cause bytes to be reused?
//
// Optional argv[1] is a JSON report measured by the real Python worker probe
// on this host (trainvm/tests/verify_worker_runtime_evidence.py drives it).
// The same assertions then run over a report nothing in this file wrote.

#include "trainvm/worker_runtime_evidence.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "trainvm/reflection_json.hpp"

namespace {

using namespace trainvm;

const std::string kHost = "sha256:" + std::string(64U, 'a');
constexpr std::string_view kBoot = "11111111-1111-4111-8111-111111111111";
constexpr std::string_view kGPU = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
constexpr std::string_view kPCI = "0000:01:00.0";
constexpr std::string_view kNonce = "nonce-1";

std::string hash(char value) {
  return "sha256:" + std::string(64U, value);
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void rejected(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const WorkerRuntimeEvidenceError&) {
    return;
  } catch (const CacheNamespaceAuthorityError&) {
    return;
  } catch (const CacheArtifactAuthorityError&) {
    return;
  }
  throw std::runtime_error(message);
}

AdapterKey adapter_key() {
  return {
      .adapter = "rwkv-lab.test-trainer",
      .version = "1.0.0",
      .runtime = ComponentRuntime::python_worker,
      .operation = "train",
      .contract = "rwkv_lab.test.v1.Train",
  };
}

AdapterRegistry adapter_registry() {
  return AdapterRegistry({AdapterProfile{
      .key = adapter_key(),
      .effect = Effect::process,
      .idempotency = Idempotency::receipt_required,
      .code_fingerprint = hash('1'),
      .required_capabilities = {"artifact.manifest.v1"},
      .lifecycle = {.stateful = true,
                    .graceful_stop = true,
                    .compile = true,
                    .resume_grade = ResumeGrade::compatible},
      .authoring = OperationAuthoringDeclaration{.inputs = {}, .outputs = {}},
  }});
}

// The deployment sealed whatever runtime the worker actually has, so the
// registry's closure fingerprint is the report's. Every case that perturbs one
// of the two therefore perturbs it deliberately.
HostLaunchRegistry launch_registry(const std::string& closure) {
  return HostLaunchRegistry(HostLaunchRegistryDocument{
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = {"/opt/trainvm"},
      .profiles = {HostLaunchProfile{
          .key = adapter_key(),
          .code_fingerprint = hash('1'),
          .bootstrap_runtime_closure_fingerprint = closure,
          .provided_capabilities = {"artifact.manifest.v1"},
          .executable_path = "/opt/trainvm/python",
          .executable_fingerprint = hash('2'),
          .code_path = "/opt/trainvm/worker.py",
          .public_arguments = {"--trainvm-bootstrap-fd=4"},
          .working_directory = "/opt/trainvm/work",
      }},
  });
}

WorkerInvocationSpec invocation(bool accelerated) {
  nlohmann::json resources = nlohmann::json::object();
  if (accelerated) {
    resources["accelerators"] = {{"count", 1U}, {"vendor", "nvidia"}};
  }
  nlohmann::json body{
      {"adapter", encode_json(adapter_key())},
      {"api_version", "trainvm.worker-invocation/v1"},
      {"attempt_id", "train@1"},
      {"controls", nlohmann::json::object()},
      {"dispatch_id", "run-1:dispatch:train:train@1"},
      {"effective_control_revision", 0U},
      {"execution", {{"compile", {{"enabled", true}}}}},
      {"host_id", kHost},
      {"inputs", {{"config_digest", hash('3')}}},
      {"node_id", "train"},
      {"observability", nlohmann::json::object()},
      {"plan_hash", std::string(64U, 'b')},
      {"plan_revision", 1U},
      {"publishes", nlohmann::json::object()},
      {"resources", std::move(resources)},
      {"run_id", "run-1"},
      {"training", nullptr},
      {"workspace", nlohmann::json::object()},
  };
  body["invocation_digest"] = "sha256:" + sha256_hex(body.dump());
  return worker_invocation_from_canonical_json(body.dump());
}

HostInventoryReceipt inventory() {
  ObservedHostResource gpu{};
  gpu.id = {.kind = HostResourceKind::accelerator,
            .vendor = HostAcceleratorVendor::nvidia,
            .stable_id = std::string(kGPU),
            .parent_id = std::nullopt};
  gpu.disposition = ResourceObservationDisposition::audited_eligible;
  gpu.compute_contexts = ResourceContextDisposition::absent;
  gpu.graphics_contexts = ResourceContextDisposition::absent;
  gpu.pci_bdf = std::string(kPCI);
  gpu.device_major = 195U;
  gpu.device_minor = 0U;
  gpu.total_memory_bytes = 96ULL << 30U;
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = kHost,
      .boot_id = std::string(kBoot),
      .broker_epoch = "broker-1",
      .begin_revision = "inventory-1",
      .end_revision = "inventory-1",
      .probes = {{.vendor = HostAcceleratorVendor::nvidia,
                  .disposition = ProbeDisposition::complete,
                  .context_details_complete = true,
                  .detail = "complete"}},
      .resources = {std::move(gpu)},
  };
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResolvedLaunchSpec launch(const HostLaunchRegistry& launches,
                          const HostInventoryReceipt& host_inventory,
                          const std::string& closure, bool accelerated) {
  std::optional<HostLaunchGrantClaim> grant;
  if (accelerated) {
    grant = HostLaunchGrantClaim{
        .request_id = "request-1",
        .grant_digest = hash('4'),
        .fences = {ResourceFence{
            .resource = host_inventory.resources.front().id,
            .generation = 7U,
            .inventory_digest = host_inventory.inventory_digest,
            .topology_digest = host_inventory.topology_digest,
        }},
    };
  }
  ResolvedLaunchIdentity identity{
      .api_version = "trainvm.resolved-launch/v4",
      .launch_event_id = "run-1:worker-launch:train:train@1",
      .run_id = "run-1",
      .node_id = "train",
      .attempt_id = "train@1",
      .launch_nonce = std::string(kNonce),
      .adapter_key = adapter_key(),
      .code_fingerprint = hash('1'),
      .bootstrap_runtime_closure_fingerprint = closure,
      .required_capabilities = {"artifact.manifest.v1"},
      .provided_capabilities = {"artifact.manifest.v1"},
      .host_registry_digest = launches.registry_digest(),
      .host_profile_digest = launches.profile_digest(adapter_key(), hash('1')),
      .concurrency_key = "gpu:0",
      .lease_id = "lease-1",
      .fencing_token = 9U,
      .host_grant = std::move(grant),
      .host = {.host_id = kHost, .boot_id = std::string(kBoot)},
      .executable = {.source_path = "/opt/trainvm/python",
                     .source_device = 1U,
                     .source_inode = 2U,
                     .source_size = 4096U,
                     .source_mode = 0100500U,
                     .source_uid = 0U,
                     .source_gid = 0U,
                     .sealed_sha256 = hash('2')},
      .code = VerifiedLaunchArtifact{.source_path = "/opt/trainvm/worker.py",
                                     .source_device = 1U,
                                     .source_inode = 3U,
                                     .source_size = 2048U,
                                     .source_mode = 0100400U,
                                     .source_uid = 0U,
                                     .source_gid = 0U,
                                     .sealed_sha256 = hash('1')},
      .public_arguments = {"--trainvm-bootstrap-fd=4"},
      .working_directory = {.source_path = "/opt/trainvm/work",
                            .device = 1U,
                            .inode = 4U,
                            .mode = 0040500U,
                            .uid = 0U,
                            .gid = 0U},
  };
  const std::string spec_digest =
      "sha256:" + sha256_hex(resolved_launch_identity_json(identity).dump());
  return {.identity = std::move(identity), .spec_digest = spec_digest};
}

CacheCompileInputManifest compile_inputs(
    const WorkerInvocationSpec& worker_invocation) {
  return {
      .api_version = "trainvm.cache-compile-inputs/v1",
      .invocation_digest = worker_invocation.invocation_digest,
      .model_topology_digest = hash('5'),
      .shape_set_digest = hash('6'),
      .dtype_precision_digest = hash('7'),
      .runtime_options_digest = hash('8'),
      .embedded_constants_digest = hash('9'),
      .compiler_configuration_digest = hash('a'),
      .checkpoint_fingerprints = {hash('b')},
  };
}

WorkerRuntimeEvidenceReport accelerated_report() {
  return {
      .api_version = "trainvm.worker-runtime-evidence/v1",
      .run_id = "run-1",
      .node_id = "train",
      .attempt_id = "train@1",
      .launch_nonce = std::string(kNonce),
      .concurrency_key = "gpu:0",
      .lease_id = "lease-1",
      .fencing_token = 9U,
      .compute_device_vendor = "nvidia",
      .compute_architecture = "sm_120",
      .compute_device_uuid = std::string(kGPU),
      .compute_device_pci_address = std::string(kPCI),
      .driver_version = "610.43.03",
      .runtime_versions = {{.name = "cuda", .version = "13.1"},
                           {.name = "python", .version = "3.13.5"},
                           {.name = "torch", .version = "2.10.0+cu130"}},
      .runtime_closure_fingerprint = hash('c'),
      .host_abi_digest = hash('d'),
      .compute_compatibility_digest = hash('e'),
  };
}

// A worker probe is a program, not a person, so a "measured" report from the
// CPU path looks exactly like this. The Python parity run replaces it with one
// this file did not write.
WorkerRuntimeEvidenceReport cpu_report() {
  WorkerRuntimeEvidenceReport report = accelerated_report();
  report.compute_device_vendor = "cpu";
  report.compute_architecture = "x86_64";
  report.compute_device_uuid = std::nullopt;
  report.compute_device_pci_address = std::nullopt;
  report.driver_version = "none";
  report.runtime_versions = {{.name = "python", .version = "3.13.5"}};
  return report;
}

void remove_test_tree(const std::filesystem::path& root) {
  if (!std::filesystem::exists(root)) return;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied)) {
    std::error_code ignored;
    std::filesystem::permissions(entry.path(),
                                 std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::add, ignored);
  }
  std::filesystem::remove_all(root);
}

struct ScopedTestTree {
  std::filesystem::path root;
  ~ScopedTestTree() { remove_test_tree(root); }
};

std::size_t receipt_count(const std::filesystem::path& root) {
  std::size_t count = 0U;
  for (const auto& entry :
       std::filesystem::directory_iterator(root / "runtime")) {
    (void)entry;
    ++count;
  }
  return count;
}

// One complete pass: transport, binding, publication, and derivation, against
// one report. Runs twice -- once over the report written above, once over the
// report a real worker probe measured on this host.
void exercise(const WorkerRuntimeEvidenceReport& report, bool accelerated,
              const std::string& label) {
  const std::string closure = report.runtime_closure_fingerprint;
  const AdapterRegistry adapters = adapter_registry();
  const HostLaunchRegistry launches = launch_registry(closure);
  const HostInventoryReceipt host_inventory = inventory();
  const WorkerInvocationSpec worker_invocation = invocation(accelerated);
  const ResolvedLaunchSpec resolved_launch =
      launch(launches, host_inventory, closure, accelerated);
  const HostIdentity host{.host_id = kHost, .boot_id = std::string(kBoot)};
  const WorkerRuntimeEvidenceBinding binding{
      .host = host,
      .launch = resolved_launch,
      .inventory = host_inventory,
      .placement_specific = accelerated,
  };
  const CacheNamespaceAuthorityRequest request{
      .invocation = worker_invocation,
      .launch = resolved_launch,
      .inventory = host_inventory,
      .compile_inputs = compile_inputs(worker_invocation),
      .placement_specific = accelerated,
  };

  // ---- The transport carries measurements, and cannot be made to carry a
  // receipt. -------------------------------------------------------------
  const nlohmann::json wire = worker_runtime_evidence_json(report);
  std::set<std::string> keys;
  for (const auto& member : wire.items()) keys.insert(member.key());
  std::set<std::string> expected{
      "api_version",
      "attempt_id",
      "compute_architecture",
      "compute_compatibility_digest",
      "compute_device_vendor",
      "concurrency_key",
      "driver_version",
      "fencing_token",
      "host_abi_digest",
      "launch_nonce",
      "lease_id",
      "node_id",
      "run_id",
      "runtime_closure_fingerprint",
      "runtime_versions",
  };
  // Placement identity is present only where the launch has one, which is the
  // same rule the namespace claim enforces.
  if (report.compute_device_uuid) expected.insert("compute_device_uuid");
  if (report.compute_device_pci_address) {
    expected.insert("compute_device_pci_address");
  }
  require(keys == expected,
          label +
              ": the worker transport carries exactly the measured facts and "
              "the identity claims the authority already knows");
  for (const std::string_view owned :
       {"host_id", "boot_id", "launch_spec_digest", "inventory_receipt_digest",
        "resource_binding_digest", "receipt_name", "namespace_digest",
        "receipt_digest"}) {
    require(!keys.contains(std::string(owned)),
            label + ": the worker transport must never grow the "
                    "authority-derived field " + std::string(owned));
  }
  require(worker_runtime_evidence_from_json(wire) == report,
          label + ": a measured report survives its canonical wire form");

  const AdmittedWorkerRuntimeEvidence admitted =
      admit_worker_runtime_evidence(report, binding);
  // A caller-authored receipt -- the exact document the sealed probe reads --
  // has no way into the transport. This is the shape a compromised worker
  // would reach for first.
  rejected(
      [&] {
        (void)worker_runtime_evidence_from_json(
            cache_runtime_probe_receipt_json(admitted.context,
                                             admitted.snapshot));
      },
      label + ": a sealed runtime receipt cannot be submitted as a worker "
              "report");
  {
    nlohmann::json grafted = wire;
    grafted["resource_binding_digest"] = hash('f');
    rejected([&] { (void)worker_runtime_evidence_from_json(grafted); },
             label + ": a report carrying an authority-derived digest is "
                     "refused rather than ignored");
  }

  // ---- The receipt identity comes from the authority, never the report. ---
  const CacheRuntimeProbeContext derived = cache_runtime_probe_context(
      host, resolved_launch, host_inventory, accelerated);
  require(admitted.context == derived &&
              admitted.snapshot.launch_spec_digest ==
                  resolved_launch.spec_digest &&
              admitted.snapshot.inventory_receipt_digest ==
                  host_inventory.receipt_digest &&
              admitted.snapshot.resource_binding_digest ==
                  derived.resource_binding_digest &&
              admitted.snapshot.host_id == kHost,
          label + ": admission derives the probe context from sealed launch "
                  "and inventory authority");

  ScopedTestTree tree{std::filesystem::temp_directory_path() /
                      ("trainvm-worker-evidence-" + label + "-" +
                       std::to_string(static_cast<long long>(::getpid())))};
  remove_test_tree(tree.root);
  std::filesystem::create_directories(tree.root / "runtime");
  std::filesystem::create_directories(tree.root / "qualification");
  const LinuxCacheEvidenceConfig evidence_config{
      .receipt_root = tree.root,
      .authority_uid = ::geteuid(),
      .maximum_receipt_bytes = 1U << 20U,
  };
  LinuxCacheEvidencePublisher publisher(evidence_config);

  // ---- Every forged binding is refused before anything is published. ------
  struct Forgery {
    std::string description;
    WorkerRuntimeEvidenceReport report;
  };
  std::vector<Forgery> forgeries;
  {
    auto forged = report;
    forged.fencing_token += 1U;
    forgeries.push_back({"a superseded fencing token", std::move(forged)});
  }
  {
    auto forged = report;
    forged.lease_id = "lease-2";
    forgeries.push_back({"a lease it does not hold", std::move(forged)});
  }
  {
    auto forged = report;
    forged.attempt_id = "train@2";
    forgeries.push_back({"another attempt", std::move(forged)});
  }
  {
    auto forged = report;
    forged.launch_nonce = "nonce-2";
    forgeries.push_back({"another launch of this attempt", std::move(forged)});
  }
  {
    auto forged = report;
    forged.run_id = "run-2";
    forgeries.push_back({"another run", std::move(forged)});
  }
  {
    auto forged = report;
    forged.runtime_closure_fingerprint = hash('f');
    forgeries.push_back({"a runtime closure the launch did not seal",
                         std::move(forged)});
  }
  if (accelerated) {
    auto uuid = report;
    uuid.compute_device_uuid = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff";
    forgeries.push_back({"a device it was not fenced to", std::move(uuid)});
    auto address = report;
    address.compute_device_pci_address = "0000:02:00.0";
    forgeries.push_back({"a device address it was not fenced to",
                         std::move(address)});
  } else {
    auto vendor = report;
    vendor.compute_device_vendor = "nvidia";
    forgeries.push_back({"an accelerator it holds no fence for",
                         std::move(vendor)});
    auto placement = report;
    placement.compute_device_uuid = std::string(kGPU);
    forgeries.push_back({"placement identity in a portable namespace",
                         std::move(placement)});
  }
  for (const Forgery& forgery : forgeries) {
    rejected(
        [&] {
          (void)publish_worker_runtime_evidence(publisher, forgery.report,
                                                binding);
        },
        label + ": a worker claiming " + forgery.description +
            " must not reach the immutable publisher");
  }
  require(receipt_count(tree.root) == 0U,
          label + ": a refused report publishes nothing at all");

  // Refused, and therefore no reuse: the derivation the cache namespace needs
  // has no receipt to read.
  LinuxSealedCacheRuntimeProbe sealed(evidence_config);
  CacheNamespaceAuthority authority(adapters, launches, host, sealed);
  rejected([&] { (void)authority.derive(request); },
           label + ": with no admitted worker evidence there is no namespace "
                   "to reuse");

  // ---- The admitted report, and only it, feeds derivation. ---------------
  const std::string name =
      publish_worker_runtime_evidence(publisher, report, binding);
  require(name == cache_runtime_probe_receipt_name(derived) &&
              receipt_count(tree.root) == 1U,
          label + ": the published receipt is named by the authority's own "
                  "context");
  const CacheNamespaceAuthorityReceipt first = authority.derive(request);
  require(first.cache_namespace.evidence.runtime_closure_fingerprint ==
                  closure &&
              first.cache_namespace.evidence.compute_device_vendor ==
                  report.compute_device_vendor &&
              first.cache_namespace.evidence.host_abi_digest ==
                  report.host_abi_digest &&
              first.launch_spec_digest == resolved_launch.spec_digest &&
              first.attempt_id == "train@1" && first.fencing_token == 9U &&
              first.resource_binding_digest == derived.resource_binding_digest,
          label + ": a worker-measured probe reaches cache namespace "
                  "derivation bound to attempt, fence, launch and devices");
  require((first.cache_namespace.evidence.compute_device_uuid.has_value() ==
           accelerated),
          label + ": placement identity survives exactly where the launch is "
                  "placement specific");
  require(publish_worker_runtime_evidence(publisher, report, binding) == name &&
              authority.derive(request) == first,
          label + ": an identical second report is exact replay, not a second "
                  "namespace");

  // ---- A later worker cannot rewrite the measurement history. ------------
  auto revised = report;
  revised.compute_architecture = "sm_999";
  rejected(
      [&] { (void)publish_worker_runtime_evidence(publisher, revised, binding); },
      label + ": a second worker measurement never replaces the receipt the "
              "first attempt is bound to");
  require(authority.derive(request) == first,
          label + ": the namespace after a refused revision is the one the "
                  "admitted evidence produced");

  // A worker that measures a genuinely different runtime does not get a
  // different namespace; it gets no namespace, which is the cold path.
  auto drifted = report;
  drifted.runtime_closure_fingerprint = hash('0');
  rejected(
      [&] { (void)publish_worker_runtime_evidence(publisher, drifted, binding); },
      label + ": drifted runtime closure is cold, not a second namespace");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    exercise(accelerated_report(), true, "accelerated");
    exercise(cpu_report(), false, "portable");
    if (argc > 1) {
      std::ifstream stream(argv[1]);
      if (!stream) {
        throw std::runtime_error("measured worker report could not be read");
      }
      std::ostringstream buffer;
      buffer << stream.rdbuf();
      const WorkerRuntimeEvidenceReport measured =
          worker_runtime_evidence_from_json(
              nlohmann::json::parse(buffer.str()));
      require(measured.compute_device_vendor == "cpu",
              "the measured parity report is the portable CPU probe");
      exercise(measured, false, "measured");
    }
  } catch (const std::exception& exception) {
    std::cerr << "worker_runtime_evidence_tests: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
