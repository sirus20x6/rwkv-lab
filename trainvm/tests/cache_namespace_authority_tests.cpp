#include "trainvm/cache_namespace_authority.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace {

using namespace trainvm;

const std::string kHost = "sha256:" + std::string(64U, 'a');
constexpr std::string_view kBoot = "11111111-1111-4111-8111-111111111111";
constexpr std::string_view kGPU =
    "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

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
  } catch (const CacheNamespaceAuthorityError&) {
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

AdapterRegistry adapter_registry(bool compile = true) {
  return AdapterRegistry({AdapterProfile{
      .key = adapter_key(),
      .effect = Effect::process,
      .idempotency = Idempotency::receipt_required,
      .code_fingerprint = hash('1'),
      .required_capabilities = {"artifact.manifest.v1"},
      .lifecycle = {.stateful = true,
                    .graceful_stop = true,
                    .compile = compile,
                    .resume_grade = ResumeGrade::compatible},
  }});
}

HostLaunchRegistry launch_registry() {
  return HostLaunchRegistry(HostLaunchRegistryDocument{
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {"/opt/trainvm"},
      .profiles = {HostLaunchProfile{
          .key = adapter_key(),
          .code_fingerprint = hash('1'),
          .executable_path = "/opt/trainvm/python",
          .executable_fingerprint = hash('2'),
          .code_path = "/opt/trainvm/worker.py",
          .public_arguments = {"--trainvm-bootstrap-fd=4"},
          .working_directory = "/opt/trainvm/work",
      }},
  });
}

WorkerInvocationSpec invocation() {
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
      {"resources", {{"accelerators", {{"count", 1U}, {"vendor", "nvidia"}}}}},
      {"run_id", "run-1"},
      {"training", nullptr},
      {"workspace", nlohmann::json::object()},
  };
  const std::string digest = "sha256:" + sha256_hex(body.dump());
  body["invocation_digest"] = digest;
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
  gpu.pci_bdf = "0000:01:00.0";
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
  FakeHostKernel kernel({{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResolvedLaunchSpec launch(const AdapterRegistry& adapters,
                          const HostLaunchRegistry& launches,
                          const HostInventoryReceipt& inventory) {
  ResourceFence fence{
      .resource = inventory.resources.front().id,
      .generation = 7U,
      .inventory_digest = inventory.inventory_digest,
      .topology_digest = inventory.topology_digest,
  };
  ResolvedLaunchIdentity identity{
      .api_version = "trainvm.resolved-launch/v1",
      .launch_event_id = "run-1:worker-launch:train:train@1",
      .run_id = "run-1",
      .node_id = "train",
      .attempt_id = "train@1",
      .launch_nonce = "nonce-1",
      .adapter_key = adapter_key(),
      .code_fingerprint = hash('1'),
      .required_capabilities = {"artifact.manifest.v1"},
      .host_registry_digest = launches.registry_digest(),
      .host_profile_digest = launches.profile_digest(adapter_key(), hash('1')),
      .concurrency_key = "gpu:0",
      .lease_id = "lease-1",
      .fencing_token = 9U,
      .host_grant = HostLaunchGrantClaim{
          .request_id = "request-1",
          .grant_digest = hash('4'),
          .fences = {std::move(fence)},
      },
      .host = {.host_id = kHost, .boot_id = std::string(kBoot)},
      .executable = {.source_path = "/opt/trainvm/python",
                     .source_device = 1U,
                     .source_inode = 2U,
                     .source_size = 4096U,
                     .source_mode = 0100500U,
                     .source_uid = 0U,
                     .source_gid = 0U,
                     .sealed_sha256 = hash('2')},
      .code = VerifiedLaunchArtifact{
          .source_path = "/opt/trainvm/worker.py",
          .source_device = 1U,
          .source_inode = 3U,
          .source_size = 2048U,
          .source_mode = 0100400U,
          .source_uid = 0U,
          .source_gid = 0U,
          .sealed_sha256 = adapters.resolve(adapter_key()).code_fingerprint,
      },
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
    const WorkerInvocationSpec& invocation) {
  return {
      .api_version = "trainvm.cache-compile-inputs/v1",
      .invocation_digest = invocation.invocation_digest,
      .model_topology_digest = hash('5'),
      .shape_set_digest = hash('6'),
      .dtype_precision_digest = hash('7'),
      .runtime_options_digest = hash('8'),
      .embedded_constants_digest = hash('9'),
      .compiler_configuration_digest = hash('a'),
      .checkpoint_fingerprints = {hash('b')},
  };
}

class Probe final : public ICacheRuntimeProbe {
 public:
  bool wrong_resource{};
  std::size_t calls{};

  CacheRuntimeProbeSnapshot capture(
      const CacheRuntimeProbeContext& context) override {
    ++calls;
    return {
        .api_version = "trainvm.cache-runtime-probe/v1",
        .host_id = context.host.host_id,
        .boot_id = context.host.boot_id,
        .launch_spec_digest = context.launch_spec_digest,
        .inventory_receipt_digest = context.inventory_receipt_digest,
        .resource_binding_digest = wrong_resource ? hash('f')
                                                  : context.resource_binding_digest,
        .compute_device_vendor = "nvidia",
        .compute_architecture = "sm_120",
        .compute_device_uuid =
            context.placement_specific
                ? std::optional<std::string>{std::string(kGPU)}
                : std::nullopt,
        .compute_device_pci_address =
            context.placement_specific
                ? std::optional<std::string>{"0000:01:00.0"}
                : std::nullopt,
        .driver_version = "610.43.03",
        .runtime_versions = {{.name = "cuda", .version = "13.1"},
                             {.name = "pytorch", .version = "2.10.0+cu130"},
                             {.name = "triton", .version = "3.6.0"}},
        .runtime_closure_fingerprint = hash('c'),
        .host_abi_digest = hash('d'),
        .compute_compatibility_digest = hash('e'),
    };
  }
};

struct Fixture {
  AdapterRegistry adapters{adapter_registry()};
  HostLaunchRegistry launches{launch_registry()};
  HostInventoryReceipt host_inventory{inventory()};
  WorkerInvocationSpec worker_invocation{invocation()};
  ResolvedLaunchSpec resolved_launch{
      launch(adapters, launches, host_inventory)};
  CacheCompileInputManifest compile_manifest{
      compile_inputs(worker_invocation)};

  CacheNamespaceAuthorityRequest request() const {
    return {.invocation = worker_invocation,
            .launch = resolved_launch,
            .inventory = host_inventory,
            .compile_inputs = compile_manifest,
            .placement_specific = true};
  }
};

}  // namespace

int main() {
  try {
    Fixture fixture;
    Probe probe;
    CacheNamespaceAuthority authority(
        fixture.adapters, fixture.launches,
        {.host_id = kHost, .boot_id = std::string(kBoot)}, probe);
    const auto first = authority.derive(fixture.request());
    const auto second = authority.derive(fixture.request());
    require(first == second && probe.calls == 2U &&
                first.cache_namespace.evidence.adapter_profile_digest ==
                    fixture.adapters.profile_digest(adapter_key()) &&
                first.cache_namespace.evidence.executable_fingerprint ==
                    hash('2') &&
                first.cache_namespace.evidence.compute_device_uuid == kGPU &&
                first.receipt_digest.starts_with("sha256:") &&
                cache_namespace_authority_receipt_json(first)
                        .at("receipt_digest") == first.receipt_digest,
            "authority builder binds registries, launch, inventory, runtime, compile inputs, and device");

    auto changed = fixture.request();
    changed.compile_inputs.shape_set_digest = hash('f');
    const auto changed_receipt = authority.derive(changed);
    require(changed_receipt.cache_namespace.namespace_digest !=
                first.cache_namespace.namespace_digest,
            "shape changes create a cold authority namespace");

    auto portable = fixture.request();
    portable.placement_specific = false;
    const auto portable_receipt = authority.derive(portable);
    require(!portable_receipt.cache_namespace.evidence.compute_device_uuid &&
                !portable_receipt.cache_namespace.evidence
                     .compute_device_pci_address &&
                portable_receipt.cache_namespace.namespace_digest !=
                    first.cache_namespace.namespace_digest,
            "portable namespaces retain compatibility but no placement identity");

    auto forged = fixture.request();
    forged.launch.identity.executable.sealed_sha256 = hash('f');
    forged.launch.spec_digest = "sha256:" +
        sha256_hex(resolved_launch_identity_json(forged.launch.identity).dump());
    rejected([&] { (void)authority.derive(forged); },
             "sealed executable drift must fail authority derivation");

    forged = fixture.request();
    forged.compile_inputs.invocation_digest = hash('f');
    rejected([&] { (void)authority.derive(forged); },
             "compile manifest must bind the exact worker invocation");

    probe.wrong_resource = true;
    rejected([&] { (void)authority.derive(fixture.request()); },
             "runtime probe must echo the selected resource binding");
    probe.wrong_resource = false;

    auto receipt = first;
    receipt.compile_inputs_digest = hash('f');
    rejected([&] { (void)cache_namespace_authority_receipt_json(receipt); },
             "serialized authority receipts reject manual mutation");

    auto no_compile_adapters = adapter_registry(false);
    CacheNamespaceAuthority no_compile(
        no_compile_adapters, fixture.launches,
        {.host_id = kHost, .boot_id = std::string(kBoot)}, probe);
    rejected([&] { (void)no_compile.derive(fixture.request()); },
             "an adapter without compile authority cannot derive a cache namespace");
  } catch (const std::exception& exception) {
    std::cerr << "cache_namespace_authority_tests: " << exception.what()
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
