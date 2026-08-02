#include "trainvm/cache_artifact_authority.hpp"
#include "trainvm/journal_cache_lease_authority.hpp"
#include "trainvm/linux_cache_evidence.hpp"
#include "trainvm/linux_immutable_cache_store.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

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

class LeaseAuthority final : public ICacheLeaseAuthority {
public:
  bool current{true};
  std::size_t calls{};

  void require_current(const ResourceLease &) override {
    ++calls;
    if (!current) {
      throw CacheArtifactAuthorityError("test lease is stale");
    }
  }
};

class ArtifactStore final : public ICacheArtifactStore {
public:
  bool mutate_on_verify{};
  std::size_t publications{};
  std::size_t verifications{};
  ImmutableCacheTreeReceipt receipt;

  ImmutableCacheTreeReceipt
  publish(const CacheNamespaceAuthorityReceipt &authority,
          const CacheArtifactCandidate &) override {
    ++publications;
    receipt = {
        .api_version = "trainvm.immutable-cache-tree/v1",
        .namespace_digest = authority.cache_namespace.namespace_digest,
        .artifact_tree_digest = hash('3'),
        .manifest_digest = hash('4'),
        .content_address =
            "cache/" + authority.cache_namespace.namespace_digest.substr(7U) +
            "/" + hash('3').substr(7U),
        .file_count = 7U,
        .total_bytes = 4096U,
        .immutable = true,
        .store_receipt_digest = hash('5'),
    };
    return receipt;
  }

  ImmutableCacheTreeReceipt verify(const std::string &) override {
    ++verifications;
    auto result = receipt;
    if (mutate_on_verify)
      result.artifact_tree_digest = hash('f');
    return result;
  }
};

CacheQualificationEvidence qualification_evidence() {
  return {
      .api_version = "trainvm.cache-qualification-evidence/v1",
      .authority_receipt_digest = hash('1'),
      .namespace_digest = hash('2'),
      .artifact_tree_digest = hash('3'),
      .workload_class = CacheWorkloadClass::training,
      .baseline_run_digest = hash('6'),
      .candidate_run_digest = hash('7'),
      .shape_coverage_digest = hash('8'),
      .transition_coverage = true,
      .baseline_instrumented = false,
      .candidate_instrumented = false,
      .output_parity = true,
      .gradient_parity = true,
      .optimizer_update_parity = true,
      .state_parity = true,
      .resumed_trajectory_parity = true,
      .determinism_parity = true,
      .content_parity = false,
      .ordering_parity = false,
      .manifest_parity = false,
      .model_quality_pass = true,
      .baseline_throughput = 100.0,
      .candidate_throughput = 125.0,
      .baseline_peak_memory_bytes = 1000U,
      .candidate_peak_memory_bytes = 1050U,
      .minimum_throughput_gain_ratio = 0.10,
      .maximum_memory_regression_ratio = 0.10,
  };
}

class QualificationAuthority final : public ICacheQualificationEvidenceSource {
public:
  CacheQualificationEvidence evidence{qualification_evidence()};
  std::size_t capture_calls{};
  std::size_t verification_calls{};
  bool trusted{true};
  bool misbind{};

  CacheQualificationEvidence
  capture(const CacheNamespaceAuthorityReceipt& authority,
          const ImmutableCacheTreeReceipt& artifact) override {
    ++capture_calls;
    CacheQualificationEvidence result = evidence;
    result.authority_receipt_digest = authority.receipt_digest;
    result.namespace_digest = artifact.namespace_digest;
    result.artifact_tree_digest = artifact.artifact_tree_digest;
    if (misbind) result.artifact_tree_digest = hash('f');
    return result;
  }

  void
  require_trusted(const CacheNamespaceAuthorityReceipt &authority,
                  const ImmutableCacheTreeReceipt &artifact,
                  const CacheQualificationReceipt &qualification) override {
    ++verification_calls;
    CacheQualificationEvidence expected = evidence;
    expected.authority_receipt_digest = authority.receipt_digest;
    expected.namespace_digest = artifact.namespace_digest;
    expected.artifact_tree_digest = artifact.artifact_tree_digest;
    if (!trusted || qualification.evidence != expected) {
      throw CacheArtifactAuthorityError(
          "test qualification evidence is not trusted");
    }
  }
};

ResourceLease lease_for(const CacheNamespaceAuthorityReceipt &receipt) {
  return {
      .concurrency_key = receipt.concurrency_key,
      .owner_run_id = receipt.run_id,
      .lease_id = receipt.lease_id,
      .fencing_token = receipt.fencing_token,
      .clock_domain = ResourceLease::kBootTimeDomain,
      .boot_id = std::string(kBoot),
      .acquired_boottime_ns = 1U,
      .expires_boottime_ns = 1000U,
      .acquired_wall_time_ns = 1U,
      .expires_wall_time_ns = 1000U,
  };
}

AuthorityTimeSample authority_time(std::int64_t value) {
  return {.wall = {.nanoseconds = value},
          .boot = {.nanoseconds = value},
          .boot_id = std::string(kBoot)};
}

void remove_test_tree(const std::filesystem::path &root) {
  if (!std::filesystem::exists(root))
    return;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(
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

void write_immutable_receipt(const std::filesystem::path& path,
                             const nlohmann::json& document) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << document.dump();
  stream.close();
  if (!stream) throw std::runtime_error("test receipt write failed");
  std::filesystem::permissions(
      path, std::filesystem::perms::owner_read |
                std::filesystem::perms::group_read,
      std::filesystem::perm_options::replace);
}

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

    ScopedTestTree evidence_tree{
        std::filesystem::temp_directory_path() /
        ("trainvm-cache-evidence-" +
         std::to_string(static_cast<long long>(::getpid())))};
    remove_test_tree(evidence_tree.root);
    std::filesystem::create_directories(evidence_tree.root / "runtime");
    std::filesystem::create_directories(evidence_tree.root / "qualification");
    const CacheRuntimeProbeContext runtime_context{
        .host = {.host_id = kHost, .boot_id = std::string(kBoot)},
        .launch_spec_digest = fixture.resolved_launch.spec_digest,
        .inventory_receipt_digest = fixture.host_inventory.receipt_digest,
        .resource_binding_digest = first.resource_binding_digest,
        .selected_resources = fixture.host_inventory.resources,
        .placement_specific = true,
    };
    const CacheRuntimeProbeSnapshot runtime_snapshot =
        probe.capture(runtime_context);
    const auto runtime_receipt_path =
        evidence_tree.root / "runtime" /
        cache_runtime_probe_receipt_name(runtime_context);
    write_immutable_receipt(
        runtime_receipt_path,
        cache_runtime_probe_receipt_json(runtime_context, runtime_snapshot));
    LinuxSealedCacheRuntimeProbe sealed_runtime({
        .receipt_root = evidence_tree.root,
        .authority_uid = ::geteuid(),
        .maximum_receipt_bytes = 1U << 20U,
    });
    CacheNamespaceAuthority sealed_namespace_authority(
        fixture.adapters, fixture.launches,
        {.host_id = kHost, .boot_id = std::string(kBoot)}, sealed_runtime);
    require(sealed_namespace_authority.derive(fixture.request()) == first,
            "an immutable runtime receipt reproduces the exact authority "
            "namespace without accepting runtime identity from the plan");
    std::filesystem::permissions(
        runtime_receipt_path, std::filesystem::perms::owner_write,
        std::filesystem::perm_options::add);
    rejected([&] { (void)sealed_runtime.capture(runtime_context); },
             "a writable runtime receipt cannot authorize a cache namespace");
    std::filesystem::permissions(
        runtime_receipt_path, std::filesystem::perms::owner_write,
        std::filesystem::perm_options::remove);
    const auto runtime_receipt_alias = evidence_tree.root / "runtime-alias";
    std::filesystem::create_hard_link(runtime_receipt_path,
                                      runtime_receipt_alias);
    rejected([&] { (void)sealed_runtime.capture(runtime_context); },
             "a multiply-linked runtime receipt is outside immutable receipt "
             "authority");
    std::filesystem::remove(runtime_receipt_alias);

    const auto qualified = qualify_cache_artifact(qualification_evidence());
    require(qualified.qualified && qualified.rejection_reasons.empty() &&
                cache_qualification_receipt_json(qualified).at(
                    "receipt_digest") == qualified.receipt_digest,
            "training cache qualification requires parity, quality, coverage, "
            "and unprofiled speed evidence");
    auto slow = qualification_evidence();
    slow.candidate_throughput = 105.0;
    slow.gradient_parity = false;
    const auto rejected_qualification = qualify_cache_artifact(slow);
    require(!rejected_qualification.qualified &&
                rejected_qualification.rejection_reasons ==
                    std::vector<std::string>{"gradient_parity_failed",
                                             "throughput_gate_failed"},
            "failed correctness and speed gates produce a deterministic "
            "rejection receipt");
    auto forged_qualification = rejected_qualification;
    forged_qualification.qualified = true;
    forged_qualification.rejection_reasons.clear();
    const nlohmann::json forged_qualification_body{
        {"api_version", forged_qualification.api_version},
        {"evidence", encode_json(forged_qualification.evidence)},
        {"qualified", forged_qualification.qualified},
        {"rejection_reasons", forged_qualification.rejection_reasons},
    };
    forged_qualification.receipt_digest =
        "sha256:" + sha256_hex(nlohmann::json{
                        {"domain", "trainvm.cache-qualification-receipt/v1"},
                        {"value", forged_qualification_body},
                    }
                                   .dump());
    rejected(
        [&] { (void)cache_qualification_receipt_json(forged_qualification); },
        "a self-hashed receipt cannot override the deterministic qualification "
        "decision");

    LeaseAuthority lease_authority;
    ArtifactStore artifact_store;
    QualificationAuthority qualification_authority;
    CacheArtifactAuthority artifact_authority(lease_authority, artifact_store,
                                              qualification_authority);
    auto publication = artifact_authority.publish({
        .authority = first,
        .publisher_lease = lease_for(first),
        .candidate = {.source_directory = "/run/trainvm/cache-candidate",
                      .maximum_file_count = 100U,
                      .maximum_total_bytes = 1U << 20U},
    });
    require(
        publication.qualification.qualified &&
            publication.artifact.file_count == 7U &&
            publication.publisher_lease.fencing_token == first.fencing_token &&
            lease_authority.calls == 2U && artifact_store.publications == 1U &&
            cache_artifact_publication_receipt_json(publication)
                    .at("publication_digest") == publication.publication_digest,
        "publication binds qualification and immutable tree creation between "
        "two live-fence checks");

    qualification_authority.evidence.candidate_throughput = 105.0;
    rejected(
        [&] {
          (void)artifact_authority.publish({
              .authority = first,
              .publisher_lease = lease_for(first),
              .candidate =
                  {
                      .source_directory = "/run/trainvm/cache-candidate",
                      .maximum_file_count = 100U,
                      .maximum_total_bytes = 1U << 20U,
                  },
          });
        },
        "a trusted qualification source must veto a candidate that misses its "
        "speed gate");
    require(qualification_authority.capture_calls == 2U &&
                artifact_store.publications == 2U &&
                lease_authority.calls == 3U,
            "qualification is captured only after a current publisher stores "
            "exact candidate bytes");
    qualification_authority.evidence = qualification_evidence();
    qualification_authority.misbind = true;
    rejected(
        [&] {
          (void)artifact_authority.publish({
              .authority = first,
              .publisher_lease = lease_for(first),
              .candidate = {
                  .source_directory = "/run/trainvm/cache-candidate",
                  .maximum_file_count = 100U,
                  .maximum_total_bytes = 1U << 20U,
              },
          });
        },
        "publication must not repair qualification evidence bound to other "
        "artifact bytes");
    qualification_authority.misbind = false;
    lease_authority.calls = 0U;

    const auto adoption = artifact_authority.adopt({
        .current_authority = first,
        .adopter_lease = lease_for(first),
        .publication = publication,
    });
    require(adoption.content_address == publication.artifact.content_address &&
                adoption.publication_digest == publication.publication_digest &&
                adoption.grant_digest.starts_with("sha256:") &&
                lease_authority.calls == 2U &&
                artifact_store.verifications == 1U &&
                qualification_authority.verification_calls == 1U,
            "adoption re-verifies immutable bytes and the current owner around "
            "the store read");

    qualification_authority.trusted = false;
    rejected(
        [&] {
          (void)artifact_authority.adopt({
              .current_authority = first,
              .adopter_lease = lease_for(first),
              .publication = publication,
          });
        },
        "adoption must re-attest qualification through its trusted evidence "
        "source");
    qualification_authority.trusted = true;

    artifact_store.mutate_on_verify = true;
    rejected(
        [&] {
          (void)artifact_authority.adopt({
              .current_authority = first,
              .adopter_lease = lease_for(first),
              .publication = publication,
          });
        },
        "immutable-store mutation must reject cache adoption");
    artifact_store.mutate_on_verify = false;

    auto wrong_lease = lease_for(first);
    wrong_lease.fencing_token += 1U;
    rejected(
        [&] {
          (void)artifact_authority.publish({
              .authority = first,
              .publisher_lease = wrong_lease,
              .candidate = {.source_directory = "/run/trainvm/cache-candidate",
                            .maximum_file_count = 100U,
                            .maximum_total_bytes = 1U << 20U},
          });
        },
        "a different publisher fence must not enter the immutable store");

    const auto store_temporary =
        std::filesystem::temp_directory_path() /
        ("trainvm-cache-store-" +
         std::to_string(static_cast<long long>(::getpid())));
    remove_test_tree(store_temporary);
    const auto source_root = store_temporary / "source";
    const auto candidate_root = source_root / "candidate";
    const auto publication_root = store_temporary / "published";
    std::filesystem::create_directories(candidate_root / "nested");
    std::filesystem::create_directories(publication_root);
    std::ofstream(candidate_root / "graph.bin", std::ios::binary)
        << "compiled-graph";
    std::ofstream(candidate_root / "nested" / "kernel.bin", std::ios::binary)
        << "compiled-kernel";
    try {
      LinuxImmutableCacheStore linux_store({
          .publication_root = publication_root,
          .allowed_source_roots = {source_root},
          .authority_uid = ::geteuid(),
          .source_uid = ::geteuid(),
          .maximum_file_count = 100U,
          .maximum_total_bytes = 1U << 20U,
          .maximum_single_file_bytes = 1U << 20U,
      });
      CacheArtifactAuthority linux_authority(lease_authority, linux_store,
                                             qualification_authority);
      const auto linux_publication = linux_authority.publish({
          .authority = first,
          .publisher_lease = lease_for(first),
          .candidate = {.source_directory = candidate_root.string(),
                        .maximum_file_count = 100U,
                        .maximum_total_bytes = 1U << 20U},
      });
      require(
          linux_publication.artifact.file_count == 2U &&
              linux_publication.artifact.total_bytes == 29U &&
              linux_store.verify(linux_publication.artifact.content_address) ==
                  linux_publication.artifact,
          "Linux cache store descriptor-copies, hashes, promotes, and "
          "re-verifies a nested cache tree");

      CacheQualificationEvidence immutable_evidence =
          qualification_evidence();
      immutable_evidence.authority_receipt_digest = first.receipt_digest;
      immutable_evidence.namespace_digest =
          linux_publication.artifact.namespace_digest;
      immutable_evidence.artifact_tree_digest =
          linux_publication.artifact.artifact_tree_digest;
      const auto qualification_receipt_path =
          evidence_tree.root / "qualification" /
          cache_qualification_evidence_receipt_name(
              first, linux_publication.artifact);
      write_immutable_receipt(
          qualification_receipt_path,
          cache_qualification_evidence_receipt_json(
              first, linux_publication.artifact, immutable_evidence));
      LinuxImmutableCacheQualificationSource immutable_qualification({
          .receipt_root = evidence_tree.root,
          .authority_uid = ::geteuid(),
          .maximum_receipt_bytes = 1U << 20U,
      });
      CacheArtifactAuthority immutable_evidence_authority(
          lease_authority, linux_store, immutable_qualification);
      const auto immutable_publication =
          immutable_evidence_authority.publish({
              .authority = first,
              .publisher_lease = lease_for(first),
              .candidate = {.source_directory = candidate_root.string(),
                            .maximum_file_count = 100U,
                            .maximum_total_bytes = 1U << 20U},
          });
      const auto immutable_adoption = immutable_evidence_authority.adopt({
          .current_authority = first,
          .adopter_lease = lease_for(first),
          .publication = immutable_publication,
      });
      require(immutable_publication.artifact == linux_publication.artifact &&
                  immutable_adoption.content_address ==
                      linux_publication.artifact.content_address,
              "immutable qualification evidence authorizes publication and "
              "is re-attested before cache adoption");
      std::filesystem::permissions(
          qualification_receipt_path,
          std::filesystem::perms::owner_write,
          std::filesystem::perm_options::add);
      rejected(
          [&] {
            (void)immutable_qualification.capture(
                first, linux_publication.artifact);
          },
          "writable qualification evidence cannot authorize cache adoption");
      std::filesystem::permissions(
          qualification_receipt_path,
          std::filesystem::perms::owner_write,
          std::filesystem::perm_options::remove);

      const auto replay = linux_authority.publish({
          .authority = first,
          .publisher_lease = lease_for(first),
          .candidate = {.source_directory = candidate_root.string(),
                        .maximum_file_count = 100U,
                        .maximum_total_bytes = 1U << 20U},
      });
      require(
          replay.artifact == linux_publication.artifact,
          "identical Linux cache publications converge on one content address");

      const std::size_t separator =
          linux_publication.artifact.content_address.find('/');
      const auto payload_file =
          publication_root / "namespaces" /
          linux_publication.artifact.content_address.substr(0U, separator) /
          "artifacts" /
          linux_publication.artifact.content_address.substr(separator + 1U) /
          "payload" / "graph.bin";
      std::filesystem::permissions(payload_file,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::add);
      rejected(
          [&] {
            (void)linux_store.verify(
                linux_publication.artifact.content_address);
          },
          "writable published cache bytes must fail immutable verification");
      std::filesystem::permissions(payload_file,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::remove);

      const auto payload_directory = payload_file.parent_path();
      const auto undeclared_directory = payload_directory / "undeclared";
      std::filesystem::permissions(payload_directory,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::add);
      std::filesystem::create_directory(undeclared_directory);
      std::filesystem::permissions(undeclared_directory,
                                   std::filesystem::perms::owner_read |
                                       std::filesystem::perms::owner_exec |
                                       std::filesystem::perms::group_read |
                                       std::filesystem::perms::group_exec,
                                   std::filesystem::perm_options::replace);
      std::filesystem::permissions(payload_directory,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::remove);
      rejected(
          [&] {
            (void)linux_store.verify(
                linux_publication.artifact.content_address);
          },
          "undeclared directories must fail immutable cache verification");
      std::filesystem::permissions(payload_directory,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::add);
      std::filesystem::permissions(undeclared_directory,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::add);
      std::filesystem::remove(undeclared_directory);
      std::filesystem::permissions(payload_directory,
                                   std::filesystem::perms::owner_write,
                                   std::filesystem::perm_options::remove);

      const auto bad_candidate = source_root / "bad";
      std::filesystem::create_directories(bad_candidate);
      std::filesystem::create_symlink(candidate_root / "graph.bin",
                                      bad_candidate / "escape");
      rejected(
          [&] {
            (void)linux_store.publish(
                first, {.source_directory = bad_candidate.string(),
                        .maximum_file_count = 100U,
                        .maximum_total_bytes = 1U << 20U});
          },
          "Linux cache publication rejects source symlinks");
    } catch (...) {
      remove_test_tree(store_temporary);
      throw;
    }
    remove_test_tree(store_temporary);

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

    const auto temporary = std::filesystem::temp_directory_path() /
                           ("trainvm-cache-lease-" +
                            std::to_string(static_cast<long long>(::getpid())));
    std::filesystem::create_directories(temporary);
    try {
      Journal journal(temporary / "journal.sqlite3", std::nullopt,
                      HostGrantEnforcement::legacy_process_free_test);
      const auto acquired = journal.acquire_lease(
          "cache:test", "run-cache", "lease-cache", authority_time(10), 100);
      require(acquired.status == LeaseAcquireStatus::acquired,
              "journal fixture acquires a cache lease");
      std::int64_t now_value = 20;
      JournalCacheLeaseAuthority journal_leases(
          journal, [&] { return authority_time(now_value); });
      journal_leases.require_current(acquired.lease);
      require(journal.release_lease(
                  acquired.lease.concurrency_key, acquired.lease.owner_run_id,
                  acquired.lease.lease_id, acquired.lease.fencing_token,
                  authority_time(30)),
              "journal fixture releases its cache lease");
      now_value = 40;
      rejected([&] { journal_leases.require_current(acquired.lease); },
               "production cache lease authority rejects a released fence");
    } catch (...) {
      std::filesystem::remove_all(temporary);
      throw;
    }
    std::filesystem::remove_all(temporary);
  } catch (const std::exception& exception) {
    std::cerr << "cache_namespace_authority_tests: " << exception.what()
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
