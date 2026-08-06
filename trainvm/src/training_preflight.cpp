#include "trainvm/training_preflight.hpp"

#include "trainvm/recipe_profile.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumEnvironmentBytes = 4U << 20U;
constexpr std::size_t kMaximumTrainingNodes = 1'024U;
constexpr std::size_t kMaximumAccelerators = 64U;
constexpr std::uint64_t kMaximumGpuQualificationMilliseconds = 60'000U;
constexpr std::uint64_t kMaximumPassiveSnapshotAgeNs = 60'000'000'000ULL;
constexpr std::uint64_t kGib = 1ULL << 30U;

bool digest_valid(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool boot_id_valid(std::string_view value) {
  if (value.size() != 36U) {
    return false;
  }
  for (std::size_t index = 0U; index < value.size(); ++index) {
    const bool hyphen =
        index == 8U || index == 13U || index == 18U || index == 23U;
    const bool lower_hex = (value[index] >= '0' && value[index] <= '9') ||
                           (value[index] >= 'a' && value[index] <= 'f');
    if (hyphen ? value[index] != '-' : !lower_hex) {
      return false;
    }
  }
  return true;
}

std::string digest(std::string_view domain, const nlohmann::json &value) {
  return "sha256:" + sha256_hex(std::string(domain) + '\0' + value.dump());
}

void diagnose(std::vector<TrainingPreflightDiagnostic> &diagnostics,
              Diagnostic::Severity severity, std::string code, std::string path,
              std::string message, std::string help) {
  diagnostics.push_back({.severity = severity,
                         .code = std::move(code),
                         .path = std::move(path),
                         .message = std::move(message),
                         .help = std::move(help)});
}

void fail(std::vector<TrainingPreflightDiagnostic> &diagnostics,
          std::string code, std::string path, std::string message,
          std::string help) {
  diagnose(diagnostics, Diagnostic::Severity::error, std::move(code),
           std::move(path), std::move(message), std::move(help));
}

bool directory_permission(const struct stat &status, std::uint32_t uid,
                          std::uint32_t gid,
                          const std::vector<std::uint32_t> &supplementary_gids,
                          mode_t requested) {
  mode_t granted = status.st_mode & 0007U;
  if (status.st_uid == static_cast<uid_t>(uid)) {
    granted = (status.st_mode >> 6U) & 0007U;
  } else if (status.st_gid == static_cast<gid_t>(gid) ||
             std::ranges::find(supplementary_gids,
                               static_cast<std::uint32_t>(status.st_gid)) !=
                 supplementary_gids.end()) {
    granted = (status.st_mode >> 3U) & 0007U;
  }
  return (granted & requested) == requested;
}

std::vector<std::filesystem::path>
path_prefixes(const std::filesystem::path &path) {
  std::vector<std::filesystem::path> prefixes;
  std::filesystem::path current;
  for (const auto &component : path) {
    current /= component;
    prefixes.push_back(current);
  }
  return prefixes;
}

struct ExistingPathInspection {
  bool complete{};
  std::filesystem::path nearest_existing;
  struct stat nearest_status{};
};

std::optional<ExistingPathInspection>
inspect_ancestor_chain(const std::filesystem::path &path, std::uint32_t uid,
                       std::uint32_t gid,
                       const std::vector<std::uint32_t> &supplementary_gids,
                       const std::string &diagnostic_path,
                       bool defer_final_permission, bool final_may_be_regular,
                       std::vector<TrainingPreflightDiagnostic> &diagnostics) {
  ExistingPathInspection result;
  const auto prefixes = path_prefixes(path);
  for (std::size_t index = 0U; index < prefixes.size(); ++index) {
    const auto &prefix = prefixes[index];
    struct stat status{};
    if (::lstat(prefix.c_str(), &status) != 0) {
      if (errno == ENOENT) {
        result.complete = false;
        return result;
      }
      fail(diagnostics, "path.inspect_failed", diagnostic_path,
           "could not inspect path component " + prefix.string() + ": " +
               std::generic_category().message(errno),
           "Repair the path or mount before submitting the run.");
      return std::nullopt;
    }
    if (S_ISLNK(status.st_mode)) {
      fail(diagnostics, "path.symlink", diagnostic_path,
           "path component is a symbolic link: " + prefix.string(),
           "Use the canonical non-symlink path already covered by content and "
           "write authority.");
      return std::nullopt;
    }
    const bool final = index + 1U == prefixes.size();
    if (!S_ISDIR(status.st_mode) &&
        !(final && final_may_be_regular && S_ISREG(status.st_mode))) {
      fail(diagnostics, "path.ancestor_not_directory", diagnostic_path,
           "path component is not a directory: " + prefix.string(),
           "Replace the component with a directory or correct the declarative "
           "path.");
      return std::nullopt;
    }
    if (S_ISDIR(status.st_mode) && (!final || !defer_final_permission) &&
        !directory_permission(status, uid, gid, supplementary_gids, 0001U)) {
      fail(diagnostics, "path.ancestor_not_searchable", diagnostic_path,
           "worker uid " + std::to_string(uid) + ":gid " + std::to_string(gid) +
               " cannot search " + prefix.string(),
           "Grant execute/search permission on the full ancestor chain to the "
           "configured hostd worker identity.");
      return std::nullopt;
    }
    result.nearest_existing = prefix;
    result.nearest_status = status;
    result.complete = final;
  }
  return result;
}

void validate_read_root(const InputContentRootIdentity &root, std::size_t index,
                        std::uint32_t uid, std::uint32_t gid,
                        const std::vector<std::uint32_t> &supplementary_gids,
                        std::vector<TrainingPreflightDiagnostic> &diagnostics) {
  const std::string path =
      "/spec/workspace/input_content_roots/" + std::to_string(index) + "/path";
  const auto inspection = inspect_ancestor_chain(
      root.path, uid, gid, supplementary_gids, path, false,
      root.kind == ContentRootKind::file, diagnostics);
  if (!inspection || !inspection->complete) {
    if (inspection && !inspection->complete) {
      fail(diagnostics, "input.missing", path,
           "locked input root does not exist: " + root.path,
           "Restore the exact locked input or re-lock the declarative "
           "experiment after deliberately changing inputs.");
    }
    return;
  }
  const bool kind_matches = (root.kind == ContentRootKind::directory &&
                             S_ISDIR(inspection->nearest_status.st_mode)) ||
                            (root.kind == ContentRootKind::file &&
                             S_ISREG(inspection->nearest_status.st_mode));
  if (!kind_matches) {
    fail(diagnostics, "input.kind_changed", path,
         "locked input root kind changed: " + root.path,
         "Restore the locked file/directory kind or deliberately re-lock the "
         "input content.");
    return;
  }
  const mode_t required =
      root.kind == ContentRootKind::directory ? 0005U : 0004U;
  if (!directory_permission(inspection->nearest_status, uid, gid,
                            supplementary_gids, required)) {
    fail(diagnostics, "input.not_readable", path,
         "hostd worker identity cannot read locked input root " + root.path,
         "Grant read and search permission to the configured worker uid/gid "
         "without changing the locked content.");
  }
}

void validate_output_directory(
    const Workspace &workspace, std::uint32_t uid, std::uint32_t gid,
    const std::vector<std::uint32_t> &supplementary_gids,
    std::vector<TrainingPreflightDiagnostic> &diagnostics) {
  const std::string path = "/spec/workspace/run_directory";
  const auto inspection = inspect_ancestor_chain(workspace.run_directory, uid,
                                                 gid, supplementary_gids, path,
                                                 true, false, diagnostics);
  if (!inspection)
    return;
  if (inspection->nearest_existing.empty()) {
    fail(diagnostics, "output.no_existing_ancestor", path,
         "run directory has no inspectable existing ancestor",
         "Create an authority-owned output root reachable by the hostd worker "
         "identity.");
    return;
  }
  if (!directory_permission(inspection->nearest_status, uid, gid,
                            supplementary_gids, 0003U)) {
    const std::string target = inspection->complete
                                   ? workspace.run_directory
                                   : inspection->nearest_existing.string();
    fail(diagnostics, "output.worker_permission", path,
         "hostd worker uid " + std::to_string(uid) + ":gid " +
             std::to_string(gid) +
             " cannot create, write, and atomically rename entries in " +
             target,
         "Change ownership/group or grant write+search permission to the exact "
         "hostd worker identity before submission.");
  }
  if ((inspection->nearest_status.st_mode & S_ISVTX) != 0 &&
      inspection->nearest_status.st_uid != static_cast<uid_t>(uid)) {
    fail(diagnostics, "output.sticky_rename_policy", path,
         "the writable output directory is sticky and is not owned by the "
         "hostd worker",
         "Use a dedicated worker-owned/group-owned run directory so checkpoint "
         "and status-file atomic renames are deterministic.");
  }
}

std::set<std::string> training_node_ids(const CompiledPlan &plan) {
  std::set<std::string> result;
  for (const auto &[node_id, node] : plan.experiment.spec.workflow.nodes) {
    if (node.invoke.training)
      result.insert(node_id);
  }
  return result;
}

bool qualitative_artifact_declared(const CompiledPlan &plan,
                                   std::string_view node_id) {
  const auto found =
      plan.experiment.spec.workflow.nodes.find(std::string(node_id));
  if (found == plan.experiment.spec.workflow.nodes.end() ||
      !found->second.publishes) {
    return false;
  }
  for (const auto &[unused_port, artifact_name] : *found->second.publishes) {
    (void)unused_port;
    const auto artifact = plan.experiment.spec.artifacts.find(artifact_name);
    if (artifact != plan.experiment.spec.artifacts.end() &&
        (artifact->second.type == ArtifactType::image_gallery ||
         artifact->second.type == ArtifactType::report)) {
      return true;
    }
  }
  return false;
}

std::uint64_t gib_to_bytes(double gib) {
  if (!std::isfinite(gib) || gib < 0.0 ||
      gib > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) /
                static_cast<double>(kGib)) {
    throw TrainingPreflightError(
        "memory policy is outside the supported range");
  }
  return static_cast<std::uint64_t>(std::ceil(gib * static_cast<double>(kGib)));
}

nlohmann::json
receipt_digest_document(const TrainingPreflightReceipt &receipt) {
  nlohmann::json value = encode_json(receipt);
  value.erase("receipt_digest");
  return value;
}

} // namespace

std::string training_preflight_node_input_digest(const CompiledPlan &plan,
                                                 std::string_view node_id) {
  const auto &nodes = plan.canonical_plan.at("spec").at("workflow").at("nodes");
  const auto found = nodes.find(std::string(node_id));
  if (found == nodes.end()) {
    throw TrainingPreflightError(
        "preflight node is absent from canonical plan");
  }
  return digest("trainvm.training-preflight-node-input/v1",
                nlohmann::json{{"plan_hash", plan.plan_hash},
                               {"node_id", node_id},
                               {"invoke", found->at("invoke")}});
}

TrainingPreflightRecipeProvenance
training_preflight_recipe_provenance(const ExpandedRecipe &expanded) {
  return {
      .registry_digest = expanded.registry_digest,
      .profile_digest = expanded.profile_digest,
      .instance_digest = expanded.instance_digest,
      .expanded_plan_digest = expanded.expanded_plan_digest,
  };
}

TrainingPreflightReceipt
run_training_preflight(const CompiledPlan &plan,
                       const TrainingPreflightEnvironment &environment) {
  TrainingPreflightReceipt receipt{
      .api_version = std::string(kTrainingPreflightReceiptApiVersion),
      .passed = false,
      .accelerator_passive = !environment.gpu_qualification.has_value(),
      .cacheable = false,
      .plan_hash = plan.plan_hash,
      .input_identity_digest = digest(
          "trainvm.training-preflight-input-identities/v1",
          encode_json(plan.experiment.spec.workspace.input_content_roots)),
      .environment_digest = digest("trainvm.training-preflight-environment/v1",
                                   encode_json(environment)),
      .cache_key = {},
      .valid_until_monotonic_ns = environment.snapshot_valid_until_monotonic_ns,
      .diagnostics = {},
      .receipt_digest = {},
  };
  auto &diagnostics = receipt.diagnostics;

  if (environment.api_version != kTrainingPreflightEnvironmentApiVersion) {
    fail(diagnostics, "preflight.api_version", "/api_version",
         "unsupported training-preflight environment api_version",
         "Generate evidence with trainvm.training-preflight-environment/v1 "
         "tooling.");
  }
  if (!digest_valid(environment.snapshot_digest)) {
    fail(diagnostics, "preflight.snapshot_identity", "/snapshot_digest",
         "passive host/runtime snapshot has no canonical SHA-256 identity",
         "Regenerate the passive snapshot through the registered runtime "
         "probe.");
  }
  if (!digest_valid(environment.host_id) ||
      !boot_id_valid(environment.boot_id)) {
    fail(diagnostics, "preflight.host_identity", "/host_id",
         "passive evidence is not bound to a canonical host and boot identity",
         "Regenerate evidence from the same hostd authority that will admit "
         "the run.");
  }
  if (environment.snapshot_observed_monotonic_ns == 0U ||
      environment.evaluation_monotonic_ns <
          environment.snapshot_observed_monotonic_ns ||
      environment.snapshot_valid_until_monotonic_ns <
          environment.evaluation_monotonic_ns ||
      environment.snapshot_valid_until_monotonic_ns -
              environment.snapshot_observed_monotonic_ns >
          kMaximumPassiveSnapshotAgeNs) {
    fail(diagnostics, "preflight.snapshot_freshness", "/snapshot_digest",
         "passive host/runtime snapshot is stale, future-dated, or valid for "
         "more than 60 seconds",
         "Refresh passive evidence immediately before submission and use the "
         "host authority's monotonic clock domain.");
  }
  if (environment.worker_uid == 0U || environment.worker_gid == 0U) {
    fail(diagnostics, "preflight.worker_identity", "/worker_uid",
         "training preflight requires the non-root hostd worker uid and gid",
         "Copy the exact worker identity from the validated hostd "
         "configuration.");
  }
  if (!digest_valid(environment.worker_principal_digest)) {
    fail(diagnostics, "preflight.worker_principal", "/worker_principal_digest",
         "worker uid/gid are not bound to an exact hostd principal profile",
         "Generate evidence from the validated hostd worker profile and "
         "include its canonical digest.");
  }
  if (environment.total_host_memory_bytes == 0U ||
      environment.available_host_memory_bytes >
          environment.total_host_memory_bytes ||
      environment.logical_cpu_count == 0U ||
      environment.logical_cpu_count > 1'048'576U) {
    fail(diagnostics, "resource.host_observation", "/total_host_memory_bytes",
         "passive host memory/CPU observation is malformed",
         "Refresh bounded host resource evidence through the host authority.");
  }
  if (plan.experiment.spec.resources.minimum_host_memory_gib &&
      environment.total_host_memory_bytes <
          gib_to_bytes(
              *plan.experiment.spec.resources.minimum_host_memory_gib)) {
    fail(diagnostics, "resource.host_memory_insufficient",
         "/spec/resources/minimum_host_memory_gib",
         "host capacity cannot satisfy the declared minimum host memory",
         "Choose a host with enough RAM or correct the declarative capacity "
         "requirement.");
  }
  if (plan.experiment.spec.resources.cpu_threads &&
      *plan.experiment.spec.resources.cpu_threads >
          static_cast<std::int64_t>(environment.logical_cpu_count)) {
    fail(diagnostics, "resource.cpu_threads_insufficient",
         "/spec/resources/cpu_threads",
         "host CPU inventory cannot satisfy the declared thread count",
         "Choose a host with enough logical CPUs or lower the bounded thread "
         "request.");
  }
  if (environment.supplementary_gids.size() > 64U ||
      !std::ranges::is_sorted(environment.supplementary_gids) ||
      std::ranges::adjacent_find(environment.supplementary_gids) !=
          environment.supplementary_gids.end() ||
      std::ranges::find(environment.supplementary_gids,
                        environment.worker_gid) !=
          environment.supplementary_gids.end()) {
    fail(diagnostics, "preflight.worker_groups", "/supplementary_gids",
         "supplementary worker groups must be the complete bounded sorted "
         "unique set and exclude the primary gid",
         "Snapshot the exact effective credentials hostd will install; use [] "
         "when hostd clears supplementary groups.");
  }
  if (environment.recipe_provenance &&
      (!digest_valid(environment.recipe_provenance->registry_digest) ||
       !digest_valid(environment.recipe_provenance->profile_digest) ||
       !digest_valid(environment.recipe_provenance->instance_digest) ||
       !digest_valid(environment.recipe_provenance->expanded_plan_digest))) {
    fail(diagnostics, "preflight.recipe_provenance", "/recipe_provenance",
         "recipe provenance contains a noncanonical digest",
         "Bind the registry, profile, instance, and expanded-plan identities "
         "emitted by recipe expansion.");
  } else if (environment.recipe_provenance &&
             environment.recipe_provenance->expanded_plan_digest !=
                 "sha256:" + plan.plan_hash) {
    fail(diagnostics, "preflight.recipe_plan_mismatch",
         "/recipe_provenance/expanded_plan_digest",
         "recipe provenance names a different compiled plan",
         "Use the ordinary CompiledPlan and the four identities from the same "
         "recipe expansion result.");
  }
  if (environment.accelerators.size() > kMaximumAccelerators) {
    fail(diagnostics, "resource.accelerator_bound", "/accelerators",
         "passive accelerator evidence exceeds its device bound",
         "Restrict the snapshot to the host inventory used for this "
         "submission.");
  }
  if (environment.training_nodes.size() > kMaximumTrainingNodes) {
    fail(diagnostics, "probe.node_bound", "/training_nodes",
         "runtime probe evidence exceeds its training-node bound",
         "Probe only training nodes present in the compiled plan.");
  }
  if (environment.gpu_qualification) {
    const auto &qualification = *environment.gpu_qualification;
    if (qualification.maximum_duration_milliseconds == 0U ||
        qualification.maximum_duration_milliseconds >
            kMaximumGpuQualificationMilliseconds ||
        !digest_valid(qualification.receipt_digest)) {
      fail(
          diagnostics, "gpu_qualification.unbounded", "/gpu_qualification",
          "explicit GPU qualification is unbounded or lacks a receipt identity",
          "Declare a nonzero duration of at most 60000 ms and supply its "
          "sealed receipt digest.");
    } else if (!qualification.passed) {
      fail(diagnostics, "gpu_qualification.failed", "/gpu_qualification/passed",
           "the explicitly requested bounded GPU qualification failed",
           "Inspect its receipt, correct the runtime/kernel issue, and qualify "
           "again.");
    }
  }

  if (plan.experiment.spec.workspace.input_content_roots) {
    const auto &roots = *plan.experiment.spec.workspace.input_content_roots;
    for (std::size_t index = 0U; index < roots.size(); ++index) {
      validate_read_root(roots[index], index, environment.worker_uid,
                         environment.worker_gid, environment.supplementary_gids,
                         diagnostics);
    }
  }
  validate_output_directory(plan.experiment.spec.workspace,
                            environment.worker_uid, environment.worker_gid,
                            environment.supplementary_gids, diagnostics);

  std::map<std::string, const TrainingNodePreflightEvidence *> supplied_nodes;
  for (std::size_t index = 0U; index < environment.training_nodes.size();
       ++index) {
    const auto &node = environment.training_nodes[index];
    if (node.node_id.empty() ||
        !supplied_nodes.emplace(node.node_id, &node).second) {
      fail(
          diagnostics, "probe.node_duplicate",
          "/training_nodes/" + std::to_string(index) + "/node_id",
          "training-node probe identity is empty or duplicated",
          "Emit exactly one passive probe receipt per compiled training node.");
    }
  }

  const auto expected_nodes = training_node_ids(plan);
  for (const auto &[node_id, unused] : supplied_nodes) {
    (void)unused;
    if (!expected_nodes.contains(node_id)) {
      fail(diagnostics, "probe.node_unknown", "/training_nodes",
           "probe evidence names non-training node " + node_id,
           "Regenerate evidence from the exact compiled plan hash.");
    }
  }

  std::optional<double> maximum_free_policy;
  for (const auto &node_id : expected_nodes) {
    const auto supplied = supplied_nodes.find(node_id);
    if (supplied == supplied_nodes.end()) {
      fail(diagnostics, "probe.missing", "/spec/workflow/nodes/" + node_id,
           "registered adapter supplied no passive preflight evidence for "
           "training node " +
               node_id,
           "Run the registered accelerator-passive family probe before "
           "submission; native inference is intentionally forbidden.");
      continue;
    }
    const auto &node = *supplied->second;
    const std::string base = "/training_nodes/" + node_id;
    const std::string expected_input =
        training_preflight_node_input_digest(plan, node_id);
    if (node.node_input_digest != expected_input) {
      fail(diagnostics, "probe.input_identity", base + "/node_input_digest",
           "adapter probe evidence does not match the compiled node invocation",
           "Discard cached evidence and probe the current plan inputs again.");
    }
    if (!digest_valid(node.runtime_profile_digest)) {
      fail(diagnostics, "runtime.profile_identity",
           base + "/runtime_profile_digest",
           "adapter probe is not bound to an exact registered runtime profile",
           "Probe with the same sealed adapter/launch profile whose digest "
           "will be locked at submission.");
    }
    if (node.minimum_free_memory_gib) {
      if (!std::isfinite(*node.minimum_free_memory_gib) ||
          *node.minimum_free_memory_gib < 0.0) {
        fail(diagnostics, "resource.free_memory_policy",
             base + "/minimum_free_memory_gib",
             "adapter free-VRAM policy is not a finite nonnegative value",
             "Correct the trainer configuration and regenerate passive probe "
             "evidence.");
      } else {
        maximum_free_policy = std::max(maximum_free_policy.value_or(0.0),
                                       *node.minimum_free_memory_gib);
      }
    } else if (plan.experiment.spec.resources.accelerators.count > 0) {
      fail(diagnostics, "resource.free_vram_policy_missing",
           base + "/minimum_free_memory_gib",
           "adapter probe did not report the trainer's effective free-VRAM "
           "policy",
           "Have the registered family probe report the exact configured "
           "threshold, using 0 only when the trainer deliberately has no "
           "free-memory gate.");
    }

    std::map<TrainingPreflightCheckKind, const TrainingPreflightCheckEvidence *>
        checks;
    for (std::size_t index = 0U; index < node.checks.size(); ++index) {
      const auto &check = node.checks[index];
      if (!checks.emplace(check.kind, &check).second) {
        fail(diagnostics, "probe.check_duplicate",
             base + "/checks/" + std::to_string(index),
             "adapter probe duplicated a semantic preflight obligation",
             "Emit exactly one result for every closed preflight check kind.");
      }
      if (!digest_valid(check.evidence_digest)) {
        fail(diagnostics, "probe.check_identity",
             base + "/checks/" + std::to_string(index),
             "adapter check lacks a canonical evidence digest",
             "Bind the check to its bounded input/output receipt.");
      }
      if (check.disposition ==
              TrainingPreflightCheckDisposition::not_applicable &&
          (!check.detail || check.detail->empty())) {
        fail(diagnostics, "probe.not_applicable_reason",
             base + "/checks/" + std::to_string(index),
             "not_applicable check has no reason",
             "Explain why this exact training operation does not use the "
             "semantic obligation.");
      }
      if (check.disposition == TrainingPreflightCheckDisposition::failed) {
        fail(diagnostics, "probe.check_failed",
             base + "/checks/" + std::to_string(index),
             "adapter preflight check failed: " +
                 check.detail.value_or(enum_to_string(check.kind)),
             "Correct the reported model, dataset, selector, evaluator, "
             "checkpoint, or runtime failure before submission.");
      }
    }
    static constexpr TrainingPreflightCheckKind all_kinds[] = {
        TrainingPreflightCheckKind::model_configuration,
        TrainingPreflightCheckKind::tokenizer,
        TrainingPreflightCheckKind::processor,
        TrainingPreflightCheckKind::dataset_schema,
        TrainingPreflightCheckKind::dataset_sample_decode,
        TrainingPreflightCheckKind::parameter_selection,
        TrainingPreflightCheckKind::kernel_runtime,
        TrainingPreflightCheckKind::checkpoint_compatibility,
        TrainingPreflightCheckKind::step_zero_evaluator,
        TrainingPreflightCheckKind::dashboard_artifacts,
    };
    for (const auto kind : all_kinds) {
      if (!checks.contains(kind)) {
        fail(diagnostics, "probe.check_missing", base + "/checks",
             "adapter probe omitted semantic obligation " +
                 enum_to_string(kind),
             "Update the registered family probe to pass, fail, or explicitly "
             "mark this obligation not_applicable.");
      }
    }
    for (const auto mandatory :
         {TrainingPreflightCheckKind::dataset_schema,
          TrainingPreflightCheckKind::dataset_sample_decode,
          TrainingPreflightCheckKind::kernel_runtime,
          TrainingPreflightCheckKind::step_zero_evaluator,
          TrainingPreflightCheckKind::dashboard_artifacts}) {
      const auto found = checks.find(mandatory);
      if (found != checks.end() &&
          found->second->disposition ==
              TrainingPreflightCheckDisposition::not_applicable) {
        fail(diagnostics, "probe.mandatory_not_applicable", base + "/checks",
             "training node cannot mark " + enum_to_string(mandatory) +
                 " not_applicable",
             "Every training run needs decoded data, runtime qualification, "
             "step-zero evidence, and declared dashboard examples.");
      }
    }
    if (!qualitative_artifact_declared(plan, node_id)) {
      fail(diagnostics, "dashboard.examples_undeclared",
           "/spec/workflow/nodes/" + node_id + "/publishes",
           "training node declares no qualitative example artifact for the "
           "dashboard",
           "Publish a versioned image-gallery or report artifact populated at "
           "step 0 and later eval steps.");
    }

    std::set<std::string> required(node.required_capabilities.begin(),
                                   node.required_capabilities.end());
    std::set<std::string> provided(node.provided_capabilities.begin(),
                                   node.provided_capabilities.end());
    if (node.required_capabilities.size() > 256U ||
        node.provided_capabilities.size() > 256U ||
        required.size() != node.required_capabilities.size() ||
        provided.size() != node.provided_capabilities.size()) {
      fail(diagnostics, "runtime.capability_set", base + "/capabilities",
           "runtime capability evidence is oversized or noncanonical",
           "Emit bounded unique capability identities from the registered "
           "runtime profile.");
    }
    for (const auto &capability : node.required_capabilities) {
      if (!provided.contains(capability)) {
        fail(diagnostics, "runtime.capability_missing",
             base + "/provided_capabilities",
             "runtime does not provide required capability " + capability,
             "Install/qualify the kernel or choose a recipe supported by the "
             "sealed runtime profile.");
      }
    }
  }

  std::set<std::string> accelerator_ids;
  for (std::size_t index = 0U; index < environment.accelerators.size();
       ++index) {
    const auto &accelerator = environment.accelerators[index];
    const std::string path = "/accelerators/" + std::to_string(index);
    if (accelerator.stable_id.empty() ||
        !accelerator_ids.insert(accelerator.stable_id).second ||
        accelerator.total_memory_bytes == 0U ||
        accelerator.free_memory_bytes > accelerator.total_memory_bytes ||
        !digest_valid(accelerator.observation_digest)) {
      fail(diagnostics, "resource.passive_observation", path,
           "passive accelerator memory observation is malformed or duplicated",
           "Refresh the host-authority memory snapshot without creating a "
           "device context.");
    }
  }

  const auto &request = plan.experiment.spec.resources.accelerators;
  const std::uint64_t minimum_total =
      request.minimum_memory_gib ? gib_to_bytes(*request.minimum_memory_gib)
                                 : 0U;
  const std::uint64_t minimum_free =
      maximum_free_policy ? gib_to_bytes(*maximum_free_policy) : 0U;
  std::size_t suitable = 0U;
  for (const auto &accelerator : environment.accelerators) {
    if (accelerator.vendor != request.vendor ||
        accelerator.total_memory_bytes < minimum_total) {
      continue;
    }
    if (accelerator.free_memory_bytes < minimum_free)
      continue;
    ++suitable;
  }
  if (request.count > 0 && suitable < static_cast<std::size_t>(request.count)) {
    const bool enough_total =
        std::ranges::count_if(
            environment.accelerators, [&](const auto &accelerator) {
              return accelerator.vendor == request.vendor &&
                     accelerator.total_memory_bytes >= minimum_total;
            }) >= request.count;
    if (enough_total && minimum_free > 0U) {
      fail(diagnostics, "resource.free_vram_insufficient",
           "/spec/resources/accelerators",
           "declared total-VRAM selector is satisfiable, but only " +
               std::to_string(suitable) +
               " accelerator(s) meet the trainer's " +
               std::to_string(maximum_free_policy.value_or(0.0)) +
               " GiB free-VRAM policy",
           "Lower the trainer free-VRAM requirement only if safe, stop "
           "conflicting GPU consumers, or select a device with enough "
           "passively observed free memory.");
    } else {
      fail(diagnostics, "resource.total_vram_insufficient",
           "/spec/resources/accelerators/minimum_memory_gib",
           "passive inventory cannot satisfy the declared accelerator "
           "count/vendor/total-VRAM selector",
           "Correct the total-memory selector or schedule the run on a "
           "matching host.");
    }
  }

  receipt.passed =
      std::ranges::none_of(diagnostics, [](const auto &diagnostic) {
        return diagnostic.severity == Diagnostic::Severity::error;
      });
  receipt.cacheable =
      receipt.passed && digest_valid(environment.snapshot_digest);
  receipt.cache_key = digest(
      "trainvm.training-preflight-cache-key/v1",
      nlohmann::json{{"plan_hash", receipt.plan_hash},
                     {"input_identity_digest", receipt.input_identity_digest},
                     {"environment_digest", receipt.environment_digest},
                     {"snapshot_digest", environment.snapshot_digest}});
  receipt.receipt_digest = digest("trainvm.training-preflight-receipt/v1",
                                  receipt_digest_document(receipt));
  return receipt;
}

TrainingPreflightEnvironment
load_training_preflight_environment(const std::filesystem::path &path) {
  const auto status = std::filesystem::symlink_status(path);
  if (!std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status)) {
    throw TrainingPreflightError(
        "training-preflight environment must be a regular non-symlink file");
  }
  const auto size = std::filesystem::file_size(path);
  if (size == 0U || size > kMaximumEnvironmentBytes) {
    throw TrainingPreflightError(
        "training-preflight environment exceeds its 4 MiB bound");
  }
  std::ifstream input(path, std::ios::binary);
  nlohmann::json document;
  input >> document;
  if (!input) {
    throw TrainingPreflightError(
        "could not parse training-preflight environment JSON");
  }
  TrainingPreflightEnvironment environment;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(document, environment, "", diagnostics) ||
      !diagnostics.empty() || encode_json(environment) != document) {
    std::ostringstream message;
    message << "training-preflight environment has an invalid closed schema";
    for (const auto &diagnostic : diagnostics) {
      message << "\n  " << diagnostic.code << ' ' << diagnostic.path << ' '
              << diagnostic.message;
    }
    throw TrainingPreflightError(message.str());
  }
  return environment;
}

} // namespace trainvm
