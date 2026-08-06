#include "trainvm/final_evaluation.hpp"

#include <algorithm>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumFinalizationHistory = 1024U;
constexpr std::size_t kMaximumFinalMembers = 1'000'000U;
constexpr std::size_t kMaximumFinalRecords = 4'000'000U;
constexpr std::size_t kMaximumFinalOutputs = 64U;
constexpr std::size_t kMaximumFinalScalars = 256U;

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool bounded_identity(std::string_view value) {
  return !value.empty() && value.size() <= 1024U &&
         std::ranges::none_of(value, [](char character) {
           return character == '\0' || character == '\n' || character == '\r';
         });
}

FinalOutputPolicy classify_output(
    const std::string& name, const OperationPortDescriptor& output) {
  if (output.type != OperationPortType::artifact || !output.artifact_type) {
    throw std::invalid_argument(
        "stateful operation finalization outputs must be typed artifacts");
  }
  FinalOutputPolicy policy{
      .output_name = name,
      .required = output.required,
      .required_when_declared = !output.required,
      .coverage = FinalCoveragePolicy::durable_nonempty,
      .errors = FinalErrorPolicy::not_applicable,
      .artifact_schema = output.artifact_schema,
  };
  switch (*output.artifact_type) {
    case ArtifactType::checkpoint:
      policy.evidence_kind = FinalEvidenceKind::checkpoint;
      policy.exact_optimizer_step = true;
      return policy;
    case ArtifactType::image_gallery:
      policy.evidence_kind = FinalEvidenceKind::examples;
      policy.exact_optimizer_step = true;
      policy.checkpoint_bound = true;
      policy.coverage = FinalCoveragePolicy::full_membership;
      policy.errors = FinalErrorPolicy::zero_unresolved_errors;
      return policy;
    case ArtifactType::report:
      if (output.artifact_schema == "rwkv-lab.final-evaluation.v1") {
        policy.evidence_kind = FinalEvidenceKind::closure;
        policy.exact_optimizer_step = true;
        policy.checkpoint_bound = true;
      } else if (output.artifact_schema ==
          "rwkv-lab.scalar-metric-result.v1") {
        policy.evidence_kind = FinalEvidenceKind::scalar;
        policy.coverage = FinalCoveragePolicy::full_membership;
        policy.errors = FinalErrorPolicy::zero_unresolved_errors;
      } else if (output.artifact_schema ==
                 "rwkv-lab.hf-test-caption-evidence-bundle.v1") {
        policy.evidence_kind = FinalEvidenceKind::test;
        policy.coverage = FinalCoveragePolicy::full_membership;
        policy.errors = FinalErrorPolicy::zero_unresolved_errors;
      } else if (output.artifact_schema ==
                 "rwkv-lab.scalar-metric-decision.v1") {
        // The only current decision operation is stateless and never reaches
        // this branch. Rejecting it if copied to stateful execution prevents a
        // decision receipt from impersonating final evaluation.
        throw std::invalid_argument(
            "stateful finalization cannot classify a decision report as evaluation");
      } else {
        policy.evidence_kind = FinalEvidenceKind::audit;
      }
      policy.exact_optimizer_step = true;
      policy.checkpoint_bound = true;
      return policy;
    case ArtifactType::opaque:
    case ArtifactType::metrics:
    case ArtifactType::path:
    case ArtifactType::dataset:
      policy.evidence_kind = FinalEvidenceKind::audit;
      return policy;
  }
  throw std::invalid_argument("stateful operation output has an invalid artifact kind");
}

std::vector<std::string> canonical_members(
    const std::vector<std::string>& values) {
  std::vector<std::string> result = values;
  std::ranges::sort(result);
  if (result.empty() || std::ranges::adjacent_find(result) != result.end() ||
      std::ranges::any_of(result, [](const std::string& value) {
        return !bounded_identity(value);
      })) {
    throw std::invalid_argument(
        "final evaluation membership must be nonempty, unique, and bounded");
  }
  return result;
}

FinalizationVerdict verdict(FinalizationDisposition disposition,
                            std::string cause,
                            std::vector<std::string> unresolved = {}) {
  std::ranges::sort(unresolved);
  return {.disposition = disposition,
          .cause = std::move(cause),
          .unresolved_members = std::move(unresolved),
          .selected_artifact_id = std::nullopt,
          .selected_artifact_fingerprint = std::nullopt};
}

}  // namespace

FinalizationPolicyRegistry::FinalizationPolicyRegistry(
    const std::vector<AdapterProfile>& profiles) {
  for (const AdapterProfile& profile : profiles) {
    if (!profile.lifecycle.stateful) continue;
    if (!profile.authoring) {
      throw std::invalid_argument(
          "stateful operation is missing finalization authoring authority");
    }
    OperationFinalizationPolicy policy{
        .key = profile.key,
        .outputs = {},
        .eval_only_recovery =
            profile.lifecycle.resume_grade == ResumeGrade::compatible ||
            profile.lifecycle.resume_grade == ResumeGrade::exact ||
            profile.lifecycle.resume_grade == ResumeGrade::terminal_checkpoint,
        .closure_output_name = std::nullopt,
        .migration_pending = true,
    };
    for (const auto& [name, output] : profile.authoring->outputs) {
      FinalOutputPolicy classified = classify_output(name, output);
      if (classified.evidence_kind == FinalEvidenceKind::closure) {
        if (policy.closure_output_name) {
          throw std::invalid_argument(
              "stateful operation declares multiple finalization closure outputs");
        }
        policy.closure_output_name = name;
        policy.migration_pending = false;
      } else {
        policy.outputs.push_back(std::move(classified));
      }
    }
    std::ranges::sort(policy.outputs, {}, &FinalOutputPolicy::output_name);
    if (!policies_.emplace(profile.key, std::move(policy)).second) {
      throw std::invalid_argument(
          "stateful operation has duplicate finalization policy identity");
    }
  }
}

const OperationFinalizationPolicy& FinalizationPolicyRegistry::resolve(
    const AdapterKey& key) const {
  const auto found = policies_.find(key);
  if (found == policies_.end()) {
    throw std::out_of_range(
        "operation has no registered stateful finalization policy");
  }
  return found->second;
}

const std::map<AdapterKey, OperationFinalizationPolicy>&
FinalizationPolicyRegistry::policies() const {
  return policies_;
}

nlohmann::json FinalizationPolicyRegistry::inventory_json() const {
  nlohmann::json operations = nlohmann::json::array();
  for (const auto& [key, policy] : policies_) {
    (void)key;
    operations.push_back(encode_json(policy));
  }
  return {{"api_version", "trainvm.finalization-inventory/v1"},
          {"operations", std::move(operations)}};
}

FinalizationVerdict reduce_final_evaluation(
    const FinalOutputPolicy& policy,
    std::uint64_t terminal_optimizer_step,
    std::string checkpoint_artifact_id,
    std::string checkpoint_fingerprint,
    const std::vector<FinalEvaluationReceipt>& history) {
  if (!policy.exact_optimizer_step || !policy.checkpoint_bound ||
      policy.coverage != FinalCoveragePolicy::full_membership ||
      policy.errors != FinalErrorPolicy::zero_unresolved_errors) {
    throw std::invalid_argument(
        "semantic final evaluation reducer requires a strict evaluation output policy");
  }
  if (!bounded_identity(checkpoint_artifact_id) ||
      !valid_digest(checkpoint_fingerprint)) {
    throw std::invalid_argument("final checkpoint identity is invalid");
  }
  if (history.empty()) {
    return verdict(FinalizationDisposition::pending,
                   "required final evaluation evidence is missing");
  }
  if (history.size() > kMaximumFinalizationHistory) {
    return verdict(FinalizationDisposition::failed,
                   "final evaluation history exceeds its receipt bound");
  }

  std::vector<FinalMemberRecord> previous_records;
  std::vector<std::string> membership;
  std::string membership_digest;
  std::map<std::string, std::uint64_t> highest_attempt;
  std::map<std::string, std::string> member_contexts;
  std::set<std::string> successful;
  std::uint64_t previous_sequence = 0U;
  std::optional<std::string> selected_id;
  std::optional<std::string> selected_fingerprint;
  std::string policy_digest;
  std::set<std::string> receipt_artifact_ids;
  std::set<std::string> receipt_artifact_fingerprints;
  std::vector<FinalOutputReceipt> previous_output_receipts;
  std::vector<FinalScalarRequirement> scalar_requirements;
  bool scalar_requirements_initialized = false;

  for (const FinalEvaluationReceipt& receipt : history) {
    if (!bounded_identity(receipt.artifact_id) ||
        !valid_digest(receipt.artifact_fingerprint) ||
        !valid_digest(receipt.policy_digest) ||
        receipt.durable_sequence == 0U ||
        receipt.durable_sequence <= previous_sequence ||
        !receipt_artifact_ids.insert(receipt.artifact_id).second ||
        !receipt_artifact_fingerprints
             .insert(receipt.artifact_fingerprint)
             .second) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation receipt durability is invalid");
    }
    previous_sequence = receipt.durable_sequence;
    if (policy_digest.empty()) {
      policy_digest = receipt.policy_digest;
    } else if (policy_digest != receipt.policy_digest) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation policy changed");
    }
    if (receipt.optimizer_step != terminal_optimizer_step) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation evidence is stale-step");
    }
    if (receipt.checkpoint_artifact_id != checkpoint_artifact_id ||
        receipt.checkpoint_fingerprint != checkpoint_fingerprint) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation evidence is bound to another checkpoint");
    }
    std::vector<std::string> receipt_members;
    if (receipt.required_members.size() > kMaximumFinalMembers ||
        receipt.records.size() > kMaximumFinalRecords ||
        receipt.output_receipts.size() > kMaximumFinalOutputs ||
        receipt.required_scalars.size() > kMaximumFinalScalars) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation receipt exceeds a collection bound");
    }
    try {
      receipt_members = canonical_members(receipt.required_members);
    } catch (const std::invalid_argument&) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation coverage membership is invalid");
    }
    if (membership.empty()) {
      membership = receipt_members;
      membership_digest = receipt.membership_digest;
    } else if (receipt_members != membership) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation coverage membership changed");
    }
    if (!valid_digest(receipt.membership_digest) ||
        receipt.membership_digest != membership_digest ||
        receipt.membership_digest != final_membership_digest(receipt_members) ||
        receipt.membership_count != receipt_members.size()) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation membership digest disagrees");
    }
    const std::size_t prior_output_receipt_count =
        previous_output_receipts.size();
    bool policy_output_receipted = false;
    for (const FinalOutputReceipt& output : receipt.output_receipts) {
      if (!bounded_identity(output.output_name) ||
          !bounded_identity(output.artifact_id) ||
          !valid_digest(output.artifact_fingerprint)) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation output receipts are invalid");
      }
      if (output.output_name == policy.output_name) {
        policy_output_receipted = true;
      }
    }
    if (receipt.output_receipts.size() < previous_output_receipts.size() ||
        !std::ranges::equal(
            previous_output_receipts.begin(), previous_output_receipts.end(),
            receipt.output_receipts.begin(),
            receipt.output_receipts.begin() + static_cast<std::ptrdiff_t>(
                                                 previous_output_receipts.size()))) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation output receipt history drifted");
    }
    std::set<std::string> output_artifact_ids;
    std::set<std::string> output_artifact_fingerprints;
    if (std::ranges::any_of(
            receipt.output_receipts, [&](const FinalOutputReceipt& output) {
              return !output_artifact_ids.insert(output.artifact_id).second ||
                     !output_artifact_fingerprints
                          .insert(output.artifact_fingerprint)
                          .second;
            })) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation output receipt history has duplicates");
    }
    previous_output_receipts = receipt.output_receipts;
    if (!policy_output_receipted) {
      return verdict(FinalizationDisposition::pending,
                     "required final evaluation output is not durably receipted",
                     receipt_members);
    }
    std::set<std::string> scalar_names;
    for (const FinalScalarRequirement& scalar : receipt.required_scalars) {
      if (!bounded_identity(scalar.metric_name) ||
          scalar.step_domain != "optimizer_step" ||
          !scalar_names.insert(scalar.metric_name).second) {
        return verdict(FinalizationDisposition::failed,
                       "final scalar requirements are invalid");
      }
    }
    if (!scalar_requirements_initialized) {
      scalar_requirements = receipt.required_scalars;
      scalar_requirements_initialized = true;
    } else if (scalar_requirements != receipt.required_scalars) {
      return verdict(FinalizationDisposition::failed,
                     "final scalar requirements drifted");
    }
    if (receipt.records.size() < previous_records.size() ||
        !std::ranges::equal(
            previous_records.begin(), previous_records.end(),
            receipt.records.begin(),
            receipt.records.begin() +
                static_cast<std::ptrdiff_t>(previous_records.size()))) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation history was rewritten");
    }
    if (!previous_records.empty() &&
        receipt.records.size() > previous_records.size() &&
        receipt.output_receipts.size() == prior_output_receipt_count) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation retry appended no output evidence");
    }

    std::vector<std::string> unresolved_before;
    std::ranges::set_difference(membership,
                                std::vector<std::string>(successful.begin(),
                                                         successful.end()),
                                std::back_inserter(unresolved_before));
    if (receipt.recovery) {
      std::vector<std::string> requested;
      try {
        requested = canonical_members(receipt.recovery->requested_members);
      } catch (const std::invalid_argument&) {
        return verdict(FinalizationDisposition::failed,
                       "eval-only recovery membership is invalid");
      }
      if (previous_records.empty() || requested != unresolved_before) {
        return verdict(FinalizationDisposition::failed,
                       "eval-only recovery did not select exactly unresolved members");
      }
      if (!valid_digest(receipt.recovery->optimizer_state_before) ||
          receipt.recovery->optimizer_state_before !=
              receipt.recovery->optimizer_state_after) {
        return verdict(FinalizationDisposition::failed,
                       "eval-only recovery mutated optimizer state");
      }
    } else if (!previous_records.empty() &&
               receipt.records.size() > previous_records.size()) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation retry omitted its eval-only recovery receipt");
    }

    for (std::size_t index = previous_records.size();
         index < receipt.records.size(); ++index) {
      const FinalMemberRecord& record = receipt.records[index];
      if (!std::ranges::binary_search(membership, record.member_id) ||
          !valid_digest(record.context_digest) || record.attempt == 0U) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation record is context-poisoned");
      }
      const auto [context, inserted] =
          member_contexts.emplace(record.member_id, record.context_digest);
      if (!inserted && context->second != record.context_digest) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation member context changed");
      }
      const auto prior = highest_attempt.find(record.member_id);
      if (prior != highest_attempt.end() && record.attempt <= prior->second) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation attempts are not append-only");
      }
      highest_attempt[record.member_id] = record.attempt;
      const bool success = record.disposition == FinalMemberDisposition::success;
      if ((success && (!record.result_digest ||
                       !valid_digest(*record.result_digest) ||
                       record.error_code)) ||
          (!success && (record.result_digest || !record.error_code ||
                        !bounded_identity(*record.error_code)))) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation record result is malformed");
      }
      if (success) successful.insert(record.member_id);
    }
    previous_records = receipt.records;
    std::set<std::string> resolved_at_receipt;
    for (const FinalMemberRecord& record : receipt.records) {
      if (record.disposition == FinalMemberDisposition::success) {
        resolved_at_receipt.insert(record.member_id);
      }
    }
    if (receipt.resolved_member_count != resolved_at_receipt.size() ||
        receipt.failed_member_count !=
            receipt_members.size() - resolved_at_receipt.size()) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation summary counts disagree with its ledger");
    }
    selected_id = receipt.artifact_id;
    selected_fingerprint = receipt.artifact_fingerprint;
  }

  std::vector<std::string> unresolved;
  std::ranges::set_difference(membership,
                              std::vector<std::string>(successful.begin(),
                                                       successful.end()),
                              std::back_inserter(unresolved));
  if (!unresolved.empty()) {
    const bool all_error = successful.empty() && !previous_records.empty();
    return verdict(FinalizationDisposition::pending,
                   all_error ? "final evaluation has no successful members"
                             : "final evaluation coverage is partial",
                   std::move(unresolved));
  }
  FinalizationVerdict result = verdict(
      FinalizationDisposition::complete,
      "required final evaluation evidence passes coverage and error policy");
  result.selected_artifact_id = std::move(selected_id);
  result.selected_artifact_fingerprint = std::move(selected_fingerprint);
  return result;
}

std::string final_membership_digest(
    const std::vector<std::string>& canonical_members_value) {
  if (canonical_members(canonical_members_value) != canonical_members_value) {
    throw std::invalid_argument(
        "membership digest input must be canonical membership");
  }
  return "sha256:" + sha256_hex(nlohmann::json(canonical_members_value).dump());
}

nlohmann::json finalization_verdict_json(
    const FinalizationVerdict& verdict_value) {
  nlohmann::json value = encode_json(verdict_value);
  value["api_version"] = "trainvm.finalization-verdict/v1";
  return value;
}

nlohmann::json final_evaluation_manifest_json(
    const FinalEvaluationReceipt& receipt) {
  nlohmann::json value = encode_json(receipt);
  value.erase("artifact_id");
  value.erase("artifact_fingerprint");
  value.erase("durable_sequence");
  value["api_version"] = "rwkv-lab.final-evaluation/v1";
  return value;
}

}  // namespace trainvm
