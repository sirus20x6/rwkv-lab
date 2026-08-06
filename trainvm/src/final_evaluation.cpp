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

constexpr std::size_t kMaximumFinalizationHistory = 64U;
constexpr std::size_t kMaximumFinalMembers = 100'000U;
constexpr std::size_t kMaximumFinalRecords = 400'000U;
constexpr std::size_t kMaximumFinalOutputs = 64U;
constexpr std::size_t kMaximumFinalScalars = 256U;
constexpr std::size_t kMaximumAggregateRecords = 2'000'000U;
constexpr std::size_t kMaximumAggregateCollectionEntries = 4'000'000U;
constexpr std::size_t kMaximumAggregateIdentityBytes = 64U * 1024U * 1024U;

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

FinalOutputPolicy classify_output(const std::string &name,
                                  const OperationPortDescriptor &output) {
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
    } else if (output.artifact_schema == "rwkv-lab.scalar-metric-result.v1") {
      policy.evidence_kind = FinalEvidenceKind::scalar;
      policy.coverage = FinalCoveragePolicy::full_membership;
      policy.errors = FinalErrorPolicy::zero_unresolved_errors;
    } else if (output.artifact_schema ==
               "rwkv-lab.hf-test-caption-evidence-bundle.v1") {
      policy.evidence_kind = FinalEvidenceKind::test;
      policy.coverage = FinalCoveragePolicy::full_membership;
      policy.errors = FinalErrorPolicy::zero_unresolved_errors;
    } else if (output.artifact_schema == "rwkv-lab.scalar-metric-decision.v1") {
      // The only current decision operation is stateless and never reaches
      // this branch. Rejecting it if copied to stateful execution prevents a
      // decision receipt from impersonating final evaluation.
      throw std::invalid_argument("stateful finalization cannot classify a "
                                  "decision report as evaluation");
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
  throw std::invalid_argument(
      "stateful operation output has an invalid artifact kind");
}

bool canonical_members(const std::vector<std::string> &values) {
  return !values.empty() && std::ranges::is_sorted(values) &&
         std::ranges::adjacent_find(values) == values.end() &&
         std::ranges::none_of(values, [](const std::string &value) {
           return !bounded_identity(value);
         });
}

bool canonical_scalars(const std::vector<FinalScalarRequirement> &values) {
  return values.size() <= kMaximumFinalScalars &&
         std::ranges::is_sorted(values, {},
                                &FinalScalarRequirement::metric_name) &&
         std::ranges::adjacent_find(values, {},
                                    &FinalScalarRequirement::metric_name) ==
             values.end() &&
         std::ranges::all_of(values, [](const FinalScalarRequirement &value) {
           return bounded_identity(value.metric_name) &&
                  value.step_domain == "optimizer_step";
         });
}

bool consume(std::size_t amount, std::size_t limit, std::size_t &total) {
  if (amount > limit - total)
    return false;
  total += amount;
  return true;
}

bool consume_identity(std::string_view value, std::size_t &total) {
  return consume(value.size(), kMaximumAggregateIdentityBytes, total);
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

} // namespace

FinalizationPolicyRegistry::FinalizationPolicyRegistry(
    const std::vector<AdapterProfile> &profiles) {
  for (const AdapterProfile &profile : profiles) {
    if (!profile.lifecycle.stateful)
      continue;
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
        .closure_required = false,
        .migration_pending = true,
    };
    for (const auto &[name, output] : profile.authoring->outputs) {
      FinalOutputPolicy classified = classify_output(name, output);
      if (classified.evidence_kind == FinalEvidenceKind::closure) {
        if (policy.closure_output_name) {
          throw std::invalid_argument("stateful operation declares multiple "
                                      "finalization closure outputs");
        }
        policy.closure_output_name = name;
        policy.closure_required = classified.required;
        policy.migration_pending = !classified.required;
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

const OperationFinalizationPolicy &
FinalizationPolicyRegistry::resolve(const AdapterKey &key) const {
  const auto found = policies_.find(key);
  if (found == policies_.end()) {
    throw std::out_of_range(
        "operation has no registered stateful finalization policy");
  }
  return found->second;
}

const std::map<AdapterKey, OperationFinalizationPolicy> &
FinalizationPolicyRegistry::policies() const {
  return policies_;
}

nlohmann::json FinalizationPolicyRegistry::inventory_json() const {
  nlohmann::json operations = nlohmann::json::array();
  for (const auto &[key, policy] : policies_) {
    (void)key;
    operations.push_back(encode_json(policy));
  }
  return {{"api_version", "trainvm.finalization-inventory/v1"},
          {"operations", std::move(operations)}};
}

std::string
finalization_policy_digest(const OperationFinalizationPolicy &policy) {
  return "sha256:" + sha256_hex(encode_json(policy).dump());
}

FinalizationVerdict
reduce_final_evaluation(const OperationFinalizationPolicy &operation_policy,
                        const FinalEvaluationExpectation &expectation,
                        const std::vector<FinalEvaluationReceipt> &history) {
  if (operation_policy.migration_pending ||
      !operation_policy.closure_output_name ||
      !operation_policy.closure_required) {
    return verdict(FinalizationDisposition::pending,
                   "operation final evaluation producer is migration-pending");
  }
  const auto selected_policy =
      std::ranges::find(operation_policy.outputs, expectation.output_name,
                        &FinalOutputPolicy::output_name);
  if (selected_policy == operation_policy.outputs.end() ||
      !selected_policy->exact_optimizer_step ||
      !selected_policy->checkpoint_bound ||
      selected_policy->coverage != FinalCoveragePolicy::full_membership ||
      selected_policy->errors != FinalErrorPolicy::zero_unresolved_errors) {
    throw std::invalid_argument(
        "semantic final evaluation requires a registered strict output policy");
  }
  if (!bounded_identity(expectation.output_name) ||
      !valid_digest(expectation.policy_digest) ||
      expectation.policy_digest !=
          finalization_policy_digest(operation_policy) ||
      !bounded_identity(expectation.checkpoint_artifact_id) ||
      !valid_digest(expectation.checkpoint_fingerprint) ||
      !valid_digest(expectation.terminal_optimizer_fingerprint) ||
      expectation.required_members.size() > kMaximumFinalMembers ||
      !canonical_members(expectation.required_members) ||
      expectation.membership_count != expectation.required_members.size() ||
      !valid_digest(expectation.membership_digest) ||
      expectation.membership_digest !=
          final_membership_digest(expectation.required_members) ||
      !canonical_scalars(expectation.required_scalars)) {
    throw std::invalid_argument(
        "controller final evaluation expectation is invalid");
  }
  std::size_t expectation_identity_bytes = 0U;
  if (!consume_identity(expectation.output_name, expectation_identity_bytes) ||
      !consume_identity(expectation.checkpoint_artifact_id,
                        expectation_identity_bytes) ||
      std::ranges::any_of(expectation.required_members,
                          [&](const std::string &member) {
                            return !consume_identity(
                                member, expectation_identity_bytes);
                          }) ||
      std::ranges::any_of(expectation.required_scalars,
                          [&](const FinalScalarRequirement &scalar) {
                            return !consume_identity(
                                scalar.metric_name, expectation_identity_bytes);
                          })) {
    throw std::invalid_argument(
        "controller final evaluation expectation exceeds its byte bound");
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
  std::map<std::string, std::uint64_t> highest_attempt;
  std::map<std::string, std::string> member_contexts;
  std::set<std::string> successful;
  std::uint64_t previous_sequence = 0U;
  std::optional<std::string> selected_id;
  std::optional<std::string> selected_fingerprint;
  std::set<std::string> receipt_artifact_ids;
  std::set<std::string> receipt_artifact_fingerprints;
  std::vector<FinalOutputReceipt> previous_output_receipts;
  std::size_t aggregate_records = 0U;
  std::size_t aggregate_entries = 0U;
  std::size_t aggregate_identity_bytes = 0U;
  bool policy_output_receipted = false;

  for (std::size_t receipt_index = 0U; receipt_index < history.size();
       ++receipt_index) {
    const FinalEvaluationReceipt &receipt = history[receipt_index];
    const FinalEvaluationManifest &manifest = receipt.manifest;
    if (!bounded_identity(receipt.artifact_id) ||
        !valid_digest(receipt.artifact_fingerprint) ||
        receipt.durable_sequence == 0U ||
        receipt.durable_sequence <= previous_sequence ||
        !receipt_artifact_ids.insert(receipt.artifact_id).second ||
        !receipt_artifact_fingerprints.insert(receipt.artifact_fingerprint)
             .second) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation receipt durability is invalid");
    }
    previous_sequence = receipt.durable_sequence;
    if (!consume_identity(receipt.artifact_id, aggregate_identity_bytes) ||
        !consume_identity(receipt.artifact_fingerprint,
                          aggregate_identity_bytes) ||
        !consume_identity(manifest.api_version, aggregate_identity_bytes) ||
        !consume_identity(manifest.policy_digest, aggregate_identity_bytes) ||
        !consume_identity(manifest.checkpoint_artifact_id,
                          aggregate_identity_bytes) ||
        !consume_identity(manifest.checkpoint_fingerprint,
                          aggregate_identity_bytes) ||
        !consume_identity(manifest.membership_digest,
                          aggregate_identity_bytes)) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation identity byte budget was exceeded");
    }
    if (manifest.api_version != "rwkv-lab.final-evaluation/v1" ||
        manifest.policy_digest != expectation.policy_digest) {
      return verdict(
          FinalizationDisposition::failed,
          "final evaluation policy or schema disagrees with authority");
    }
    if (manifest.optimizer_step != expectation.optimizer_step) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation evidence is stale-step");
    }
    if (manifest.checkpoint_artifact_id != expectation.checkpoint_artifact_id ||
        manifest.checkpoint_fingerprint != expectation.checkpoint_fingerprint) {
      return verdict(
          FinalizationDisposition::failed,
          "final evaluation evidence is bound to another checkpoint");
    }
    if (manifest.required_members.size() > kMaximumFinalMembers ||
        manifest.records.size() > kMaximumFinalRecords ||
        manifest.output_receipts.size() > kMaximumFinalOutputs ||
        manifest.required_scalars.size() > kMaximumFinalScalars ||
        (manifest.recovery &&
         manifest.recovery->requested_members.size() > kMaximumFinalMembers) ||
        !consume(manifest.records.size(), kMaximumAggregateRecords,
                 aggregate_records) ||
        !consume(manifest.required_members.size() + manifest.records.size() +
                     manifest.output_receipts.size() +
                     manifest.required_scalars.size() +
                     (manifest.recovery
                          ? manifest.recovery->requested_members.size()
                          : 0U),
                 kMaximumAggregateCollectionEntries, aggregate_entries)) {
      return verdict(
          FinalizationDisposition::failed,
          "final evaluation receipt exceeds an aggregate work bound");
    }
    if (manifest.required_members != expectation.required_members ||
        manifest.membership_digest != expectation.membership_digest ||
        manifest.membership_count != expectation.membership_count) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation membership disagrees with authority");
    }
    if (manifest.required_scalars != expectation.required_scalars) {
      return verdict(FinalizationDisposition::failed,
                     "final scalar requirements disagree with authority");
    }

    const std::size_t prior_output_receipt_count =
        previous_output_receipts.size();
    if (manifest.output_receipts.size() < prior_output_receipt_count ||
        !std::ranges::equal(
            previous_output_receipts.begin(), previous_output_receipts.end(),
            manifest.output_receipts.begin(),
            manifest.output_receipts.begin() +
                static_cast<std::ptrdiff_t>(prior_output_receipt_count))) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation output receipt history drifted");
    }
    if (receipt_index == 0U &&
        (!std::ranges::is_sorted(manifest.output_receipts, {},
                                 &FinalOutputReceipt::output_name) ||
         std::ranges::adjacent_find(manifest.output_receipts, {},
                                    &FinalOutputReceipt::output_name) !=
             manifest.output_receipts.end())) {
      return verdict(FinalizationDisposition::failed,
                     "initial final output receipts are not canonical");
    }
    if (receipt_index > 0U &&
        (manifest.output_receipts.size() != prior_output_receipt_count + 1U ||
         manifest.output_receipts.back().output_name !=
             expectation.output_name)) {
      return verdict(
          FinalizationDisposition::failed,
          "final evaluation retry appended ambiguous output evidence");
    }
    std::set<std::string> output_artifact_ids;
    std::set<std::string> output_artifact_fingerprints;
    for (const FinalOutputReceipt &output : manifest.output_receipts) {
      const auto declared =
          std::ranges::find(operation_policy.outputs, output.output_name,
                            &FinalOutputPolicy::output_name);
      if (declared == operation_policy.outputs.end() ||
          !bounded_identity(output.artifact_id) ||
          !valid_digest(output.artifact_fingerprint) ||
          !output_artifact_ids.insert(output.artifact_id).second ||
          !output_artifact_fingerprints.insert(output.artifact_fingerprint)
               .second ||
          !consume_identity(output.output_name, aggregate_identity_bytes) ||
          !consume_identity(output.artifact_id, aggregate_identity_bytes) ||
          !consume_identity(output.artifact_fingerprint,
                            aggregate_identity_bytes)) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation output receipts are invalid");
      }
      if (output.output_name == expectation.output_name) {
        policy_output_receipted = true;
      }
    }

    if (manifest.records.size() < previous_records.size() ||
        !std::ranges::equal(
            previous_records.begin(), previous_records.end(),
            manifest.records.begin(),
            manifest.records.begin() +
                static_cast<std::ptrdiff_t>(previous_records.size()))) {
      return verdict(FinalizationDisposition::failed,
                     "final evaluation history was rewritten");
    }
    std::vector<std::string> unresolved_before;
    std::ranges::set_difference(
        expectation.required_members,
        std::vector<std::string>(successful.begin(), successful.end()),
        std::back_inserter(unresolved_before));
    const std::size_t appended_record_count =
        manifest.records.size() - previous_records.size();
    if (receipt_index == 0U) {
      if (manifest.recovery ||
          !std::ranges::is_sorted(manifest.records, {},
                                  &FinalMemberRecord::member_id) ||
          std::ranges::adjacent_find(manifest.records, {},
                                     &FinalMemberRecord::member_id) !=
              manifest.records.end()) {
        return verdict(FinalizationDisposition::failed,
                       "initial final member ledger is not canonical");
      }
    } else {
      if (!operation_policy.eval_only_recovery || !manifest.recovery ||
          manifest.recovery->requested_members != unresolved_before ||
          appended_record_count != unresolved_before.size() ||
          manifest.recovery->optimizer_state_before !=
              expectation.terminal_optimizer_fingerprint ||
          manifest.recovery->optimizer_state_after !=
              expectation.terminal_optimizer_fingerprint) {
        return verdict(
            FinalizationDisposition::failed,
            "eval-only recovery disagrees with controller authority");
      }
      for (std::size_t offset = 0U; offset < appended_record_count; ++offset) {
        if (manifest.records[previous_records.size() + offset].member_id !=
            unresolved_before[offset]) {
          return verdict(
              FinalizationDisposition::failed,
              "eval-only recovery evaluated members outside its exact scope");
        }
      }
    }

    for (std::size_t index = previous_records.size();
         index < manifest.records.size(); ++index) {
      const FinalMemberRecord &record = manifest.records[index];
      if (!std::ranges::binary_search(expectation.required_members,
                                      record.member_id) ||
          !valid_digest(record.context_digest) || record.attempt == 0U ||
          !consume_identity(record.member_id, aggregate_identity_bytes) ||
          !consume_identity(record.context_digest, aggregate_identity_bytes) ||
          (record.result_digest &&
           !consume_identity(*record.result_digest,
                             aggregate_identity_bytes))) {
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
      const bool success =
          record.disposition == FinalMemberDisposition::success;
      if ((success &&
           (!record.result_digest || !valid_digest(*record.result_digest) ||
            record.error_code)) ||
          (!success && (record.result_digest || !record.error_code ||
                        !bounded_identity(*record.error_code)))) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation record result is malformed");
      }
      if (record.error_code &&
          !consume_identity(*record.error_code, aggregate_identity_bytes)) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation identity byte budget was exceeded");
      }
      if (success)
        successful.insert(record.member_id);
    }
    for (const std::string &member : manifest.required_members) {
      if (!consume_identity(member, aggregate_identity_bytes)) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation identity byte budget was exceeded");
      }
    }
    for (const FinalScalarRequirement &scalar : manifest.required_scalars) {
      if (!consume_identity(scalar.metric_name, aggregate_identity_bytes) ||
          !consume_identity(scalar.step_domain, aggregate_identity_bytes)) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation identity byte budget was exceeded");
      }
    }
    if (manifest.recovery) {
      if (!consume_identity(manifest.recovery->optimizer_state_before,
                            aggregate_identity_bytes) ||
          !consume_identity(manifest.recovery->optimizer_state_after,
                            aggregate_identity_bytes)) {
        return verdict(FinalizationDisposition::failed,
                       "final evaluation identity byte budget was exceeded");
      }
      for (const std::string &member : manifest.recovery->requested_members) {
        if (!consume_identity(member, aggregate_identity_bytes)) {
          return verdict(FinalizationDisposition::failed,
                         "final evaluation identity byte budget was exceeded");
        }
      }
    }
    previous_records = manifest.records;
    std::set<std::string> resolved_at_receipt;
    for (const FinalMemberRecord &record : manifest.records) {
      if (record.disposition == FinalMemberDisposition::success) {
        resolved_at_receipt.insert(record.member_id);
      }
    }
    if (manifest.resolved_member_count != resolved_at_receipt.size() ||
        manifest.failed_member_count !=
            expectation.required_members.size() - resolved_at_receipt.size()) {
      return verdict(
          FinalizationDisposition::failed,
          "final evaluation summary counts disagree with its ledger");
    }
    previous_output_receipts = manifest.output_receipts;
    selected_id = receipt.artifact_id;
    selected_fingerprint = receipt.artifact_fingerprint;
  }

  std::vector<std::string> unresolved;
  std::ranges::set_difference(
      expectation.required_members,
      std::vector<std::string>(successful.begin(), successful.end()),
      std::back_inserter(unresolved));
  if (!policy_output_receipted) {
    return verdict(FinalizationDisposition::pending,
                   "required final evaluation output is not durably receipted",
                   expectation.required_members);
  }
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
    const std::vector<std::string> &canonical_members_value) {
  if (!canonical_members(canonical_members_value)) {
    throw std::invalid_argument(
        "membership digest input must be canonical membership");
  }
  return "sha256:" + sha256_hex(nlohmann::json(canonical_members_value).dump());
}

nlohmann::json
finalization_verdict_json(const FinalizationVerdict &verdict_value) {
  nlohmann::json value = encode_json(verdict_value);
  value["api_version"] = "trainvm.finalization-verdict/v1";
  return value;
}

nlohmann::json
final_evaluation_manifest_json(const FinalEvaluationManifest &manifest) {
  return encode_json(manifest);
}

} // namespace trainvm
