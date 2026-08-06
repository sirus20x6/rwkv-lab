#include "trainvm/final_evaluation.hpp"

#include <algorithm>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/rwkv_lab_worker_contract.hpp"

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

std::string digest(char character) {
  return "sha256:" + std::string(64U, character);
}

trainvm::FinalOutputPolicy strict_test_policy() {
  return {
      .output_name = "test_eval",
      .evidence_kind = trainvm::FinalEvidenceKind::test,
      .required = true,
      .required_when_declared = false,
      .exact_optimizer_step = true,
      .checkpoint_bound = true,
      .coverage = trainvm::FinalCoveragePolicy::full_membership,
      .errors = trainvm::FinalErrorPolicy::zero_unresolved_errors,
      .artifact_schema =
          "rwkv-lab.hf-test-caption-evidence-bundle.v1",
  };
}

trainvm::FinalMemberRecord error(std::string member, std::uint64_t attempt = 1U) {
  return {.member_id = std::move(member),
          .context_digest = digest('c'),
          .attempt = attempt,
          .disposition = trainvm::FinalMemberDisposition::error,
          .result_digest = std::nullopt,
          .error_code = "generation_failed"};
}

trainvm::FinalMemberRecord success(std::string member,
                                   std::uint64_t attempt = 1U) {
  return {.member_id = std::move(member),
          .context_digest = digest('c'),
          .attempt = attempt,
          .disposition = trainvm::FinalMemberDisposition::success,
          .result_digest = digest('d'),
          .error_code = std::nullopt};
}

trainvm::FinalEvaluationReceipt receipt(
    std::uint64_t sequence, std::vector<std::string> members,
    std::vector<trainvm::FinalMemberRecord> records) {
  std::ranges::sort(members);
  std::set<std::string> resolved;
  for (const auto& record : records) {
    if (record.disposition == trainvm::FinalMemberDisposition::success) {
      resolved.insert(record.member_id);
    }
  }
  return {.artifact_id = "artifact-eval-" + std::to_string(sequence),
          .artifact_fingerprint = digest('e'),
          .durable_sequence = sequence,
          .policy_digest = digest('5'),
          .optimizer_step = 745U,
          .checkpoint_artifact_id = "checkpoint-step-745",
          .checkpoint_fingerprint = digest('f'),
          .membership_digest = trainvm::final_membership_digest(members),
          .membership_count = members.size(),
          .resolved_member_count = resolved.size(),
          .failed_member_count = members.size() - resolved.size(),
          .required_members = std::move(members),
          .output_receipts = {{.output_name = "test_eval",
                               .artifact_id = "test-evidence-" +
                                              std::to_string(sequence),
                               .artifact_fingerprint = digest('4')}},
          .required_scalars = {},
          .records = std::move(records),
          .recovery = std::nullopt};
}

}  // namespace

int main() {
  try {
    const auto worker = trainvm::rwkv_lab_worker_contract(digest('a'));
    const trainvm::FinalizationPolicyRegistry registry(
        worker.adapter_registry.profiles);

    std::set<trainvm::AdapterKey> stateful;
    for (const trainvm::AdapterProfile& profile :
         worker.adapter_registry.profiles) {
      if (!profile.lifecycle.stateful) continue;
      stateful.insert(profile.key);
      const auto& policy = registry.resolve(profile.key);
      require(policy.key == profile.key,
              "stateful profile resolved another finalization identity");
      require(policy.migration_pending ==
                      !policy.closure_output_name.has_value(),
              "closure producer gaps must be explicit in the registry inventory");
      require(policy.outputs.size() +
                      (policy.closure_output_name ? 1U : 0U) ==
                  profile.authoring->outputs.size(),
              "every stateful operation output must have one finalization classification");
      for (const auto& [name, descriptor] : profile.authoring->outputs) {
        if (policy.closure_output_name == name) continue;
        const auto found = std::ranges::find(policy.outputs, name,
                                             &trainvm::FinalOutputPolicy::output_name);
        require(found != policy.outputs.end() &&
                    found->required == descriptor.required,
                "registry output requirement must survive finalization classification");
      }
    }
    std::set<trainvm::AdapterKey> inventoried;
    for (const auto& [key, policy] : registry.policies()) {
      (void)policy;
      inventoried.insert(key);
    }
    require(inventoried == stateful,
            "finalization inventory must be derived from every stateful registry profile without a count pin");
    require(registry.inventory_json().at("operations").size() == stateful.size(),
            "dashboard inventory JSON must expose every stateful policy");

    const auto& hf = registry.resolve(std::ranges::find_if(
        worker.adapter_registry.profiles, [](const auto& profile) {
          return profile.key.adapter == "rwkv-lab.hf-multimodal-sft";
        })->key);
    require(std::ranges::any_of(hf.outputs, [](const auto& output) {
              return output.output_name == "test_eval" && output.required &&
                     output.evidence_kind == trainvm::FinalEvidenceKind::test &&
                     output.exact_optimizer_step && output.checkpoint_bound &&
                     output.coverage ==
                         trainvm::FinalCoveragePolicy::full_membership &&
                     output.errors ==
                         trainvm::FinalErrorPolicy::zero_unresolved_errors;
            }),
            "Qwen/HF test evidence must be a strict final completion barrier");

    std::vector<std::string> qwen_members;
    std::vector<trainvm::FinalMemberRecord> qwen_errors;
    for (std::size_t index = 0; index < 674U; ++index) {
      const std::string id = "qwen-caption-" + std::to_string(index);
      qwen_members.push_back(id);
      qwen_errors.push_back(error(id));
    }
    auto all_error = receipt(1U, qwen_members, qwen_errors);
    const auto all_error_verdict = trainvm::reduce_final_evaluation(
        strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
        {all_error});
    require(all_error_verdict.disposition ==
                    trainvm::FinalizationDisposition::pending &&
                all_error_verdict.unresolved_members.size() == 674U &&
                all_error_verdict.cause.find("no successful") !=
                    std::string::npos,
            "the production Qwen 674/674 generation-error incident must never complete");

    auto recovered = all_error;
    recovered.artifact_id = "artifact-eval-recovered";
    recovered.artifact_fingerprint = digest('9');
    recovered.durable_sequence = 2U;
    recovered.recovery = trainvm::EvalOnlyRecoveryReceipt{
        .requested_members = qwen_members,
        .optimizer_state_before = digest('8'),
        .optimizer_state_after = digest('8')};
    recovered.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-recovered",
         .artifact_fingerprint = digest('2')});
    for (const std::string& member : qwen_members) {
      recovered.records.push_back(success(member, 2U));
    }
    recovered.resolved_member_count = qwen_members.size();
    recovered.failed_member_count = 0U;
    const auto recovery_verdict = trainvm::reduce_final_evaluation(
        strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
        {all_error, recovered});
    require(recovery_verdict.disposition ==
                    trainvm::FinalizationDisposition::complete &&
                recovery_verdict.selected_artifact_id ==
                    std::optional<std::string>{"artifact-eval-recovered"},
            "eval-only recovery must permit completion after exactly unresolved rows succeed");

    auto mutated = recovered;
    mutated.recovery->optimizer_state_after = digest('7');
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {all_error, mutated})
                .disposition == trainvm::FinalizationDisposition::failed,
            "eval-only recovery must prove zero optimizer mutation");

    auto wrong_members = recovered;
    wrong_members.recovery->requested_members.pop_back();
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {all_error, wrong_members})
                .disposition == trainvm::FinalizationDisposition::failed,
            "eval-only recovery must retry every and only unresolved member");

    auto partial = receipt(1U, {"a", "b"}, {success("a"), error("b")});
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {partial})
                    .disposition == trainvm::FinalizationDisposition::pending,
            "partial final evidence must remain pending");

    auto stale = partial;
    stale.optimizer_step = 744U;
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {stale})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "stale-step evidence must fail finalization");

    auto poisoned_retry = partial;
    poisoned_retry.artifact_id = "artifact-poisoned-retry";
    poisoned_retry.durable_sequence = 2U;
    auto changed_context = success("b", 2U);
    changed_context.context_digest = digest('3');
    poisoned_retry.records.push_back(std::move(changed_context));
    poisoned_retry.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-poisoned",
         .artifact_fingerprint = digest('2')});
    poisoned_retry.resolved_member_count = 2U;
    poisoned_retry.failed_member_count = 0U;
    poisoned_retry.recovery = trainvm::EvalOnlyRecoveryReceipt{
        .requested_members = {"b"},
        .optimizer_state_before = digest('8'),
        .optimizer_state_after = digest('8')};
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {partial, poisoned_retry})
                .disposition == trainvm::FinalizationDisposition::failed,
            "context-poisoned retry evidence must fail finalization");

    auto wrong_checkpoint = partial;
    wrong_checkpoint.checkpoint_artifact_id = "checkpoint-step-700";
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {wrong_checkpoint})
                .disposition == trainvm::FinalizationDisposition::failed,
            "evidence from another checkpoint must fail finalization");

    auto scalar_base = partial;
    scalar_base.required_scalars = {
        {.metric_name = "eval.loss", .step_domain = "optimizer_step"}};
    auto scalar_drift = scalar_base;
    scalar_drift.artifact_id = "artifact-scalar-drift";
    scalar_drift.artifact_fingerprint = digest('3');
    scalar_drift.durable_sequence = 2U;
    scalar_drift.required_scalars = {
        {.metric_name = "eval.perplexity", .step_domain = "optimizer_step"}};
    scalar_drift.recovery = trainvm::EvalOnlyRecoveryReceipt{
        .requested_members = {"b"},
        .optimizer_state_before = digest('8'),
        .optimizer_state_after = digest('8')};
    scalar_drift.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-scalar-drift",
         .artifact_fingerprint = digest('2')});
    scalar_drift.records.push_back(success("b", 2U));
    scalar_drift.resolved_member_count = 2U;
    scalar_drift.failed_member_count = 0U;
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {scalar_base, scalar_drift})
                .disposition == trainvm::FinalizationDisposition::failed,
            "scalar requirements must not drift between recovery receipts");

    auto output_drift = scalar_drift;
    output_drift.required_scalars = scalar_base.required_scalars;
    output_drift.output_receipts.front().artifact_id = "rewritten-parent";
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {scalar_base, output_drift})
                .disposition == trainvm::FinalizationDisposition::failed,
            "parent output receipt history must remain append-only");

    auto duplicate_closure = recovered;
    duplicate_closure.artifact_id = all_error.artifact_id;
    require(trainvm::reduce_final_evaluation(
                strict_test_policy(), 745U, "checkpoint-step-745", digest('f'),
                {all_error, duplicate_closure})
                .disposition == trainvm::FinalizationDisposition::failed,
            "closure artifact identities must be unique across history");

    require(trainvm::finalization_verdict_json(all_error_verdict)
                    .at("cause") == all_error_verdict.cause,
            "dashboard verdict must expose the exact finalization cause");
    const nlohmann::json manifest =
        trainvm::final_evaluation_manifest_json(all_error);
    require(manifest.at("api_version") ==
                    "rwkv-lab.final-evaluation/v1" &&
                !manifest.contains("artifact_id") &&
                !manifest.contains("artifact_fingerprint") &&
                !manifest.contains("durable_sequence"),
            "closure manifest must exclude controller-derived durability identity");
    std::cout << "final evaluation tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
