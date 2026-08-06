#include "trainvm/final_evaluation.hpp"

#include <algorithm>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/reflection_json.hpp"
#include "trainvm/rwkv_lab_worker_contract.hpp"

namespace {

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
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
      .artifact_schema = "rwkv-lab.hf-test-caption-evidence-bundle.v1",
  };
}

trainvm::OperationFinalizationPolicy strict_operation_policy() {
  return {
      .key = {.adapter = "test.adapter",
              .version = "1.0.0",
              .runtime = trainvm::ComponentRuntime::python_worker,
              .operation = "train",
              .contract = "test.FinalEvaluation"},
      .outputs = {strict_test_policy()},
      .eval_only_recovery = true,
      .closure_output_name = "final_evaluation",
      .closure_required = true,
      .migration_pending = false,
  };
}

trainvm::FinalMemberRecord error(std::string member,
                                 std::uint64_t attempt = 1U) {
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

trainvm::FinalEvaluationReceipt
receipt(const trainvm::OperationFinalizationPolicy &policy,
        std::uint64_t sequence, std::vector<std::string> members,
        std::vector<trainvm::FinalMemberRecord> records) {
  std::ranges::sort(members);
  std::set<std::string> resolved;
  for (const auto &record : records) {
    if (record.disposition == trainvm::FinalMemberDisposition::success) {
      resolved.insert(record.member_id);
    }
  }
  return {.artifact_id = "artifact-eval-" + std::to_string(sequence),
          .artifact_fingerprint = digest('e'),
          .durable_sequence = sequence,
          .manifest = {
              .api_version = "rwkv-lab.final-evaluation/v1",
              .policy_digest = trainvm::finalization_policy_digest(policy),
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
              .recovery = std::nullopt}};
}

trainvm::FinalEvaluationExpectation
expectation(const trainvm::OperationFinalizationPolicy &policy,
            std::vector<std::string> members,
            std::vector<trainvm::FinalScalarRequirement> scalars = {}) {
  std::ranges::sort(members);
  std::ranges::sort(scalars, {}, &trainvm::FinalScalarRequirement::metric_name);
  return {
      .output_name = "test_eval",
      .policy_digest = trainvm::finalization_policy_digest(policy),
      .optimizer_step = 745U,
      .checkpoint_artifact_id = "checkpoint-step-745",
      .checkpoint_fingerprint = digest('f'),
      .required_members = members,
      .membership_digest = trainvm::final_membership_digest(members),
      .membership_count = members.size(),
      .required_scalars = std::move(scalars),
      .terminal_optimizer_fingerprint = digest('8'),
  };
}

} // namespace

int main() {
  try {
    const auto worker = trainvm::rwkv_lab_worker_contract(digest('a'));
    const trainvm::FinalizationPolicyRegistry registry(
        worker.adapter_registry.profiles);

    std::set<trainvm::AdapterKey> stateful;
    for (const trainvm::AdapterProfile &profile :
         worker.adapter_registry.profiles) {
      if (!profile.lifecycle.stateful)
        continue;
      stateful.insert(profile.key);
      const auto &policy = registry.resolve(profile.key);
      require(policy.key == profile.key,
              "stateful profile resolved another finalization identity");
      require(
          policy.migration_pending == !policy.closure_required &&
              (!policy.closure_required ||
               policy.closure_output_name.has_value()),
          "closure producer gaps must be explicit in the registry inventory");
      require(policy.outputs.size() + (policy.closure_output_name ? 1U : 0U) ==
                  profile.authoring->outputs.size(),
              "every stateful operation output must have one finalization "
              "classification");
      for (const auto &[name, descriptor] : profile.authoring->outputs) {
        if (policy.closure_output_name == name)
          continue;
        const auto found = std::ranges::find(
            policy.outputs, name, &trainvm::FinalOutputPolicy::output_name);
        require(found != policy.outputs.end() &&
                    found->required == descriptor.required,
                "registry output requirement must survive finalization "
                "classification");
      }
    }
    std::set<trainvm::AdapterKey> inventoried;
    for (const auto &[key, policy] : registry.policies()) {
      (void)policy;
      inventoried.insert(key);
    }
    require(inventoried == stateful,
            "finalization inventory must be derived from every stateful "
            "registry profile without a count pin");
    require(registry.inventory_json().at("operations").size() ==
                stateful.size(),
            "dashboard inventory JSON must expose every stateful policy");

    auto optional_closure_profile = *std::ranges::find_if(
        worker.adapter_registry.profiles, [](const auto &profile) {
          return profile.key.adapter == "rwkv-lab.hf-multimodal-sft";
        });
    optional_closure_profile.authoring->outputs.emplace(
        "final_evaluation",
        trainvm::OperationPortDescriptor{
            .type = trainvm::OperationPortType::artifact,
            .required = false,
            .artifact_type = trainvm::ArtifactType::report,
            .artifact_schema = "rwkv-lab.final-evaluation.v1",
            .description = std::nullopt});
    const trainvm::FinalizationPolicyRegistry optional_closure_registry(
        {optional_closure_profile});
    const auto &optional_closure =
        optional_closure_registry.resolve(optional_closure_profile.key);
    require(optional_closure.closure_output_name ==
                    std::optional<std::string>{"final_evaluation"} &&
                !optional_closure.closure_required &&
                optional_closure.migration_pending,
            "an optional closure must not clear migration-pending authority");
    optional_closure_profile.authoring->outputs.at("final_evaluation")
        .required = true;
    const trainvm::FinalizationPolicyRegistry required_closure_registry(
        {optional_closure_profile});
    const auto &required_closure =
        required_closure_registry.resolve(optional_closure_profile.key);
    require(required_closure.closure_required &&
                !required_closure.migration_pending,
            "only a required canonical closure may clear migration-pending");

    const auto &hf = registry.resolve(
        std::ranges::find_if(worker.adapter_registry.profiles, [](const auto &
                                                                      profile) {
          return profile.key.adapter == "rwkv-lab.hf-multimodal-sft";
        })->key);
    require(std::ranges::any_of(
                hf.outputs,
                [](const auto &output) {
                  return output.output_name == "test_eval" && output.required &&
                         output.evidence_kind ==
                             trainvm::FinalEvidenceKind::test &&
                         output.exact_optimizer_step &&
                         output.checkpoint_bound &&
                         output.coverage ==
                             trainvm::FinalCoveragePolicy::full_membership &&
                         output.errors ==
                             trainvm::FinalErrorPolicy::zero_unresolved_errors;
                }),
            "Qwen/HF test evidence must be a strict final completion barrier");

    const auto strict = strict_operation_policy();

    std::vector<std::string> qwen_members;
    std::vector<trainvm::FinalMemberRecord> qwen_errors;
    for (std::size_t index = 0; index < 674U; ++index) {
      const std::string id = "qwen-caption-" + std::to_string(index);
      qwen_members.push_back(id);
      qwen_errors.push_back(error(id));
    }
    std::ranges::sort(qwen_members);
    std::ranges::sort(qwen_errors, {}, &trainvm::FinalMemberRecord::member_id);
    const auto qwen_expectation = expectation(strict, qwen_members);
    auto all_error = receipt(strict, 1U, qwen_members, qwen_errors);
    const auto all_error_verdict =
        trainvm::reduce_final_evaluation(strict, qwen_expectation, {all_error});
    require(all_error_verdict.disposition ==
                    trainvm::FinalizationDisposition::pending &&
                all_error_verdict.unresolved_members.size() == 674U &&
                all_error_verdict.cause.find("no successful") !=
                    std::string::npos,
            "the production Qwen 674/674 generation-error incident must never "
            "complete");

    auto recovered = all_error;
    recovered.artifact_id = "artifact-eval-recovered";
    recovered.artifact_fingerprint = digest('9');
    recovered.durable_sequence = 2U;
    recovered.manifest.recovery =
        trainvm::EvalOnlyRecoveryReceipt{.requested_members = qwen_members,
                                         .optimizer_state_before = digest('8'),
                                         .optimizer_state_after = digest('8')};
    recovered.manifest.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-recovered",
         .artifact_fingerprint = digest('2')});
    for (const std::string &member : qwen_members) {
      recovered.manifest.records.push_back(success(member, 2U));
    }
    recovered.manifest.resolved_member_count = qwen_members.size();
    recovered.manifest.failed_member_count = 0U;
    const auto recovery_verdict = trainvm::reduce_final_evaluation(
        strict, qwen_expectation, {all_error, recovered});
    require(recovery_verdict.disposition ==
                    trainvm::FinalizationDisposition::complete &&
                recovery_verdict.selected_artifact_id ==
                    std::optional<std::string>{"artifact-eval-recovered"},
            "eval-only recovery must permit completion after exactly "
            "unresolved rows succeed");

    auto mutated = recovered;
    mutated.manifest.recovery->optimizer_state_after = digest('7');
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             {all_error, mutated})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "eval-only recovery must prove zero optimizer mutation");

    auto wrong_members = recovered;
    wrong_members.manifest.recovery->requested_members.pop_back();
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             {all_error, wrong_members})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "eval-only recovery must retry every and only unresolved member");

    auto noncanonical_recovery = recovered;
    std::ranges::reverse(
        noncanonical_recovery.manifest.recovery->requested_members);
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             {all_error, noncanonical_recovery})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "recovery membership must use the one canonical wire ordering");

    auto worker_selected_policy = all_error;
    worker_selected_policy.manifest.policy_digest = digest('6');
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             {worker_selected_policy})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "worker-selected policy identity must not become authority");

    auto optimizer_spoof = recovered;
    optimizer_spoof.manifest.recovery->optimizer_state_before = digest('7');
    optimizer_spoof.manifest.recovery->optimizer_state_after = digest('7');
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             {all_error, optimizer_spoof})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "equal worker optimizer assertions must still match controller "
            "authority");

    auto migration_pending = strict;
    migration_pending.closure_required = false;
    migration_pending.migration_pending = true;
    require(trainvm::reduce_final_evaluation(migration_pending,
                                             qwen_expectation, {recovered})
                    .disposition == trainvm::FinalizationDisposition::pending,
            "migration-pending operations must never complete");

    auto no_recovery = strict;
    no_recovery.eval_only_recovery = false;
    const auto no_recovery_expectation = expectation(no_recovery, qwen_members);
    auto no_recovery_base = all_error;
    auto no_recovery_retry = recovered;
    no_recovery_base.manifest.policy_digest =
        no_recovery_expectation.policy_digest;
    no_recovery_retry.manifest.policy_digest =
        no_recovery_expectation.policy_digest;
    require(
        trainvm::reduce_final_evaluation(no_recovery, no_recovery_expectation,
                                         {no_recovery_base, no_recovery_retry})
                .disposition == trainvm::FinalizationDisposition::failed,
        "operation policy must explicitly authorize eval-only recovery");

    const auto small_expectation = expectation(strict, {"a", "b"});
    auto partial = receipt(strict, 1U, {"a", "b"}, {success("a"), error("b")});
    require(
        trainvm::reduce_final_evaluation(strict, small_expectation, {partial})
                .disposition == trainvm::FinalizationDisposition::pending,
        "partial final evidence must remain pending");

    auto undersized = receipt(strict, 1U, {"a"}, {success("a")});
    require(trainvm::reduce_final_evaluation(strict, small_expectation,
                                             {undersized})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "worker-selected undersized membership must not satisfy authority");

    auto noncanonical_members = partial;
    std::ranges::reverse(noncanonical_members.manifest.required_members);
    require(trainvm::reduce_final_evaluation(strict, small_expectation,
                                             {noncanonical_members})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "noncanonical wire membership must fail closed");

    auto stale = partial;
    stale.manifest.optimizer_step = 744U;
    require(trainvm::reduce_final_evaluation(strict, small_expectation, {stale})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "stale-step evidence must fail finalization");

    auto poisoned_retry = partial;
    poisoned_retry.artifact_id = "artifact-poisoned-retry";
    poisoned_retry.durable_sequence = 2U;
    auto changed_context = success("b", 2U);
    changed_context.context_digest = digest('3');
    poisoned_retry.manifest.records.push_back(std::move(changed_context));
    poisoned_retry.manifest.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-poisoned",
         .artifact_fingerprint = digest('2')});
    poisoned_retry.manifest.resolved_member_count = 2U;
    poisoned_retry.manifest.failed_member_count = 0U;
    poisoned_retry.manifest.recovery =
        trainvm::EvalOnlyRecoveryReceipt{.requested_members = {"b"},
                                         .optimizer_state_before = digest('8'),
                                         .optimizer_state_after = digest('8')};
    require(trainvm::reduce_final_evaluation(strict, small_expectation,
                                             {partial, poisoned_retry})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "context-poisoned retry evidence must fail finalization");

    auto out_of_scope_retry = partial;
    out_of_scope_retry.artifact_id = "artifact-out-of-scope-retry";
    out_of_scope_retry.artifact_fingerprint = digest('1');
    out_of_scope_retry.durable_sequence = 2U;
    out_of_scope_retry.manifest.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-out-of-scope",
         .artifact_fingerprint = digest('2')});
    out_of_scope_retry.manifest.recovery =
        trainvm::EvalOnlyRecoveryReceipt{.requested_members = {"b"},
                                         .optimizer_state_before = digest('8'),
                                         .optimizer_state_after = digest('8')};
    out_of_scope_retry.manifest.records.push_back(success("b", 2U));
    out_of_scope_retry.manifest.records.push_back(success("a", 2U));
    out_of_scope_retry.manifest.resolved_member_count = 2U;
    out_of_scope_retry.manifest.failed_member_count = 0U;
    require(trainvm::reduce_final_evaluation(strict, small_expectation,
                                             {partial, out_of_scope_retry})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "retry rows must contain every and only unresolved members");

    auto wrong_checkpoint = partial;
    wrong_checkpoint.manifest.checkpoint_artifact_id = "checkpoint-step-700";
    require(trainvm::reduce_final_evaluation(strict, small_expectation,
                                             {wrong_checkpoint})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "evidence from another checkpoint must fail finalization");

    auto scalar_base = partial;
    scalar_base.manifest.required_scalars = {
        {.metric_name = "eval.loss", .step_domain = "optimizer_step"}};
    const auto scalar_expectation =
        expectation(strict, {"a", "b"}, scalar_base.manifest.required_scalars);
    auto scalar_drift = scalar_base;
    scalar_drift.artifact_id = "artifact-scalar-drift";
    scalar_drift.artifact_fingerprint = digest('3');
    scalar_drift.durable_sequence = 2U;
    scalar_drift.manifest.required_scalars = {
        {.metric_name = "eval.perplexity", .step_domain = "optimizer_step"}};
    scalar_drift.manifest.recovery =
        trainvm::EvalOnlyRecoveryReceipt{.requested_members = {"b"},
                                         .optimizer_state_before = digest('8'),
                                         .optimizer_state_after = digest('8')};
    scalar_drift.manifest.output_receipts.push_back(
        {.output_name = "test_eval",
         .artifact_id = "test-evidence-scalar-drift",
         .artifact_fingerprint = digest('2')});
    scalar_drift.manifest.records.push_back(success("b", 2U));
    scalar_drift.manifest.resolved_member_count = 2U;
    scalar_drift.manifest.failed_member_count = 0U;
    require(trainvm::reduce_final_evaluation(strict, scalar_expectation,
                                             {scalar_base, scalar_drift})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "scalar requirements must match controller authority");

    auto output_drift = scalar_drift;
    output_drift.manifest.required_scalars =
        scalar_base.manifest.required_scalars;
    output_drift.manifest.output_receipts.front().artifact_id =
        "rewritten-parent";
    require(trainvm::reduce_final_evaluation(strict, scalar_expectation,
                                             {scalar_base, output_drift})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "parent output receipt history must remain append-only");

    auto duplicate_closure = recovered;
    duplicate_closure.artifact_id = all_error.artifact_id;
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             {all_error, duplicate_closure})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "closure artifact identities must be unique across history");

    std::vector<trainvm::FinalEvaluationReceipt> excessive_history(65U,
                                                                   all_error);
    require(trainvm::reduce_final_evaluation(strict, qwen_expectation,
                                             excessive_history)
                    .disposition == trainvm::FinalizationDisposition::failed,
            "receipt history must have a hard pre-reduction bound");

    auto excessive_members = partial;
    excessive_members.manifest.required_members.assign(100'001U, "x");
    require(trainvm::reduce_final_evaluation(strict, small_expectation,
                                             {excessive_members})
                    .disposition == trainvm::FinalizationDisposition::failed,
            "manifest collections must have hard work bounds");

    require(trainvm::finalization_verdict_json(all_error_verdict).at("cause") ==
                all_error_verdict.cause,
            "dashboard verdict must expose the exact finalization cause");
    const nlohmann::json manifest =
        trainvm::final_evaluation_manifest_json(all_error.manifest);
    require(
        manifest.at("api_version") == "rwkv-lab.final-evaluation/v1" &&
            !manifest.contains("artifact_id") &&
            !manifest.contains("artifact_fingerprint") &&
            !manifest.contains("durable_sequence"),
        "closure manifest must exclude controller-derived durability identity");
    trainvm::FinalEvaluationManifest decoded_manifest;
    std::vector<trainvm::Diagnostic> manifest_diagnostics;
    require(trainvm::decode_json(manifest, decoded_manifest, "",
                                 manifest_diagnostics) &&
                manifest_diagnostics.empty() &&
                decoded_manifest == all_error.manifest,
            "semantic manifest reflection must round-trip exactly");
    const nlohmann::json controller_receipt = trainvm::encode_json(all_error);
    trainvm::FinalEvaluationReceipt decoded_receipt;
    std::vector<trainvm::Diagnostic> receipt_diagnostics;
    require(trainvm::decode_json(controller_receipt, decoded_receipt, "",
                                 receipt_diagnostics) &&
                receipt_diagnostics.empty() && decoded_receipt == all_error,
            "controller receipt reflection must round-trip exactly");
    auto forged_manifest = manifest;
    forged_manifest["artifact_id"] = "worker-selected-controller-id";
    trainvm::FinalEvaluationManifest rejected_manifest;
    std::vector<trainvm::Diagnostic> forged_diagnostics;
    require(!trainvm::decode_json(forged_manifest, rejected_manifest, "",
                                  forged_diagnostics),
            "semantic manifest must reject controller-owned fields");
    std::cout << "final evaluation tests passed\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
