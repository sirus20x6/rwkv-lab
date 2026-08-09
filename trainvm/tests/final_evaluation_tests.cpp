#include "trainvm/final_evaluation.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/adapter_invocation.hpp"
#include "trainvm/eval_examples_contract.hpp"
#include "trainvm/input_content_authority.hpp"
#include "trainvm/journal.hpp"
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
        std::vector<trainvm::FinalMemberRecord> records,
        std::string output_name = "test_eval") {
  std::ranges::sort(members);
  std::set<std::string> resolved;
  for (const auto &record : records) {
    if (record.disposition == trainvm::FinalMemberDisposition::success) {
      resolved.insert(record.member_id);
    }
  }
  std::vector<trainvm::FinalOutputReceipt> output_receipts;
  for (const auto &declared : policy.outputs) {
    if (!declared.required && declared.output_name != output_name)
      continue;
    output_receipts.push_back(
        {.output_name = declared.output_name,
         .artifact_id =
             declared.output_name + "-evidence-" + std::to_string(sequence),
         .artifact_fingerprint =
             "sha256:" + trainvm::sha256_hex(declared.output_name + ":" +
                                             std::to_string(sequence))});
  }
  std::ranges::sort(output_receipts, {},
                    &trainvm::FinalOutputReceipt::output_name);
  trainvm::FinalEvaluationReceipt result{
      .artifact_id = "artifact-eval-" + std::to_string(sequence),
      .artifact_fingerprint = digest('e'),
      .durable_sequence = sequence,
      .durable_output_receipts = {},
      .durable_scalar_observations = {},
      .manifest = {.api_version = "rwkv-lab.final-evaluation/v1",
                   .policy_digest = trainvm::finalization_policy_digest(policy),
                   .optimizer_step = 745U,
                   .checkpoint_artifact_id = "checkpoint-step-745",
                   .checkpoint_fingerprint = digest('f'),
                   .membership_digest =
                       trainvm::final_membership_digest(members),
                   .membership_count = members.size(),
                   .resolved_member_count = resolved.size(),
                   .failed_member_count = members.size() - resolved.size(),
                   .required_members = std::move(members),
                   .output_receipts = std::move(output_receipts),
                   .required_scalars = {},
                   .records = std::move(records),
                   .recovery = std::nullopt}};
  result.durable_output_receipts = result.manifest.output_receipts;
  return result;
}

trainvm::FinalEvaluationExpectation
expectation(const trainvm::OperationFinalizationPolicy &policy,
            std::vector<std::string> members,
            std::vector<trainvm::FinalScalarRequirement> scalars = {},
            std::string output_name = "test_eval") {
  std::ranges::sort(members);
  std::ranges::sort(scalars, {}, &trainvm::FinalScalarRequirement::metric_name);
  std::vector<std::string> required_output_names;
  for (const auto &declared : policy.outputs) {
    if (declared.required || declared.output_name == output_name)
      required_output_names.push_back(declared.output_name);
  }
  std::ranges::sort(required_output_names);
  return {
      .output_name = std::move(output_name),
      .required_output_names = std::move(required_output_names),
      .policy_digest = trainvm::finalization_policy_digest(policy),
      .optimizer_step = 745U,
      .checkpoint_artifact_id = "checkpoint-step-745",
      .checkpoint_fingerprint = digest('f'),
      .required_members = members,
      .member_contexts =
          [&members] {
            std::vector<trainvm::FinalMemberContext> contexts;
            contexts.reserve(members.size());
            for (const std::string &member : members) {
              contexts.push_back(
                  {.member_id = member, .context_digest = digest('c')});
            }
            return contexts;
          }(),
      .membership_digest = trainvm::final_membership_digest(members),
      .membership_count = members.size(),
      .required_scalars = std::move(scalars),
      .terminal_optimizer_fingerprint =
          policy.eval_only_recovery
              ? std::optional<std::string>{digest('8')}
              : std::nullopt,
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
    optional_closure_profile.authoring->outputs.at("final_evaluation")
        .required = false;
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
    optional_closure_profile.authoring->outputs.emplace(
        "semantic_examples",
        trainvm::OperationPortDescriptor{
            .type = trainvm::OperationPortType::artifact,
            .required = true,
            .artifact_type = trainvm::ArtifactType::eval_examples,
            .artifact_schema = std::string(trainvm::kEvalExamplesSchema),
            .description = std::nullopt});
    const trainvm::FinalizationPolicyRegistry required_closure_registry(
        {optional_closure_profile});
    const auto &required_closure =
        required_closure_registry.resolve(optional_closure_profile.key);
    require(required_closure.closure_required &&
                !required_closure.migration_pending,
            "only a required canonical closure may clear migration-pending");
    require(std::ranges::any_of(
                required_closure.outputs,
                [](const auto &output) {
                  return output.output_name == "semantic_examples" &&
                         output.evidence_kind ==
                             trainvm::FinalEvidenceKind::examples &&
                         output.exact_optimizer_step &&
                         output.checkpoint_bound &&
                         output.coverage ==
                             trainvm::FinalCoveragePolicy::full_membership &&
                         output.errors ==
                             trainvm::FinalErrorPolicy::zero_unresolved_errors;
                }),
            "merged eval_examples artifacts must inventory as strict examples "
            "evidence");

    const auto semantic_expectation =
        expectation(required_closure, {"heldout-1"}, {}, "semantic_examples");
    const auto semantic_receipt =
        receipt(required_closure, 1U, {"heldout-1"}, {success("heldout-1")},
                "semantic_examples");
    require(trainvm::reduce_final_evaluation(
                required_closure, semantic_expectation, {semantic_receipt})
                    .disposition == trainvm::FinalizationDisposition::complete,
            "merged eval_examples evidence must reduce through its strict "
            "inventory policy");

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
    auto telemetry_profile = *std::ranges::find_if(
        worker.adapter_registry.profiles, [](const auto &profile) {
          return profile.key.adapter == "rwkv-lab.hf-multimodal-sft";
        });
    telemetry_profile.authoring->outputs.emplace(
        "metrics", trainvm::OperationPortDescriptor{
                       .type = trainvm::OperationPortType::artifact,
                       .required = false,
                       .artifact_type = trainvm::ArtifactType::metrics,
                       .artifact_schema = "test.append-only-metrics.v1",
                       .description = std::nullopt});
    const trainvm::FinalizationPolicyRegistry telemetry_registry(
        {telemetry_profile});
    const auto &telemetry_policy =
        telemetry_registry.resolve(telemetry_profile.key);
    require(std::ranges::any_of(
                telemetry_policy.outputs,
                [](const auto &output) {
                  return output.output_name == "metrics" &&
                         output.evidence_kind ==
                             trainvm::FinalEvidenceKind::audit &&
                         !output.required_when_declared;
                }),
            "append-only metrics must not impersonate terminal evidence");
    require(hf.closure_required && !hf.migration_pending &&
                hf.closure_output_name ==
                    std::optional<std::string>{"final_evaluation"},
            "HF must leave migration-pending only with its required closure");

    const std::filesystem::path authority_root =
        std::filesystem::temp_directory_path() /
        "trainvm-final-evaluation-authority-test";
    std::filesystem::remove_all(authority_root);
    std::filesystem::create_directories(authority_root);
    trainvm::WorkerInvocationSpec authority_invocation{
        .api_version = std::string(trainvm::kWorkerInvocationApiVersion),
        .run_id = "authority-run",
        .host_id = "authority-host",
        .plan_hash = digest('1'),
        .plan_revision = 1U,
        .node_id = "train",
        .attempt_id = "train@1",
        .dispatch_id = "authority-dispatch",
        .adapter = hf.key,
        .workspace = {{"run_directory", (authority_root / "run").string()}},
        .resources = nlohmann::json::object(),
        .inputs = nlohmann::json::object(),
        .controls = nlohmann::json::object(),
        .effective_control_revision = 0U,
        .publishes =
            {{"checkpoint", {{"logical_name", "checkpoint"}}},
             {"eval_gallery", {{"logical_name", "eval_gallery"}}},
             {"metrics", {{"logical_name", "metrics"}}},
             {"test_eval", {{"logical_name", "test_eval"}}},
             {"eval_examples", {{"logical_name", "eval_examples"}}},
             {"final_evaluation",
              {{"logical_name", "final_evaluation"}}}},
        .observability =
            {{"metrics",
              nlohmann::json::array(
                  {{{"name", "eval.loss"},
                    {"step_domain", "optimizer_step"}},
                   {{"name", "eval.worker_extra"},
                    {"step_domain", "optimizer_step"}}})}},
        .execution = nlohmann::json::object(),
        .training = nlohmann::json::object(),
        .resume = nlohmann::json::object(),
        .invocation_digest = digest('2')};
    nlohmann::json components = nlohmann::json::object();
    for (const std::string slot :
         {"artifact_renderer", "data", "evaluator", "generation_policy",
          "processor", "sample_mapping", "test_split"}) {
      components[slot] = {{"descriptor_digest", digest('3')},
                          {"configuration", nlohmann::json::object()}};
    }
    components["evaluator"]["configuration"]["metrics"] = {"loss"};
    components["data"]["configuration"] =
        {{"dataset_root", authority_root.string()},
         {"content_fingerprint", digest('4')},
         {"id_column", "id"}};
    authority_invocation.training["components"] = components;
    const auto seal_authority_dataset = [&](const std::string& test_bytes,
                                            std::optional<std::string>
                                                receipt_override =
                                                    std::nullopt) {
      {
        std::ofstream output(authority_root / "test.jsonl",
                             std::ios::binary | std::ios::trunc);
        output.write(test_bytes.data(),
                     static_cast<std::streamsize>(test_bytes.size()));
      }
      const std::size_t rows = static_cast<std::size_t>(
          std::ranges::count(test_bytes, '\n'));
      const std::string receipt = receipt_override.value_or(
          nlohmann::json(
              {{"counts", {{"test", rows}}},
               {"files",
                {{"test.jsonl",
                  {{"rows", rows},
                   {"sha256", trainvm::sha256_hex(test_bytes)}}}}}})
              .dump());
      {
        std::ofstream output(authority_root / "manifest.json",
                             std::ios::binary | std::ios::trunc);
        output.write(receipt.data(),
                     static_cast<std::streamsize>(receipt.size()));
      }
      const auto identity =
          trainvm::measure_input_content_root(authority_root);
      authority_invocation.workspace["input_content_roots"] =
          nlohmann::json::array({trainvm::encode_json(identity)});
      authority_invocation.training["components"]["data"]["configuration"]
                                  ["content_fingerprint"] =
          identity.tree_sha256;
    };
    const trainvm::Event terminal_checkpoint{
        .event_id = "terminal-checkpoint-event",
        .run_id = authority_invocation.run_id,
        .run_revision = 1U,
        .plan_revision = 1U,
        .node_id = authority_invocation.node_id,
        .attempt_id = authority_invocation.attempt_id,
        .worker_sequence = 1U,
        .event_type = "artifact.published",
        .event_version = 1U,
        .wall_time_ns = 1U,
        .monotonic_time_ns = 1U,
        .optimizer_step = 745U,
        .payload = {{"logical_name", "checkpoint"},
                    {"kind", "checkpoint"},
                    {"complete", true},
                    {"artifact_id", "checkpoint-step-745"},
                    {"fingerprint", digest('5')}}};
    const std::string valid_test_row =
        R"({"id":"member-a","split":"test"})" "\n";
    seal_authority_dataset(valid_test_row);
    const auto authority_expectation =
        trainvm::derive_hf_final_evaluation_expectation(
            telemetry_policy, authority_invocation, 745U,
            {terminal_checkpoint});
    const std::vector<trainvm::FinalScalarRequirement>
        expected_authority_scalars{
            {.metric_name = "eval.loss",
             .step_domain = "optimizer_step"}};
    nlohmann::json expected_component_digests = nlohmann::json::object();
    for (const std::string slot :
         {"artifact_renderer", "data", "evaluator", "generation_policy",
          "processor", "sample_mapping", "test_split"})
      expected_component_digests[slot] = digest('3');
    const std::string expected_member_context =
        "sha256:" + trainvm::sha256_hex(
                        nlohmann::json(
                            {{"api_version",
                              "rwkv-lab.hf-final-member-context/v1"},
                             {"components", expected_component_digests},
                             {"member_id", "member-a"}})
                            .dump());
    require(authority_expectation.required_members ==
                    std::vector<std::string>{"member-a"} &&
                authority_expectation.member_contexts.front().context_digest ==
                    expected_member_context &&
                authority_expectation.required_scalars ==
                    expected_authority_scalars &&
                !std::ranges::binary_search(
                    authority_expectation.required_output_names,
                    std::string("final_evaluation")) &&
                !std::ranges::binary_search(
                    authority_expectation.required_output_names,
                    std::string("metrics")) &&
                !authority_expectation.terminal_optimizer_fingerprint,
            "HF authority freezes test membership and evaluator metrics while "
            "ignoring worker-only observable extras and closure self-parents");
    auto resume_invocation = authority_invocation;
    resume_invocation.attempt_id = "train@2";
    auto resume_checkpoint_event = terminal_checkpoint;
    resume_checkpoint_event.attempt_id = "train@1";
    resume_checkpoint_event.payload["fingerprint_algorithm"] =
        "manifest_sha256";
    resume_checkpoint_event.payload["producer_node_id"] = "train";
    resume_checkpoint_event.payload["producer_attempt_id"] = "train@1";
    resume_invocation.resume =
        {{"api_version", "trainvm.resume-checkpoint/v1"},
         {"checkpoint", resume_checkpoint_event.payload},
         {"optimizer_step", 745U},
         {"pause_command_id", "pause-1"},
         {"resume_command_id", "resume-1"}};
    const auto resumed_expectation =
        trainvm::derive_hf_final_evaluation_expectation(
            telemetry_policy, resume_invocation, 745U,
            {resume_checkpoint_event});
    require(resumed_expectation.checkpoint_artifact_id ==
                    authority_expectation.checkpoint_artifact_id &&
                resumed_expectation.checkpoint_fingerprint ==
                    authority_expectation.checkpoint_fingerprint,
            "terminal replacement attempts inherit only the controller-selected "
            "exact resume checkpoint");
    const auto authority_rejects = [&](const std::string& test_bytes,
                                       std::optional<std::string>
                                           receipt_override = std::nullopt) {
      seal_authority_dataset(test_bytes, std::move(receipt_override));
      try {
        (void)trainvm::derive_hf_final_evaluation_expectation(
            telemetry_policy, authority_invocation, 745U,
            {terminal_checkpoint});
        return false;
      } catch (const std::invalid_argument&) {
        return true;
      }
    };
    require(authority_rejects(
                R"({"id":"member-a","id":"member-b","split":"test"})"
                "\n"),
            "frozen dataset rows reject duplicate JSON keys");
    std::string deep_row = R"({"id":"member-a","split":"test","x":)";
    deep_row.append(34U, '[');
    deep_row += '0';
    deep_row.append(34U, ']');
    deep_row += "}\n";
    require(authority_rejects(deep_row),
            "frozen dataset rows reject excessive JSON depth");
    std::string node_row =
        R"({"id":"member-a","split":"test","x":[)";
    for (std::size_t index = 0U; index < 10'001U; ++index)
      node_row += index == 0U ? "0" : ",0";
    node_row += "]}\n";
    require(authority_rejects(node_row),
            "frozen dataset rows reject excessive JSON nodes");
    const std::string oversized_row =
        R"({"id":"member-a","split":"test","blob":")" +
        std::string(1024U * 1024U, 'x') + "\"}\n";
    require(authority_rejects(oversized_row),
            "frozen dataset rows reject oversized lines before parsing");
    const std::string duplicate_receipt =
        R"({"counts":{"test":1},"counts":{"test":1},"files":{"test.jsonl":{"rows":1,"sha256":")" +
        trainvm::sha256_hex(valid_test_row) + "\"}}}";
    require(authority_rejects(valid_test_row, duplicate_receipt),
            "frozen dataset receipts reject duplicate JSON keys");
    std::filesystem::remove_all(authority_root);

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
    recovered.durable_output_receipts = recovered.manifest.output_receipts;
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
    auto forged_optimizer_expectation = no_recovery_expectation;
    forged_optimizer_expectation.terminal_optimizer_fingerprint = digest('8');
    bool forged_optimizer_rejected = false;
    try {
      (void)trainvm::reduce_final_evaluation(
          no_recovery, forged_optimizer_expectation, {all_error});
    } catch (const std::invalid_argument&) {
      forged_optimizer_rejected = true;
    }
    require(forged_optimizer_rejected,
            "a checkpoint digest cannot impersonate independently verified "
            "optimizer-state authority");
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
    poisoned_retry.durable_output_receipts =
        poisoned_retry.manifest.output_receipts;
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
    out_of_scope_retry.durable_output_receipts =
        out_of_scope_retry.manifest.output_receipts;
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
    scalar_base.durable_scalar_observations = {
        {.metric_name = "eval.loss",
         .step_domain = "optimizer_step",
         .optimizer_step = 745U,
         .event_id = "metric-eval-loss-step-745"}};
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
    scalar_drift.durable_output_receipts =
        scalar_drift.manifest.output_receipts;
    scalar_drift.durable_scalar_observations =
        scalar_base.durable_scalar_observations;
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
    output_drift.durable_output_receipts =
        output_drift.manifest.output_receipts;
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
    excessive_members.manifest.required_members.assign(10'001U, "x");
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
    const std::string canonical_manifest_bytes = manifest.dump();
    require(trainvm::decode_final_evaluation_manifest(
                canonical_manifest_bytes) == all_error.manifest,
            "raw semantic manifest authority decoder must round-trip canonical "
            "bytes");
    const auto decoder_rejects = [](std::string_view source) {
      try {
        (void)trainvm::decode_final_evaluation_manifest(source);
        return false;
      } catch (const std::invalid_argument &) {
        return true;
      }
    };
    require(
        decoder_rejects(manifest.dump(2)),
        "raw semantic manifest decoder must reject alternate JSON encodings");
    std::string duplicate_key_manifest = canonical_manifest_bytes;
    duplicate_key_manifest.replace(
        1U, 0U,
        R"("api_version":"rwkv-lab.final-evaluation/v1",)" );
    require(decoder_rejects(duplicate_key_manifest),
            "raw semantic manifest decoder must reject duplicate JSON keys");
    const std::string oversized_manifest(
        trainvm::kMaximumFinalEvaluationManifestBytes + 1U, ' ');
    require(
        decoder_rejects(oversized_manifest),
        "raw semantic manifest bytes must be rejected before JSON allocation");
    auto near_member_bound = all_error.manifest;
    near_member_bound.required_members.clear();
    near_member_bound.records.clear();
    for (std::size_t index = 0U; index < 10'000U; ++index) {
      const std::string member =
          "member-" + std::string(5U - std::to_string(index).size(), '0') +
          std::to_string(index);
      near_member_bound.required_members.push_back(member);
      near_member_bound.records.push_back(error(member));
    }
    near_member_bound.membership_count =
        near_member_bound.required_members.size();
    near_member_bound.membership_digest =
        trainvm::final_membership_digest(near_member_bound.required_members);
    near_member_bound.resolved_member_count = 0U;
    near_member_bound.failed_member_count =
        near_member_bound.required_members.size();
    const std::string near_member_bound_bytes =
        trainvm::final_evaluation_manifest_json(near_member_bound).dump();
    require(near_member_bound_bytes.size() <
                    trainvm::kMaximumFinalEvaluationManifestBytes &&
                trainvm::decode_final_evaluation_manifest(
                    near_member_bound_bytes) == near_member_bound,
            "the parser bound must admit a canonical manifest at the explicit "
            "10,000-member semantic maximum");
    std::string structural_bomb = "[";
    structural_bomb.reserve(1'200'002U);
    for (std::size_t index = 0U; index < 600'000U; ++index)
      structural_bomb += "0,";
    structural_bomb += "0]";
    require(structural_bomb.size() <
                    trainvm::kMaximumFinalEvaluationManifestBytes &&
                decoder_rejects(structural_bomb),
            "in-budget JSON node bombs must fail during bounded parsing");
    std::string depth_bomb(66U, '[');
    depth_bomb += '0';
    depth_bomb.append(66U, ']');
    require(decoder_rejects(depth_bomb),
            "in-budget JSON depth bombs must fail during bounded parsing");
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
    require(
        decoder_rejects(forged_manifest.dump()),
        "raw semantic manifest decoder must reject controller-owned fields");
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
