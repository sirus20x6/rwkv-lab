#include "trainvm/final_evaluation.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <linux/openat2.h>
#include <memory>
#include <openssl/evp.h>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/adapter_invocation.hpp"
#include "trainvm/document.hpp"
#include "trainvm/input_content_authority.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumFinalizationHistory = 64U;
// The first migrated family evaluates 674 held-out members.  Keep the public
// closure envelope comfortably above that while making the semantic maxima
// actually representable inside the raw/parser bounds below.
constexpr std::size_t kMaximumFinalMembers = 10'000U;
constexpr std::size_t kMaximumFinalRecords = 40'000U;
constexpr std::size_t kMaximumFinalOutputs = 64U;
constexpr std::size_t kMaximumFinalScalars = 256U;
constexpr std::size_t kMaximumAggregateRecords = 2'000'000U;
constexpr std::size_t kMaximumAggregateCollectionEntries = 4'000'000U;
constexpr std::size_t kMaximumAggregateIdentityBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumManifestParserNodes = 500'000U;
constexpr int kMaximumManifestParserDepth = 64;
constexpr std::uint64_t kMaximumFrozenDatasetReceiptBytes = 4U * 1024U * 1024U;
constexpr std::uint64_t kMaximumFrozenTestManifestBytes = 64U * 1024U * 1024U;

class FinalDescriptor final {
public:
  explicit FinalDescriptor(int value = -1) : value_(value) {}
  FinalDescriptor(const FinalDescriptor &) = delete;
  FinalDescriptor &operator=(const FinalDescriptor &) = delete;
  FinalDescriptor(FinalDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  ~FinalDescriptor() {
    if (value_ >= 0)
      ::close(value_);
  }
  int get() const { return value_; }
  explicit operator bool() const { return value_ >= 0; }

private:
  int value_;
};

FinalDescriptor final_open_beneath(int root, std::string_view relative,
                                   int flags) {
  const std::string owned(relative);
  open_how how{};
  how.flags = static_cast<std::uint64_t>(flags | O_CLOEXEC | O_NOFOLLOW);
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
  const long result =
      ::syscall(SYS_openat2, root, owned.c_str(), &how, sizeof(how));
  if (result < 0)
    throw std::invalid_argument("final evaluation path cannot be pinned");
  return FinalDescriptor(static_cast<int>(result));
}

std::string final_descriptor_bytes(int descriptor, std::uint64_t maximum) {
  struct stat before {};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_nlink != 1 || before.st_size < 0 ||
      static_cast<std::uint64_t>(before.st_size) > maximum ||
      ::lseek(descriptor, 0, SEEK_SET) < 0)
    throw std::invalid_argument("final evaluation file is invalid");
  std::string result(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0U;
  while (offset < result.size()) {
    const ssize_t count =
        ::read(descriptor, result.data() + offset, result.size() - offset);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      throw std::invalid_argument("final evaluation file read failed");
    }
    if (count == 0)
      throw std::invalid_argument("final evaluation file was truncated");
    offset += static_cast<std::size_t>(count);
  }
  struct stat after {};
  if (::fstat(descriptor, &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      after.st_nlink != 1 ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
    throw std::invalid_argument("final evaluation file changed while reading");
  return result;
}

std::string final_file_bytes(int root, std::string_view relative,
                             std::uint64_t maximum) {
  FinalDescriptor file = final_open_beneath(root, relative, O_RDONLY);
  return final_descriptor_bytes(file.get(), maximum);
}

nlohmann::json parse_bounded_authority_json(std::string_view bytes,
                                            std::size_t maximum_nodes,
                                            int maximum_depth) {
  std::size_t nodes = 0U;
  std::vector<std::optional<std::set<std::string>>> containers;
  try {
    const nlohmann::json::parser_callback_t callback =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json &value) {
          if (depth > maximum_depth)
            throw std::invalid_argument(
                "final evaluation authority JSON exceeds its depth bound");
          if (event == nlohmann::json::parse_event_t::object_start) {
            containers.emplace_back(std::set<std::string>{});
            ++nodes;
          } else if (event == nlohmann::json::parse_event_t::array_start) {
            containers.emplace_back(std::nullopt);
            ++nodes;
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (containers.empty() || !containers.back() ||
                !containers.back()->insert(value.get<std::string>()).second)
              throw std::invalid_argument(
                  "final evaluation authority JSON has duplicate keys");
          } else if (event == nlohmann::json::parse_event_t::value) {
            ++nodes;
          } else if (event == nlohmann::json::parse_event_t::object_end ||
                     event == nlohmann::json::parse_event_t::array_end) {
            if (containers.empty())
              throw std::invalid_argument(
                  "final evaluation authority JSON nesting is invalid");
            containers.pop_back();
          }
          if (nodes > maximum_nodes)
            throw std::invalid_argument(
                "final evaluation authority JSON exceeds its node bound");
          return true;
        };
    nlohmann::json result =
        nlohmann::json::parse(bytes.begin(), bytes.end(), callback);
    if (!containers.empty())
      throw std::invalid_argument(
          "final evaluation authority JSON nesting is incomplete");
    return result;
  } catch (const nlohmann::json::exception &) {
    throw std::invalid_argument(
        "final evaluation authority JSON is malformed");
  }
}

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
  case ArtifactType::eval_examples:
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

bool canonical_member_contexts(const std::vector<FinalMemberContext> &values) {
  return values.size() <= kMaximumFinalMembers &&
         std::ranges::is_sorted(values, {}, &FinalMemberContext::member_id) &&
         std::ranges::adjacent_find(
             values, {}, &FinalMemberContext::member_id) == values.end() &&
         std::ranges::all_of(values, [](const FinalMemberContext &value) {
           return bounded_identity(value.member_id) &&
                  valid_digest(value.context_digest);
         });
}

bool canonical_output_names(const std::vector<std::string> &values) {
  return !values.empty() && values.size() <= kMaximumFinalOutputs &&
         std::ranges::is_sorted(values) &&
         std::ranges::adjacent_find(values) == values.end() &&
         std::ranges::none_of(values, [](const std::string &value) {
           return !bounded_identity(value);
         });
}

bool canonical_scalar_observations(
    const std::vector<FinalScalarObservation> &values) {
  return values.size() <= kMaximumFinalScalars &&
         std::ranges::is_sorted(values, {},
                                &FinalScalarObservation::metric_name) &&
         std::ranges::adjacent_find(values, {},
                                    &FinalScalarObservation::metric_name) ==
             values.end() &&
         std::ranges::all_of(values, [](const FinalScalarObservation &value) {
           return bounded_identity(value.metric_name) &&
                  value.step_domain == "optimizer_step" &&
                  bounded_identity(value.event_id);
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
        // Resume-grade alone does not prove that an eval-only retry preserves
        // optimizer bytes. Families opt in only after a controller-verifiable
        // optimizer-state receipt exists.
        .eval_only_recovery = false,
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
      !canonical_output_names(expectation.required_output_names) ||
      !std::ranges::binary_search(expectation.required_output_names,
                                  expectation.output_name) ||
      !valid_digest(expectation.policy_digest) ||
      expectation.policy_digest !=
          finalization_policy_digest(operation_policy) ||
      !bounded_identity(expectation.checkpoint_artifact_id) ||
      !valid_digest(expectation.checkpoint_fingerprint) ||
      (operation_policy.eval_only_recovery
           ? (!expectation.terminal_optimizer_fingerprint ||
              !valid_digest(*expectation.terminal_optimizer_fingerprint))
           : expectation.terminal_optimizer_fingerprint.has_value()) ||
      expectation.required_members.size() > kMaximumFinalMembers ||
      !canonical_members(expectation.required_members) ||
      !canonical_member_contexts(expectation.member_contexts) ||
      expectation.member_contexts.size() !=
          expectation.required_members.size() ||
      expectation.membership_count != expectation.required_members.size() ||
      !valid_digest(expectation.membership_digest) ||
      expectation.membership_digest !=
          final_membership_digest(expectation.required_members) ||
      !canonical_scalars(expectation.required_scalars)) {
    throw std::invalid_argument(
        "controller final evaluation expectation is invalid");
  }
  for (std::size_t index = 0U; index < expectation.required_members.size();
       ++index) {
    if (expectation.member_contexts[index].member_id !=
        expectation.required_members[index]) {
      throw std::invalid_argument(
          "controller final member contexts disagree with membership");
    }
  }
  for (const std::string &required_output : expectation.required_output_names) {
    if (std::ranges::find(operation_policy.outputs, required_output,
                          &FinalOutputPolicy::output_name) ==
        operation_policy.outputs.end()) {
      throw std::invalid_argument(
          "controller final output expectation is not registered");
    }
  }
  for (const FinalOutputPolicy &policy_output : operation_policy.outputs) {
    if (policy_output.required &&
        !std::ranges::binary_search(expectation.required_output_names,
                                    policy_output.output_name)) {
      throw std::invalid_argument(
          "controller final output expectation omits a required output");
    }
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
      std::ranges::any_of(expectation.required_output_names,
                          [&](const std::string &output) {
                            return !consume_identity(
                                output, expectation_identity_bytes);
                          }) ||
      std::ranges::any_of(
          expectation.member_contexts,
          [&](const FinalMemberContext &context) {
            return !consume_identity(context.member_id,
                                     expectation_identity_bytes) ||
                   !consume_identity(context.context_digest,
                                     expectation_identity_bytes);
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
  std::set<std::string> receipted_output_names;

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
    if (receipt.durable_output_receipts != manifest.output_receipts) {
      return verdict(FinalizationDisposition::failed,
                     "worker output receipts disagree with durable parents");
    }
    if (!canonical_scalar_observations(receipt.durable_scalar_observations) ||
        receipt.durable_scalar_observations.size() !=
            expectation.required_scalars.size()) {
      return verdict(FinalizationDisposition::failed,
                     "terminal scalar durability is incomplete");
    }
    for (std::size_t scalar_index = 0U;
         scalar_index < expectation.required_scalars.size(); ++scalar_index) {
      const FinalScalarRequirement &required =
          expectation.required_scalars[scalar_index];
      const FinalScalarObservation &observed =
          receipt.durable_scalar_observations[scalar_index];
      if (observed.metric_name != required.metric_name ||
          observed.step_domain != required.step_domain ||
          observed.optimizer_step != expectation.optimizer_step) {
        return verdict(FinalizationDisposition::failed,
                       "terminal scalar durability disagrees with authority");
      }
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
      receipted_output_names.insert(output.output_name);
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
              *expectation.terminal_optimizer_fingerprint ||
          manifest.recovery->optimizer_state_after !=
              *expectation.terminal_optimizer_fingerprint) {
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
      const auto expected_context = std::ranges::lower_bound(
          expectation.member_contexts, record.member_id, {},
          &FinalMemberContext::member_id);
      if (expected_context == expectation.member_contexts.end() ||
          expected_context->member_id != record.member_id ||
          expected_context->context_digest != record.context_digest) {
        return verdict(
            FinalizationDisposition::failed,
            "final evaluation member context disagrees with authority");
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
  if (!std::ranges::includes(receipted_output_names,
                             expectation.required_output_names)) {
    return verdict(
        FinalizationDisposition::pending,
        "required final evaluation outputs are not durably receipted",
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

FinalEvaluationManifest
decode_final_evaluation_manifest(std::string_view bytes) {
  if (bytes.empty() || bytes.size() > kMaximumFinalEvaluationManifestBytes) {
    throw std::invalid_argument(
        "final evaluation manifest bytes are empty or exceed their bound");
  }
  nlohmann::json document;
  try {
    std::size_t parser_nodes = 0U;
    const nlohmann::json::parser_callback_t bounded_parser =
        [&](int depth, nlohmann::json::parse_event_t event, nlohmann::json &) {
          const bool allocates_node =
              event == nlohmann::json::parse_event_t::object_start ||
              event == nlohmann::json::parse_event_t::array_start ||
              event == nlohmann::json::parse_event_t::value;
          if (depth > kMaximumManifestParserDepth ||
              (allocates_node &&
               ++parser_nodes > kMaximumManifestParserNodes)) {
            throw std::invalid_argument(
                "final evaluation manifest JSON exceeds structural limits");
          }
          return true;
        };
    document =
        nlohmann::json::parse(bytes.begin(), bytes.end(), bounded_parser);
  } catch (const nlohmann::json::exception &) {
    throw std::invalid_argument("final evaluation manifest JSON is invalid");
  }
  FinalEvaluationManifest manifest;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(document, manifest, "", diagnostics) ||
      !diagnostics.empty()) {
    throw std::invalid_argument(
        "final evaluation manifest does not match its reflected schema");
  }
  if (encode_json(manifest).dump() != bytes) {
    throw std::invalid_argument(
        "final evaluation manifest bytes are not canonical");
  }
  if (manifest.api_version != "rwkv-lab.final-evaluation/v1" ||
      !valid_digest(manifest.policy_digest) ||
      !bounded_identity(manifest.checkpoint_artifact_id) ||
      !valid_digest(manifest.checkpoint_fingerprint) ||
      manifest.required_members.size() > kMaximumFinalMembers ||
      !canonical_members(manifest.required_members) ||
      !valid_digest(manifest.membership_digest) ||
      manifest.membership_digest !=
          final_membership_digest(manifest.required_members) ||
      manifest.membership_count != manifest.required_members.size() ||
      manifest.records.size() > kMaximumFinalRecords ||
      manifest.output_receipts.size() > kMaximumFinalOutputs ||
      !canonical_scalars(manifest.required_scalars) ||
      manifest.resolved_member_count > manifest.membership_count ||
      manifest.failed_member_count !=
          manifest.membership_count - manifest.resolved_member_count ||
      (manifest.recovery &&
       (manifest.recovery->requested_members.size() > kMaximumFinalMembers ||
        !canonical_members(manifest.recovery->requested_members) ||
        !valid_digest(manifest.recovery->optimizer_state_before) ||
        !valid_digest(manifest.recovery->optimizer_state_after)))) {
    throw std::invalid_argument(
        "final evaluation manifest semantic shape is invalid");
  }
  std::set<std::string> output_artifact_ids;
  std::set<std::string> output_artifact_fingerprints;
  for (const FinalOutputReceipt &output : manifest.output_receipts) {
    if (!bounded_identity(output.output_name) ||
        !bounded_identity(output.artifact_id) ||
        !valid_digest(output.artifact_fingerprint) ||
        !output_artifact_ids.insert(output.artifact_id).second ||
        !output_artifact_fingerprints.insert(output.artifact_fingerprint)
             .second) {
      throw std::invalid_argument(
          "final evaluation manifest output receipt is invalid");
    }
  }
  for (const FinalMemberRecord &record : manifest.records) {
    const bool success = record.disposition == FinalMemberDisposition::success;
    if (!std::ranges::binary_search(manifest.required_members,
                                    record.member_id) ||
        !valid_digest(record.context_digest) || record.attempt == 0U ||
        (success &&
         (!record.result_digest || !valid_digest(*record.result_digest) ||
          record.error_code)) ||
        (!success && (record.result_digest || !record.error_code ||
                      !bounded_identity(*record.error_code)))) {
      throw std::invalid_argument(
          "final evaluation manifest member record is invalid");
    }
  }
  return manifest;
}

FinalEvaluationExpectation derive_hf_final_evaluation_expectation(
    const OperationFinalizationPolicy &policy,
    const WorkerInvocationSpec &invocation, std::uint64_t terminal_step,
    const std::vector<Event> &durable_events) {
  if (policy.key != invocation.adapter || policy.migration_pending ||
      policy.closure_output_name !=
          std::optional<std::string>{"final_evaluation"} ||
      !policy.closure_required || !invocation.training.is_object() ||
      !invocation.training.contains("components") ||
      !invocation.training.at("components").is_object() ||
      !invocation.workspace.is_object() ||
      !invocation.publishes.is_object() ||
      !invocation.observability.is_object())
    throw std::invalid_argument(
        "HF finalization authority lacks an immutable invocation");

  std::vector<std::string> required_output_names;
  for (const FinalOutputPolicy &output : policy.outputs) {
    const bool declared = invocation.publishes.contains(output.output_name);
    if (output.required && !declared)
      throw std::invalid_argument(
          "HF finalization invocation omits a required policy output");
    if (output.required || (output.required_when_declared && declared))
      required_output_names.push_back(output.output_name);
  }
  std::ranges::sort(required_output_names);
  if (!std::ranges::binary_search(required_output_names,
                                  std::string("test_eval")))
    throw std::invalid_argument(
        "HF finalization lacks strict test-evaluation authority");

  const Event *checkpoint = nullptr;
  for (const Event &event : durable_events) {
    if (event.event_type != "artifact.published" ||
        event.node_id != invocation.node_id ||
        event.attempt_id != invocation.attempt_id ||
        event.optimizer_step != terminal_step ||
        event.payload.value("logical_name", std::string{}) !=
            invocation.publishes.at("checkpoint")
                .value("logical_name", std::string{}) ||
        event.payload.value("kind", std::string{}) != "checkpoint" ||
        !event.payload.value("complete", false))
      continue;
    if (checkpoint != nullptr)
      throw std::invalid_argument(
          "HF finalization has ambiguous terminal checkpoints");
    checkpoint = &event;
  }
  if (checkpoint == nullptr && invocation.resume.is_object() &&
      invocation.resume.value("api_version", std::string{}) ==
          "trainvm.resume-checkpoint/v1" &&
      invocation.resume.value("optimizer_step", std::uint64_t{}) ==
          terminal_step &&
      invocation.resume.contains("checkpoint") &&
      invocation.resume.at("checkpoint").is_object()) {
    const nlohmann::json &selected = invocation.resume.at("checkpoint");
    const std::string selected_id =
        selected.value("artifact_id", std::string{});
    const std::string selected_attempt =
        selected.value("producer_attempt_id", std::string{});
    const auto matches_selected = [&](const Event &event) {
      return event.event_type == "artifact.published" &&
             event.payload.value("artifact_id", std::string{}) == selected_id;
    };
    if (!bounded_identity(selected_id) ||
        !bounded_identity(selected_attempt) ||
        selected_attempt == invocation.attempt_id ||
        selected.value("producer_node_id", std::string{}) !=
            invocation.node_id ||
        selected.value("logical_name", std::string{}) !=
            invocation.publishes.at("checkpoint")
                .value("logical_name", std::string{}) ||
        selected.value("kind", std::string{}) != "checkpoint" ||
        selected.value("fingerprint_algorithm", std::string{}) !=
            "manifest_sha256" ||
        !selected.value("complete", false) ||
        !valid_digest(selected.value("fingerprint", std::string{})) ||
        std::ranges::count_if(durable_events, matches_selected) != 1)
      throw std::invalid_argument(
          "HF finalization selected resume checkpoint is invalid");
    const auto durable_selected =
        std::ranges::find_if(durable_events, matches_selected);
    if (durable_selected == durable_events.end() ||
        durable_selected->node_id != invocation.node_id ||
        durable_selected->attempt_id != selected_attempt ||
        durable_selected->optimizer_step != terminal_step ||
        durable_selected->payload != selected)
      throw std::invalid_argument(
          "HF finalization selected resume checkpoint lacks exact durability");
    checkpoint = &*durable_selected;
  }
  if (checkpoint == nullptr)
    throw std::invalid_argument(
        "HF finalization has no exact terminal checkpoint");
  const std::string checkpoint_id =
      checkpoint->payload.value("artifact_id", std::string{});
  const std::string checkpoint_fingerprint =
      checkpoint->payload.value("fingerprint", std::string{});

  const nlohmann::json &components =
      invocation.training.at("components");
  constexpr std::array<std::string_view, 7> context_slots{
      "artifact_renderer", "data",       "evaluator", "generation_policy",
      "processor",         "sample_mapping", "test_split"};
  nlohmann::json component_digests = nlohmann::json::object();
  for (const std::string_view slot : context_slots) {
    const auto component = components.find(std::string(slot));
    if (component == components.end() || !component->is_object() ||
        !valid_digest(component->value("descriptor_digest", std::string{})))
      throw std::invalid_argument(
          "HF final member context lacks a component descriptor");
    component_digests[std::string(slot)] =
        component->at("descriptor_digest");
  }
  const nlohmann::json &data = components.at("data");
  if (!data.contains("configuration") ||
      !data.at("configuration").is_object())
    throw std::invalid_argument("HF finalization data authority is malformed");
  const nlohmann::json &configuration = data.at("configuration");
  const std::filesystem::path dataset_root(
      configuration.value("dataset_root", std::string{}));
  const std::string content_fingerprint =
      configuration.value("content_fingerprint", std::string{});
  const std::string id_column =
      configuration.value("id_column", std::string{});
  if (!dataset_root.is_absolute() ||
      dataset_root.lexically_normal() != dataset_root ||
      !valid_digest(content_fingerprint) || !bounded_identity(id_column) ||
      !invocation.workspace.contains("input_content_roots") ||
      !invocation.workspace.at("input_content_roots").is_array())
    throw std::invalid_argument(
        "HF finalization dataset root lacks admitted content authority");
  const auto admitted = std::ranges::find_if(
      invocation.workspace.at("input_content_roots"),
      [&](const nlohmann::json &identity) {
        return identity.is_object() &&
               identity.value("path", std::string{}) == dataset_root.string();
      });
  if (admitted == invocation.workspace.at("input_content_roots").end() ||
      admitted->value("api_version", std::string{}) !=
          "trainvm.input-content-root/v1" ||
      admitted->value("kind", std::string{}) != "directory" ||
      admitted->value("tree_sha256", std::string{}) != content_fingerprint)
    throw std::invalid_argument(
        "HF finalization dataset root disagrees with admission");
  const InputContentRootIdentity remeasured =
      measure_input_content_root(dataset_root);
  InputContentRootIdentity admitted_identity;
  std::vector<Diagnostic> root_diagnostics;
  if (!decode_json(*admitted, admitted_identity, "", root_diagnostics) ||
      !root_diagnostics.empty() || remeasured != admitted_identity)
    throw std::invalid_argument(
        "HF finalization dataset content drifted after admission");

  FinalDescriptor filesystem_root(
      ::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!filesystem_root)
    throw std::runtime_error("HF finalization cannot pin filesystem root");
  const std::string dataset_relative =
      dataset_root.lexically_relative("/").generic_string();
  FinalDescriptor dataset = final_open_beneath(
      filesystem_root.get(), dataset_relative, O_PATH | O_DIRECTORY);
  const std::string receipt_bytes = final_file_bytes(
      dataset.get(), "manifest.json", kMaximumFrozenDatasetReceiptBytes);
  const std::string test_bytes = final_file_bytes(
      dataset.get(), "test.jsonl", kMaximumFrozenTestManifestBytes);
  const nlohmann::json receipt =
      parse_bounded_authority_json(receipt_bytes, 100'000U, 32);
  if (!receipt.is_object() || !receipt.contains("counts") ||
      !receipt.at("counts").is_object() || !receipt.contains("files") ||
      !receipt.at("files").is_object() ||
      !receipt.at("files").contains("test.jsonl") ||
      !receipt.at("files").at("test.jsonl").is_object() ||
      !receipt.at("counts").contains("test") ||
      !receipt.at("counts").at("test").is_number_unsigned() ||
      !receipt.at("files").at("test.jsonl").contains("rows") ||
      !receipt.at("files").at("test.jsonl").at("rows").is_number_unsigned() ||
      receipt.at("files").at("test.jsonl").value("sha256", std::string{}) !=
          sha256_hex(test_bytes) ||
      receipt.at("counts").at("test") !=
          receipt.at("files").at("test.jsonl").at("rows"))
    throw std::invalid_argument(
        "HF frozen test manifest disagrees with its receipt");

  std::vector<FinalMemberContext> member_contexts;
  std::set<std::string> seen_members;
  std::istringstream lines(test_bytes);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty() || line.size() > 1024U * 1024U)
      throw std::invalid_argument(
          "HF frozen test manifest row is empty or exceeds its bound");
    const nlohmann::json row =
        parse_bounded_authority_json(line, 10'000U, 32);
    const std::string member = row.value(id_column, std::string{});
    if (!row.is_object() || row.value("split", std::string{}) != "test" ||
        !bounded_identity(member) || !seen_members.insert(member).second)
      throw std::invalid_argument(
          "HF frozen test membership is invalid or duplicated");
    // The data descriptor commits the entire admitted root.  Binding its
    // digest and the member ID avoids asking controller C++ to reproduce a
    // Python processor merely to establish completion authority.
    const nlohmann::json context{
        {"api_version", "rwkv-lab.hf-final-member-context/v1"},
        {"components", component_digests},
        {"member_id", member},
    };
    member_contexts.push_back(
        {.member_id = member,
         .context_digest = "sha256:" + sha256_hex(context.dump())});
  }
  if (member_contexts.size() !=
      receipt.at("counts").at("test").get<std::uint64_t>())
    throw std::invalid_argument(
        "HF frozen test row count disagrees with its receipt");
  std::ranges::sort(member_contexts, {}, &FinalMemberContext::member_id);
  std::vector<std::string> required_members;
  required_members.reserve(member_contexts.size());
  std::ranges::transform(member_contexts,
                         std::back_inserter(required_members),
                         &FinalMemberContext::member_id);

  std::vector<FinalScalarRequirement> required_scalars;
  if (!invocation.observability.contains("metrics") ||
      !invocation.observability.at("metrics").is_array() ||
      !components.at("evaluator").contains("configuration") ||
      !components.at("evaluator").at("configuration").is_object() ||
      !components.at("evaluator").at("configuration").contains("metrics") ||
      !components.at("evaluator")
           .at("configuration")
           .at("metrics")
           .is_array())
    throw std::invalid_argument(
        "HF finalization metric composition is malformed");
  for (const nlohmann::json &declared :
       components.at("evaluator").at("configuration").at("metrics")) {
    if (!declared.is_string() || declared.get_ref<const std::string &>().empty())
      throw std::invalid_argument(
          "HF evaluator metric identity is malformed");
    std::string name = declared.get<std::string>();
    if (!name.starts_with("eval."))
      name = "eval." + name;
    const std::size_t matches = static_cast<std::size_t>(std::ranges::count_if(
        invocation.observability.at("metrics"),
        [&](const nlohmann::json &metric) {
          return metric.is_object() &&
                 metric.value("name", std::string{}) == name &&
                 metric.value("step_domain", std::string{}) ==
                     "optimizer_step";
        }));
    if (matches != 1U)
      throw std::invalid_argument(
          "HF evaluator metric lacks one exact observable scalar");
    required_scalars.push_back(
        {.metric_name = std::move(name),
         .step_domain = "optimizer_step"});
  }
  std::ranges::sort(required_scalars, {},
                    &FinalScalarRequirement::metric_name);
  return {
      .output_name = "test_eval",
      .required_output_names = std::move(required_output_names),
      .policy_digest = finalization_policy_digest(policy),
      .optimizer_step = terminal_step,
      .checkpoint_artifact_id = checkpoint_id,
      .checkpoint_fingerprint = checkpoint_fingerprint,
      .required_members = required_members,
      .member_contexts = std::move(member_contexts),
      .membership_digest = final_membership_digest(required_members),
      .membership_count = required_members.size(),
      .required_scalars = std::move(required_scalars),
      .terminal_optimizer_fingerprint = std::nullopt,
  };
}

std::vector<FinalEvaluationReceipt> resolve_final_evaluation_receipts(
    const WorkerInvocationSpec &invocation,
    const FinalEvaluationExpectation &expectation,
    const std::vector<Event> &durable_events) {
  constexpr std::string_view file_prefix = "file://";
  const std::filesystem::path run_root(
      invocation.workspace.value("run_directory", std::string{}));
  if (!run_root.is_absolute() || run_root.lexically_normal() != run_root)
    throw std::invalid_argument(
        "final evaluation run workspace is noncanonical");
  FinalDescriptor filesystem_root(
      ::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!filesystem_root)
    throw std::runtime_error("final evaluation cannot pin filesystem root");
  std::vector<FinalEvaluationReceipt> result;
  for (const Event &closure : durable_events) {
    if (closure.event_type != "artifact.published" ||
        closure.node_id != invocation.node_id ||
        closure.attempt_id != invocation.attempt_id ||
        closure.optimizer_step != expectation.optimizer_step ||
        closure.payload.value("schema", std::string{}) !=
            "rwkv-lab.final-evaluation.v1")
      continue;
    const std::string artifact_id =
        closure.payload.value("artifact_id", std::string{});
    const std::string fingerprint =
        closure.payload.value("fingerprint", std::string{});
    const std::string uri = closure.payload.value("uri", std::string{});
    const std::uint64_t size =
        closure.payload.value("size_bytes", std::uint64_t{});
    if (closure.worker_sequence == 0U ||
        closure.payload.value("logical_name", std::string{}) !=
            invocation.publishes.at("final_evaluation")
                .value("logical_name", std::string{}) ||
        closure.payload.value("kind", std::string{}) != "report" ||
        closure.payload.value("fingerprint_algorithm", std::string{}) !=
            "manifest_sha256" ||
        !closure.payload.value("complete", false) ||
        !bounded_identity(artifact_id) || !valid_digest(fingerprint) ||
        size == 0U || size > kMaximumFinalEvaluationManifestBytes ||
        !uri.starts_with(file_prefix) || uri.find('%') != std::string::npos)
      throw std::invalid_argument(
          "final evaluation artifact identity is invalid");
    const std::filesystem::path path(uri.substr(file_prefix.size()));
    const std::filesystem::path expected =
        run_root / "trainvm_artifacts" / "final_evaluation" / artifact_id /
        "manifest.json";
    if (!path.is_absolute() || path.lexically_normal() != path ||
        path != expected)
      throw std::invalid_argument(
          "final evaluation artifact escaped its immutable revision");
    const std::string bytes = final_file_bytes(
        filesystem_root.get(), path.lexically_relative("/").generic_string(),
        size);
    if (bytes.size() != size ||
        fingerprint != "sha256:" + sha256_hex(bytes))
      throw std::invalid_argument(
          "final evaluation artifact bytes disagree with durability");
    FinalEvaluationManifest manifest =
        decode_final_evaluation_manifest(bytes);

    std::set<std::string> declared_parents;
    const nlohmann::json parents =
        closure.payload.value("parent_artifact_ids", nlohmann::json::array());
    if (!parents.is_array() ||
        std::ranges::any_of(parents, [&](const nlohmann::json &parent) {
          return !parent.is_string() ||
                 !declared_parents.insert(parent.get<std::string>()).second;
        }))
      throw std::invalid_argument(
          "final evaluation closure parents are invalid");
    std::vector<FinalOutputReceipt> durable_outputs;
    for (const FinalOutputReceipt &claimed : manifest.output_receipts) {
      if (!declared_parents.contains(claimed.artifact_id))
        throw std::invalid_argument(
            "final evaluation output is not a durable closure parent");
      if (!invocation.publishes.contains(claimed.output_name) ||
          !invocation.publishes.at(claimed.output_name).is_object())
        throw std::invalid_argument(
            "final evaluation output is not declared by the invocation");
      const nlohmann::json *resume_checkpoint = nullptr;
      if (claimed.output_name == "checkpoint" &&
          invocation.resume.is_object() &&
          invocation.resume.value("api_version", std::string{}) ==
              "trainvm.resume-checkpoint/v1" &&
          invocation.resume.contains("checkpoint") &&
          invocation.resume.at("checkpoint").is_object() &&
          invocation.resume.at("checkpoint")
                  .value("artifact_id", std::string{}) ==
              claimed.artifact_id)
        resume_checkpoint = &invocation.resume.at("checkpoint");
      const std::string parent_attempt =
          resume_checkpoint == nullptr
              ? invocation.attempt_id
              : resume_checkpoint->value("producer_attempt_id",
                                         std::string{});
      const auto matches_parent = [&](const Event &event) {
            return event.event_type == "artifact.published" &&
                   event.node_id == invocation.node_id &&
                   event.attempt_id == parent_attempt &&
                   event.payload.value("artifact_id", std::string{}) ==
                       claimed.artifact_id;
          };
      if (std::ranges::count_if(durable_events, matches_parent) != 1)
        throw std::invalid_argument(
            "final evaluation parent event identity is ambiguous");
      const auto parent = std::ranges::find_if(durable_events, matches_parent);
      if (parent == durable_events.end() ||
          parent->worker_sequence == 0U ||
          (resume_checkpoint == nullptr &&
           parent->worker_sequence >= closure.worker_sequence) ||
          !parent->payload.value("complete", false) ||
          parent->payload.value("fingerprint_algorithm", std::string{}) !=
              "manifest_sha256" ||
          parent->payload.value("logical_name", std::string{}) !=
              invocation.publishes.at(claimed.output_name)
                  .value("logical_name", std::string{}) ||
          parent->payload.value("fingerprint", std::string{}) !=
              claimed.artifact_fingerprint ||
          (resume_checkpoint != nullptr &&
           parent->payload != *resume_checkpoint) ||
          parent->optimizer_step != expectation.optimizer_step)
        throw std::invalid_argument(
            "final evaluation output receipt disagrees with durable parent");
      durable_outputs.push_back(claimed);
    }
    std::set<std::string> output_ids;
    for (const FinalOutputReceipt &output : manifest.output_receipts)
      output_ids.insert(output.artifact_id);
    if (output_ids != declared_parents)
      throw std::invalid_argument(
          "final evaluation closure has unclaimed durable parents");

    std::vector<FinalScalarObservation> scalar_observations;
    for (const FinalScalarRequirement &required :
         expectation.required_scalars) {
      const Event *selected = nullptr;
      for (const Event &event : durable_events) {
        if (event.event_type == "metric.sampled" &&
            event.node_id == invocation.node_id &&
            event.attempt_id == invocation.attempt_id &&
            event.optimizer_step == expectation.optimizer_step &&
            event.payload.value("name", std::string{}) ==
                required.metric_name &&
            event.payload.value("step_domain", std::string{}) ==
                required.step_domain &&
            event.worker_sequence < closure.worker_sequence &&
            (selected == nullptr ||
             event.worker_sequence > selected->worker_sequence))
          selected = &event;
      }
      if (selected == nullptr)
        throw std::invalid_argument(
            "final evaluation lacks an exact terminal scalar event");
      scalar_observations.push_back(
          {.metric_name = required.metric_name,
           .step_domain = required.step_domain,
           .optimizer_step = expectation.optimizer_step,
           .event_id = selected->event_id});
    }
    result.push_back(
        {.artifact_id = artifact_id,
         .artifact_fingerprint = fingerprint,
         .durable_sequence = closure.worker_sequence,
         .durable_output_receipts = std::move(durable_outputs),
         .durable_scalar_observations = std::move(scalar_observations),
         .manifest = std::move(manifest)});
  }
  std::ranges::sort(result, {}, &FinalEvaluationReceipt::durable_sequence);
  return result;
}

} // namespace trainvm
