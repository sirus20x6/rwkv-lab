#include "trainvm/post_training_authority.hpp"

#include <algorithm>
#include <set>

namespace trainvm {
namespace {

LifecycleAdmissionRefusal refuse(std::string code, std::string message) {
  return LifecycleAdmissionRefusal{.code = std::move(code),
                                   .message = std::move(message)};
}

bool arm_calls_outside_this_host(PostTrainingArmKind kind) {
  // An external verifier or service is not ours to replay. Even with a fixed
  // seed, the arm's result depends on a response we neither produced nor
  // recorded the inputs of.
  return kind == PostTrainingArmKind::direct_rlvr ||
         kind == PostTrainingArmKind::external_ltx;
}

}  // namespace

std::string_view post_training_arm_kind_name(PostTrainingArmKind kind) {
  switch (kind) {
    case PostTrainingArmKind::supervised_finetune:
      return "supervised_finetune";
    case PostTrainingArmKind::recursive_post_training:
      return "recursive_post_training";
    case PostTrainingArmKind::external_ltx:
      return "external_ltx";
    case PostTrainingArmKind::direct_rlvr:
      return "direct_rlvr";
  }
  return "unknown";
}

std::string_view run_bound_kind_name(RunBoundKind kind) {
  switch (kind) {
    case RunBoundKind::optimizer_steps:
      return "optimizer_steps";
    case RunBoundKind::training_tokens:
      return "training_tokens";
    case RunBoundKind::wall_clock_seconds:
      return "wall_clock_seconds";
    case RunBoundKind::external_signal:
      return "external_signal";
  }
  return "unknown";
}

std::string_view reproducibility_claim_name(ReproducibilityClaim claim) {
  switch (claim) {
    case ReproducibilityClaim::none:
      return "none";
    case ReproducibilityClaim::seeded:
      return "seeded";
    case ReproducibilityClaim::exact:
      return "exact";
  }
  return "unknown";
}

bool run_bound_is_reproducible(RunBoundKind kind) {
  switch (kind) {
    case RunBoundKind::optimizer_steps:
    case RunBoundKind::training_tokens:
      return true;
    case RunBoundKind::wall_clock_seconds:
    case RunBoundKind::external_signal:
      return false;
  }
  return false;
}

ReproducibilityClaim supportable_reproducibility_claim(
    const PostTrainingArm& arm) {
  // A run with no declared end has no endpoint to reproduce.
  if (arm.bounds.empty()) return ReproducibilityClaim::none;
  const bool endpoint_repeats = std::ranges::all_of(
      arm.bounds,
      [](const RunBound& bound) { return run_bound_is_reproducible(bound.kind); });
  if (!endpoint_repeats) return ReproducibilityClaim::none;
  if (!arm.seed) return ReproducibilityClaim::none;
  if (arm_calls_outside_this_host(arm.kind) || !arm.external_mutations.empty())
    return ReproducibilityClaim::seeded;
  return ReproducibilityClaim::exact;
}

std::optional<LifecycleAdmissionRefusal> admit_post_training_arm(
    const PostTrainingArm& arm, ResumeGrade resume_grade) {
  if (arm.arm_id.empty())
    return refuse("post-training-arm-unidentified",
                  "a post-training arm must carry an identifier");

  if (arm.bounds.empty())
    return refuse(
        "post-training-arm-unbounded",
        "a post-training arm must declare what ends it: an unbounded arm "
        "cannot be budgeted, compared, or resumed honestly");

  std::set<RunBoundKind> declared;
  for (const RunBound& bound : arm.bounds) {
    if (!declared.insert(bound.kind).second)
      return refuse("post-training-bound-duplicated",
                    std::string("the arm declares the bound ") +
                        std::string(run_bound_kind_name(bound.kind)) +
                        " more than once");
    // An external signal has no magnitude; every other bound is a quantity,
    // and zero of it is not a bound.
    if (bound.kind != RunBoundKind::external_signal && bound.magnitude == 0U)
      return refuse("post-training-bound-empty",
                    std::string("the bound ") +
                        std::string(run_bound_kind_name(bound.kind)) +
                        " must declare a magnitude");
  }

  // The card's first clause. A wall-clock arm stops wherever the machine
  // happened to be, so no seed makes its endpoint repeat, and presenting it as
  // deterministic invites a comparison against a seeded arm that is not valid.
  // ReproducibilityClaim is declared weakest-first, so a claim stronger than
  // what the declarations support compares greater. Claiming something weaker
  // than supported is always allowed: under-claiming misleads nobody.
  const ReproducibilityClaim supportable = supportable_reproducibility_claim(arm);
  if (arm.claim > supportable) {
    return refuse(
        "post-training-claim-unsupported",
        std::string("the arm claims reproducibility '") +
            std::string(reproducibility_claim_name(arm.claim)) +
            "' but its declarations support at most '" +
            std::string(reproducibility_claim_name(supportable)) + "'");
  }

  // A reward model is half the objective. Two RLVR runs against different
  // verifiers optimize different things, so an unbound verifier makes the
  // arm's result uninterpretable rather than merely unreproducible.
  if (arm.kind == PostTrainingArmKind::direct_rlvr &&
      (!arm.verifier_identity || arm.verifier_identity->empty()))
    return refuse("post-training-verifier-unbound",
                  "an RLVR arm must bind its verifier identity: the verifier "
                  "is part of the objective, not part of the environment");

  // A verifier named by an arm that has none is provenance nobody can check.
  if (arm.verifier_identity && arm.kind != PostTrainingArmKind::direct_rlvr)
    return refuse("post-training-verifier-unexpected",
                  std::string("a ") +
                      std::string(post_training_arm_kind_name(arm.kind)) +
                      " arm has no verifier and may not name one");

  std::set<std::string> mutation_targets;
  for (const ExternalMutation& mutation : arm.external_mutations) {
    if (mutation.target.empty() || mutation.effect.empty())
      return refuse("post-training-mutation-undescribed",
                    "an external mutation must name its target and effect");
    if (!mutation_targets.insert(mutation.target + "\x1f" + mutation.effect)
             .second)
      return refuse("post-training-mutation-duplicated",
                    "the arm declares the same external mutation twice");
    // The card's second clause. Authorization is granted before the run, not
    // inferred afterwards from the fact that the run completed.
    if (!mutation.authorized)
      return refuse(
          "post-training-mutation-unauthorized",
          "the arm would change '" + mutation.target +
              "' outside this host without authorization");
  }

  // An arm that mutates nothing outside the host but is of a kind defined by
  // doing so has under-declared, and the authorization gate above would then
  // have nothing to check.
  if (arm_calls_outside_this_host(arm.kind) && arm.external_mutations.empty())
    return refuse(
        "post-training-external-effect-undeclared",
        std::string("a ") + std::string(post_training_arm_kind_name(arm.kind)) +
            " arm acts outside this host and must declare what it changes");

  // Resume honesty: the adapter's grade decides whether a resumed run is the
  // same trajectory. An arm may not tell operators otherwise.
  if (arm.claims_trajectory_preserving_resume &&
      !resume_preserves_trajectory(resume_grade))
    return refuse(
        "post-training-resume-claim-unsupported",
        "the arm claims a trajectory-preserving resume, but its adapter's "
        "resume grade only supports restarting");

  return std::nullopt;
}

std::optional<LifecycleAdmissionRefusal> qualify_post_training_completion(
    const PostTrainingArm& arm,
    const std::vector<ExternalMutation>& performed_mutations) {
  for (const ExternalMutation& performed : performed_mutations) {
    const auto declared = std::ranges::find_if(
        arm.external_mutations, [&performed](const ExternalMutation& candidate) {
          return candidate.target == performed.target &&
                 candidate.effect == performed.effect;
        });
    // Something outside this host changed that the arm never asked to change.
    if (declared == arm.external_mutations.end())
      return refuse("post-training-mutation-undeclared",
                    "the run changed '" + performed.target +
                        "' outside this host without ever declaring it");
    if (!performed.receipt_id || performed.receipt_id->empty())
      return refuse("post-training-mutation-unreceipted",
                    "the run changed '" + performed.target +
                        "' outside this host without a receipt");
  }

  // Every authorized mutation must be accounted for, either as performed or
  // by the run reporting it did not happen. Silence is not evidence.
  for (const ExternalMutation& declared : arm.external_mutations) {
    const bool reported = std::ranges::any_of(
        performed_mutations, [&declared](const ExternalMutation& candidate) {
          return candidate.target == declared.target &&
                 candidate.effect == declared.effect;
        });
    if (!reported)
      return refuse("post-training-mutation-unaccounted",
                    "the arm was authorized to change '" + declared.target +
                        "' and the run never reported whether it did");
  }
  return std::nullopt;
}

}  // namespace trainvm
