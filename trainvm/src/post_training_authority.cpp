#include "trainvm/post_training_authority.hpp"

#include <algorithm>
#include <set>

namespace trainvm {
namespace {

LifecycleAdmissionRefusal refuse(std::string code, std::string message) {
  return LifecycleAdmissionRefusal{.code = std::move(code),
                                   .message = std::move(message)};
}

// Local to the refusal messages. Effect has no shared name function, and
// adding one to the model surface for a diagnostic string would widen it for
// every consumer to serve this one.
std::string_view effect_name(Effect effect) {
  switch (effect) {
    case Effect::read_only:
      return "read_only";
    case Effect::workspace_write:
      return "workspace_write";
    case Effect::process:
      return "process";
    case Effect::resource:
      return "resource";
    case Effect::external:
      return "external";
  }
  return "unknown";
}

// Whether this arm reaches outside the host is the registry's answer. Reading
// it off the arm's kind instead would let a mislabelled arm decide its own
// admission, and would miss an adapter that is declared external for a reason
// the kind vocabulary does not name.
bool arm_calls_outside_this_host(Effect declared_effect) {
  // An external verifier or service is not ours to replay. Even with a fixed
  // seed, the arm's result depends on a response we neither produced nor
  // recorded the inputs of.
  return declared_effect == Effect::external;
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

std::optional<PostTrainingArmKind> post_training_arm_kind_from_name(
    std::string_view name) {
  for (const PostTrainingArmKind kind :
       {PostTrainingArmKind::supervised_finetune,
        PostTrainingArmKind::recursive_post_training,
        PostTrainingArmKind::external_ltx, PostTrainingArmKind::direct_rlvr}) {
    if (post_training_arm_kind_name(kind) == name) return kind;
  }
  return std::nullopt;
}

std::optional<RunBoundKind> run_bound_kind_from_name(std::string_view name) {
  for (const RunBoundKind kind :
       {RunBoundKind::optimizer_steps, RunBoundKind::training_tokens,
        RunBoundKind::wall_clock_seconds, RunBoundKind::external_signal}) {
    if (run_bound_kind_name(kind) == name) return kind;
  }
  return std::nullopt;
}

std::optional<ReproducibilityClaim> reproducibility_claim_from_name(
    std::string_view name) {
  for (const ReproducibilityClaim claim :
       {ReproducibilityClaim::none, ReproducibilityClaim::seeded,
        ReproducibilityClaim::exact}) {
    if (reproducibility_claim_name(claim) == name) return claim;
  }
  return std::nullopt;
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
    const PostTrainingArm& arm, Effect declared_effect) {
  // A run with no declared end has no endpoint to reproduce.
  if (arm.bounds.empty()) return ReproducibilityClaim::none;
  const bool endpoint_repeats = std::ranges::all_of(
      arm.bounds,
      [](const RunBound& bound) { return run_bound_is_reproducible(bound.kind); });
  if (!endpoint_repeats) return ReproducibilityClaim::none;
  if (!arm.seed) return ReproducibilityClaim::none;
  if (arm_calls_outside_this_host(declared_effect) ||
      !arm.external_mutations.empty())
    return ReproducibilityClaim::seeded;
  return ReproducibilityClaim::exact;
}

PostTrainingArmLowering lower_post_training_arm(
    const PostTrainingArmDeclaration& declared) {
  PostTrainingArmLowering lowering;
  lowering.arm.arm_id = declared.arm_id;
  lowering.arm.seed = declared.seed;
  lowering.arm.verifier_identity = declared.verifier_identity;
  lowering.arm.claims_trajectory_preserving_resume =
      declared.claims_trajectory_preserving_resume.value_or(false);

  if (const auto kind = post_training_arm_kind_from_name(declared.kind))
    lowering.arm.kind = *kind;
  else
    lowering.unknown_kind = declared.kind;

  if (const auto claim =
          reproducibility_claim_from_name(declared.reproducibility_claim))
    lowering.arm.claim = *claim;
  else
    lowering.unknown_claim = declared.reproducibility_claim;

  for (std::size_t index = 0U; index < declared.bounds.size(); ++index) {
    const PostTrainingBoundDeclaration& bound = declared.bounds[index];
    if (const auto kind = run_bound_kind_from_name(bound.kind)) {
      lowering.arm.bounds.push_back(
          {.kind = *kind, .magnitude = bound.magnitude.value_or(0U)});
    } else {
      lowering.unknown_bounds.emplace_back(index, bound.kind);
    }
  }

  for (const PostTrainingMutationDeclaration& mutation :
       declared.external_mutations.value_or(
           std::vector<PostTrainingMutationDeclaration>{})) {
    lowering.arm.external_mutations.push_back({.target = mutation.target,
                                               .effect = mutation.effect,
                                               .authorized = mutation.authorized,
                                               .receipt_id = std::nullopt});
  }
  return lowering;
}

nlohmann::json post_training_arm_json(const PostTrainingArm& arm) {
  nlohmann::json bounds = nlohmann::json::array();
  for (const RunBound& bound : arm.bounds) {
    bounds.push_back({{"kind", std::string(run_bound_kind_name(bound.kind))},
                      {"magnitude", bound.magnitude}});
  }
  nlohmann::json mutations = nlohmann::json::array();
  for (const ExternalMutation& mutation : arm.external_mutations) {
    mutations.push_back({{"target", mutation.target},
                         {"effect", mutation.effect},
                         {"authorized", mutation.authorized}});
  }
  nlohmann::json result{
      {"api_version", "trainvm.post-training-arm/v1"},
      {"arm_id", arm.arm_id},
      {"kind", std::string(post_training_arm_kind_name(arm.kind))},
      {"bounds", std::move(bounds)},
      {"reproducibility_claim",
       std::string(reproducibility_claim_name(arm.claim))},
      {"claims_trajectory_preserving_resume",
       arm.claims_trajectory_preserving_resume},
  };
  // Omitted rather than emitted as null, so the worker sees the same shape a
  // document without them would produce.
  if (arm.seed) result["seed"] = *arm.seed;
  if (arm.verifier_identity) result["verifier_identity"] = *arm.verifier_identity;
  if (!arm.external_mutations.empty()) result["external_mutations"] = mutations;
  return result;
}

std::optional<LifecycleAdmissionRefusal>
validate_post_training_arm_declaration(const PostTrainingArm& arm) {
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
  // Effect::process is the profile-independent ceiling: a profile can only
  // LOWER what an arm may claim (an external adapter caps it at seeded),
  // never raise it. So refusing a claim above this ceiling can never refuse
  // something the launch gate would have allowed.
  const ReproducibilityClaim supportable =
      supportable_reproducibility_claim(arm, Effect::process);
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

  return std::nullopt;
}

std::optional<LifecycleAdmissionRefusal> admit_post_training_arm(
    const PostTrainingArm& arm, ResumeGrade resume_grade,
    Effect declared_effect) {
  // Re-run the profile-independent rules rather than trusting that a document
  // was compiled first. Admission must not depend on the compile path having
  // run, and the two must not be able to drift apart.
  if (auto refusal = validate_post_training_arm_declaration(arm)) return refusal;

  // Checked before the reproducibility label, because an arm acting outside
  // what its adapter is admitted for is the more fundamental problem: telling
  // the operator it mislabelled its determinism would bury that.
  //
  // The adapter is declared external, so something out there changes when this
  // runs. An arm that names nothing leaves the authorization gate below with
  // nothing to check, which is indistinguishable from having no gate.
  if (arm_calls_outside_this_host(declared_effect) &&
      arm.external_mutations.empty())
    return refuse(
        "post-training-external-effect-undeclared",
        std::string("a ") + std::string(post_training_arm_kind_name(arm.kind)) +
            " arm on an adapter declared external must say what it changes");

  // The mirror, and the reason to read the effect off the registry rather than
  // the arm: an adapter admitted only to write its workspace may not reach
  // outside it, whatever the arm calls itself. Admitting this would let an arm
  // widen its own adapter's effect by declaring a mutation.
  if (!arm_calls_outside_this_host(declared_effect) &&
      !arm.external_mutations.empty())
    return refuse(
        "post-training-external-effect-unadmitted",
        std::string("the arm declares an external mutation, but its adapter is "
                    "declared ") +
            std::string(effect_name(declared_effect)) +
            " and is not admitted to act outside this host");

  // Now with the adapter's own effect, which can only tighten the ceiling the
  // declaration check already applied.
  const ReproducibilityClaim admitted =
      supportable_reproducibility_claim(arm, declared_effect);
  if (arm.claim > admitted) {
    return refuse(
        "post-training-claim-unsupported",
        std::string("the arm claims reproducibility '") +
            std::string(reproducibility_claim_name(arm.claim)) +
            "' but its adapter supports at most '" +
            std::string(reproducibility_claim_name(admitted)) + "'");
  }

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

std::optional<LifecycleAdmissionRefusal> admit_post_training_arm(
    const PostTrainingArm& arm, const AdapterProfile& profile) {
  return admit_post_training_arm(arm, profile.lifecycle.resume_grade,
                                 profile.effect);
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
