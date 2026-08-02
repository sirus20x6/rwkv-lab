#include "trainvm/post_training_authority.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_refused(const std::optional<LifecycleAdmissionRefusal>& refusal,
                     const std::string& expected_code,
                     const std::string& message) {
  if (!refusal) throw std::runtime_error(message);
  if (refusal->code != expected_code) {
    throw std::runtime_error(message + " (refused for the wrong reason: " +
                             refusal->code + ")");
  }
}

// A supervised fine-tune bounded by optimizer steps: the one shape that can
// honestly claim to repeat exactly.
PostTrainingArm finetune() {
  return {
      .arm_id = "arm.rwkv-finetune-a",
      .kind = PostTrainingArmKind::supervised_finetune,
      .bounds = {{.kind = RunBoundKind::optimizer_steps, .magnitude = 10000U}},
      .claim = ReproducibilityClaim::exact,
      .seed = 7U,
      .verifier_identity = std::nullopt,
      .external_mutations = {},
      .claims_trajectory_preserving_resume = true,
  };
}

PostTrainingArm rlvr() {
  return {
      .arm_id = "arm.direct-rlvr-a",
      .kind = PostTrainingArmKind::direct_rlvr,
      .bounds = {{.kind = RunBoundKind::optimizer_steps, .magnitude = 2000U}},
      .claim = ReproducibilityClaim::seeded,
      .seed = 11U,
      .verifier_identity = "verifier.unit-tests@sha256-abc",
      .external_mutations = {{.target = "verifier.unit-tests",
                              .effect = "record_judgement",
                              .authorized = true,
                              .receipt_id = std::nullopt}},
      .claims_trajectory_preserving_resume = false,
  };
}

void an_honest_arm_is_admitted() {
  require(!admit_post_training_arm(finetune(), ResumeGrade::exact),
          "a step-bounded seeded fine-tune is admissible");
  require(!admit_post_training_arm(rlvr(), ResumeGrade::compatible),
          "an authorized RLVR arm with a bound verifier is admissible");
  // Without this, every refusal below could be firing for an unrelated reason.
}

// The card's first clause.
void a_wall_clock_arm_cannot_be_labelled_deterministic() {
  auto timed = finetune();
  timed.bounds = {{.kind = RunBoundKind::wall_clock_seconds, .magnitude = 3600U}};
  require_refused(admit_post_training_arm(timed, ResumeGrade::exact),
                  "post-training-claim-unsupported",
                  "a wall-clock-bounded arm claimed exact reproducibility");

  // A seed does not rescue it: the seed fixes the trajectory, not where the
  // timer happens to stop.
  timed.claim = ReproducibilityClaim::seeded;
  require_refused(admit_post_training_arm(timed, ResumeGrade::exact),
                  "post-training-claim-unsupported",
                  "a seed was allowed to launder a wall-clock endpoint");

  // Declared honestly, the same arm is fine. The authority refuses the label,
  // not the experiment.
  timed.claim = ReproducibilityClaim::none;
  require(!admit_post_training_arm(timed, ResumeGrade::exact),
          "an honestly-labelled wall-clock arm is admissible");
  require(supportable_reproducibility_claim(timed) == ReproducibilityClaim::none,
          "a wall-clock arm supports no reproducibility claim");

  // A step bound alongside a wall-clock bound is still wall-clock terminated:
  // whichever fires first ends the run.
  auto mixed = finetune();
  mixed.bounds.push_back(
      {.kind = RunBoundKind::wall_clock_seconds, .magnitude = 60U});
  require_refused(admit_post_training_arm(mixed, ResumeGrade::exact),
                  "post-training-claim-unsupported",
                  "a mixed step/wall-clock arm claimed exact reproducibility");

  // An external stop signal is the same problem wearing different clothes.
  auto signalled = finetune();
  signalled.bounds = {{.kind = RunBoundKind::external_signal, .magnitude = 0U}};
  require_refused(admit_post_training_arm(signalled, ResumeGrade::exact),
                  "post-training-claim-unsupported",
                  "an externally-stopped arm claimed exact reproducibility");

  // Under-claiming is always allowed — it misleads nobody.
  auto modest = finetune();
  modest.claim = ReproducibilityClaim::none;
  require(!admit_post_training_arm(modest, ResumeGrade::exact),
          "an arm that under-claims is admissible");
}

// Reaching outside the host costs the exact claim even when the endpoint is
// perfectly reproducible.
void an_arm_that_calls_outside_cannot_claim_exact() {
  auto exact_rlvr = rlvr();
  exact_rlvr.claim = ReproducibilityClaim::exact;
  require_refused(admit_post_training_arm(exact_rlvr, ResumeGrade::compatible),
                  "post-training-claim-unsupported",
                  "an RLVR arm claimed bit-exact reproducibility");
  require(supportable_reproducibility_claim(rlvr()) ==
              ReproducibilityClaim::seeded,
          "a step-bounded seeded RLVR arm supports the seeded claim");

  auto unseeded = finetune();
  unseeded.seed = std::nullopt;
  require_refused(admit_post_training_arm(unseeded, ResumeGrade::exact),
                  "post-training-claim-unsupported",
                  "an arm with no seed claimed exact reproducibility");
}

// The card's second clause.
void external_mutation_must_be_authorized_then_receipted() {
  auto unauthorized = rlvr();
  unauthorized.external_mutations[0].authorized = false;
  require_refused(admit_post_training_arm(unauthorized, ResumeGrade::compatible),
                  "post-training-mutation-unauthorized",
                  "an unauthorized external mutation was admitted");

  auto undeclared = rlvr();
  undeclared.external_mutations.clear();
  require_refused(admit_post_training_arm(undeclared, ResumeGrade::compatible),
                  "post-training-external-effect-undeclared",
                  "an RLVR arm declared no external effect at all");

  const auto arm = rlvr();
  // Completion: the same mutation, now carrying its receipt.
  std::vector<ExternalMutation> performed = arm.external_mutations;
  performed[0].receipt_id = "receipt.verifier-0001";
  require(!qualify_post_training_completion(arm, performed),
          "a receipted mutation qualifies");

  auto unreceipted = performed;
  unreceipted[0].receipt_id = std::nullopt;
  require_refused(qualify_post_training_completion(arm, unreceipted),
                  "post-training-mutation-unreceipted",
                  "a mutation with no receipt was qualified");

  auto empty_receipt = performed;
  empty_receipt[0].receipt_id = "";
  require_refused(qualify_post_training_completion(arm, empty_receipt),
                  "post-training-mutation-unreceipted",
                  "an empty receipt id was accepted as evidence");

  // The run touched something it never asked to touch.
  auto extra = performed;
  extra.push_back({.target = "model-registry.production",
                   .effect = "publish",
                   .authorized = true,
                   .receipt_id = "receipt.publish-0001"});
  require_refused(qualify_post_training_completion(arm, extra),
                  "post-training-mutation-undeclared",
                  "an undeclared external mutation was qualified");

  // Authorized and then never mentioned again. Silence is not evidence that
  // nothing happened.
  require_refused(qualify_post_training_completion(arm, {}),
                  "post-training-mutation-unaccounted",
                  "an authorized mutation was allowed to go unreported");
}

// An RLVR reward is only as identified as its verifier.
void the_verifier_is_part_of_the_objective() {
  auto anonymous = rlvr();
  anonymous.verifier_identity = std::nullopt;
  require_refused(admit_post_training_arm(anonymous, ResumeGrade::compatible),
                  "post-training-verifier-unbound",
                  "an RLVR arm ran against an unidentified verifier");

  auto blank = rlvr();
  blank.verifier_identity = "";
  require_refused(admit_post_training_arm(blank, ResumeGrade::compatible),
                  "post-training-verifier-unbound",
                  "an empty verifier identity was accepted");

  auto invented = finetune();
  invented.verifier_identity = "verifier.unit-tests@sha256-abc";
  require_refused(admit_post_training_arm(invented, ResumeGrade::exact),
                  "post-training-verifier-unexpected",
                  "a supervised fine-tune invented a verifier");
}

void an_arm_must_declare_what_ends_it() {
  auto unbounded = finetune();
  unbounded.bounds.clear();
  require_refused(admit_post_training_arm(unbounded, ResumeGrade::exact),
                  "post-training-arm-unbounded",
                  "an arm with no declared end was admitted");

  auto zero = finetune();
  zero.bounds[0].magnitude = 0U;
  require_refused(admit_post_training_arm(zero, ResumeGrade::exact),
                  "post-training-bound-empty",
                  "a bound of zero steps was accepted as a bound");

  auto repeated = finetune();
  repeated.bounds.push_back(repeated.bounds.front());
  require_refused(admit_post_training_arm(repeated, ResumeGrade::exact),
                  "post-training-bound-duplicated",
                  "two conflicting step bounds were accepted");

  auto anonymous = finetune();
  anonymous.arm_id.clear();
  require_refused(admit_post_training_arm(anonymous, ResumeGrade::exact),
                  "post-training-arm-unidentified",
                  "an arm with no identifier was admitted");
}

// Resume honesty rides on the adapter's grade, not on what the arm would
// prefer to tell operators.
void a_restart_only_adapter_cannot_promise_a_continued_trajectory() {
  // `compatible` belongs in this list, which is the whole point: it can resume
  // from a checkpoint, so it is tempting to present as continuing the run, but
  // resume_from_checkpoint_supported and resume_preserves_trajectory are
  // different questions and only `exact` answers the second one.
  for (const ResumeGrade grade :
       {ResumeGrade::none, ResumeGrade::restart_only,
        ResumeGrade::terminal_checkpoint, ResumeGrade::compatible}) {
    require_refused(
        admit_post_training_arm(finetune(), grade),
        "post-training-resume-claim-unsupported",
        "an adapter that cannot preserve a trajectory promised to preserve it");
  }
  require(!admit_post_training_arm(finetune(), ResumeGrade::exact),
          "an adapter that preserves trajectory may say so");

  // Not claiming it is always allowed.
  auto modest = finetune();
  modest.claims_trajectory_preserving_resume = false;
  require(!admit_post_training_arm(modest, ResumeGrade::restart_only),
          "an arm that makes no resume claim is admissible on any grade");
}

}  // namespace

int main() {
  try {
    an_honest_arm_is_admitted();
    a_wall_clock_arm_cannot_be_labelled_deterministic();
    an_arm_that_calls_outside_cannot_claim_exact();
    external_mutation_must_be_authorized_then_receipted();
    the_verifier_is_part_of_the_objective();
    an_arm_must_declare_what_ends_it();
    a_restart_only_adapter_cannot_promise_a_continued_trajectory();
    std::cout << "post-training authority tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "post-training authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
