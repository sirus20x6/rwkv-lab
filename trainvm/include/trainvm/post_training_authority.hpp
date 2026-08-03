#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/lifecycle_admission.hpp"
#include "trainvm/model.hpp"

namespace trainvm {

// Post-training arms differ from pretraining in two ways that the schema has
// so far left implicit, and both of them are ways a run can lie about itself.
//
// The first is stopping. A supervised fine-tune bounded by optimizer steps
// stops at the same place every time; an arm bounded by wall clock stops
// wherever the machine happened to be when the timer expired. Those two arms
// are not the same experiment with different budgets — the second has no
// reproducible endpoint at all, and labelling it deterministic invites a
// comparison that cannot be made.
//
// The second is external effect. RLVR calls a verifier and LTX calls a service
// outside this host. Those calls change state nobody here owns, so they must be
// authorized before the run and receipted after it, rather than inferred from
// the fact that a run completed.
enum class PostTrainingArmKind {
  supervised_finetune,
  recursive_post_training,
  external_ltx,
  direct_rlvr,
};

// What ends the run. The distinction that matters is not the unit but whether
// two runs of the same arm, from the same inputs and seed, stop in the same
// place.
enum class RunBoundKind {
  optimizer_steps,
  training_tokens,
  wall_clock_seconds,
  external_signal,
};

struct RunBound final {
  RunBoundKind kind{};
  std::uint64_t magnitude{};

  bool operator==(const RunBound&) const = default;
};

// What the arm is allowed to claim about repeating itself.
//
//   none   — repeating it may land somewhere else, and that is declared.
//   seeded — same seed, same inputs, same endpoint; scheduling may still
//            reorder reductions, so results are close rather than identical.
//   exact  — bit-identical on a repeat.
//
// These are claims, not measurements. The authority's job is to refuse a claim
// the arm's own declarations contradict, which is the only part of this that
// can be checked without running anything.
enum class ReproducibilityClaim {
  none,
  seeded,
  exact,
};

// A change this arm makes outside the host: a verifier that records a
// judgement, a service that stores a generation. Authorization is granted
// before the run; the receipt is evidence it happened as authorized.
struct ExternalMutation final {
  std::string target;
  std::string effect;
  bool authorized{};
  std::optional<std::string> receipt_id;

  bool operator==(const ExternalMutation&) const = default;
};

struct PostTrainingArm final {
  std::string arm_id;
  PostTrainingArmKind kind{};
  std::vector<RunBound> bounds;
  ReproducibilityClaim claim{};
  std::optional<std::uint64_t> seed;
  // Bound by digest, because an RLVR reward depends on the verifier as much as
  // on the policy: a changed verifier is a changed objective.
  std::optional<std::string> verifier_identity;
  std::vector<ExternalMutation> external_mutations;
  // What the arm tells operators about resuming. Refused when the adapter's
  // resume grade cannot support it.
  bool claims_trajectory_preserving_resume{};

  bool operator==(const PostTrainingArm&) const = default;
};

// True when repeating this bound lands in the same place. Wall-clock and
// external-signal bounds do not.
[[nodiscard]] bool run_bound_is_reproducible(RunBoundKind kind);

// Returns a refusal when the arm's declarations contradict either each other
// or the adapter's own. Reuses the lifecycle refusal shape so the dashboard
// renders these the same way it already renders control refusals.
//
// declared_effect is the adapter profile's Effect. Whether an arm reaches
// outside this host is the registry's answer, not something inferred from the
// arm's kind: an adapter admitted only for workspace_write may not perform
// external mutation whatever it calls itself, and an adapter declared external
// must say what it changes.
[[nodiscard]] std::optional<LifecycleAdmissionRefusal> admit_post_training_arm(
    const PostTrainingArm& arm, ResumeGrade resume_grade, Effect declared_effect);

// The form callers should use. Both of the values above belong to the resolved
// adapter profile, so passing them separately invites a caller to supply the
// pair it wants rather than the pair the registry holds — which is the failure
// this authority exists to refuse. The overload above stays for tests that
// need to sweep grades and effects the registry does not currently contain.
[[nodiscard]] std::optional<LifecycleAdmissionRefusal> admit_post_training_arm(
    const PostTrainingArm& arm, const AdapterProfile& profile);

// The subset of the rules that depend on the arm alone. Document compilation
// has no adapter registry — compile_document takes only JSON, because the
// registry is authority-owned and applied later — so this is what can be
// enforced while an author is still writing.
//
// It is deliberately a STRICT SUBSET, not a cheaper version: an arm that
// passes here can still be refused at launch by the effect and resume-grade
// rules, which need the profile. Callers must not present a compile-time pass
// as admission. Every rule here is also re-run by admit_post_training_arm, so
// the compile diagnostic and the launch gate cannot disagree about the rules
// they share.
[[nodiscard]] std::optional<LifecycleAdmissionRefusal>
validate_post_training_arm_declaration(const PostTrainingArm& arm);

// The strongest claim this arm's own declarations can support. Callers that
// want a label rather than a refusal should ask for this and use it, instead
// of proposing a claim and hoping.
[[nodiscard]] ReproducibilityClaim supportable_reproducibility_claim(
    const PostTrainingArm& arm, Effect declared_effect);

// Checked when the run finishes. Admission proves every external mutation was
// authorized; this proves each one was receipted, and that the run did not
// perform one it never declared.
[[nodiscard]] std::optional<LifecycleAdmissionRefusal>
qualify_post_training_completion(
    const PostTrainingArm& arm,
    const std::vector<ExternalMutation>& performed_mutations);

[[nodiscard]] std::string_view post_training_arm_kind_name(
    PostTrainingArmKind kind);

// Name-to-value for the document surface. Returning nullopt rather than a
// default is the point: an unknown name must produce a diagnostic that repeats
// it back, not decode to something the author did not write.
[[nodiscard]] std::optional<PostTrainingArmKind> post_training_arm_kind_from_name(
    std::string_view name);
[[nodiscard]] std::optional<RunBoundKind> run_bound_kind_from_name(
    std::string_view name);
[[nodiscard]] std::optional<ReproducibilityClaim> reproducibility_claim_from_name(
    std::string_view name);
[[nodiscard]] std::string_view run_bound_kind_name(RunBoundKind kind);
[[nodiscard]] std::string_view reproducibility_claim_name(
    ReproducibilityClaim claim);

}  // namespace trainvm
