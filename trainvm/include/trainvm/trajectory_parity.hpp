#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trainvm {

// Why this is not a boolean.
//
// A fused optimizer kernel and a foreach reference agree to float32 epsilon on
// a single step and then diverge, because the training trajectory amplifies
// that agreement-level difference. Measured on an 8x2048 linear stack, batch
// 64, torch fused AdamW versus the foreach reference:
//
//     after  1 step   max|delta| 3.7e-09   relative 1.6e-07
//     after  2 steps  max|delta| 4.0e-08   relative 1.7e-06
//     after  5 steps  max|delta| 3.4e-05   relative 1.3e-03
//     after 25 steps  max|delta| 4.9e-03   relative 1.4e-01
//
// That is not a bug and no kernel change removes it: any implementation that
// is not bit-identical does the same. Read as bit-identity, a trajectory gate
// can never pass, so every fused candidate is rejected for a reason unrelated
// to its quality. Read loosely, a candidate that genuinely damages training
// passes because nobody said what "same trajectory" means.
//
// So the gate carries three states rather than two, and each verdict is
// derived from the statistics recorded beside it rather than asserted:
//
//   equivalent                 the trajectories are bit-identical. This is
//                              exactly what the old boolean `true` meant, and
//                              it is not widened by anything below.
//   diverged_within_tolerance  the trajectories differ, and the difference is
//                              bounded by a stated, testable criterion.
//   diverged                   not shown to be either.
enum class TrajectoryParityVerdict {
  equivalent,
  diverged_within_tolerance,
  diverged,
};

// How `diverged_within_tolerance` was established.
//
// divergence_rate is primary. The measurements above are near-perfectly
// geometric in the pre-saturation regime -- the first three points fit
// ln(deviation) = a + lambda*step with r^2 = 1.000 -- so `lambda` is a real
// per-step divergence rate. The reference is run against itself under a
// different seed and its own rate estimated identically. A candidate that
// diverges no faster than the reference diverges from itself is equivalent in
// the only sense that matters, because the seed is already free.
//
// checkpoint_quality is the fallback for when a rate cannot be estimated --
// the trajectory saturated, too few samples, a deviation of exactly zero part
// way through. It compares the declared eval metric at a checkpoint over
// paired seeds and requires the interval on the paired delta to lie inside the
// effect class's tolerance. That is an equivalence claim, not a "we failed to
// find a difference" claim.
//
// bit_identical is the degenerate case and the only route to `equivalent`.
enum class TrajectoryEquivalenceCriterion {
  bit_identical,
  divergence_rate,
  checkpoint_quality,
};

// Tolerance is stated per effect class rather than globally, because the
// classes do not have the same right to drift. These are the trajectory-
// bearing rows of the qualification contract's component table.
//
//   optimizer_update   a reassociated update of the same arithmetic. It may
//                      not amplify faster than seed noise, plus estimation
//                      headroom.
//   gradient_policy    accumulation/clipping/synchronization reassociation:
//                      summation order changes, the summands do not.
//   precision_scaling  a dtype or scaler change has a genuinely larger
//                      per-step floor, so it is allowed to amplify faster --
//                      and is the class most likely to need the quality gate.
//   schedule_state     a schedule value is computed from the step index, not
//                      accumulated. Any divergence is a defect, so the
//                      tolerance is zero and only bit-identity passes.
//   curriculum_phase   sample membership and order are discrete. "Within
//                      tolerance" is not a meaningful statement about them.
enum class TrajectoryEffectClass {
  optimizer_update,
  gradient_policy,
  precision_scaling,
  schedule_state,
  curriculum_phase,
};

struct TrajectoryEffectTolerance {
  // Ceiling on candidate_rate / reference_rate. Zero forbids the rate route.
  double maximum_divergence_rate_ratio{};
  // Ceiling on |relative bound of the paired-delta interval|. Zero forbids the
  // checkpoint-quality route.
  double maximum_quality_deviation{};

  bool operator==(const TrajectoryEffectTolerance&) const = default;
};

[[nodiscard]] TrajectoryEffectTolerance
trajectory_effect_tolerance(TrajectoryEffectClass effect_class);

struct TrajectoryDivergenceSample {
  std::uint64_t step{};
  double relative_deviation{};

  bool operator==(const TrajectoryDivergenceSample&) const = default;
};

struct TrajectoryQualityObservation {
  std::uint64_t seed{};
  double baseline_metric{};
  double candidate_metric{};

  bool operator==(const TrajectoryQualityObservation&) const = default;
};

// The evidence a candidate presents. `verdict` is declared by the producer and
// must equal the verdict the recorded statistics support; a document whose
// label disagrees with its own numbers is malformed, not rejected.
struct TrajectoryParityEvidence {
  TrajectoryParityVerdict verdict{TrajectoryParityVerdict::diverged};
  TrajectoryEquivalenceCriterion criterion{
      TrajectoryEquivalenceCriterion::divergence_rate};
  TrajectoryEffectClass effect_class{TrajectoryEffectClass::optimizer_update};
  // Candidate against the reference, and the reference against itself under a
  // different seed. Both are relative deviations of the declared summary
  // statistic, sampled at strictly increasing steps.
  std::vector<TrajectoryDivergenceSample> candidate_divergence;
  std::vector<TrajectoryDivergenceSample> reference_divergence;
  // Paired per-seed eval metric at the declared checkpoint.
  std::vector<TrajectoryQualityObservation> checkpoint_quality;
  // Fixes the bootstrap and sign-flip draws so a receipt is reproducible.
  std::uint64_t analysis_seed{};

  bool operator==(const TrajectoryParityEvidence&) const = default;
};

// Everything the gate derived, so a later reader can tell why a candidate was
// accepted rather than only that it was.
struct TrajectoryParityAssessment {
  TrajectoryParityVerdict verdict{TrajectoryParityVerdict::diverged};
  TrajectoryEquivalenceCriterion criterion{
      TrajectoryEquivalenceCriterion::bit_identical};
  TrajectoryEffectClass effect_class{TrajectoryEffectClass::optimizer_update};
  TrajectoryEffectTolerance tolerance;
  double candidate_divergence_rate{};
  double reference_divergence_rate{};
  double divergence_rate_ratio{};
  double candidate_fit_quality{};
  double reference_fit_quality{};
  double quality_relative_delta{};
  double quality_relative_interval_low{};
  double quality_relative_interval_high{};
  double quality_p_value{};
  std::uint64_t quality_seed_count{};

  bool operator==(const TrajectoryParityAssessment&) const = default;
};

class TrajectoryParityError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// The log-linear fit the rate criterion is built on. Returns the slope per
// step and the coefficient of determination of the fit.
struct TrajectoryDivergenceFit {
  double rate{};
  double fit_quality{};

  bool operator==(const TrajectoryDivergenceFit&) const = default;
};

[[nodiscard]] TrajectoryDivergenceFit
fit_trajectory_divergence(const std::vector<TrajectoryDivergenceSample>& samples);

// Derives the verdict from the evidence. Deterministic: the same evidence
// always produces the same assessment, including its bootstrap interval.
[[nodiscard]] TrajectoryParityAssessment
assess_trajectory_parity(const TrajectoryParityEvidence& evidence);

[[nodiscard]] std::string
trajectory_parity_verdict_name(TrajectoryParityVerdict verdict);

// A log-linear fit below this is not a geometric divergence, so the rate the
// criterion would compare is not a rate. Refuse rather than compare it.
inline constexpr double kMinimumDivergenceFitQuality = 0.90;
// Below three points a coefficient of determination is not informative: two
// points fit a line exactly whatever they are.
inline constexpr std::size_t kMinimumDivergenceSamples = 3U;
// Fewer paired seeds than this cannot support an interval narrow enough to
// mean anything, and the permutation test has 2^n arrangements to draw from.
inline constexpr std::size_t kMinimumQualitySeeds = 5U;

}  // namespace trainvm
