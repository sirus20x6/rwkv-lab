#include "trainvm/trajectory_parity.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using trainvm::assess_trajectory_parity;
using trainvm::fit_trajectory_divergence;
using trainvm::TrajectoryDivergenceSample;
using trainvm::TrajectoryEffectClass;
using trainvm::TrajectoryEquivalenceCriterion;
using trainvm::TrajectoryParityError;
using trainvm::TrajectoryParityEvidence;
using trainvm::TrajectoryParityVerdict;
using trainvm::TrajectoryQualityObservation;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance,
          const std::string& message) {
  require(std::abs(actual - expected) <= tolerance,
          message + ": expected " + std::to_string(expected) + ", got " +
              std::to_string(actual));
}

template <typename Callable>
void rejected(Callable&& callable, std::string_view expected,
              const std::string& message) {
  try {
    callable();
  } catch (const TrajectoryParityError& error) {
    require(std::string_view(error.what()).find(expected) !=
                std::string_view::npos,
            message + ": wrong error: " + error.what());
    return;
  }
  throw std::runtime_error(message + ": input was accepted");
}

// The measurement this gate exists for: torch fused AdamW against the foreach
// reference, 8x2048 linear stack, batch 64. Relative deviation of the
// parameter vector after 1, 2, 5 and 25 steps.
std::vector<TrajectoryDivergenceSample> measured_fused_adamw() {
  return {{.step = 1U, .relative_deviation = 1.6e-07},
          {.step = 2U, .relative_deviation = 1.7e-06},
          {.step = 5U, .relative_deviation = 1.3e-03},
          {.step = 25U, .relative_deviation = 1.4e-01}};
}

std::vector<TrajectoryDivergenceSample> measured_fused_adamw_unsaturated() {
  std::vector<TrajectoryDivergenceSample> samples = measured_fused_adamw();
  samples.pop_back();
  return samples;
}

TrajectoryParityEvidence rate_evidence(
    std::vector<TrajectoryDivergenceSample> candidate,
    std::vector<TrajectoryDivergenceSample> reference,
    TrajectoryParityVerdict verdict,
    TrajectoryEffectClass effect_class =
        TrajectoryEffectClass::optimizer_update) {
  return {
      .verdict = verdict,
      .criterion = TrajectoryEquivalenceCriterion::divergence_rate,
      .effect_class = effect_class,
      .candidate_divergence = std::move(candidate),
      .reference_divergence = std::move(reference),
      .checkpoint_quality = {},
      .analysis_seed = 7U,
  };
}

// A geometric series with a declared per-step rate, which is what a reference
// run against itself under a different seed produces.
std::vector<TrajectoryDivergenceSample> geometric(double first, double rate,
                                                  std::vector<std::uint64_t>
                                                      steps) {
  std::vector<TrajectoryDivergenceSample> samples;
  for (std::uint64_t step : steps) {
    samples.push_back(
        {.step = step,
         .relative_deviation =
             first * std::exp(rate * (static_cast<double>(step) - 1.0))});
  }
  return samples;
}

void the_measured_divergence_is_geometric_until_it_saturates() {
  const trainvm::TrajectoryDivergenceFit clean =
      fit_trajectory_divergence(measured_fused_adamw_unsaturated());
  near(clean.rate, 2.242, 0.01,
       "the pre-saturation fused-AdamW divergence rate");
  require(clean.fit_quality > 0.999,
          "the first three points are geometric to within rounding");

  // The fourth point has saturated -- a relative deviation cannot keep
  // growing past order one -- so the log-linear model no longer describes the
  // series, and its slope is not a divergence rate.
  const trainvm::TrajectoryDivergenceFit saturated =
      fit_trajectory_divergence(measured_fused_adamw());
  near(saturated.fit_quality, 0.766, 0.01,
       "the saturated series is a poor log-linear fit");
  require(saturated.fit_quality < trainvm::kMinimumDivergenceFitQuality,
          "a saturated series must fall below the usable-fit floor");
}

void a_fused_kernel_qualifies_when_it_diverges_no_faster_than_the_seed() {
  // The reference against itself under a different seed, at a comparable rate.
  const auto reference = geometric(2.0e-07, 2.30, {1U, 2U, 5U});
  const trainvm::TrajectoryParityAssessment accepted =
      assess_trajectory_parity(rate_evidence(
          measured_fused_adamw_unsaturated(), reference,
          TrajectoryParityVerdict::diverged_within_tolerance));
  require(accepted.verdict == TrajectoryParityVerdict::diverged_within_tolerance,
          "a candidate diverging no faster than the reference's own seed noise "
          "is equivalent in the only sense that matters");
  require(accepted.divergence_rate_ratio < 1.0,
          "the fused candidate amplifies more slowly than the seed does");
  near(accepted.tolerance.maximum_divergence_rate_ratio, 1.25, 1e-12,
       "the optimizer_update rate tolerance");

  // The same kernel against a reference that barely moves between seeds is a
  // different claim, and fails.
  const auto quiet_reference = geometric(2.0e-07, 0.50, {1U, 2U, 5U});
  const trainvm::TrajectoryParityAssessment refused = assess_trajectory_parity(
      rate_evidence(measured_fused_adamw_unsaturated(), quiet_reference,
                    TrajectoryParityVerdict::diverged));
  require(refused.verdict == TrajectoryParityVerdict::diverged &&
              refused.divergence_rate_ratio > 4.0,
          "a candidate that amplifies four times faster than the seed is not "
          "equivalent");
}

void a_saturated_series_cannot_be_graded_on_its_rate() {
  const auto reference = geometric(2.0e-07, 2.30, {1U, 2U, 5U, 25U});
  const trainvm::TrajectoryParityAssessment assessment =
      assess_trajectory_parity(rate_evidence(
          measured_fused_adamw(), reference, TrajectoryParityVerdict::diverged));
  require(assessment.verdict == TrajectoryParityVerdict::diverged,
          "an ungradeable rate fails closed rather than passing on a slope "
          "that describes nothing");
  require(assessment.candidate_fit_quality <
              trainvm::kMinimumDivergenceFitQuality,
          "the recorded fit quality says why");
}

void bit_identity_is_the_only_route_to_equivalent() {
  TrajectoryParityEvidence evidence{
      .verdict = TrajectoryParityVerdict::equivalent,
      .criterion = TrajectoryEquivalenceCriterion::bit_identical,
      .effect_class = TrajectoryEffectClass::optimizer_update,
      .candidate_divergence = {{.step = 1U, .relative_deviation = 0.0},
                               {.step = 2U, .relative_deviation = 0.0},
                               {.step = 5U, .relative_deviation = 0.0}},
      .reference_divergence = {},
      .checkpoint_quality = {},
      .analysis_seed = 0U,
  };
  require(assess_trajectory_parity(evidence).verdict ==
              TrajectoryParityVerdict::equivalent,
          "a zero deviation throughout is equivalence");

  // Widening what the field can express must not widen what `equivalent`
  // means: the rate route tops out at diverged_within_tolerance.
  const auto reference = geometric(2.0e-07, 2.30, {1U, 2U, 5U});
  require(assess_trajectory_parity(
              rate_evidence(measured_fused_adamw_unsaturated(), reference,
                            TrajectoryParityVerdict::diverged_within_tolerance))
                  .verdict != TrajectoryParityVerdict::equivalent,
          "a statistical pass is never promoted to bit-identity");

  // And a bit-identity claim the samples contradict is refused.
  evidence.candidate_divergence[2].relative_deviation = 1e-12;
  require(assess_trajectory_parity(evidence).verdict ==
              TrajectoryParityVerdict::diverged,
          "an unsupported bit-identity claim is rejected");

  // Declaring a looser criterion over bit-identical samples is malformed
  // rather than merely wrong, because it would hide the strongest result.
  rejected(
      [&] {
        (void)assess_trajectory_parity(rate_evidence(
            {{.step = 1U, .relative_deviation = 0.0},
             {.step = 2U, .relative_deviation = 0.0},
             {.step = 5U, .relative_deviation = 0.0}},
            reference, TrajectoryParityVerdict::equivalent));
      },
      "bit_identical criterion",
      "bit-identical samples must be declared as such");
}

void classes_that_cannot_drift_have_a_zero_tolerance() {
  const auto reference = geometric(2.0e-07, 2.30, {1U, 2U, 5U});
  for (TrajectoryEffectClass effect_class :
       {TrajectoryEffectClass::schedule_state,
        TrajectoryEffectClass::curriculum_phase}) {
    const trainvm::TrajectoryEffectTolerance tolerance =
        trainvm::trajectory_effect_tolerance(effect_class);
    require(tolerance.maximum_divergence_rate_ratio == 0.0 &&
                tolerance.maximum_quality_deviation == 0.0,
            "a computed schedule value and a discrete curriculum phase have no "
            "tolerance band");
    require(assess_trajectory_parity(
                rate_evidence(measured_fused_adamw_unsaturated(), reference,
                              TrajectoryParityVerdict::diverged, effect_class))
                    .verdict == TrajectoryParityVerdict::diverged,
            "the rate route is closed for a zero-tolerance class");
  }

  // The looser classes are ordered, and none of them reaches the point where a
  // candidate may amplify unboundedly.
  const double optimizer =
      trainvm::trajectory_effect_tolerance(
          TrajectoryEffectClass::optimizer_update)
          .maximum_divergence_rate_ratio;
  const double gradient =
      trainvm::trajectory_effect_tolerance(
          TrajectoryEffectClass::gradient_policy)
          .maximum_divergence_rate_ratio;
  const double precision =
      trainvm::trajectory_effect_tolerance(
          TrajectoryEffectClass::precision_scaling)
          .maximum_divergence_rate_ratio;
  require(optimizer < gradient && gradient < precision && precision <= 2.0,
          "tolerance is stated per effect class and stays bounded");
}

void the_quality_route_is_an_equivalence_claim_not_a_null_result() {
  const auto observations = [](std::vector<double> baseline,
                               std::vector<double> candidate) {
    std::vector<TrajectoryQualityObservation> rows;
    for (std::size_t index = 0; index < baseline.size(); ++index) {
      rows.push_back({.seed = static_cast<std::uint64_t>(index) + 1U,
                      .baseline_metric = baseline[index],
                      .candidate_metric = candidate[index]});
    }
    return rows;
  };

  TrajectoryParityEvidence evidence{
      .verdict = TrajectoryParityVerdict::diverged_within_tolerance,
      .criterion = TrajectoryEquivalenceCriterion::checkpoint_quality,
      .effect_class = TrajectoryEffectClass::precision_scaling,
      .candidate_divergence = measured_fused_adamw(),
      .reference_divergence = {},
      .checkpoint_quality = observations(
          {2.500, 2.510, 2.495, 2.505, 2.498, 2.502},
          {2.501, 2.509, 2.497, 2.504, 2.499, 2.503}),
      .analysis_seed = 11U,
  };
  const trainvm::TrajectoryParityAssessment tight =
      assess_trajectory_parity(evidence);
  require(tight.verdict == TrajectoryParityVerdict::diverged_within_tolerance,
          "a paired interval inside the tolerance band is an equivalence "
          "result");
  require(tight.quality_seed_count == 6U,
          "the seed count that produced the interval is recorded");
  require(std::abs(tight.quality_relative_interval_low) <=
                  tight.tolerance.maximum_quality_deviation &&
              std::abs(tight.quality_relative_interval_high) <=
                  tight.tolerance.maximum_quality_deviation,
          "the whole interval, not just its centre, is inside the band");

  // A noisy pair whose delta averages to nearly zero must NOT pass: a wide
  // interval straddling zero is a failure to measure, not equivalence. This is
  // the mistake a p-value gate would make here.
  evidence.checkpoint_quality = observations(
      {2.500, 2.510, 2.495, 2.505, 2.498, 2.502},
      {2.750, 2.270, 2.740, 2.270, 2.745, 2.265});
  evidence.verdict = TrajectoryParityVerdict::diverged;
  const trainvm::TrajectoryParityAssessment noisy =
      assess_trajectory_parity(evidence);
  require(noisy.verdict == TrajectoryParityVerdict::diverged,
          "a wide interval around zero is not an equivalence result");
  require(std::abs(noisy.quality_relative_delta) < 0.01 &&
              noisy.quality_p_value > 0.05,
          "and it would have passed a difference test, which is the point");

  // Too few seeds to support an interval at all.
  evidence.checkpoint_quality = observations({2.5, 2.5, 2.5}, {2.5, 2.5, 2.5});
  require(assess_trajectory_parity(evidence).verdict ==
              TrajectoryParityVerdict::diverged,
          "an underpowered comparison fails closed");
}

void malformed_evidence_is_refused_rather_than_graded() {
  rejected(
      [] {
        (void)assess_trajectory_parity(rate_evidence(
            {{.step = 5U, .relative_deviation = 1e-6},
             {.step = 2U, .relative_deviation = 1e-5},
             {.step = 9U, .relative_deviation = 1e-4}},
            {}, TrajectoryParityVerdict::diverged));
      },
      "candidate divergence series", "steps must increase");
  rejected(
      [] {
        (void)assess_trajectory_parity(rate_evidence(
            {{.step = 1U, .relative_deviation = -1.0},
             {.step = 2U, .relative_deviation = 1e-5},
             {.step = 3U, .relative_deviation = 1e-4}},
            {}, TrajectoryParityVerdict::diverged));
      },
      "candidate divergence series", "a deviation cannot be negative");
  rejected(
      [] {
        (void)fit_trajectory_divergence({{.step = 1U, .relative_deviation = 1e-6},
                                   {.step = 2U, .relative_deviation = 1e-5}});
      },
      "at least three", "two points fit a line whatever they are");
}

}  // namespace

int main() {
  try {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"measured divergence", the_measured_divergence_is_geometric_until_it_saturates},
        {"rate criterion", a_fused_kernel_qualifies_when_it_diverges_no_faster_than_the_seed},
        {"saturation guard", a_saturated_series_cannot_be_graded_on_its_rate},
        {"equivalent is bit-identity", bit_identity_is_the_only_route_to_equivalent},
        {"per-class tolerance", classes_that_cannot_drift_have_a_zero_tolerance},
        {"quality criterion", the_quality_route_is_an_equivalence_claim_not_a_null_result},
        {"malformed evidence", malformed_evidence_is_refused_rather_than_graded},
    };
    for (const auto& [name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
    std::cout << "All trajectory parity tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
