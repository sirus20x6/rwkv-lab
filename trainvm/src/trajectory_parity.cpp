#include "trainvm/trajectory_parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "trainvm/experiment_analysis.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumDivergenceSamples = 4'096U;
constexpr std::size_t kMaximumQualityObservations = 64U;

// Deliberately smaller than the analysis defaults. The verdict is recomputed
// every time a receipt is validated, and the sign-flip test is exact rather
// than sampled at these sample counts anyway, so a larger draw buys nothing.
constexpr std::size_t kBootstrapSamples = 2'000U;
constexpr std::size_t kPermutationSamples = 4'000U;

void require(bool condition, const char* message) {
  if (!condition) {
    throw TrajectoryParityError(message);
  }
}

void validate_series(const std::vector<TrajectoryDivergenceSample>& samples,
                     const char* name) {
  require(samples.size() <= kMaximumDivergenceSamples, name);
  bool first = true;
  std::uint64_t previous = 0U;
  for (const TrajectoryDivergenceSample& sample : samples) {
    require(std::isfinite(sample.relative_deviation) &&
                sample.relative_deviation >= 0.0,
            name);
    require(first || sample.step > previous, name);
    previous = sample.step;
    first = false;
  }
}

bool all_zero(const std::vector<TrajectoryDivergenceSample>& samples) {
  return std::ranges::all_of(samples, [](const TrajectoryDivergenceSample& s) {
    return s.relative_deviation == 0.0;
  });
}

bool fittable(const std::vector<TrajectoryDivergenceSample>& samples) {
  return samples.size() >= kMinimumDivergenceSamples &&
         std::ranges::all_of(samples,
                             [](const TrajectoryDivergenceSample& s) {
                               return s.relative_deviation > 0.0;
                             });
}

}  // namespace

TrajectoryEffectTolerance
trajectory_effect_tolerance(TrajectoryEffectClass effect_class) {
  switch (effect_class) {
    case TrajectoryEffectClass::optimizer_update:
      // A reassociated update of the same arithmetic. The 25% headroom is for
      // the rate estimate, not for the candidate: it must not amplify faster
      // than the reference amplifies against its own seed.
      return {.maximum_divergence_rate_ratio = 1.25,
              .maximum_quality_deviation = 0.010};
    case TrajectoryEffectClass::gradient_policy:
      // Summation order changes across accumulation and synchronization
      // boundaries; the summands do not.
      return {.maximum_divergence_rate_ratio = 1.50,
              .maximum_quality_deviation = 0.010};
    case TrajectoryEffectClass::precision_scaling:
      // A dtype or scaler change has a larger per-step floor by construction,
      // so it may amplify faster and is graded on quality more often.
      return {.maximum_divergence_rate_ratio = 2.00,
              .maximum_quality_deviation = 0.020};
    case TrajectoryEffectClass::schedule_state:
    case TrajectoryEffectClass::curriculum_phase:
      // Computed, not accumulated; discrete, not continuous. There is no
      // "within tolerance" here, and a zero tolerance says so in the receipt
      // rather than leaving a reader to infer it.
      return {.maximum_divergence_rate_ratio = 0.0,
              .maximum_quality_deviation = 0.0};
  }
  throw TrajectoryParityError("unknown trajectory effect class");
}

std::string trajectory_parity_verdict_name(TrajectoryParityVerdict verdict) {
  switch (verdict) {
    case TrajectoryParityVerdict::equivalent:
      return "equivalent";
    case TrajectoryParityVerdict::diverged_within_tolerance:
      return "diverged_within_tolerance";
    case TrajectoryParityVerdict::diverged:
      return "diverged";
  }
  throw TrajectoryParityError("unknown trajectory parity verdict");
}

TrajectoryDivergenceFit fit_trajectory_divergence(
    const std::vector<TrajectoryDivergenceSample>& samples) {
  require(fittable(samples),
          "a divergence rate needs at least three strictly positive samples");
  const auto count = static_cast<double>(samples.size());
  double step_total = 0.0;
  double log_total = 0.0;
  for (const TrajectoryDivergenceSample& sample : samples) {
    step_total += static_cast<double>(sample.step);
    log_total += std::log(sample.relative_deviation);
  }
  const double step_mean = step_total / count;
  const double log_mean = log_total / count;
  double step_variance = 0.0;
  double log_variance = 0.0;
  double covariance = 0.0;
  for (const TrajectoryDivergenceSample& sample : samples) {
    const double step_deviation = static_cast<double>(sample.step) - step_mean;
    const double log_deviation =
        std::log(sample.relative_deviation) - log_mean;
    step_variance += step_deviation * step_deviation;
    log_variance += log_deviation * log_deviation;
    covariance += step_deviation * log_deviation;
  }
  require(step_variance > 0.0, "a divergence rate needs distinct steps");
  const double rate = covariance / step_variance;
  // A perfectly flat series has no variance to explain. It is a clean fit of a
  // zero rate, not an unexplained one.
  const double fit_quality =
      log_variance > 0.0 ? (covariance * covariance) /
                               (step_variance * log_variance)
                         : 1.0;
  require(std::isfinite(rate) && std::isfinite(fit_quality),
          "the divergence fit is not finite");
  return {.rate = rate, .fit_quality = std::clamp(fit_quality, 0.0, 1.0)};
}

TrajectoryParityAssessment
assess_trajectory_parity(const TrajectoryParityEvidence& evidence) {
  const TrajectoryEffectTolerance tolerance =
      trajectory_effect_tolerance(evidence.effect_class);
  validate_series(evidence.candidate_divergence, "candidate divergence series");
  validate_series(evidence.reference_divergence, "reference divergence series");
  require(evidence.checkpoint_quality.size() <= kMaximumQualityObservations,
          "checkpoint quality observations exceed the supported bound");
  {
    bool first = true;
    std::uint64_t previous = 0U;
    for (const TrajectoryQualityObservation& observation :
         evidence.checkpoint_quality) {
      require(std::isfinite(observation.baseline_metric) &&
                  std::isfinite(observation.candidate_metric),
              "a checkpoint quality observation is not finite");
      require(first || observation.seed > previous,
              "checkpoint quality seeds must be distinct and ordered");
      previous = observation.seed;
      first = false;
    }
  }

  TrajectoryParityAssessment assessment{
      .verdict = TrajectoryParityVerdict::diverged,
      .criterion = evidence.criterion,
      .effect_class = evidence.effect_class,
      .tolerance = tolerance,
  };

  // Bit-identity is decided before the criterion, so it cannot be reached by
  // declaring a looser one. It is also the only route to `equivalent`, which
  // is what stops the third state from widening the second.
  const bool identical = !evidence.candidate_divergence.empty() &&
                         all_zero(evidence.candidate_divergence);
  if (identical) {
    require(evidence.criterion == TrajectoryEquivalenceCriterion::bit_identical,
            "a bit-identical trajectory must be declared under the "
            "bit_identical criterion");
    assessment.verdict = TrajectoryParityVerdict::equivalent;
    assessment.candidate_fit_quality = 1.0;
    return assessment;
  }

  switch (evidence.criterion) {
    case TrajectoryEquivalenceCriterion::bit_identical:
      // Claimed and not observed. A rejection, not a malformed document.
      require(!evidence.candidate_divergence.empty(),
              "a bit_identical claim must carry the samples that support it");
      return assessment;

    case TrajectoryEquivalenceCriterion::divergence_rate: {
      if (tolerance.maximum_divergence_rate_ratio <= 0.0) {
        return assessment;
      }
      if (!fittable(evidence.candidate_divergence) ||
          !fittable(evidence.reference_divergence)) {
        return assessment;
      }
      const TrajectoryDivergenceFit candidate =
          fit_trajectory_divergence(evidence.candidate_divergence);
      const TrajectoryDivergenceFit reference =
          fit_trajectory_divergence(evidence.reference_divergence);
      assessment.candidate_divergence_rate = candidate.rate;
      assessment.reference_divergence_rate = reference.rate;
      assessment.candidate_fit_quality = candidate.fit_quality;
      assessment.reference_fit_quality = reference.fit_quality;
      // A poor log-linear fit means the series is not geometric -- it
      // saturated, or it never was -- so the slope is not a divergence rate
      // and comparing it would be comparing nothing. The card's own four
      // points fit at r^2 = 0.766 because the last one has saturated; the
      // first three fit at r^2 = 1.000.
      if (candidate.fit_quality < kMinimumDivergenceFitQuality ||
          reference.fit_quality < kMinimumDivergenceFitQuality) {
        return assessment;
      }
      // A reference that does not diverge from itself grants no headroom.
      if (!(reference.rate > 0.0)) {
        return assessment;
      }
      assessment.divergence_rate_ratio = candidate.rate / reference.rate;
      if (assessment.divergence_rate_ratio <=
          tolerance.maximum_divergence_rate_ratio) {
        assessment.verdict = TrajectoryParityVerdict::diverged_within_tolerance;
      }
      return assessment;
    }

    case TrajectoryEquivalenceCriterion::checkpoint_quality: {
      if (tolerance.maximum_quality_deviation <= 0.0) {
        return assessment;
      }
      if (evidence.checkpoint_quality.size() < kMinimumQualitySeeds) {
        return assessment;
      }
      std::vector<double> baseline;
      std::vector<double> candidate;
      baseline.reserve(evidence.checkpoint_quality.size());
      candidate.reserve(evidence.checkpoint_quality.size());
      for (const TrajectoryQualityObservation& observation :
           evidence.checkpoint_quality) {
        baseline.push_back(observation.baseline_metric);
        candidate.push_back(observation.candidate_metric);
      }
      assessment.quality_seed_count =
          static_cast<std::uint64_t>(baseline.size());
      const PairedStatistics statistics = paired_statistics(
          baseline, candidate,
          {.bootstrap_samples = kBootstrapSamples,
           .permutation_samples = kPermutationSamples,
           .seed = evidence.analysis_seed,
           .alpha = 0.05});
      const double scale = std::abs(statistics.baseline_mean);
      if (!(scale > 0.0)) {
        // Nothing to be relative to; a relative tolerance is meaningless.
        return assessment;
      }
      assessment.quality_relative_delta = statistics.delta / scale;
      assessment.quality_relative_interval_low = statistics.ci_low / scale;
      assessment.quality_relative_interval_high = statistics.ci_high / scale;
      assessment.quality_p_value = statistics.p_value;
      // An equivalence claim, not a failure to reject: the whole interval on
      // the paired delta must sit inside the tolerance band. A wide interval
      // that happens to straddle zero fails here, which is the point.
      const double bound =
          std::max(std::abs(assessment.quality_relative_interval_low),
                   std::abs(assessment.quality_relative_interval_high));
      if (bound <= tolerance.maximum_quality_deviation) {
        assessment.verdict = TrajectoryParityVerdict::diverged_within_tolerance;
      }
      return assessment;
    }
  }
  throw TrajectoryParityError("unknown trajectory equivalence criterion");
}

}  // namespace trainvm
