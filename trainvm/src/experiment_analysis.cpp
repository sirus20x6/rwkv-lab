#include "trainvm/experiment_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <numbers>
#include <set>
#include <utility>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumDraws = 1'000'000U;
constexpr std::size_t kMaximumPairedSamples = 1'000'000U;

void require_probability(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0 || value >= 1.0) {
    throw ExperimentAnalysisError(std::string(name) +
                                  " must be finite and between zero and one");
  }
}

void require_finite(std::span<const double> values, const char* name) {
  if (values.empty()) {
    throw ExperimentAnalysisError(std::string(name) + " must not be empty");
  }
  if (values.size() > kMaximumPairedSamples) {
    throw ExperimentAnalysisError(std::string(name) + " exceeds the sample limit");
  }
  if (!std::ranges::all_of(values, [](double value) { return std::isfinite(value); })) {
    throw ExperimentAnalysisError(std::string(name) + " contains a non-finite value");
  }
}

double mean(std::span<const double> values) {
  const long double total = std::accumulate(values.begin(), values.end(), 0.0L);
  const long double result = total / static_cast<long double>(values.size());
  if (!std::isfinite(result) ||
      std::abs(result) > static_cast<long double>(std::numeric_limits<double>::max())) {
    throw ExperimentAnalysisError("sample mean exceeds the supported numeric range");
  }
  return static_cast<double>(result);
}

class StableRandom {
 public:
  explicit StableRandom(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  std::size_t bounded(std::size_t bound) {
    const auto unsigned_bound = static_cast<std::uint64_t>(bound);
    const auto threshold = static_cast<std::uint64_t>(-unsigned_bound) % unsigned_bound;
    while (true) {
      const auto value = next();
      if (value >= threshold) {
        return static_cast<std::size_t>(value % unsigned_bound);
      }
    }
  }

  double sign() { return (next() & 1U) == 0U ? -1.0 : 1.0; }

 private:
  std::uint64_t state_;
};

double linear_quantile(std::vector<double> values, double probability) {
  std::ranges::sort(values);
  if (values.size() == 1U) return values.front();
  const double position = probability * static_cast<double>(values.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] + fraction * (values[upper] - values[lower]);
}

// Peter J. Acklam's inverse-normal approximation, followed by one Halley step.
double inverse_normal_cdf(double probability) {
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 = 2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 = 1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 = 2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 = 1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 = 6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 = 4.374664141464968e+00;
  constexpr double c6 = 2.938163982698783e+00;
  constexpr double d1 = 7.784695709041462e-03;
  constexpr double d2 = 3.224671290700398e-01;
  constexpr double d3 = 2.445134137142996e+00;
  constexpr double d4 = 3.754408661907416e+00;
  constexpr double low = 0.02425;
  constexpr double high = 1.0 - low;

  double value{};
  if (probability < low) {
    const double q = std::sqrt(-2.0 * std::log(probability));
    value = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  } else if (probability <= high) {
    const double q = probability - 0.5;
    const double r = q * q;
    value = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
    value = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double error = 0.5 * std::erfc(-value / std::sqrt(2.0)) - probability;
  const double correction = error * std::sqrt(2.0 * std::numbers::pi) *
                            std::exp(value * value / 2.0);
  return value - correction / (1.0 + value * correction / 2.0);
}

double cumulative_alpha(std::size_t look, std::size_t total_looks, double alpha,
                        AlphaSpendingMethod method) {
  const double information = static_cast<double>(look) /
                             static_cast<double>(total_looks);
  switch (method) {
    case AlphaSpendingMethod::obrien_fleming: {
      const double z = inverse_normal_cdf(1.0 - alpha / 2.0);
      const double spent = 2.0 *
                           (1.0 - 0.5 * std::erfc(-z / std::sqrt(2.0 * information)));
      return std::min(alpha, spent);
    }
    case AlphaSpendingMethod::pocock:
      return alpha * std::log(1.0 + (std::numbers::e - 1.0) * information);
    case AlphaSpendingMethod::linear:
      return alpha * information;
  }
  throw ExperimentAnalysisError("unknown alpha-spending method");
}

}  // namespace

PairedStatistics paired_statistics(std::span<const double> baseline,
                                   std::span<const double> candidate,
                                   const PairedAnalysisOptions& options) {
  require_finite(baseline, "baseline");
  require_finite(candidate, "candidate");
  if (baseline.size() != candidate.size()) {
    throw ExperimentAnalysisError("paired samples must have equal length");
  }
  require_probability(options.alpha, "alpha");
  if (options.bootstrap_samples == 0U || options.bootstrap_samples > kMaximumDraws) {
    throw ExperimentAnalysisError("bootstrap_samples must be in [1, 1000000]");
  }
  if (options.permutation_samples == 0U ||
      options.permutation_samples > kMaximumDraws) {
    throw ExperimentAnalysisError("permutation_samples must be in [1, 1000000]");
  }

  std::vector<double> deltas;
  deltas.reserve(baseline.size());
  for (std::size_t index = 0; index < baseline.size(); ++index) {
    const double delta = candidate[index] - baseline[index];
    if (!std::isfinite(delta)) {
      throw ExperimentAnalysisError("paired delta overflowed");
    }
    deltas.push_back(delta);
  }
  const double baseline_mean = mean(baseline);
  const double candidate_mean = mean(candidate);
  const double delta_mean = mean(deltas);
  long double deviation_sum{};
  for (double delta : deltas) {
    const long double deviation = static_cast<long double>(delta) - delta_mean;
    deviation_sum += deviation * deviation;
  }
  const long double standard_deviation_wide =
      deltas.size() > 1U
          ? std::sqrt(deviation_sum /
                      static_cast<long double>(deltas.size() - 1U))
          : 0.0L;
  if (!std::isfinite(standard_deviation_wide) ||
      standard_deviation_wide >
          static_cast<long double>(std::numeric_limits<double>::max())) {
    throw ExperimentAnalysisError("sample deviation exceeds the supported numeric range");
  }
  const double standard_deviation = static_cast<double>(standard_deviation_wide);
  const double effect_size = standard_deviation > 0.0
                                 ? delta_mean / standard_deviation
                                 : (delta_mean == 0.0
                                        ? 0.0
                                        : std::copysign(
                                              std::numeric_limits<double>::infinity(),
                                              delta_mean));

  StableRandom random(options.seed);
  double ci_low = delta_mean;
  double ci_high = delta_mean;
  if (deltas.size() > 1U) {
    std::vector<double> bootstrap_means;
    bootstrap_means.reserve(options.bootstrap_samples);
    for (std::size_t draw = 0; draw < options.bootstrap_samples; ++draw) {
      long double total{};
      for (std::size_t index = 0; index < deltas.size(); ++index) {
        total += deltas[random.bounded(deltas.size())];
      }
      bootstrap_means.push_back(static_cast<double>(
          total / static_cast<long double>(deltas.size())));
    }
    ci_low = linear_quantile(bootstrap_means, options.alpha / 2.0);
    ci_high = linear_quantile(std::move(bootstrap_means), 1.0 - options.alpha / 2.0);
  }

  const double observed = std::abs(delta_mean);
  std::size_t null_count{};
  std::size_t extreme_count{};
  if (deltas.size() <= 16U) {
    null_count = std::size_t{1U} << deltas.size();
    for (std::size_t mask = 0; mask < null_count; ++mask) {
      long double total{};
      for (std::size_t index = 0; index < deltas.size(); ++index) {
        const double sign = ((mask >> index) & 1U) == 0U ? -1.0 : 1.0;
        total += sign * deltas[index];
      }
      if (std::abs(total / static_cast<long double>(deltas.size())) >=
          static_cast<long double>(observed - 1e-15)) {
        ++extreme_count;
      }
    }
  } else {
    null_count = options.permutation_samples;
    for (std::size_t draw = 0; draw < null_count; ++draw) {
      long double total{};
      for (double delta : deltas) total += random.sign() * delta;
      if (std::abs(total / static_cast<long double>(deltas.size())) >=
          static_cast<long double>(observed - 1e-15)) {
        ++extreme_count;
      }
    }
  }
  const double p_value = static_cast<double>(extreme_count + 1U) /
                         static_cast<double>(null_count + 1U);

  std::size_t recommended_n{};
  if (std::abs(delta_mean) > 0.0 && standard_deviation > 0.0) {
    const double estimate = std::pow(2.8 * standard_deviation /
                                         std::abs(delta_mean),
                                     2.0);
    if (!std::isfinite(estimate) || estimate >
                                         static_cast<double>(kMaximumPairedSamples)) {
      recommended_n = kMaximumPairedSamples;
    } else {
      recommended_n = std::max(deltas.size(),
                               static_cast<std::size_t>(std::ceil(estimate)));
    }
  } else if (standard_deviation == 0.0 && delta_mean != 0.0) {
    recommended_n = deltas.size();
  } else {
    recommended_n = std::max(deltas.size() + 1U, std::size_t{8U});
  }

  return {
      .n = deltas.size(),
      .baseline_mean = baseline_mean,
      .candidate_mean = candidate_mean,
      .delta = delta_mean,
      .ci_low = ci_low,
      .ci_high = ci_high,
      .p_value = p_value,
      .effect_size = effect_size,
      .recommended_n = recommended_n,
      .significant = ci_low > 0.0 || ci_high < 0.0,
      .paired_deltas = std::move(deltas),
  };
}

std::vector<HypothesisDecision> holm_adjust(
    std::span<const HypothesisEvidence> hypotheses, double alpha) {
  require_probability(alpha, "alpha");
  std::set<std::string> names;
  for (const auto& hypothesis : hypotheses) {
    if (hypothesis.name.empty() || !names.insert(hypothesis.name).second) {
      throw ExperimentAnalysisError("hypothesis names must be non-empty and unique");
    }
    if (!std::isfinite(hypothesis.p_value) || hypothesis.p_value < 0.0 ||
        hypothesis.p_value > 1.0) {
      throw ExperimentAnalysisError("hypothesis p-values must be finite probabilities");
    }
  }
  std::vector<std::size_t> order(hypotheses.size());
  std::iota(order.begin(), order.end(), 0U);
  std::ranges::sort(order, {}, [&](std::size_t index) {
    return std::pair{hypotheses[index].p_value, hypotheses[index].name};
  });
  std::vector<HypothesisDecision> output(hypotheses.size());
  double running{};
  bool reject_chain = true;
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    const auto index = order[rank];
    const auto remaining = order.size() - rank;
    const auto& hypothesis = hypotheses[index];
    const double adjusted = std::min(
        1.0, std::max(running, static_cast<double>(remaining) * hypothesis.p_value));
    running = adjusted;
    reject_chain = reject_chain &&
                   hypothesis.p_value <= alpha / static_cast<double>(remaining);
    output[index] = {
        .name = hypothesis.name,
        .p_value = hypothesis.p_value,
        .p_adjusted = adjusted,
        .significant = hypothesis.interval_excludes_zero && reject_chain,
    };
  }
  return output;
}

AlphaSpend alpha_spending(std::size_t look, std::size_t total_looks,
                          double alpha, AlphaSpendingMethod method) {
  if (total_looks == 0U || look == 0U || look > total_looks) {
    throw ExperimentAnalysisError("look must be in [1, total_looks]");
  }
  require_probability(alpha, "alpha");
  const double spent = cumulative_alpha(look, total_looks, alpha, method);
  const double previous = look > 1U
                              ? cumulative_alpha(look - 1U, total_looks, alpha, method)
                              : 0.0;
  return {
      .look = look,
      .total_looks = total_looks,
      .information_fraction = static_cast<double>(look) /
                              static_cast<double>(total_looks),
      .method = method,
      .family_alpha = alpha,
      .cumulative = spent,
      .increment = std::max(spent - previous,
                            std::numeric_limits<double>::epsilon()),
  };
}

std::vector<SequentialHypothesisDecision> sequential_holm(
    std::span<const HypothesisEvidence> hypotheses, std::size_t look,
    std::size_t total_looks, double alpha, AlphaSpendingMethod method) {
  const auto spending = alpha_spending(look, total_looks, alpha, method);
  auto decisions = holm_adjust(hypotheses, spending.increment);
  std::vector<SequentialHypothesisDecision> output;
  output.reserve(decisions.size());
  for (auto& decision : decisions) {
    decision.significant = decision.significant &&
                           decision.p_adjusted <= spending.increment;
    output.push_back({.decision = std::move(decision), .spending = spending});
  }
  return output;
}

std::vector<bool> pareto_front(std::span<const ObjectiveRow> rows,
                               std::span<const ParetoObjective> objectives) {
  if (objectives.empty()) {
    throw ExperimentAnalysisError("at least one Pareto objective is required");
  }
  std::set<std::string> names;
  for (const auto& objective : objectives) {
    if (objective.name.empty() || !names.insert(objective.name).second) {
      throw ExperimentAnalysisError("Pareto objective names must be non-empty and unique");
    }
  }
  const auto value_for = [](const ObjectiveRow& row,
                            const std::string& name) -> std::optional<double> {
    const auto found = row.find(name);
    if (found == row.end() || !found->second.has_value() ||
        !std::isfinite(*found->second)) {
      return std::nullopt;
    }
    return *found->second;
  };
  std::vector<bool> result(rows.size(), false);
  for (std::size_t index = 0; index < rows.size(); ++index) {
    bool complete = true;
    for (const auto& objective : objectives) {
      complete = complete && value_for(rows[index], objective.name).has_value();
    }
    if (!complete) continue;
    bool dominated = false;
    for (std::size_t other_index = 0; other_index < rows.size(); ++other_index) {
      if (index == other_index) continue;
      bool other_complete = true;
      bool weak = true;
      bool strict = false;
      for (const auto& objective : objectives) {
        const auto current = value_for(rows[index], objective.name);
        const auto other = value_for(rows[other_index], objective.name);
        if (!other.has_value()) {
          other_complete = false;
          break;
        }
        if (objective.direction == ObjectiveDirection::maximize) {
          weak = weak && *other >= *current;
          strict = strict || *other > *current;
        } else {
          weak = weak && *other <= *current;
          strict = strict || *other < *current;
        }
      }
      if (other_complete && weak && strict) {
        dominated = true;
        break;
      }
    }
    result[index] = !dominated;
  }
  return result;
}

}  // namespace trainvm
