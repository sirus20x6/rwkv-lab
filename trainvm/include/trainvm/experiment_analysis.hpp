#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace trainvm {

class ExperimentAnalysisError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct PairedAnalysisOptions {
  std::size_t bootstrap_samples{10'000U};
  std::size_t permutation_samples{20'000U};
  std::uint64_t seed{0U};
  double alpha{0.05};
};

struct PairedStatistics {
  std::size_t n{};
  double baseline_mean{};
  double candidate_mean{};
  double delta{};
  double ci_low{};
  double ci_high{};
  double p_value{};
  double effect_size{};
  std::size_t recommended_n{};
  bool significant{};
  std::vector<double> paired_deltas;
};

[[nodiscard]] PairedStatistics paired_statistics(
    std::span<const double> baseline, std::span<const double> candidate,
    const PairedAnalysisOptions& options = {});

struct HypothesisEvidence {
  std::string name;
  double p_value{};
  bool interval_excludes_zero{};
};

struct HypothesisDecision {
  std::string name;
  double p_value{};
  double p_adjusted{};
  bool significant{};
};

[[nodiscard]] std::vector<HypothesisDecision> holm_adjust(
    std::span<const HypothesisEvidence> hypotheses, double alpha = 0.05);

enum class AlphaSpendingMethod {
  obrien_fleming,
  pocock,
  linear,
};

struct AlphaSpend {
  std::size_t look{};
  std::size_t total_looks{};
  double information_fraction{};
  AlphaSpendingMethod method{AlphaSpendingMethod::obrien_fleming};
  double family_alpha{};
  double cumulative{};
  double increment{};
};

[[nodiscard]] AlphaSpend alpha_spending(
    std::size_t look, std::size_t total_looks, double alpha = 0.05,
    AlphaSpendingMethod method = AlphaSpendingMethod::obrien_fleming);

struct SequentialHypothesisDecision {
  HypothesisDecision decision;
  AlphaSpend spending;
};

[[nodiscard]] std::vector<SequentialHypothesisDecision> sequential_holm(
    std::span<const HypothesisEvidence> hypotheses, std::size_t look,
    std::size_t total_looks, double alpha = 0.05,
    AlphaSpendingMethod method = AlphaSpendingMethod::obrien_fleming);

enum class ObjectiveDirection {
  maximize,
  minimize,
};

struct ParetoObjective {
  std::string name;
  ObjectiveDirection direction{ObjectiveDirection::maximize};
};

using ObjectiveRow = std::map<std::string, std::optional<double>>;

[[nodiscard]] std::vector<bool> pareto_front(
    std::span<const ObjectiveRow> rows,
    std::span<const ParetoObjective> objectives);

struct CampaignSummary {
  std::int64_t id{};
  double created_ts{};
  std::string task;
  std::string phase;
  std::string status;
  std::optional<std::int64_t> parent_id;
  std::string git_sha;
  std::size_t arm_count{};
  std::size_t trial_count{};
};

struct CampaignComparison {
  std::string arm_name;
  std::size_t n{};
  double delta{};
  double ci_low{};
  double ci_high{};
  double p_adjusted{};
  double effect_size{};
  bool significant{};
  bool confirmed{};
};

struct CampaignComparisonReport {
  std::int64_t campaign_id{};
  std::string task;
  std::string phase;
  std::string metric;
  std::vector<CampaignComparison> comparisons;
};

struct LegacyAggregateComparison {
  std::string config;
  double mean{};
  double standard_deviation{};
  std::string git_sha;
  double timestamp{};
  std::size_t seeds{};
  std::optional<double> delta_from_baseline;
  bool significant{};
};

struct LegacyComparisonReport {
  std::string task;
  std::string metric;
  std::string baseline;
  bool baseline_present{};
  std::vector<LegacyAggregateComparison> comparisons;
};

struct ExperimentRegistrySnapshot {
  std::string api_version{"trainvm.experiment-registry-snapshot/v1"};
  std::string source;
  std::vector<CampaignSummary> campaigns;
  std::optional<CampaignComparisonReport> latest_comparison;
  std::optional<LegacyComparisonReport> legacy_comparison;
};

struct ExperimentRegistryQuery {
  std::size_t campaign_limit{20U};
  std::optional<std::string> task;
  std::string metric{"acc"};
  std::string baseline{"baseline"};
};

// Opens an existing Python experiment registry with SQLite's read-only flag and
// reads all requested views inside one snapshot transaction. It never creates,
// migrates, checkpoints, or otherwise writes the source database.
[[nodiscard]] ExperimentRegistrySnapshot read_experiment_registry(
    const std::filesystem::path& database_path,
    const ExperimentRegistryQuery& query = {});

}  // namespace trainvm
