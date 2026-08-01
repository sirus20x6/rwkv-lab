#include "trainvm/experiment_analysis.hpp"
#include "trainvm/reflection_json.hpp"

#include <sqlite3.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

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
  } catch (const trainvm::ExperimentAnalysisError& error) {
    require(std::string_view(error.what()).find(expected) != std::string_view::npos,
            message + ": wrong error: " + error.what());
    return;
  }
  throw std::runtime_error(message + ": input was accepted");
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::atomic<std::uint64_t> sequence{};
    path_ = std::filesystem::temp_directory_path() /
            ("trainvm-experiment-analysis-" + std::to_string(::getpid()) + "-" +
             std::to_string(sequence.fetch_add(1U)));
    if (!std::filesystem::create_directory(path_)) {
      throw std::runtime_error("could not create test directory");
    }
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] std::filesystem::path file(std::string_view name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

void sqlite_execute(const std::filesystem::path& path, const std::string& sql) {
  sqlite3* database = nullptr;
  if (sqlite3_open(path.c_str(), &database) != SQLITE_OK) {
    const std::string detail = database == nullptr ? "unknown" : sqlite3_errmsg(database);
    if (database != nullptr) sqlite3_close(database);
    throw std::runtime_error("could not create SQLite fixture: " + detail);
  }
  char* raw_message = nullptr;
  const int status = sqlite3_exec(database, sql.c_str(), nullptr, nullptr, &raw_message);
  const std::string message = raw_message == nullptr ? sqlite3_errmsg(database) : raw_message;
  sqlite3_free(raw_message);
  sqlite3_close(database);
  if (status != SQLITE_OK) throw std::runtime_error("fixture SQL failed: " + message);
}

std::int64_t sqlite_integer(const std::filesystem::path& path, const char* sql) {
  sqlite3* database = nullptr;
  require(sqlite3_open_v2(path.c_str(), &database, SQLITE_OPEN_READONLY, nullptr) ==
              SQLITE_OK,
          "fixture opens read-only");
  sqlite3_stmt* statement = nullptr;
  require(sqlite3_prepare_v2(database, sql, -1, &statement, nullptr) == SQLITE_OK,
          "fixture scalar prepares");
  require(sqlite3_step(statement) == SQLITE_ROW &&
              sqlite3_column_type(statement, 0) == SQLITE_INTEGER,
          "fixture scalar returns integer");
  const auto result = sqlite3_column_int64(statement, 0);
  sqlite3_finalize(statement);
  sqlite3_close(database);
  return result;
}

void paired_statistics_are_deterministic_and_bounded() {
  const std::vector<double> baseline(8U, 0.0);
  const std::vector<double> candidate(8U, 1.0);
  const auto exact = trainvm::paired_statistics(baseline, candidate);
  require(exact.n == 8U && exact.paired_deltas == candidate,
          "paired analysis retains exact deltas");
  near(exact.baseline_mean, 0.0, 0.0, "baseline mean");
  near(exact.candidate_mean, 1.0, 0.0, "candidate mean");
  near(exact.ci_low, 1.0, 0.0, "constant bootstrap low bound");
  near(exact.ci_high, 1.0, 0.0, "constant bootstrap high bound");
  near(exact.p_value, 3.0 / 257.0, 1e-15, "exact sign permutation p-value");
  require(std::isinf(exact.effect_size) && exact.effect_size > 0.0 &&
              exact.significant && exact.recommended_n == 8U,
          "zero-variance positive effects retain Python semantics");

  const std::vector<double> varied_baseline{0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                             10, 11, 12, 13, 14, 15, 16};
  const std::vector<double> varied_candidate{1, 1, 4, 4, 7, 7, 10, 10, 13,
                                              13, 16, 16, 19, 19, 22, 22, 25};
  const trainvm::PairedAnalysisOptions options{
      .bootstrap_samples = 2'000U,
      .permutation_samples = 4'000U,
      .seed = 42U,
      .alpha = 0.05,
  };
  const auto first = trainvm::paired_statistics(varied_baseline,
                                                 varied_candidate, options);
  const auto second = trainvm::paired_statistics(varied_baseline,
                                                  varied_candidate, options);
  require(first.ci_low == second.ci_low && first.ci_high == second.ci_high &&
              first.p_value == second.p_value,
          "native randomized analysis is bit-deterministic for a fixed seed");
  require(first.ci_low <= first.delta && first.delta <= first.ci_high &&
              first.p_value >= 0.0 && first.p_value <= 1.0,
          "randomized estimates preserve probability and interval bounds");

  rejected(
      [&] {
        (void)trainvm::paired_statistics(
            std::vector<double>{1.0}, std::vector<double>{1.0, 2.0});
      },
      "equal length", "unpaired sample shapes fail closed");
  rejected(
      [&] {
        (void)trainvm::paired_statistics(
            std::vector<double>{std::numeric_limits<double>::quiet_NaN()},
            std::vector<double>{1.0});
      },
      "non-finite", "non-finite samples fail closed");
  auto invalid_options = options;
  invalid_options.bootstrap_samples = 0U;
  rejected(
      [&] {
        (void)trainvm::paired_statistics(varied_baseline, varied_candidate,
                                         invalid_options);
      },
      "bootstrap_samples", "zero bootstrap budget fails closed");
}

void multiple_testing_matches_registered_decisions() {
  const std::vector<trainvm::HypothesisEvidence> evidence{
      {.name = "first", .p_value = 0.01, .interval_excludes_zero = true},
      {.name = "second", .p_value = 0.03, .interval_excludes_zero = true},
      {.name = "third", .p_value = 0.04, .interval_excludes_zero = true},
  };
  const auto decisions = trainvm::holm_adjust(evidence);
  near(decisions[0].p_adjusted, 0.03, 1e-15, "first Holm adjustment");
  near(decisions[1].p_adjusted, 0.06, 1e-15, "second Holm adjustment");
  near(decisions[2].p_adjusted, 0.06, 1e-15, "third Holm adjustment");
  require(decisions[0].significant && !decisions[1].significant &&
              !decisions[2].significant,
          "Holm rejection stops after the first failed boundary");

  const auto linear = trainvm::alpha_spending(
      2U, 4U, 0.04, trainvm::AlphaSpendingMethod::linear);
  near(linear.information_fraction, 0.5, 0.0, "linear information fraction");
  near(linear.cumulative, 0.02, 1e-15, "linear cumulative alpha");
  near(linear.increment, 0.01, 1e-15, "linear incremental alpha");
  const auto obf_first = trainvm::alpha_spending(1U, 4U);
  const auto obf_final = trainvm::alpha_spending(4U, 4U);
  require(obf_first.cumulative > 0.0 && obf_first.cumulative < 0.05 &&
              obf_final.cumulative <= 0.05,
          "O'Brien-Fleming spending is bounded and conservative");
  near(obf_final.cumulative, 0.05, 2e-8,
       "O'Brien-Fleming spends the family alpha at the final look");
  near(obf_first.cumulative, 8.857543832130332e-05, 1e-12,
       "O'Brien-Fleming matches the Python decision helper");
  const auto sequential = trainvm::sequential_holm(
      evidence, 1U, 4U, 0.05, trainvm::AlphaSpendingMethod::linear);
  require(sequential.size() == evidence.size() &&
              sequential[0].spending.increment == 0.0125,
          "sequential Holm binds every decision to its preregistered spend");

  rejected(
      [&] {
        (void)trainvm::holm_adjust(std::vector<trainvm::HypothesisEvidence>{
            {.name = "same", .p_value = 0.1, .interval_excludes_zero = true},
            {.name = "same", .p_value = 0.2, .interval_excludes_zero = true},
        });
      },
      "unique", "duplicate hypotheses fail closed");
  rejected([&] { (void)trainvm::alpha_spending(0U, 4U); }, "look",
           "invalid interim look fails closed");
}

void pareto_front_handles_missing_and_mixed_objectives() {
  const std::vector<trainvm::ObjectiveRow> rows{
      {{"acc", 0.90}, {"seconds", 10.0}},
      {{"acc", 0.92}, {"seconds", 12.0}},
      {{"acc", 0.88}, {"seconds", 15.0}},
      {{"acc", std::nullopt}, {"seconds", 5.0}},
      {{"acc", 0.90}, {"seconds", 10.0}},
  };
  const std::vector<trainvm::ParetoObjective> objectives{
      {.name = "acc", .direction = trainvm::ObjectiveDirection::maximize},
      {.name = "seconds", .direction = trainvm::ObjectiveDirection::minimize},
  };
  const auto front = trainvm::pareto_front(rows, objectives);
  require(front == std::vector<bool>({true, true, false, false, true}),
          "Pareto analysis preserves ties and rejects incomplete rows");
  rejected(
      [&] {
        (void)trainvm::pareto_front(
            rows, std::vector<trainvm::ParetoObjective>{});
      },
      "at least one", "empty Pareto objective set fails closed");
}

void registry_snapshot_is_typed_ordered_and_read_only() {
  TemporaryDirectory directory;
  const auto database = directory.file("experiments.db");
  sqlite_execute(database, R"sql(
    CREATE TABLE campaigns(
      id INTEGER PRIMARY KEY, created_ts REAL NOT NULL, task TEXT NOT NULL,
      phase TEXT NOT NULL, status TEXT NOT NULL, parent_id INTEGER, git_sha TEXT);
    CREATE TABLE arms(id INTEGER PRIMARY KEY, campaign_id INTEGER NOT NULL, name TEXT NOT NULL);
    CREATE TABLE trials(id INTEGER PRIMARY KEY, campaign_id INTEGER NOT NULL);
    CREATE TABLE comparisons(
      id INTEGER PRIMARY KEY, campaign_id INTEGER NOT NULL, arm_id INTEGER NOT NULL,
      metric TEXT NOT NULL, n INTEGER NOT NULL, delta REAL, ci_low REAL, ci_high REAL,
      p_adjusted REAL, effect_size REAL, significant INTEGER NOT NULL, confirmed INTEGER NOT NULL);
    CREATE TABLE results(
      id INTEGER PRIMARY KEY, ts REAL, git_sha TEXT, task TEXT, config TEXT,
      seeds INTEGER, steps INTEGER, metrics_json TEXT);
    INSERT INTO campaigns VALUES
      (1,100.0,'recall:16','explore','complete',NULL,'aaa'),
      (2,200.0,'recall:16','confirm','running',1,'bbb'),
      (3,300.0,'vision','explore','complete',NULL,NULL);
    INSERT INTO arms VALUES (10,2,'wide'),(11,2,'fast'),(12,1,'old');
    INSERT INTO trials VALUES (20,2),(21,2),(22,1);
    INSERT INTO comparisons VALUES
      (30,2,10,'acc',8,0.10,0.02,0.18,0.03,0.9,1,0),
      (31,2,11,'acc',8,0.20,0.11,0.29,0.01,1.2,1,1),
      (32,2,11,'loss',8,-0.20,-0.29,-0.11,0.01,-1.2,1,1);
    INSERT INTO results VALUES
      (40,10.0,'old','legacy-task','baseline',4,100,'{"acc":[0.50,0.02]}'),
      (41,11.0,'new','legacy-task','candidate',4,100,'{"acc":[0.60,0.03]}'),
      (42,12.0,'newer','legacy-task','candidate',8,200,'{"acc":[0.65,0.02]}'),
      (43,13.0,'latest','legacy-task','missing',8,200,'{"loss":[1.0,0.1]}');
  )sql");
  const auto schema_before = sqlite_integer(database, "PRAGMA schema_version");
  const auto size_before = std::filesystem::file_size(database);
  const auto write_before = std::filesystem::last_write_time(database);
  require(::chmod(database.c_str(), S_IRUSR | S_IRGRP | S_IROTH) == 0,
          "fixture becomes filesystem read-only");

  const auto snapshot = trainvm::read_experiment_registry(
      database,
      {.campaign_limit = 2U, .task = "recall:16", .metric = "acc"});
  require(snapshot.api_version == "trainvm.experiment-registry-snapshot/v1" &&
              snapshot.source ==
                  std::filesystem::absolute(database).generic_string() &&
              snapshot.campaigns.size() == 2U,
          "registry snapshot is versioned, identified, and bounded");
  require(snapshot.campaigns[0].id == 3 && snapshot.campaigns[0].arm_count == 0U &&
              snapshot.campaigns[1].id == 2 &&
              snapshot.campaigns[1].parent_id == 1 &&
              snapshot.campaigns[1].arm_count == 2U &&
              snapshot.campaigns[1].trial_count == 2U,
          "campaigns use deterministic newest-first ordering and exact counts");
  require(snapshot.latest_comparison.has_value() &&
              snapshot.latest_comparison->campaign_id == 2 &&
              snapshot.latest_comparison->phase == "confirm" &&
              snapshot.latest_comparison->comparisons.size() == 2U &&
              snapshot.latest_comparison->comparisons[0].arm_name == "fast" &&
              snapshot.latest_comparison->comparisons[0].confirmed &&
              snapshot.latest_comparison->comparisons[1].arm_name == "wide",
          "latest paired comparisons are filtered and delta ordered");
  const auto encoded_snapshot = trainvm::encode_json(snapshot);
  require(encoded_snapshot["api_version"] == snapshot.api_version &&
              encoded_snapshot["latest_comparison"]["comparisons"].size() == 2U &&
              !encoded_snapshot.contains("legacy_comparison"),
          "C++26 reflection emits the versioned registry diagnostic without bespoke fields");

  const auto legacy = trainvm::read_experiment_registry(
      database,
      {.campaign_limit = 2U,
       .task = "legacy-task",
       .metric = "acc",
       .baseline = "baseline"});
  require(!legacy.latest_comparison.has_value() &&
              legacy.legacy_comparison.has_value() &&
              legacy.legacy_comparison->baseline_present &&
              legacy.legacy_comparison->comparisons.size() == 2U &&
              legacy.legacy_comparison->comparisons[0].config == "candidate" &&
              legacy.legacy_comparison->comparisons[0].mean == 0.65 &&
              legacy.legacy_comparison->comparisons[0].seeds == 8U &&
              legacy.legacy_comparison->comparisons[0].delta_from_baseline.has_value() &&
              legacy.legacy_comparison->comparisons[0].significant,
          "legacy fallback keeps the latest row per config and ranks by mean");
  near(*legacy.legacy_comparison->comparisons[0].delta_from_baseline, 0.15,
       1e-15, "legacy baseline delta");
  const auto injection = trainvm::read_experiment_registry(
      database,
      {.campaign_limit = 2U,
       .task = "legacy-task' OR 1=1 --",
       .metric = "acc",
       .baseline = "baseline"});
  require(!injection.latest_comparison.has_value() &&
              !injection.legacy_comparison.has_value(),
          "task values are bound data and cannot alter registry SQL");

  require(sqlite_integer(database, "PRAGMA schema_version") == schema_before &&
              std::filesystem::file_size(database) == size_before &&
              std::filesystem::last_write_time(database) == write_before,
          "native registry analysis never mutates the source database");
  require(!std::filesystem::exists(database.string() + "-journal") &&
              !std::filesystem::exists(database.string() + "-wal") &&
              !std::filesystem::exists(database.string() + "-shm"),
          "read-only analysis creates no SQLite side files");
}

void registry_reader_fails_closed_on_malformed_state() {
  TemporaryDirectory directory;
  const auto empty = directory.file("empty.db");
  sqlite_execute(empty, "CREATE TABLE unrelated(value TEXT);");
  const auto empty_snapshot = trainvm::read_experiment_registry(empty);
  require(empty_snapshot.campaigns.empty() &&
              !empty_snapshot.latest_comparison.has_value() &&
              !empty_snapshot.legacy_comparison.has_value(),
          "legacy or unrelated registries are not modified to add new tables");
  require(sqlite_integer(
              empty,
              "SELECT count(*) FROM sqlite_schema WHERE type='table'") == 1,
          "read-only analysis does not install the Python schema");

  const auto malformed = directory.file("malformed.db");
  sqlite_execute(malformed, R"sql(
    CREATE TABLE campaigns(
      id INTEGER PRIMARY KEY, created_ts REAL, task TEXT, phase TEXT,
      status TEXT, parent_id INTEGER, git_sha TEXT);
    CREATE TABLE arms(id INTEGER PRIMARY KEY, campaign_id INTEGER, name TEXT);
    CREATE TABLE trials(id INTEGER PRIMARY KEY, campaign_id INTEGER);
    CREATE TABLE comparisons(
      id INTEGER PRIMARY KEY, campaign_id INTEGER, arm_id INTEGER, metric TEXT,
      n INTEGER, delta REAL, ci_low REAL, ci_high REAL, p_adjusted REAL,
      effect_size REAL, significant INTEGER, confirmed INTEGER);
    INSERT INTO campaigns VALUES (1,1.0,'task','phase','running',NULL,'sha');
    INSERT INTO arms VALUES (1,1,'arm');
    INSERT INTO comparisons VALUES
      (1,1,1,'acc',2,'not-a-number',0.1,0.2,0.03,1.0,1,0);
  )sql");
  rejected(
      [&] {
        (void)trainvm::read_experiment_registry(
            malformed, {.campaign_limit = 20U, .task = "task", .metric = "acc"});
      },
      "numeric", "malformed comparison types fail closed");

  const auto malformed_legacy = directory.file("malformed-legacy.db");
  sqlite_execute(malformed_legacy, R"sql(
    CREATE TABLE results(
      id INTEGER PRIMARY KEY, ts REAL, git_sha TEXT, task TEXT, config TEXT,
      seeds INTEGER, steps INTEGER, metrics_json TEXT);
    INSERT INTO results VALUES
      (1,1.0,'sha','task','baseline',2,10,'{"acc":[0.5,}');
  )sql");
  rejected(
      [&] {
        (void)trainvm::read_experiment_registry(
            malformed_legacy,
            {.campaign_limit = 20U,
             .task = "task",
             .metric = "acc",
             .baseline = "baseline"});
      },
      "invalid JSON", "malformed legacy metrics fail closed");

  rejected(
      [&] {
        (void)trainvm::read_experiment_registry(
            empty, {.campaign_limit = 0U, .task = std::nullopt, .metric = "acc"});
      },
      "campaign_limit", "unbounded registry query fails closed");
  rejected(
      [&] {
        (void)trainvm::read_experiment_registry(directory.file("missing.db"));
      },
      "could not open", "missing registries are never created");
  require(!std::filesystem::exists(directory.file("missing.db")),
          "failed read-only open leaves no database behind");
}

}  // namespace

int main() {
  try {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"paired statistics", paired_statistics_are_deterministic_and_bounded},
        {"multiple testing", multiple_testing_matches_registered_decisions},
        {"Pareto front", pareto_front_handles_missing_and_mixed_objectives},
        {"registry snapshot", registry_snapshot_is_typed_ordered_and_read_only},
        {"registry rejection", registry_reader_fails_closed_on_malformed_state},
    };
    for (const auto& [name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
    std::cout << "All experiment analysis tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
