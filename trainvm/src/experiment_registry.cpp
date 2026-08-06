#include "trainvm/experiment_analysis.hpp"

#include <sqlite3.h>

#include "trainvm/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumCampaigns = 1'000U;
constexpr std::size_t kMaximumComparisons = 10'000U;
constexpr std::size_t kMaximumLegacyResults = 100'000U;
constexpr std::size_t kMaximumTextBytes = 16'384U;
constexpr std::size_t kMaximumMetricsJsonBytes = 1'048'576U;
constexpr int kMaximumProgressCallbacks = 10'000;

struct DatabaseCloser {
  void operator()(sqlite3* database) const noexcept {
    if (database != nullptr) (void)sqlite3_close_v2(database);
  }
};

struct StatementCloser {
  void operator()(sqlite3_stmt* statement) const noexcept {
    if (statement != nullptr) (void)sqlite3_finalize(statement);
  }
};

using Database = std::unique_ptr<sqlite3, DatabaseCloser>;
using Statement = std::unique_ptr<sqlite3_stmt, StatementCloser>;

[[noreturn]] void sqlite_error(sqlite3* database, std::string_view context) {
  throw ExperimentAnalysisError(std::string(context) + ": " +
                                (database == nullptr ? "unknown SQLite error"
                                                     : sqlite3_errmsg(database)));
}

void execute(sqlite3* database, const char* sql, std::string_view context) {
  char* raw_message = nullptr;
  const int status = sqlite3_exec(database, sql, nullptr, nullptr, &raw_message);
  if (status == SQLITE_OK) return;
  const std::string message = raw_message == nullptr ? sqlite3_errmsg(database)
                                                      : raw_message;
  sqlite3_free(raw_message);
  throw ExperimentAnalysisError(std::string(context) + ": " + message);
}

Statement prepare(sqlite3* database, const char* sql, std::string_view context) {
  sqlite3_stmt* raw = nullptr;
  const char* tail = nullptr;
  if (sqlite3_prepare_v3(database, sql, -1, SQLITE_PREPARE_PERSISTENT, &raw,
                         &tail) != SQLITE_OK) {
    sqlite_error(database, context);
  }
  Statement statement(raw);
  if (tail == nullptr || *tail != '\0') {
    throw ExperimentAnalysisError(std::string(context) +
                                  ": internal SQL contains trailing statements");
  }
  return statement;
}

void bind_text(sqlite3* database, sqlite3_stmt* statement, int index,
               const std::string& value, std::string_view context) {
  if (sqlite3_bind_text64(statement, index, value.data(),
                          static_cast<sqlite3_uint64>(value.size()),
                          SQLITE_TRANSIENT, SQLITE_UTF8) != SQLITE_OK) {
    sqlite_error(database, context);
  }
}

void bind_size(sqlite3* database, sqlite3_stmt* statement, int index,
               std::size_t value, std::string_view context) {
  if (value > static_cast<std::size_t>(std::numeric_limits<sqlite3_int64>::max())) {
    throw ExperimentAnalysisError(std::string(context) + ": integer is out of range");
  }
  if (sqlite3_bind_int64(statement, index, static_cast<sqlite3_int64>(value)) !=
      SQLITE_OK) {
    sqlite_error(database, context);
  }
}

std::int64_t required_integer(sqlite3_stmt* statement, int column,
                              std::string_view field) {
  if (sqlite3_column_type(statement, column) != SQLITE_INTEGER) {
    throw ExperimentAnalysisError(std::string(field) + " must be an integer");
  }
  return sqlite3_column_int64(statement, column);
}

std::size_t required_count(sqlite3_stmt* statement, int column,
                           std::string_view field) {
  const auto value = required_integer(statement, column, field);
  if (value < 0) {
    throw ExperimentAnalysisError(std::string(field) + " must not be negative");
  }
  return static_cast<std::size_t>(value);
}

double required_number(sqlite3_stmt* statement, int column,
                       std::string_view field) {
  const int type = sqlite3_column_type(statement, column);
  if (type != SQLITE_FLOAT && type != SQLITE_INTEGER) {
    throw ExperimentAnalysisError(std::string(field) + " must be numeric");
  }
  const double value = sqlite3_column_double(statement, column);
  if (!std::isfinite(value)) {
    throw ExperimentAnalysisError(std::string(field) + " must be finite");
  }
  return value;
}

std::string bounded_text(sqlite3_stmt* statement, int column,
                         std::string_view field, bool allow_empty,
                         std::size_t maximum_bytes) {
  if (sqlite3_column_type(statement, column) != SQLITE_TEXT) {
    throw ExperimentAnalysisError(std::string(field) + " must be text");
  }
  const auto bytes = sqlite3_column_bytes(statement, column);
  if (bytes < 0 || static_cast<std::size_t>(bytes) > maximum_bytes) {
    throw ExperimentAnalysisError(std::string(field) + " exceeds the text limit");
  }
  const auto* raw = sqlite3_column_text(statement, column);
  if (raw == nullptr) {
    throw ExperimentAnalysisError(std::string(field) + " is unreadable");
  }
  std::string result(reinterpret_cast<const char*>(raw),
                     static_cast<std::size_t>(bytes));
  if (result.find('\0') != std::string::npos) {
    throw ExperimentAnalysisError(std::string(field) + " contains an embedded NUL");
  }
  if (!allow_empty && result.empty()) {
    throw ExperimentAnalysisError(std::string(field) + " must not be empty");
  }
  return result;
}

std::string required_text(sqlite3_stmt* statement, int column,
                          std::string_view field, bool allow_empty = false) {
  return bounded_text(statement, column, field, allow_empty, kMaximumTextBytes);
}

std::string nullable_text(sqlite3_stmt* statement, int column,
                          std::string_view field) {
  if (sqlite3_column_type(statement, column) == SQLITE_NULL) return {};
  return required_text(statement, column, field, true);
}

bool required_boolean(sqlite3_stmt* statement, int column,
                      std::string_view field) {
  const auto value = required_integer(statement, column, field);
  if (value != 0 && value != 1) {
    throw ExperimentAnalysisError(std::string(field) + " must be zero or one");
  }
  return value == 1;
}

void require_done(sqlite3* database, sqlite3_stmt* statement,
                  std::string_view context) {
  const int status = sqlite3_step(statement);
  if (status != SQLITE_DONE) sqlite_error(database, context);
}

bool table_exists(sqlite3* database, const std::string& table) {
  auto statement = prepare(database,
      "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?1 LIMIT 1",
      "could not inspect experiment registry schema");
  bind_text(database, statement.get(), 1, table,
            "could not bind experiment registry table name");
  const int status = sqlite3_step(statement.get());
  if (status == SQLITE_ROW) {
    require_done(database, statement.get(),
                 "could not finish experiment registry schema inspection");
    return true;
  }
  if (status != SQLITE_DONE) {
    sqlite_error(database, "could not inspect experiment registry schema");
  }
  return false;
}

int read_only_authorizer(void*, int action, const char*, const char*, const char*,
                         const char*) {
  switch (action) {
    case SQLITE_SELECT:
    case SQLITE_READ:
    case SQLITE_FUNCTION:
    case SQLITE_TRANSACTION:
    case SQLITE_RECURSIVE:
      return SQLITE_OK;
    default:
      return SQLITE_DENY;
  }
}

int operation_budget(void* context) {
  auto* remaining = static_cast<int*>(context);
  --*remaining;
  return *remaining <= 0 ? 1 : 0;
}

Database open_read_only(const std::filesystem::path& path) {
  if (path.empty()) {
    throw ExperimentAnalysisError("experiment registry path must not be empty");
  }
  sqlite3* raw = nullptr;
  const std::string native = path.string();
  const int status = sqlite3_open_v2(native.c_str(), &raw,
                                     SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX,
                                     nullptr);
  Database database(raw);
  if (status != SQLITE_OK) sqlite_error(database.get(), "could not open experiment registry");
  if (sqlite3_db_readonly(database.get(), "main") != 1) {
    throw ExperimentAnalysisError("experiment registry did not open read-only");
  }
  if (sqlite3_extended_result_codes(database.get(), 1) != SQLITE_OK ||
      sqlite3_busy_timeout(database.get(), 1'000) != SQLITE_OK) {
    sqlite_error(database.get(), "could not configure experiment registry reader");
  }
  execute(database.get(), "PRAGMA query_only=ON", "could not enforce read-only query mode");
  if (sqlite3_set_authorizer(database.get(), read_only_authorizer, nullptr) != SQLITE_OK) {
    sqlite_error(database.get(), "could not install experiment registry read authorizer");
  }
  return database;
}

std::vector<CampaignSummary> read_campaigns(sqlite3* database,
                                            std::size_t limit) {
  if (!table_exists(database, "campaigns")) return {};
  auto statement = prepare(
      database,
      "SELECT c.id,c.created_ts,c.task,c.phase,c.status,c.parent_id,c.git_sha,"
      "count(DISTINCT a.id),count(DISTINCT t.id) "
      "FROM campaigns c LEFT JOIN arms a ON a.campaign_id=c.id "
      "LEFT JOIN trials t ON t.campaign_id=c.id GROUP BY c.id "
      "ORDER BY c.created_ts DESC,c.id DESC LIMIT ?1",
      "could not query experiment campaigns");
  bind_size(database, statement.get(), 1, limit,
            "could not bind experiment campaign limit");
  std::vector<CampaignSummary> campaigns;
  campaigns.reserve(limit);
  while (true) {
    const int status = sqlite3_step(statement.get());
    if (status == SQLITE_DONE) break;
    if (status != SQLITE_ROW) sqlite_error(database, "could not read experiment campaigns");
    const auto parent_type = sqlite3_column_type(statement.get(), 5);
    std::optional<std::int64_t> parent;
    if (parent_type == SQLITE_INTEGER) {
      parent = sqlite3_column_int64(statement.get(), 5);
      if (*parent <= 0) {
        throw ExperimentAnalysisError("campaign parent_id must be positive when present");
      }
    } else if (parent_type != SQLITE_NULL) {
      throw ExperimentAnalysisError("campaign parent_id must be an integer or null");
    }
    const auto id = required_integer(statement.get(), 0, "campaign id");
    if (id <= 0) throw ExperimentAnalysisError("campaign id must be positive");
    campaigns.push_back({
        .id = id,
        .created_ts = required_number(statement.get(), 1, "campaign created_ts"),
        .task = required_text(statement.get(), 2, "campaign task"),
        .phase = required_text(statement.get(), 3, "campaign phase"),
        .status = required_text(statement.get(), 4, "campaign status"),
        .parent_id = parent,
        .git_sha = nullable_text(statement.get(), 6, "campaign git_sha"),
        .arm_count = required_count(statement.get(), 7, "campaign arm count"),
        .trial_count = required_count(statement.get(), 8, "campaign trial count"),
    });
  }
  return campaigns;
}

std::optional<CampaignComparisonReport> read_latest_comparison(
    sqlite3* database, const std::string& task, const std::string& metric) {
  if (!table_exists(database, "campaigns") ||
      !table_exists(database, "comparisons") ||
      !table_exists(database, "arms")) {
    return std::nullopt;
  }
  auto latest = prepare(
      database,
      "SELECT id,phase FROM campaigns WHERE task=?1 "
      "ORDER BY created_ts DESC,id DESC LIMIT 1",
      "could not query latest experiment campaign");
  bind_text(database, latest.get(), 1, task,
            "could not bind experiment task");
  const int latest_status = sqlite3_step(latest.get());
  if (latest_status == SQLITE_DONE) return std::nullopt;
  if (latest_status != SQLITE_ROW) {
    sqlite_error(database, "could not read latest experiment campaign");
  }
  const auto campaign_id = required_integer(latest.get(), 0, "campaign id");
  if (campaign_id <= 0) {
    throw ExperimentAnalysisError("campaign id must be positive");
  }
  const auto phase = required_text(latest.get(), 1, "campaign phase");
  require_done(database, latest.get(), "could not finish latest campaign query");

  auto comparisons = prepare(
      database,
      "SELECT a.name,p.n,p.delta,p.ci_low,p.ci_high,p.p_adjusted,"
      "p.effect_size,p.significant,p.confirmed FROM comparisons p "
      "JOIN arms a ON a.id=p.arm_id WHERE p.campaign_id=?1 AND p.metric=?2 "
      "ORDER BY p.delta DESC,a.name ASC LIMIT ?3",
      "could not query experiment comparisons");
  if (sqlite3_bind_int64(comparisons.get(), 1, campaign_id) != SQLITE_OK) {
    sqlite_error(database, "could not bind experiment campaign id");
  }
  bind_text(database, comparisons.get(), 2, metric,
            "could not bind experiment metric");
  bind_size(database, comparisons.get(), 3, kMaximumComparisons + 1U,
            "could not bind experiment comparison limit");
  CampaignComparisonReport report{
      .campaign_id = campaign_id,
      .task = task,
      .phase = phase,
      .metric = metric,
      .comparisons = {},
  };
  while (true) {
    const int status = sqlite3_step(comparisons.get());
    if (status == SQLITE_DONE) break;
    if (status != SQLITE_ROW) {
      sqlite_error(database, "could not read experiment comparisons");
    }
    if (report.comparisons.size() == kMaximumComparisons) {
      throw ExperimentAnalysisError("experiment comparison count exceeds limit");
    }
    report.comparisons.push_back({
        .arm_name = required_text(comparisons.get(), 0, "comparison arm name"),
        .n = required_count(comparisons.get(), 1, "comparison n"),
        .delta = required_number(comparisons.get(), 2, "comparison delta"),
        .ci_low = required_number(comparisons.get(), 3, "comparison ci_low"),
        .ci_high = required_number(comparisons.get(), 4, "comparison ci_high"),
        .p_adjusted = required_number(comparisons.get(), 5, "comparison p_adjusted"),
        .effect_size = required_number(comparisons.get(), 6, "comparison effect_size"),
        .significant = required_boolean(comparisons.get(), 7,
                                        "comparison significant"),
        .confirmed = required_boolean(comparisons.get(), 8,
                                      "comparison confirmed"),
    });
    const auto& value = report.comparisons.back();
    if (value.n == 0U || value.ci_low > value.ci_high ||
        value.p_adjusted < 0.0 || value.p_adjusted > 1.0) {
      throw ExperimentAnalysisError("experiment comparison violates statistical invariants");
    }
  }
  if (report.comparisons.empty()) return std::nullopt;
  return report;
}

struct LegacyRow {
  double timestamp{};
  std::string git_sha;
  std::size_t seeds{};
  std::optional<std::pair<double, double>> metric;
};

std::optional<std::pair<double, double>> parse_legacy_metric(
    const std::string& encoded, const std::string& metric) {
  nlohmann::json document;
  try {
    document = nlohmann::json::parse(encoded);
  } catch (const nlohmann::json::exception& error) {
    throw ExperimentAnalysisError("legacy metrics_json is invalid JSON: " +
                                  std::string(error.what()));
  }
  if (!document.is_object()) {
    throw ExperimentAnalysisError("legacy metrics_json must be an object");
  }
  const auto found = document.find(metric);
  if (found == document.end()) return std::nullopt;
  if (!found->is_array() || found->size() < 2U ||
      !(*found)[0].is_number() || !(*found)[1].is_number()) {
    throw ExperimentAnalysisError("legacy metric must be a [mean, std] numeric array");
  }
  const double metric_mean = (*found)[0].get<double>();
  const double deviation = (*found)[1].get<double>();
  if (!std::isfinite(metric_mean) || !std::isfinite(deviation) || deviation < 0.0) {
    throw ExperimentAnalysisError("legacy metric mean/std must be finite and std non-negative");
  }
  return std::pair{metric_mean, deviation};
}

std::optional<LegacyComparisonReport> read_legacy_comparison(
    sqlite3* database, const std::string& task, const std::string& metric,
    const std::string& baseline) {
  if (!table_exists(database, "results")) return std::nullopt;
  auto statement = prepare(
      database,
      "SELECT ts,git_sha,config,seeds,metrics_json FROM results "
      "WHERE task=?1 ORDER BY ts ASC,id ASC LIMIT ?2",
      "could not query legacy experiment results");
  bind_text(database, statement.get(), 1, task,
            "could not bind legacy experiment task");
  bind_size(database, statement.get(), 2, kMaximumLegacyResults + 1U,
            "could not bind legacy result limit");
  std::map<std::string, LegacyRow> latest;
  std::size_t row_count{};
  while (true) {
    const int status = sqlite3_step(statement.get());
    if (status == SQLITE_DONE) break;
    if (status != SQLITE_ROW) {
      sqlite_error(database, "could not read legacy experiment results");
    }
    if (row_count++ == kMaximumLegacyResults) {
      throw ExperimentAnalysisError("legacy experiment result count exceeds limit");
    }
    const auto config = required_text(statement.get(), 2, "legacy config");
    const auto seeds = required_count(statement.get(), 3, "legacy seeds");
    if (seeds == 0U) {
      throw ExperimentAnalysisError("legacy seeds must be positive");
    }
    const auto metrics_json = bounded_text(statement.get(), 4,
                                            "legacy metrics_json", false,
                                            kMaximumMetricsJsonBytes);
    latest[config] = {
        .timestamp = required_number(statement.get(), 0, "legacy timestamp"),
        .git_sha = nullable_text(statement.get(), 1, "legacy git_sha"),
        .seeds = seeds,
        .metric = parse_legacy_metric(metrics_json, metric),
    };
  }
  LegacyComparisonReport report{
      .task = task,
      .metric = metric,
      .baseline = baseline,
      .baseline_present = false,
      .comparisons = {},
  };
  const auto baseline_row = latest.find(baseline);
  const std::optional<std::pair<double, double>> baseline_metric =
      baseline_row == latest.end() ? std::nullopt : baseline_row->second.metric;
  report.baseline_present = baseline_metric.has_value();
  for (const auto& [config, row] : latest) {
    if (!row.metric.has_value()) continue;
    std::optional<double> delta;
    bool significant = false;
    if (baseline_metric.has_value() && config != baseline) {
      delta = row.metric->first - baseline_metric->first;
      significant = std::abs(*delta) >
                    row.metric->second + baseline_metric->second;
    }
    report.comparisons.push_back({
        .config = config,
        .mean = row.metric->first,
        .standard_deviation = row.metric->second,
        .git_sha = row.git_sha,
        .timestamp = row.timestamp,
        .seeds = row.seeds,
        .delta_from_baseline = delta,
        .significant = significant,
    });
  }
  std::ranges::sort(report.comparisons, {},
                    [](const LegacyAggregateComparison& comparison) {
                      return std::pair{-comparison.mean, comparison.config};
                    });
  if (report.comparisons.empty()) return std::nullopt;
  return report;
}

}  // namespace

ExperimentRegistrySnapshot read_experiment_registry(
    const std::filesystem::path& database_path,
    const ExperimentRegistryQuery& query) {
  if (query.campaign_limit == 0U || query.campaign_limit > kMaximumCampaigns) {
    throw ExperimentAnalysisError("campaign_limit must be in [1, 1000]");
  }
  if (query.metric.empty() || query.metric.size() > 256U ||
      query.metric.find('\0') != std::string::npos) {
    throw ExperimentAnalysisError("metric must be a non-empty bounded string");
  }
  if (query.baseline.empty() || query.baseline.size() > 1'024U ||
      query.baseline.find('\0') != std::string::npos) {
    throw ExperimentAnalysisError("baseline must be a non-empty bounded string");
  }
  if (query.task.has_value() &&
      (query.task->empty() || query.task->size() > 1'024U ||
       query.task->find('\0') != std::string::npos)) {
    throw ExperimentAnalysisError("task must be a non-empty bounded string when present");
  }
  auto database = open_read_only(database_path);
  int operation_count = kMaximumProgressCallbacks;
  sqlite3_progress_handler(database.get(), 1'000, operation_budget,
                           &operation_count);
  execute(database.get(), "BEGIN", "could not begin experiment registry snapshot");
  try {
    ExperimentRegistrySnapshot snapshot{
        .api_version = "trainvm.experiment-registry-snapshot/v1",
        .source = std::filesystem::absolute(database_path)
                      .lexically_normal()
                      .generic_string(),
        .campaigns = read_campaigns(database.get(), query.campaign_limit),
        .latest_comparison = std::nullopt,
        .legacy_comparison = std::nullopt,
    };
    if (query.task.has_value()) {
      snapshot.latest_comparison = read_latest_comparison(
          database.get(), *query.task, query.metric);
      if (!snapshot.latest_comparison.has_value()) {
        snapshot.legacy_comparison = read_legacy_comparison(
            database.get(), *query.task, query.metric, query.baseline);
      }
    }
    execute(database.get(), "COMMIT", "could not finish experiment registry snapshot");
    return snapshot;
  } catch (...) {
    sqlite3_set_authorizer(database.get(), nullptr, nullptr);
    (void)sqlite3_exec(database.get(), "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}

}  // namespace trainvm
