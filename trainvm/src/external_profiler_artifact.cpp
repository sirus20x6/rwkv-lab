#include "trainvm/external_profiler_artifact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <sqlite3.h>
#include <string>
#include <utility>

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumWindowBytes = 64U * 1024U;
constexpr std::size_t kMaximumKernelSamples = 1'000'000U;
constexpr std::size_t kMaximumOperatorRows = 256U;
constexpr double kMaximumWindowMicroseconds = 7.0 * 24.0 * 60.0 * 60.0 *
                                             1'000'000.0;

[[noreturn]] void reject(std::string message) {
  throw ExternalProfilerArtifactError(std::move(message));
}

bool digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool bounded_text(std::string_view value, std::size_t maximum) {
  return !value.empty() && value.size() <= maximum &&
         !value.contains('\0') && !value.contains('\n') &&
         !value.contains('\r');
}

ProfilerBackend external_backend(std::string_view value) {
  if (value == "nsys") return ProfilerBackend::nsys;
  if (value == "ncu") return ProfilerBackend::ncu;
  reject("external profiler window backend is invalid");
}

void exact_fields(const nlohmann::json& value) {
  constexpr std::array<std::string_view, 14U> fields{
      "api_version",
      "attempt_id",
      "authority_digest",
      "backend",
      "canonical_window_digest",
      "capture_steps",
      "captured_step_wall_time_us",
      "input_stall_us",
      "invocation_digest",
      "node_id",
      "optimizer_steps",
      "run_id",
      "skip_steps",
      "warmup_steps",
  };
  if (!value.is_object() || value.size() != fields.size() ||
      std::ranges::any_of(fields, [&](std::string_view field) {
        return !value.contains(std::string(field));
      })) {
    reject("external profiler window fields are inexact");
  }
}

class Database final {
 public:
  explicit Database(const std::string& path) {
    if (sqlite3_open_v2(path.c_str(), &value_, SQLITE_OPEN_READONLY, nullptr) !=
        SQLITE_OK) {
      const std::string message = value_ ? sqlite3_errmsg(value_)
                                         : "allocation failed";
      if (value_) sqlite3_close(value_);
      value_ = nullptr;
      reject("could not open NSYS SQLite export: " + message);
    }
  }
  ~Database() {
    if (value_) sqlite3_close(value_);
  }
  Database(const Database&) = delete;
  Database& operator=(const Database&) = delete;
  [[nodiscard]] sqlite3* get() const { return value_; }

 private:
  sqlite3* value_{};
};

class Statement final {
 public:
  Statement(sqlite3* database, const std::string& sql) {
    if (sqlite3_prepare_v2(database, sql.c_str(), -1, &value_, nullptr) !=
        SQLITE_OK) {
      reject("could not query NSYS SQLite export: " +
             std::string(sqlite3_errmsg(database)));
    }
  }
  ~Statement() { sqlite3_finalize(value_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;
  [[nodiscard]] sqlite3_stmt* get() const { return value_; }

 private:
  sqlite3_stmt* value_{};
};

std::set<std::string, std::less<>> table_columns(sqlite3* database,
                                                 std::string_view table) {
  Statement query(database, "PRAGMA table_info(" + std::string(table) + ")");
  std::set<std::string, std::less<>> result;
  for (;;) {
    const int status = sqlite3_step(query.get());
    if (status == SQLITE_DONE) break;
    if (status != SQLITE_ROW) reject("could not inspect NSYS kernel schema");
    const unsigned char* raw = sqlite3_column_text(query.get(), 1);
    if (raw == nullptr) reject("NSYS kernel schema has a null column name");
    result.emplace(reinterpret_cast<const char*>(raw));
  }
  return result;
}

std::string string_id(sqlite3* database, std::int64_t id) {
  Statement query(database, "SELECT value FROM StringIds WHERE id=?");
  sqlite3_bind_int64(query.get(), 1, id);
  if (sqlite3_step(query.get()) != SQLITE_ROW) return "<unknown>";
  const unsigned char* raw = sqlite3_column_text(query.get(), 0);
  return raw == nullptr ? "<unknown>"
                        : std::string(reinterpret_cast<const char*>(raw));
}

std::string kernel_name(sqlite3* database, sqlite3_stmt* row) {
  if (sqlite3_column_type(row, 2) == SQLITE_INTEGER)
    return string_id(database, sqlite3_column_int64(row, 2));
  const unsigned char* raw = sqlite3_column_text(row, 2);
  return raw == nullptr ? "<unknown>"
                        : std::string(reinterpret_cast<const char*>(raw));
}

}  // namespace

ExternalProfilerWindow external_profiler_window_from_canonical_json(
    std::string_view value) {
  if (value.empty() || value.size() > kMaximumWindowBytes)
    reject("external profiler window size is invalid");
  try {
    const nlohmann::json parsed = nlohmann::json::parse(value);
    if (parsed.dump() != value)
      reject("external profiler window is not canonical JSON");
    exact_fields(parsed);
    if (parsed.at("api_version").get<std::string>() !=
        kExternalProfilerWindowApiVersion)
      reject("external profiler window API is unsupported");
    nlohmann::json body = parsed;
    body.erase("canonical_window_digest");
    const std::string canonical_digest =
        parsed.at("canonical_window_digest").get<std::string>();
    if (!digest(canonical_digest) ||
        canonical_digest != "sha256:" + sha256_hex(body.dump()))
      reject("external profiler window digest is invalid");

    std::vector<std::optional<double>> input_stall_us;
    const auto& input_stall = parsed.at("input_stall_us");
    if (!input_stall.is_array())
      reject("external profiler input-stall evidence is not an array");
    input_stall_us.reserve(input_stall.size());
    for (const auto& sample : input_stall) {
      if (sample.is_null()) {
        input_stall_us.push_back(std::nullopt);
      } else if (sample.is_number()) {
        input_stall_us.push_back(sample.get<double>());
      } else {
        reject("external profiler input-stall sample is invalid");
      }
    }

    ExternalProfilerWindow result{
        .backend = external_backend(parsed.at("backend").get<std::string>()),
        .run_id = parsed.at("run_id").get<std::string>(),
        .node_id = parsed.at("node_id").get<std::string>(),
        .attempt_id = parsed.at("attempt_id").get<std::string>(),
        .authority_digest =
            parsed.at("authority_digest").get<std::string>(),
        .invocation_digest =
            parsed.at("invocation_digest").get<std::string>(),
        .skip_steps = parsed.at("skip_steps").get<std::uint64_t>(),
        .warmup_steps = parsed.at("warmup_steps").get<std::uint64_t>(),
        .capture_steps = parsed.at("capture_steps").get<std::uint64_t>(),
        .captured_step_wall_time_us =
            parsed.at("captured_step_wall_time_us").get<double>(),
        .optimizer_steps =
            parsed.at("optimizer_steps").get<std::vector<std::uint64_t>>(),
        .input_stall_us = std::move(input_stall_us),
        .canonical_window_digest = canonical_digest,
    };
    if (!bounded_text(result.run_id, 1024U) ||
        !bounded_text(result.node_id, 1024U) ||
        !bounded_text(result.attempt_id, 1024U) ||
        !digest(result.authority_digest) ||
        !digest(result.invocation_digest) || result.capture_steps == 0U ||
        result.capture_steps > 128U || result.skip_steps > 256U ||
        result.warmup_steps > 256U ||
        result.skip_steps + result.warmup_steps + result.capture_steps >
            512U ||
        result.optimizer_steps.size() != result.capture_steps ||
        result.input_stall_us.size() != result.capture_steps ||
        !std::ranges::is_sorted(result.optimizer_steps) ||
        std::ranges::adjacent_find(result.optimizer_steps) !=
            result.optimizer_steps.end() ||
        !std::isfinite(result.captured_step_wall_time_us) ||
        result.captured_step_wall_time_us <= 0.0 ||
        result.captured_step_wall_time_us > kMaximumWindowMicroseconds ||
        std::ranges::any_of(
            result.input_stall_us, [](const std::optional<double>& sample) {
              return sample && (!std::isfinite(*sample) || *sample < 0.0);
            })) {
      reject("external profiler window semantics are invalid");
    }
    return result;
  } catch (const ExternalProfilerArtifactError&) {
    throw;
  } catch (...) {
    reject("external profiler window decoding failed closed");
  }
}

ExternalProfilerSummary normalize_external_profiler_samples(
    const ExternalProfilerWindow& window,
    std::vector<ExternalProfilerKernelSample> samples) {
  if (samples.empty() || samples.size() > kMaximumKernelSamples)
    reject("external profiler kernel sample count is invalid");
  std::ranges::sort(samples, {},
                    &ExternalProfilerKernelSample::start_ns);
  std::map<std::string, ExternalProfilerOperatorSummary, std::less<>> grouped;
  long double accelerator_ns = 0.0L;
  long double active_ns = 0.0L;
  std::uint64_t union_begin = 0U;
  std::uint64_t union_end = 0U;
  bool union_started = false;
  for (const auto& sample : samples) {
    if (!bounded_text(sample.name, 512U) || sample.end_ns <= sample.start_ns)
      reject("external profiler kernel sample is invalid");
    const std::uint64_t duration = sample.end_ns - sample.start_ns;
    accelerator_ns += static_cast<long double>(duration);
    auto& row = grouped[sample.name];
    row.name = sample.name;
    ++row.calls;
    row.accelerator_time_us += static_cast<double>(duration) / 1000.0;
    if (!union_started) {
      union_begin = sample.start_ns;
      union_end = sample.end_ns;
      union_started = true;
    } else if (sample.start_ns <= union_end) {
      union_end = std::max(union_end, sample.end_ns);
    } else {
      active_ns += static_cast<long double>(union_end - union_begin);
      union_begin = sample.start_ns;
      union_end = sample.end_ns;
    }
  }
  if (union_started)
    active_ns += static_cast<long double>(union_end - union_begin);
  const double active_us = static_cast<double>(active_ns / 1000.0L);
  if (!std::isfinite(active_us) ||
      active_us > window.captured_step_wall_time_us * 1.01)
    reject("external profiler active time exceeds its worker window");

  std::vector<ExternalProfilerOperatorSummary> rows;
  rows.reserve(grouped.size());
  for (auto& [name, row] : grouped) {
    (void)name;
    rows.push_back(std::move(row));
  }
  std::ranges::sort(rows, [](const auto& left, const auto& right) {
    if (left.accelerator_time_us != right.accelerator_time_us)
      return left.accelerator_time_us > right.accelerator_time_us;
    return left.name < right.name;
  });
  if (rows.size() > kMaximumOperatorRows) rows.resize(kMaximumOperatorRows);

  std::optional<double> input_stall_time;
  if (std::ranges::all_of(window.input_stall_us,
                          [](const auto& value) { return value.has_value(); })) {
    double total = 0.0;
    for (const auto& value : window.input_stall_us) total += *value;
    if (!std::isfinite(total) ||
        total > window.captured_step_wall_time_us * 1.01)
      reject("external profiler input stall exceeds its worker window");
    input_stall_time = std::min(total, window.captured_step_wall_time_us);
  } else if (std::ranges::any_of(
                 window.input_stall_us,
                 [](const auto& value) { return value.has_value(); })) {
    reject("external profiler input-stall evidence is incomplete");
  }

  const double accelerator_us =
      static_cast<double>(accelerator_ns / 1000.0L);
  if (!std::isfinite(accelerator_us))
    reject("external profiler accelerator time is invalid");
  return {
      .cpu_time_us = 0.0,
      .accelerator_time_us = accelerator_us,
      .kernel_or_operator_count =
          static_cast<std::uint64_t>(grouped.size()),
      .accelerator_launch_count =
          static_cast<std::uint64_t>(samples.size()),
      .captured_step_wall_time_us = window.captured_step_wall_time_us,
      .gpu_active_ratio =
          std::min(1.0, active_us / window.captured_step_wall_time_us),
      .gpu_active_time_us = active_us,
      .input_stall_ratio =
          input_stall_time
              ? std::optional<double>(*input_stall_time /
                                      window.captured_step_wall_time_us)
              : std::nullopt,
      .input_stall_time_us = input_stall_time,
      .top_operators = std::move(rows),
  };
}

std::vector<ExternalProfilerKernelSample> read_nsys_profiler_samples(
    const std::string& sqlite_path) {
  Database database(sqlite_path);
  const auto columns =
      table_columns(database.get(), "CUPTI_ACTIVITY_KIND_KERNEL");
  if (!columns.contains("start") || !columns.contains("end"))
    reject("NSYS SQLite export has no bounded kernel timeline");
  const std::string name_column =
      columns.contains("demangledName")
          ? "demangledName"
          : columns.contains("shortName") ? "shortName"
                                           : columns.contains("name")
                                                 ? "name"
                                                 : std::string{};
  if (name_column.empty())
    reject("NSYS SQLite export has no kernel identity column");
  Statement query(database.get(),
                  "SELECT start,end," + name_column +
                      " FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start");
  std::vector<ExternalProfilerKernelSample> result;
  for (;;) {
    const int status = sqlite3_step(query.get());
    if (status == SQLITE_DONE) break;
    if (status != SQLITE_ROW)
      reject("could not read NSYS kernel activity rows");
    if (result.size() == kMaximumKernelSamples)
      reject("NSYS kernel activity exceeds its row bound");
    const sqlite3_int64 start = sqlite3_column_int64(query.get(), 0);
    const sqlite3_int64 end = sqlite3_column_int64(query.get(), 1);
    if (start < 0 || end <= start)
      reject("NSYS kernel activity has an invalid interval");
    result.push_back({
        .name = kernel_name(database.get(), query.get()),
        .start_ns = static_cast<std::uint64_t>(start),
        .end_ns = static_cast<std::uint64_t>(end),
    });
  }
  if (result.empty()) reject("NSYS capture contains no kernel activity");
  return result;
}

nlohmann::json external_profiler_summary_json(
    const ExternalProfilerSummary& summary) {
  nlohmann::json rows = nlohmann::json::array();
  for (const auto& row : summary.top_operators) {
    rows.push_back({{"accelerator_time_us", row.accelerator_time_us},
                    {"calls", row.calls},
                    {"cpu_time_us", row.cpu_time_us},
                    {"name", row.name}});
  }
  nlohmann::json result{
      {"accelerator_launch_count", summary.accelerator_launch_count},
      {"accelerator_time_us", summary.accelerator_time_us},
      {"captured_step_wall_time_us",
       summary.captured_step_wall_time_us},
      {"cpu_time_us", summary.cpu_time_us},
      {"gpu_active_ratio", summary.gpu_active_ratio},
      {"gpu_active_time_us", summary.gpu_active_time_us},
      {"kernel_or_operator_count", summary.kernel_or_operator_count},
      {"top_operators", std::move(rows)},
  };
  if (summary.input_stall_ratio) {
    result["input_stall_ratio"] = *summary.input_stall_ratio;
    result["input_stall_time_us"] = *summary.input_stall_time_us;
  }
  return result;
}

}  // namespace trainvm
