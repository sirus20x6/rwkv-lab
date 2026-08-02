#include "trainvm/external_profiler_artifact.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <ranges>
#include <set>
#include <sqlite3.h>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <unistd.h>

#include "trainvm/document.hpp"
#include "trainvm/profiler_launch_profiles.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumWindowBytes = 64U * 1024U;
constexpr std::size_t kMaximumKernelSamples = 1'000'000U;
constexpr std::size_t kMaximumOperatorRows = 256U;
constexpr std::uint64_t kMaximumRawTraceBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumNcuCsvBytes = 512ULL * 1024ULL * 1024ULL;
constexpr double kMaximumWindowMicroseconds = 7.0 * 24.0 * 60.0 * 60.0 *
                                             1'000'000.0;

[[noreturn]] void reject(std::string message) {
  throw ExternalProfilerArtifactError(std::move(message));
}

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0) ::close(value_);
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  [[nodiscard]] int get() const { return value_; }

 private:
  int value_;
};

struct FileDigest final {
  std::string digest;
  std::uint64_t size{};
  struct stat evidence {};
};

bool same_file_evidence(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_size == right.st_size &&
         left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
         left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
         left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
         left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
}

std::string digest_bytes(const unsigned char* value, unsigned int size) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  for (unsigned int index = 0U; index < size; ++index) {
    result.push_back(digits[value[index] >> 4U]);
    result.push_back(digits[value[index] & 0x0fU]);
  }
  return result;
}

FileDigest digest_regular_file(const std::filesystem::path& path,
                               std::uint64_t maximum) {
  Descriptor descriptor(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  struct stat before {};
  if (descriptor.get() < 0 || ::fstat(descriptor.get(), &before) != 0 ||
      !S_ISREG(before.st_mode) || before.st_size <= 0 ||
      static_cast<std::uint64_t>(before.st_size) > maximum)
    reject("external profiler file is absent, unsafe, or outside its bound");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    reject("external profiler file digest initialization failed");
  std::array<unsigned char, 1024U * 1024U> buffer{};
  for (;;) {
    const ssize_t count = ::read(descriptor.get(), buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0 ||
        (count > 0 && EVP_DigestUpdate(context.get(), buffer.data(),
                                      static_cast<std::size_t>(count)) != 1))
      reject("external profiler file digest failed");
    if (count == 0) break;
  }
  struct stat after {};
  if (::fstat(descriptor.get(), &after) != 0 ||
      !same_file_evidence(before, after))
    reject("external profiler file changed while being hashed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest_value{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest_value.data(), &digest_size) != 1 ||
      digest_size != 32U)
    reject("external profiler file digest finalization failed");
  return {.digest = digest_bytes(digest_value.data(), digest_size),
          .size = static_cast<std::uint64_t>(before.st_size),
          .evidence = before};
}

void copy_regular_file(const std::filesystem::path& source,
                       const std::filesystem::path& destination,
                       const FileDigest& expected) {
  Descriptor input(::open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  Descriptor output(::open(destination.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                                                   O_CLOEXEC | O_NOFOLLOW,
                           0400));
  struct stat before {};
  if (input.get() < 0 || output.get() < 0 ||
      ::fstat(input.get(), &before) != 0 ||
      !same_file_evidence(before, expected.evidence))
    reject("external profiler raw trace changed before publication");
  std::array<unsigned char, 1024U * 1024U> buffer{};
  std::uint64_t copied = 0U;
  for (;;) {
    const ssize_t count = ::read(input.get(), buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) reject("external profiler raw trace copy failed");
    if (count == 0) break;
    std::size_t offset = 0U;
    while (offset < static_cast<std::size_t>(count)) {
      const ssize_t written = ::write(
          output.get(), buffer.data() + offset,
          static_cast<std::size_t>(count) - offset);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) reject("external profiler artifact write failed");
      offset += static_cast<std::size_t>(written);
    }
    copied += static_cast<std::uint64_t>(count);
  }
  struct stat after {};
  if (copied != expected.size || ::fstat(input.get(), &after) != 0 ||
      !same_file_evidence(before, after) || ::fsync(output.get()) != 0)
    reject("external profiler raw trace changed during publication");
}

std::string read_bounded_regular_file(const std::filesystem::path& path,
                                      std::size_t maximum) {
  const FileDigest evidence = digest_regular_file(path, maximum);
  Descriptor descriptor(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  struct stat before {};
  if (descriptor.get() < 0 || ::fstat(descriptor.get(), &before) != 0 ||
      !same_file_evidence(before, evidence.evidence))
    reject("external profiler evidence changed before reading");
  std::string result(evidence.size, '\0');
  std::size_t offset = 0U;
  while (offset < result.size()) {
    const ssize_t count =
        ::read(descriptor.get(), result.data() + offset, result.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) reject("external profiler evidence read was incomplete");
    offset += static_cast<std::size_t>(count);
  }
  struct stat after {};
  if (::fstat(descriptor.get(), &after) != 0 ||
      !same_file_evidence(before, after))
    reject("external profiler evidence changed while being read");
  return result;
}

void write_new_file(const std::filesystem::path& path, std::string_view value) {
  Descriptor descriptor(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                                                  O_CLOEXEC | O_NOFOLLOW,
                               0400));
  if (descriptor.get() < 0) reject("external profiler manifest create failed");
  std::size_t offset = 0U;
  while (offset < value.size()) {
    const ssize_t count = ::write(descriptor.get(), value.data() + offset,
                                  value.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) reject("external profiler manifest write failed");
    offset += static_cast<std::size_t>(count);
  }
  if (::fsync(descriptor.get()) != 0)
    reject("external profiler manifest sync failed");
}

void sync_directory(const std::filesystem::path& path) {
  Descriptor descriptor(::open(path.c_str(), O_RDONLY | O_DIRECTORY |
                                                  O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0 || ::fsync(descriptor.get()) != 0)
    reject("external profiler artifact directory sync failed");
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

std::vector<std::string> csv_fields(std::string_view record) {
  std::vector<std::string> result;
  std::size_t offset = 0U;
  while (offset <= record.size()) {
    std::string field;
    if (offset < record.size() && record[offset] == '"') {
      ++offset;
      bool closed = false;
      while (offset < record.size()) {
        if (record[offset] != '"') {
          field.push_back(record[offset++]);
          continue;
        }
        if (offset + 1U < record.size() && record[offset + 1U] == '"') {
          field.push_back('"');
          offset += 2U;
          continue;
        }
        ++offset;
        closed = true;
        break;
      }
      if (!closed || (offset < record.size() && record[offset] != ','))
        reject("NCU CSV contains an invalid quoted field");
    } else {
      const std::size_t end = record.find(',', offset);
      const std::size_t length =
          end == std::string_view::npos ? record.size() - offset
                                        : end - offset;
      field.assign(record.substr(offset, length));
      if (field.contains('"'))
        reject("NCU CSV contains an invalid quote");
      offset = end == std::string_view::npos ? record.size() : end;
    }
    result.push_back(std::move(field));
    if (offset == record.size()) break;
    ++offset;
    if (offset == record.size()) result.emplace_back();
  }
  return result;
}

std::size_t column_index(const std::vector<std::string>& columns,
                         std::string_view name) {
  const auto found = std::ranges::find(columns, name);
  return found == columns.end()
      ? std::numeric_limits<std::size_t>::max()
      : static_cast<std::size_t>(std::distance(columns.begin(), found));
}

double duration_nanoseconds(std::string value, std::string_view unit) {
  value.erase(std::remove(value.begin(), value.end(), ','), value.end());
  std::size_t consumed = 0U;
  double number = 0.0;
  try {
    number = std::stod(value, &consumed);
  } catch (...) {
    reject("NCU CSV duration is not numeric");
  }
  if (consumed != value.size() || !std::isfinite(number) || number < 0.0)
    reject("NCU CSV duration is invalid");
  if (unit == "ns" || unit == "nsecond" || unit == "nanosecond" ||
      unit == "nanoseconds")
    return number;
  if (unit == "us" || unit == "usecond" || unit == "microsecond" ||
      unit == "microseconds")
    return number * 1'000.0;
  if (unit == "ms" || unit == "msecond" || unit == "millisecond" ||
      unit == "milliseconds")
    return number * 1'000'000.0;
  if (unit == "s" || unit == "second" || unit == "seconds")
    return number * 1'000'000'000.0;
  reject("NCU CSV duration unit is unsupported");
}

std::pair<std::optional<double>, std::optional<double>> input_stall_summary(
    const ExternalProfilerWindow& window) {
  if (std::ranges::all_of(window.input_stall_us,
                          [](const auto& value) { return value.has_value(); })) {
    double total = 0.0;
    for (const auto& value : window.input_stall_us) total += *value;
    if (!std::isfinite(total) ||
        total > window.captured_step_wall_time_us * 1.01)
      reject("external profiler input stall exceeds its worker window");
    total = std::min(total, window.captured_step_wall_time_us);
    return {total / window.captured_step_wall_time_us, total};
  }
  if (std::ranges::any_of(window.input_stall_us,
                          [](const auto& value) { return value.has_value(); }))
    reject("external profiler input-stall evidence is incomplete");
  return {std::nullopt, std::nullopt};
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

  const auto [input_stall_ratio, input_stall_time] =
      input_stall_summary(window);

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
      .input_stall_ratio = input_stall_ratio,
      .input_stall_time_us = input_stall_time,
      .top_operators = std::move(rows),
  };
}

ExternalProfilerSummary read_ncu_profiler_summary(
    const std::string& csv_path, const ExternalProfilerWindow& window) {
  if (window.backend != ProfilerBackend::ncu)
    reject("NCU CSV cannot normalize another profiler backend");
  std::ifstream source(csv_path);
  if (!source) reject("could not open NCU CSV output");
  std::vector<std::string> columns;
  std::size_t kernel_index = std::numeric_limits<std::size_t>::max();
  std::size_t metric_index = std::numeric_limits<std::size_t>::max();
  std::size_t unit_index = std::numeric_limits<std::size_t>::max();
  std::size_t value_index = std::numeric_limits<std::size_t>::max();
  std::map<std::string, ExternalProfilerOperatorSummary, std::less<>> grouped;
  std::uint64_t launches = 0U;
  long double accelerator_ns = 0.0L;
  std::string line;
  while (std::getline(source, line)) {
    if (line.size() > 1024U * 1024U)
      reject("NCU CSV row exceeds its bound");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto fields = csv_fields(line);
    if (columns.empty()) {
      const auto candidate_kernel = column_index(fields, "Kernel Name");
      const auto candidate_metric = column_index(fields, "Metric Name");
      const auto candidate_unit = column_index(fields, "Metric Unit");
      const auto candidate_value = column_index(fields, "Metric Value");
      if (candidate_kernel == std::numeric_limits<std::size_t>::max() ||
          candidate_metric == std::numeric_limits<std::size_t>::max() ||
          candidate_unit == std::numeric_limits<std::size_t>::max() ||
          candidate_value == std::numeric_limits<std::size_t>::max())
        continue;
      columns = fields;
      kernel_index = candidate_kernel;
      metric_index = candidate_metric;
      unit_index = candidate_unit;
      value_index = candidate_value;
      continue;
    }
    if (fields == columns) continue;
    if (fields.size() != columns.size())
      reject("NCU CSV data row disagrees with its header");
    const std::string& metric = fields[metric_index];
    if (metric != "gpu__time_duration" &&
        metric != "gpu__time_duration.sum")
      continue;
    const std::string& name = fields[kernel_index];
    if (!bounded_text(name, 512U) || launches == kMaximumKernelSamples)
      reject("NCU CSV kernel rows exceed their bounds");
    const double duration_ns =
        duration_nanoseconds(fields[value_index], fields[unit_index]);
    accelerator_ns += duration_ns;
    auto& row = grouped[name];
    row.name = name;
    ++row.calls;
    row.accelerator_time_us += duration_ns / 1000.0;
    ++launches;
  }
  if (!source.eof() || columns.empty() || launches == 0U)
    reject("NCU CSV contains no bounded kernel-duration evidence");
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
  const double accelerator_us =
      static_cast<double>(accelerator_ns / 1000.0L);
  if (!std::isfinite(accelerator_us))
    reject("NCU CSV accelerator time is invalid");
  const auto [input_stall_ratio, input_stall_time] =
      input_stall_summary(window);
  return {
      .cpu_time_us = 0.0,
      .accelerator_time_us = accelerator_us,
      .kernel_or_operator_count =
          static_cast<std::uint64_t>(grouped.size()),
      .accelerator_launch_count = launches,
      .captured_step_wall_time_us = window.captured_step_wall_time_us,
      .gpu_active_ratio = std::nullopt,
      .gpu_active_time_us = std::nullopt,
      .input_stall_ratio = input_stall_ratio,
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
      {"kernel_or_operator_count", summary.kernel_or_operator_count},
      {"top_operators", std::move(rows)},
  };
  if (summary.gpu_active_ratio) {
    result["gpu_active_ratio"] = *summary.gpu_active_ratio;
    result["gpu_active_time_us"] = *summary.gpu_active_time_us;
  }
  if (summary.input_stall_ratio) {
    result["input_stall_ratio"] = *summary.input_stall_ratio;
    result["input_stall_time_us"] = *summary.input_stall_time_us;
  }
  return result;
}

ExternalProfilerPublishedArtifact publish_external_profiler_artifact(
    const ExternalProfilerPublicationRequest& request) {
  if (request.backend == ProfilerBackend::torch ||
      !bounded_text(request.run_id, 1024U) ||
      !bounded_text(request.node_id, 1024U) ||
      !bounded_text(request.attempt_id, 1024U) ||
      !digest(request.authority_digest) ||
      !digest(request.invocation_digest) || !request.capture.enabled ||
      request.capture.backend != request.backend ||
      !request.capture.warmup_steps || !request.capture.skip_steps ||
      !request.capture.capture_steps || !request.capture.activities ||
      *request.capture.activities !=
          std::vector<ProfilerActivity>{ProfilerActivity::accelerator} ||
      request.capture.record_shapes || request.capture.profile_memory ||
      request.capture.with_stack || request.run_directory.empty() ||
      request.raw_output_prefix.empty())
    reject("external profiler publication authority is incomplete");
  if (*request.capture.warmup_steps < 0 ||
      *request.capture.skip_steps < 0 ||
      *request.capture.capture_steps <= 0)
    reject("external profiler publication schedule is invalid");
  const auto run_directory = request.run_directory.lexically_normal();
  if (!run_directory.is_absolute() || run_directory != request.run_directory)
    reject("external profiler run directory is not canonical");
  std::error_code filesystem_error;
  const auto canonical_run = std::filesystem::canonical(run_directory,
                                                         filesystem_error);
  if (filesystem_error || canonical_run != run_directory)
    reject("external profiler run directory identity is unavailable");
  const std::string launch_id = request.run_id + ":worker-launch:" +
                                request.node_id + ":" + request.attempt_id;
  const auto root = run_directory / "trainvm_artifacts" / "gpu_traces";
  const auto expected_prefix =
      root / ".external" / sha256_hex(launch_id);
  if (request.raw_output_prefix.lexically_normal() != expected_prefix ||
      request.raw_output_prefix != request.raw_output_prefix.lexically_normal())
    reject("external profiler raw prefix disagrees with launch authority");

  const auto window_path =
      std::filesystem::path(request.raw_output_prefix.string() +
                            ".window.json");
  const ExternalProfilerWindow window =
      external_profiler_window_from_canonical_json(
          read_bounded_regular_file(window_path, kMaximumWindowBytes));
  if (window.backend != request.backend || window.run_id != request.run_id ||
      window.node_id != request.node_id ||
      window.attempt_id != request.attempt_id ||
      window.authority_digest != request.authority_digest ||
      window.invocation_digest != request.invocation_digest ||
      window.skip_steps !=
          static_cast<std::uint64_t>(*request.capture.skip_steps) ||
      window.warmup_steps !=
          static_cast<std::uint64_t>(*request.capture.warmup_steps) ||
      window.capture_steps !=
          static_cast<std::uint64_t>(*request.capture.capture_steps))
    reject("external profiler worker window disagrees with host authority");

  std::filesystem::path raw_path;
  std::string trace_file_name;
  ExternalProfilerSummary summary;
  if (request.backend == ProfilerBackend::nsys) {
    raw_path = request.raw_output_prefix.string() + ".sqlite";
    trace_file_name = "trace.sqlite";
    const FileDigest before =
        digest_regular_file(raw_path, kMaximumRawTraceBytes);
    summary = normalize_external_profiler_samples(
        window, read_nsys_profiler_samples(raw_path.string()));
    const FileDigest after =
        digest_regular_file(raw_path, kMaximumRawTraceBytes);
    if (before.digest != after.digest ||
        !same_file_evidence(before.evidence, after.evidence))
      reject("NSYS export changed while being normalized");
  } else if (request.backend == ProfilerBackend::ncu) {
    raw_path = request.raw_output_prefix.string() + ".ncu-rep";
    trace_file_name = "trace.ncu-rep";
    const auto csv_path =
        std::filesystem::path(request.raw_output_prefix.string() + ".csv");
    const FileDigest before =
        digest_regular_file(csv_path, kMaximumNcuCsvBytes);
    summary = read_ncu_profiler_summary(csv_path.string(), window);
    const FileDigest after =
        digest_regular_file(csv_path, kMaximumNcuCsvBytes);
    if (before.digest != after.digest ||
        !same_file_evidence(before.evidence, after.evidence))
      reject("NCU CSV changed while being normalized");
  } else {
    reject("external profiler publication backend is unsupported");
  }
  const FileDigest raw =
      digest_regular_file(raw_path, kMaximumRawTraceBytes);

  nlohmann::json activities = nlohmann::json::array();
  for (const ProfilerActivity activity : *request.capture.activities)
    activities.push_back(activity == ProfilerActivity::cpu ? "cpu"
                                                            : "accelerator");
  nlohmann::json body{
      {"activities", std::move(activities)},
      {"api_version", "trainvm.gpu-trace.v1"},
      {"attempt_id", request.attempt_id},
      {"backend", std::string(profiler_backend_name(request.backend))},
      {"capture_steps", window.capture_steps},
      {"first_optimizer_step", window.optimizer_steps.front()},
      {"instrumented_timing", true},
      {"invocation_digest", request.invocation_digest},
      {"last_optimizer_step", window.optimizer_steps.back()},
      {"node_id", request.node_id},
      {"options",
       {{"profile_memory", false},
        {"record_shapes", false},
        {"with_stack", false}}},
      {"run_id", request.run_id},
      {"sensitivity", "restricted"},
      {"skip_steps", window.skip_steps},
      {"summary", external_profiler_summary_json(summary)},
      {"trace_file_name", trace_file_name},
      {"trace_sha256", raw.digest},
      {"trace_size_bytes", raw.size},
      {"warmup_steps", window.warmup_steps},
  };
  const std::string canonical_manifest_digest =
      "sha256:" + sha256_hex(body.dump());
  body["canonical_manifest_digest"] = canonical_manifest_digest;
  const std::string manifest_bytes = body.dump();
  const std::string artifact_id =
      "gpu-trace-" + sha256_hex(manifest_bytes);
  const auto revision = root / artifact_id;

  std::filesystem::create_directories(root, filesystem_error);
  if (filesystem_error ||
      std::filesystem::canonical(root, filesystem_error) != root ||
      filesystem_error)
    reject("external profiler publication root is unsafe");
  std::filesystem::path temporary;
  for (unsigned int index = 0U; index < 100U; ++index) {
    temporary = root / (".revision-" + artifact_id + "-" +
                        std::to_string(static_cast<long long>(::getpid())) +
                        "-" + std::to_string(index));
    if (std::filesystem::create_directory(temporary, filesystem_error)) break;
    if (filesystem_error && filesystem_error !=
                                std::errc::file_exists)
      reject("external profiler staging directory could not be created");
    filesystem_error.clear();
    temporary.clear();
  }
  if (temporary.empty())
    reject("external profiler staging identities are exhausted");
  try {
    copy_regular_file(raw_path, temporary / trace_file_name, raw);
    write_new_file(temporary / "manifest.json", manifest_bytes);
    sync_directory(temporary);
    std::filesystem::permissions(
        temporary, std::filesystem::perms::owner_read |
                       std::filesystem::perms::owner_exec |
                       std::filesystem::perms::group_read |
                       std::filesystem::perms::group_exec,
        std::filesystem::perm_options::replace, filesystem_error);
    if (filesystem_error)
      reject("external profiler artifact permissions failed");
    std::filesystem::rename(temporary, revision, filesystem_error);
    if (filesystem_error) {
      filesystem_error.clear();
      const auto existing_manifest = revision / "manifest.json";
      if (read_bounded_regular_file(existing_manifest, 4U * 1024U * 1024U) !=
              manifest_bytes ||
          digest_regular_file(revision / trace_file_name,
                              kMaximumRawTraceBytes)
                  .digest != raw.digest)
        reject("external profiler artifact identity already has other bytes");
      std::filesystem::permissions(
          temporary, std::filesystem::perms::owner_all,
          std::filesystem::perm_options::add, filesystem_error);
      if (filesystem_error)
        reject("external profiler replay staging permissions failed");
      std::filesystem::remove_all(temporary, filesystem_error);
      if (filesystem_error)
        reject("external profiler replay staging cleanup failed");
    }
    sync_directory(root);
  } catch (...) {
    if (!temporary.empty() && std::filesystem::exists(temporary)) {
      std::error_code ignored;
      std::filesystem::permissions(
          temporary, std::filesystem::perms::owner_all,
          std::filesystem::perm_options::add, ignored);
      std::filesystem::remove_all(temporary, ignored);
    }
    throw;
  }
  const auto manifest_path = revision / "manifest.json";
  const FileDigest manifest =
      digest_regular_file(manifest_path, 4U * 1024U * 1024U);
  if (manifest.digest != "sha256:" + sha256_hex(manifest_bytes))
    reject("published external profiler manifest is not content-addressed");
  return {.artifact_id = artifact_id,
          .manifest_path = manifest_path,
          .manifest_fingerprint = manifest.digest,
          .size_bytes = raw.size + manifest.size};
}

}  // namespace trainvm
