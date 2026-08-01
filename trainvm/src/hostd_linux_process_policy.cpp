#include "trainvm/hostd_linux_process_policy.hpp"

#include <charconv>
#include <ranges>
#include <set>
#include <utility>
#include <vector>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

using Json = nlohmann::json;

[[noreturn]] void reject(std::string message) {
  throw LinuxProcessPolicyError(std::move(message));
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::int64_t parse_cpu(std::string_view text) {
  std::int64_t result = -1;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size() || result < 0 ||
      result > 1'048'575) {
    reject("CPU-list entry is malformed or out of range");
  }
  return result;
}

using CpuRange = std::pair<std::int64_t, std::int64_t>;

std::vector<CpuRange> parse_ranges(std::string_view source) {
  if (source.empty() || source.size() > 4096U)
    reject("CPU list is empty or unbounded");
  std::vector<CpuRange> ranges;
  std::size_t offset = 0U;
  while (offset < source.size()) {
    const std::size_t comma = source.find(',', offset);
    const std::string_view item = source.substr(
        offset, comma == std::string_view::npos ? source.size() - offset
                                                : comma - offset);
    if (item.empty()) reject("CPU list contains an empty range");
    const std::size_t dash = item.find('-');
    if (dash != std::string_view::npos &&
        item.find('-', dash + 1U) != std::string_view::npos)
      reject("CPU list range has more than one separator");
    const std::int64_t first =
        parse_cpu(dash == std::string_view::npos ? item
                                                 : item.substr(0U, dash));
    const std::int64_t last =
        dash == std::string_view::npos ? first : parse_cpu(item.substr(dash + 1U));
    if (first > last) reject("CPU list range is descending");
    if (!ranges.empty() && first <= ranges.back().second + 1)
      reject("CPU list ranges overlap or are adjacent");
    ranges.emplace_back(first, last);
    if (comma == std::string_view::npos) break;
    if (comma + 1U == source.size()) reject("CPU list has a trailing separator");
    offset = comma + 1U;
  }
  return ranges;
}

std::vector<CpuRange> vector_ranges(const std::vector<std::int64_t>& cpus) {
  if (cpus.empty() || cpus.size() > 1024U ||
      !std::ranges::is_sorted(cpus) ||
      std::ranges::adjacent_find(cpus) != cpus.end() ||
      std::ranges::any_of(cpus, [](std::int64_t cpu) {
        return cpu < 0 || cpu > 1'048'575;
      })) {
    reject("structured CPU list is not sorted, unique, and bounded");
  }
  std::vector<CpuRange> ranges;
  for (const std::int64_t cpu : cpus) {
    if (!ranges.empty() && cpu == ranges.back().second + 1) {
      ranges.back().second = cpu;
    } else {
      ranges.emplace_back(cpu, cpu);
    }
  }
  return ranges;
}

std::string encode_ranges(const std::vector<CpuRange>& ranges) {
  std::string result;
  for (const auto& [first, last] : ranges) {
    if (!result.empty()) result.push_back(',');
    result += std::to_string(first);
    if (last != first) result += "-" + std::to_string(last);
  }
  return result;
}

template <typename Value>
void bounded(const std::optional<Value>& value, std::string_view name,
             Value minimum, Value maximum) {
  if (value && (*value < minimum || *value > maximum))
    reject(std::string(name) + " is outside its declared bound");
}

Json policy_body(const LinuxProcessPolicy& policy) {
  return {{"api_version", policy.api_version},
          {"cpuset", policy.cpuset ? Json(*policy.cpuset) : Json(nullptr)},
          {"cpu_weight",
           policy.cpu_weight ? Json(*policy.cpu_weight) : Json(nullptr)},
          {"io_weight", policy.io_weight ? Json(*policy.io_weight) : Json(nullptr)},
          {"nice", policy.nice ? Json(*policy.nice) : Json(nullptr)},
          {"omp_threads",
           policy.omp_threads ? Json(*policy.omp_threads) : Json(nullptr)},
          {"preprocessing_workers",
           policy.preprocessing_workers ? Json(*policy.preprocessing_workers)
                                        : Json(nullptr)}};
}

std::string digest(const LinuxProcessPolicy& policy) {
  std::string material("trainvm.linux-process-policy/v1");
  material.push_back('\0');
  material += policy_body(policy).dump();
  return "sha256:" + sha256_hex(material);
}

template <typename Value>
std::optional<Value> optional(const Json& source, std::string_view name) {
  const Json& value = source.at(std::string(name));
  return value.is_null() ? std::nullopt
                         : std::optional<Value>(value.get<Value>());
}

}  // namespace

LinuxProcessPolicy compile_linux_process_policy(
    const std::optional<CpuIoPolicy>& source) {
  LinuxProcessPolicy result{
      .api_version = std::string(kLinuxProcessPolicyApiVersion),
      .cpuset = std::nullopt,
      .cpu_weight = std::nullopt,
      .io_weight = std::nullopt,
      .omp_threads = std::nullopt,
      .preprocessing_workers = std::nullopt,
      .nice = std::nullopt,
      .policy_digest = {},
  };
  if (source) {
    if (source->cpuset && source->cpus)
      reject("cpuset and cpus are mutually exclusive");
    if (source->cpuset)
      result.cpuset = encode_ranges(parse_ranges(*source->cpuset));
    if (source->cpus)
      result.cpuset = encode_ranges(vector_ranges(*source->cpus));
    result.cpu_weight = source->cpu_weight;
    result.io_weight = source->io_weight;
    result.omp_threads = source->omp_threads;
    result.preprocessing_workers = source->preprocessing_workers;
    result.nice = source->nice;
  }
  bounded(result.cpu_weight, "cpu_weight", std::int64_t{1},
          std::int64_t{10'000});
  bounded(result.io_weight, "io_weight", std::int64_t{1},
          std::int64_t{10'000});
  bounded(result.omp_threads, "omp_threads", std::int64_t{1},
          std::int64_t{65'536});
  bounded(result.preprocessing_workers, "preprocessing_workers",
          std::int64_t{0}, std::int64_t{65'536});
  bounded(result.nice, "nice", std::int64_t{-20}, std::int64_t{19});
  result.policy_digest = digest(result);
  return result;
}

void validate_linux_process_policy(const LinuxProcessPolicy& policy) {
  if (policy.api_version != kLinuxProcessPolicyApiVersion ||
      !valid_digest(policy.policy_digest) ||
      (policy.cpuset &&
       encode_ranges(parse_ranges(*policy.cpuset)) != *policy.cpuset)) {
    reject("Linux process policy is malformed or noncanonical");
  }
  const CpuIoPolicy source{
      .cpuset = policy.cpuset,
      .cpus = std::nullopt,
      .cpu_weight = policy.cpu_weight,
      .io_weight = policy.io_weight,
      .omp_threads = policy.omp_threads,
      .preprocessing_workers = policy.preprocessing_workers,
      .nice = policy.nice,
  };
  if (compile_linux_process_policy(source) != policy)
    reject("Linux process policy is not exact compiler output");
}

nlohmann::json linux_process_policy_json(const LinuxProcessPolicy& policy) {
  validate_linux_process_policy(policy);
  Json result = policy_body(policy);
  result["policy_digest"] = policy.policy_digest;
  return result;
}

LinuxProcessPolicy linux_process_policy_from_json(const nlohmann::json& source) {
  static const std::set<std::string> fields{
      "api_version",          "cpuset", "cpu_weight", "io_weight",
      "nice",                 "omp_threads",
      "policy_digest",        "preprocessing_workers"};
  if (!source.is_object() || source.size() != fields.size() ||
      !std::ranges::all_of(fields, [&](const std::string& field) {
        return source.contains(field);
      })) {
    reject("Linux process policy JSON fields are inexact");
  }
  try {
    LinuxProcessPolicy result{
        .api_version = source.at("api_version").get<std::string>(),
        .cpuset = optional<std::string>(source, "cpuset"),
        .cpu_weight = optional<std::int64_t>(source, "cpu_weight"),
        .io_weight = optional<std::int64_t>(source, "io_weight"),
        .omp_threads = optional<std::int64_t>(source, "omp_threads"),
        .preprocessing_workers =
            optional<std::int64_t>(source, "preprocessing_workers"),
        .nice = optional<std::int64_t>(source, "nice"),
        .policy_digest = source.at("policy_digest").get<std::string>(),
    };
    if (linux_process_policy_json(result) != source)
      reject("Linux process policy JSON is not canonical");
    return result;
  } catch (const LinuxProcessPolicyError&) {
    throw;
  } catch (const Json::exception&) {
    reject("Linux process policy JSON has invalid field types");
  }
}

}  // namespace trainvm
