#include "trainvm/input_content_authority.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    constexpr std::string_view prefix = "/tmp/trainvm-cache-bench-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    char *created = ::mkdtemp(pattern.data());
    if (created == nullptr)
      throw std::runtime_error("could not create benchmark directory");
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::uint64_t file_count(int argc, char **argv) {
  if (argc == 1)
    return 20'000U;
  if (argc != 2)
    throw std::invalid_argument("usage: input_content_cache_benchmark [files]");
  std::uint64_t result{};
  const std::string_view source(argv[1]);
  const auto parsed = std::from_chars(source.data(), source.data() + source.size(),
                                      result);
  if (parsed.ec != std::errc{} || parsed.ptr != source.data() + source.size() ||
      result == 0U ||
      result > trainvm::InputContentMeasurementCache::kDefaultMaximumEntries)
    throw std::invalid_argument("files must be between 1 and cache capacity");
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::uint64_t files = file_count(argc, argv);
    TemporaryDirectory temporary;
    for (std::uint64_t index = 0U; index < files; ++index) {
      std::ofstream output(
          temporary.path() / (std::to_string(index) + ".bin"),
          std::ios::binary);
      if (!output)
        throw std::runtime_error("could not create benchmark input");
    }

    trainvm::InputContentMeasurementCache cache;
    auto cold = cache.begin_transaction();
    trainvm::InputContentMeasurementStats cold_stats;
    (void)trainvm::measure_input_content_root(temporary.path(), &cold_stats,
                                              &cold);
    const auto cold_commit = cold.commit();
    if (cold_commit.entries_after != files || cold_stats.cache_misses != files)
      throw std::runtime_error(
          "filesystem cache semantics are unavailable for this benchmark");

    auto warm = cache.begin_transaction();
    trainvm::InputContentMeasurementStats warm_stats;
    const auto measurement_started = std::chrono::steady_clock::now();
    (void)trainvm::measure_input_content_root(temporary.path(), &warm_stats,
                                              &warm);
    const auto commit_started = std::chrono::steady_clock::now();
    const auto warm_commit = warm.commit();
    const auto finished = std::chrono::steady_clock::now();
    if (warm_stats.cache_hits != files || warm_stats.bytes_hashed != 0U ||
        warm_commit.entries_after != files ||
        warm_commit.staged_entries != 0U)
      throw std::runtime_error("benchmark did not exercise an all-hit commit");

    const auto microseconds = [](const auto duration) {
      return std::chrono::duration_cast<std::chrono::microseconds>(duration)
          .count();
    };
    std::cout << "entries=" << warm_commit.entries_after
              << " hits=" << warm_stats.cache_hits << " measurement_us="
              << microseconds(commit_started - measurement_started)
              << " commit_us=" << microseconds(finished - commit_started)
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "input content cache benchmark failure: " << error.what()
              << '\n';
    return 1;
  }
}
