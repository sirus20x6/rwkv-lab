#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace trainvm {

inline constexpr std::string_view kInputContentRootApiVersion =
    "trainvm.input-content-root/v1";
inline constexpr std::string_view kInputContentRootSetApiVersion =
    "trainvm.input-content-root-set/v1";
inline constexpr std::string_view kInputContentMeasurementCacheApiVersion =
    "trainvm.input-content-measurement-cache/v1";

enum class ContentRootKind {
  file,
  directory,
};

struct InputContentRootIdentity final {
  std::string api_version;
  std::string path;
  ContentRootKind kind{};
  std::uint64_t file_count{};
  std::uint64_t total_bytes{};
  std::string tree_sha256;

  bool operator==(const InputContentRootIdentity &) const = default;
};

struct InputContentRootSet final {
  std::string api_version;
  std::vector<std::string> paths;

  bool operator==(const InputContentRootSet &) const = default;
};

// Operational telemetry is deliberately separate from the immutable content
// identity. A cold preview and a warm fenced launch must compile to the same
// plan hash even though the second measurement can reuse trusted process-local
// evidence.
struct InputContentMeasurementStats final {
  std::uint64_t cache_hits{};
  std::uint64_t cache_misses{};
  std::uint64_t cache_bypasses{};
  std::uint64_t staging_saturations{};
  std::uint64_t bytes_hashed{};
  std::uint64_t elapsed_nanoseconds{};

  bool operator==(const InputContentMeasurementStats &) const = default;
};

struct InputContentFilesystemIdentity final {
  std::uint64_t filesystem_type{};
  std::uint64_t unique_mount_id{};

  bool operator==(const InputContentFilesystemIdentity &) const = default;
};

struct InputContentMeasurementCacheCommitStats final {
  std::uint64_t capacity{};
  std::uint64_t entries_before{};
  std::uint64_t entries_after{};
  std::uint64_t staged_entries{};
  std::uint64_t staging_saturations{};
  std::uint64_t evictions{};
  std::uint64_t saturations{};
  std::uint64_t corruptions{};

  bool
  operator==(const InputContentMeasurementCacheCommitStats &) const = default;
};

class InputContentMeasurementTransaction;

// This cache belongs to one long-lived controller authority. Measurements are
// staged in a transaction and become reusable only after the caller has
// successfully compiled the locked plan. It is never shared with a worker.
class InputContentMeasurementCache final {
public:
  struct Impl;

  static constexpr std::uint64_t kDefaultMaximumEntries = 1U << 20U;
  using FilesystemIdentitySource =
      std::function<std::optional<InputContentFilesystemIdentity>(
          int descriptor)>;
  using IntegrityFaultForTesting = std::function<bool()>;
  using PublicationFaultForTesting = std::function<bool()>;

  explicit InputContentMeasurementCache(
      std::uint64_t maximum_entries = kDefaultMaximumEntries,
      FilesystemIdentitySource filesystem_identity = {},
      IntegrityFaultForTesting integrity_fault_for_testing = {},
      PublicationFaultForTesting publication_fault_for_testing = {});
  ~InputContentMeasurementCache();

  InputContentMeasurementCache(const InputContentMeasurementCache &) = delete;
  InputContentMeasurementCache &
  operator=(const InputContentMeasurementCache &) = delete;

  [[nodiscard]] InputContentMeasurementTransaction begin_transaction();
  [[nodiscard]] std::string policy_digest() const;

private:
  std::unique_ptr<Impl> impl_;

  friend class InputContentMeasurementTransaction;
};

class InputContentMeasurementTransaction final {
public:
  struct Impl;

  ~InputContentMeasurementTransaction();
  InputContentMeasurementTransaction(
      InputContentMeasurementTransaction &&) noexcept;
  InputContentMeasurementTransaction &
  operator=(InputContentMeasurementTransaction &&) noexcept;
  InputContentMeasurementTransaction(
      const InputContentMeasurementTransaction &) = delete;
  InputContentMeasurementTransaction &
  operator=(const InputContentMeasurementTransaction &) = delete;

  [[nodiscard]] InputContentMeasurementCacheCommitStats commit();

private:
  explicit InputContentMeasurementTransaction(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class InputContentMeasurementCache;
  friend InputContentRootIdentity
  measure_input_content_root(const std::filesystem::path &,
                             InputContentMeasurementStats *,
                             InputContentMeasurementTransaction *);
};

[[nodiscard]] InputContentRootIdentity measure_input_content_root(
    const std::filesystem::path &path,
    InputContentMeasurementStats *measurement_stats = nullptr,
    InputContentMeasurementTransaction *transaction = nullptr);

[[nodiscard]] std::vector<InputContentRootIdentity>
measure_input_content_root_set(
    const InputContentRootSet &root_set,
    std::vector<InputContentMeasurementStats> *measurement_stats = nullptr,
    InputContentMeasurementTransaction *transaction = nullptr);

} // namespace trainvm
