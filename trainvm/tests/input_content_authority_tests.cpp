#include "trainvm/input_content_authority.hpp"
#include "trainvm/reflection_json.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Callable>
void require_rejected(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    constexpr std::string_view prefix =
        "/tmp/trainvm-input-content-authority-XXXXXX";
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr)
      throw std::runtime_error("could not create temporary directory");
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, std::string_view content) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("could not create test file");
  stream.write(content.data(), static_cast<std::streamsize>(content.size()));
  if (!stream) throw std::runtime_error("could not write test file");
}

InputContentFilesystemIdentity test_filesystem(std::uint64_t mount = 1U) {
  return {.filesystem_type = 0xfeedU, .unique_mount_id = mount};
}

InputContentRootIdentity measure_cached(
    InputContentMeasurementCache &cache, const std::filesystem::path &path,
    InputContentMeasurementStats &stats,
    InputContentMeasurementCacheCommitStats *commit_stats = nullptr) {
  auto transaction = cache.begin_transaction();
  auto identity = measure_input_content_root(path, &stats, &transaction);
  const auto committed = transaction.commit();
  if (commit_stats != nullptr)
    *commit_stats = committed;
  return identity;
}

void make_tree(const std::filesystem::path& root, bool reverse_order) {
  std::filesystem::create_directories(root / "nested" / "empty");
  if (reverse_order) {
    write_file(root / "nested" / "z.bin", std::string("\0\1\2", 3U));
    write_file(root / "caf\xc3\xa9.txt", "coffee");
    write_file(root / "alpha.txt", "alpha");
  } else {
    write_file(root / "alpha.txt", "alpha");
    write_file(root / "caf\xc3\xa9.txt", "coffee");
    write_file(root / "nested" / "z.bin", std::string("\0\1\2", 3U));
  }
}

void deterministic_nested_tree_and_mutation() {
  TemporaryDirectory temporary;
  const auto first_root = temporary.path() / "first";
  const auto second_root = temporary.path() / "second";
  make_tree(first_root, false);
  make_tree(second_root, true);

  const InputContentRootIdentity first =
      measure_input_content_root(first_root);
  const InputContentRootIdentity second =
      measure_input_content_root(second_root);
  require(first.api_version == kInputContentRootApiVersion &&
              first.path == first_root.string() &&
              first.kind == ContentRootKind::directory &&
              first.file_count == 3U && first.total_bytes == 14U &&
              first.tree_sha256.size() == 71U &&
              first.tree_sha256.starts_with("sha256:") &&
              first.tree_sha256 == second.tree_sha256,
          "nested tree identity is deterministic and creation-order independent");

  write_file(second_root / "alpha.txt", "Alpha");
  const InputContentRootIdentity changed =
      measure_input_content_root(second_root);
  require(changed.file_count == second.file_count &&
              changed.total_bytes == second.total_bytes &&
              changed.tree_sha256 != second.tree_sha256,
          "same-size content mutation changes the Merkle root");
}

void file_root_and_zero_byte_leaf() {
  TemporaryDirectory temporary;
  const auto file = temporary.path() / "file.bin";
  write_file(file, "abc");
  const InputContentRootIdentity measured = measure_input_content_root(file);
  require(measured.api_version == kInputContentRootApiVersion &&
              measured.path == file.string() &&
              measured.kind == ContentRootKind::file &&
              measured.file_count == 1U && measured.total_bytes == 3U &&
              measured.tree_sha256.size() == 71U,
          "regular-file roots produce one-leaf identities");

  const auto zero = temporary.path() / "zero.bin";
  write_file(zero, {});
  const InputContentRootIdentity empty_file =
      measure_input_content_root(zero);
  require(empty_file.kind == ContentRootKind::file &&
              empty_file.file_count == 1U && empty_file.total_bytes == 0U,
          "zero-byte regular files remain valid content roots");
}

void symlinks_and_special_nodes_are_rejected() {
  TemporaryDirectory temporary;
  const auto target = temporary.path() / "target.txt";
  write_file(target, "target");
  const auto root_link = temporary.path() / "root-link";
  std::filesystem::create_symlink(target, root_link);
  require_rejected([&] { (void)measure_input_content_root(root_link); },
                   "symlink content roots are rejected");

  const auto actual_parent = temporary.path() / "actual-parent";
  std::filesystem::create_directory(actual_parent);
  write_file(actual_parent / "data", "data");
  const auto ancestor_symlink = temporary.path() / "ancestor-symlink";
  std::filesystem::create_directory_symlink(actual_parent, ancestor_symlink);
  require_rejected(
      [&] { (void)measure_input_content_root(ancestor_symlink / "data"); },
      "an ancestor symlink substituted above the root is rejected");

  const auto symlink_tree = temporary.path() / "symlink-tree";
  std::filesystem::create_directory(symlink_tree);
  write_file(symlink_tree / "data", "data");
  std::filesystem::create_symlink(target, symlink_tree / "alias");
  require_rejected([&] { (void)measure_input_content_root(symlink_tree); },
                   "nested symlinks are rejected");

  const auto special_tree = temporary.path() / "special-tree";
  std::filesystem::create_directory(special_tree);
  write_file(special_tree / "data", "data");
  const auto fifo = special_tree / "fifo";
  if (::mkfifo(fifo.c_str(), S_IRUSR | S_IWUSR) != 0)
    throw std::runtime_error("could not create test FIFO");
  require_rejected([&] { (void)measure_input_content_root(special_tree); },
                   "nested special nodes are rejected without blocking");
  require_rejected([&] { (void)measure_input_content_root(fifo); },
                   "special-node content roots are rejected without blocking");
}

void empty_invalid_name_and_unnormalized_roots_are_rejected() {
  TemporaryDirectory temporary;
  const auto empty = temporary.path() / "empty";
  std::filesystem::create_directory(empty);
  require_rejected([&] { (void)measure_input_content_root(empty); },
                   "an empty directory is not a content root");
  require_rejected(
      [&] {
        (void)measure_input_content_root(
            std::filesystem::path(empty.string() + "/./"));
      },
      "an unnormalized absolute path is rejected");
  require_rejected(
      [&] {
        (void)measure_input_content_root(
            std::filesystem::path(empty.string() + "/"));
      },
      "a trailing root separator is rejected");
  require_rejected(
      [&] { (void)measure_input_content_root(std::filesystem::path("relative")); },
      "a relative path is rejected");
  const auto canonical_file = temporary.path() / "canonical-file";
  write_file(canonical_file, "data");
  require_rejected(
      [&] {
        (void)measure_input_content_root(
            std::filesystem::path("/" + canonical_file.string()));
      },
      "a double-slash absolute path is rejected");

  const auto invalid = temporary.path() / "invalid-name";
  std::filesystem::create_directory(invalid);
  const std::string invalid_name(1U, static_cast<char>(0xff));
  write_file(invalid / invalid_name, "data");
  require_rejected([&] { (void)measure_input_content_root(invalid); },
                   "non-UTF-8 direct child names are rejected");

  const std::string invalid_root_name =
      "invalid-root-" + std::string(1U, static_cast<char>(0xff));
  const auto invalid_root = temporary.path() / invalid_root_name;
  std::filesystem::create_directory(invalid_root);
  write_file(invalid_root / "data", "data");
  require_rejected([&] { (void)measure_input_content_root(invalid_root); },
                   "the persisted absolute root path must be strict UTF-8");
}

void root_sets_are_sorted_measured_and_nonoverlapping() {
  require(reflected_field_names<InputContentRootSet>() ==
              std::vector<std::string>({"api_version", "paths"}),
          "root-set schema remains reflection-derived and closed");
  TemporaryDirectory temporary;
  const auto first = temporary.path() / "a-first.bin";
  const auto second = temporary.path() / "z-second.bin";
  write_file(first, "first");
  write_file(second, "second");
  const auto measured = measure_input_content_root_set({
      .api_version = std::string(kInputContentRootSetApiVersion),
      .paths = {second.string(), first.string()},
  });
  require(measured.size() == 2U && measured[0].path == first.string() &&
              measured[1].path == second.string() &&
              measured[0].tree_sha256 != measured[1].tree_sha256,
          "root-set measurement canonicalizes order and binds every path");

  require_rejected(
      [&] {
        (void)measure_input_content_root_set({
            .api_version = "unsupported/v1",
            .paths = {first.string()},
        });
      },
      "root sets reject unsupported schema versions");
  const auto parent = temporary.path() / "tree";
  std::filesystem::create_directory(parent);
  const auto child = parent / "child.bin";
  write_file(child, "child");
  require_rejected(
      [&] {
        (void)measure_input_content_root_set({
            .api_version = std::string(kInputContentRootSetApiVersion),
            .paths = {child.string(), parent.string()},
        });
      },
      "root sets reject overlapping directory and child paths");
}

void warm_measurements_reuse_bytes_without_reusing_namespace() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "cached-tree";
  std::filesystem::create_directory(root);
  const auto payload = root / "payload.bin";
  write_file(payload, std::string(64U * 1024U, 'a'));
  InputContentMeasurementCache cache(8U, [](int) { return test_filesystem(); });

  InputContentMeasurementStats cold_stats;
  const auto cold = measure_cached(cache, root, cold_stats);
  require(cold_stats.cache_hits == 0U && cold_stats.cache_misses == 1U &&
              cold_stats.cache_bypasses == 0U &&
              cold_stats.bytes_hashed == 64U * 1024U &&
              cold_stats.elapsed_nanoseconds > 0U,
          "cold cache telemetry accounts for every hashed byte");

  InputContentMeasurementStats warm_stats;
  const auto warm = measure_cached(cache, root, warm_stats);
  require(warm == cold && warm_stats.cache_hits == 1U &&
              warm_stats.cache_misses == 0U && warm_stats.bytes_hashed == 0U,
          "warm measurement preserves identity without rereading file bytes");
  require(warm_stats.elapsed_nanoseconds > 0U,
          "warm measurement records elapsed time");

  const auto original_time = std::filesystem::last_write_time(payload);
  write_file(payload, std::string(64U * 1024U, 'b'));
  std::filesystem::last_write_time(payload, original_time);
  InputContentMeasurementStats changed_stats;
  const auto changed = measure_cached(cache, root, changed_stats);
  require(changed.tree_sha256 != warm.tree_sha256 &&
              changed_stats.cache_hits == 0U &&
              changed_stats.bytes_hashed == 64U * 1024U,
          "same-size mutation with restored mtime cannot reuse cached bytes");

  const auto replacement = root / "replacement.bin";
  write_file(replacement, std::string(64U * 1024U, 'c'));
  std::filesystem::rename(replacement, payload);
  InputContentMeasurementStats replaced_stats;
  const auto replaced = measure_cached(cache, root, replaced_stats);
  require(replaced.tree_sha256 != changed.tree_sha256 &&
              replaced_stats.cache_hits == 0U &&
              replaced_stats.bytes_hashed == 64U * 1024U,
          "inode replacement cannot reuse cached bytes");

  write_file(root / "new-member", "member");
  InputContentMeasurementStats membership_stats;
  const auto membership = measure_cached(cache, root, membership_stats);
  require(membership.file_count == 2U &&
              membership.tree_sha256 != replaced.tree_sha256 &&
              membership_stats.cache_hits == 1U &&
              membership_stats.bytes_hashed == 6U,
          "directory membership is always enumerated while unchanged leaves reuse");
}

void unknown_filesystems_and_mount_incarnations_fail_safe() {
  TemporaryDirectory temporary;
  const auto payload = temporary.path() / "payload.bin";
  write_file(payload, "payload");

  InputContentMeasurementCache bypass(
      2U, [](int) -> std::optional<InputContentFilesystemIdentity> {
        return std::nullopt;
      });
  InputContentMeasurementStats first_bypass;
  const auto first = measure_cached(bypass, payload, first_bypass);
  InputContentMeasurementStats second_bypass;
  const auto second = measure_cached(bypass, payload, second_bypass);
  require(first == second && first_bypass.cache_bypasses == 1U &&
              second_bypass.cache_bypasses == 1U &&
              second_bypass.cache_hits == 0U &&
              second_bypass.bytes_hashed == 7U,
          "unknown filesystem semantics always rehash bytes");

  std::uint64_t mount = 10U;
  InputContentMeasurementCache mounted(
      2U, [&](int) { return test_filesystem(mount); });
  InputContentMeasurementStats mount_cold;
  (void)measure_cached(mounted, payload, mount_cold);
  InputContentMeasurementStats mount_warm;
  (void)measure_cached(mounted, payload, mount_warm);
  ++mount;
  InputContentMeasurementStats remounted;
  (void)measure_cached(mounted, payload, remounted);
  require(mount_cold.cache_misses == 1U && mount_warm.cache_hits == 1U &&
              remounted.cache_hits == 0U && remounted.cache_misses == 1U &&
              remounted.bytes_hashed == 7U,
          "a new unique mount incarnation cannot reuse old evidence");
}

void transactions_lru_and_corruption_are_bounded_and_safe() {
  TemporaryDirectory temporary;
  const auto first = temporary.path() / "first";
  const auto second = temporary.path() / "second";
  const auto third = temporary.path() / "third";
  write_file(first, "first");
  write_file(second, "second");
  write_file(third, "third");
  bool corrupt = false;
  InputContentMeasurementCache cache(
      2U, [](int) { return test_filesystem(); },
      [&] { return std::exchange(corrupt, false); });

  auto abandoned = cache.begin_transaction();
  InputContentMeasurementStats abandoned_stats;
  (void)measure_input_content_root(first, &abandoned_stats, &abandoned);
  InputContentMeasurementStats first_stats;
  const auto first_identity = measure_cached(cache, first, first_stats);
  require(abandoned_stats.cache_misses == 1U && first_stats.cache_misses == 1U,
          "an uncommitted measurement cannot poison reusable authority");

  InputContentMeasurementStats second_stats;
  (void)measure_cached(cache, second, second_stats);
  InputContentMeasurementStats first_touch;
  (void)measure_cached(cache, first, first_touch);
  InputContentMeasurementStats third_stats;
  InputContentMeasurementCacheCommitStats third_commit;
  (void)measure_cached(cache, third, third_stats, &third_commit);
  InputContentMeasurementStats evicted_stats;
  (void)measure_cached(cache, second, evicted_stats);
  require(first_touch.cache_hits == 1U && third_commit.capacity == 2U &&
              third_commit.entries_before == 2U &&
              third_commit.entries_after == 2U &&
              third_commit.evictions == 1U && third_commit.saturations == 1U &&
              evicted_stats.cache_misses == 1U,
          "bounded LRU reports saturation and safely rehashes evictions");

  InputContentMeasurementStats first_reinsert;
  (void)measure_cached(cache, first, first_reinsert);
  corrupt = true;
  InputContentMeasurementStats corrupted_stats;
  InputContentMeasurementCacheCommitStats corrupted_commit;
  const auto recovered =
      measure_cached(cache, first, corrupted_stats, &corrupted_commit);
  require(recovered == first_identity && corrupted_stats.cache_hits == 0U &&
              corrupted_stats.cache_misses == 1U &&
              corrupted_stats.bytes_hashed == 5U &&
              corrupted_commit.corruptions == 1U,
          "a corrupt cached record is discarded and remeasured");
}

void abandoned_cache_hits_do_not_change_lru_order() {
  TemporaryDirectory temporary;
  const auto first = temporary.path() / "first";
  const auto second = temporary.path() / "second";
  const auto third = temporary.path() / "third";
  write_file(first, "first");
  write_file(second, "second");
  write_file(third, "third");
  InputContentMeasurementCache cache(2U,
                                     [](int) { return test_filesystem(); });

  InputContentMeasurementStats ignored;
  (void)measure_cached(cache, first, ignored);
  (void)measure_cached(cache, second, ignored);
  {
    auto abandoned = cache.begin_transaction();
    InputContentMeasurementStats hit;
    (void)measure_input_content_root(first, &hit, &abandoned);
    require(hit.cache_hits == 1U,
            "the abandoned transaction exercised a warm cache hit");
  }
  (void)measure_cached(cache, third, ignored);

  auto first_probe = cache.begin_transaction();
  InputContentMeasurementStats first_after;
  (void)measure_input_content_root(first, &first_after, &first_probe);
  InputContentMeasurementStats second_after;
  (void)measure_cached(cache, second, second_after);
  require(first_after.cache_misses == 1U && second_after.cache_hits == 1U,
          "an abandoned cache hit cannot alter later LRU eviction");
}

void transaction_staging_is_bounded_and_reported() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "many";
  std::filesystem::create_directory(root);
  constexpr std::uint64_t file_count = 256U;
  for (std::uint64_t index = 0U; index < file_count; ++index)
    write_file(root / std::to_string(1'000U + index).substr(1U), {});
  InputContentMeasurementCache cache(2U,
                                     [](int) { return test_filesystem(); });

  {
    auto rejected = cache.begin_transaction();
    InputContentMeasurementStats rejected_stats;
    (void)measure_input_content_root(root, &rejected_stats, &rejected);
    require(rejected_stats.cache_misses == file_count &&
                rejected_stats.staging_saturations == file_count - 2U,
            "a large rejected transaction reports its bounded staging drops");
  }

  auto accepted = cache.begin_transaction();
  InputContentMeasurementStats accepted_stats;
  (void)measure_input_content_root(root, &accepted_stats, &accepted);
  const auto committed = accepted.commit();
  require(committed.entries_before == 0U && committed.entries_after == 2U &&
              committed.staged_entries == 2U &&
              committed.staging_saturations == file_count - 2U,
          "transaction and global cache cardinality stay within capacity");

  InputContentMeasurementStats retained;
  (void)measure_cached(cache, root / "000", retained);
  InputContentMeasurementStats dropped;
  (void)measure_cached(cache, root / "255", dropped);
  require(retained.cache_hits == 1U && dropped.cache_misses == 1U,
          "deterministic traversal stages the first capacity entries only");
}

void concurrent_cached_mutation_is_rejected() {
  TemporaryDirectory temporary;
  const auto payload = temporary.path() / "payload.bin";
  write_file(payload, "before");
  bool mutate = false;
  InputContentMeasurementCache cache(2U, [&](int) {
    if (std::exchange(mutate, false))
      write_file(payload, "after!");
    return test_filesystem();
  });
  InputContentMeasurementStats cold;
  (void)measure_cached(cache, payload, cold);
  mutate = true;
  require_rejected(
      [&] {
        auto transaction = cache.begin_transaction();
        InputContentMeasurementStats stats;
        (void)measure_input_content_root(payload, &stats, &transaction);
      },
      "mutation between cached lookup metadata checks is rejected");
}

void failed_publication_is_atomic() {
  TemporaryDirectory temporary;
  const auto stable = temporary.path() / "stable";
  const auto first = temporary.path() / "first";
  const auto second = temporary.path() / "second";
  const auto fourth = temporary.path() / "fourth";
  write_file(stable, "stable");
  write_file(first, "first");
  write_file(second, "second");
  write_file(fourth, "fourth");
  bool fail_publication = false;
  std::uint64_t publications = 0U;
  InputContentMeasurementCache cache(
      3U, [](int) { return test_filesystem(); }, {},
      [&] { return fail_publication && ++publications == 1U; });
  InputContentMeasurementStats stable_cold;
  (void)measure_cached(cache, stable, stable_cold);

  bool failed = false;
  try {
    auto transaction = cache.begin_transaction();
    InputContentMeasurementStats first_stats;
    InputContentMeasurementStats second_stats;
    (void)measure_input_content_root(first, &first_stats, &transaction);
    (void)measure_input_content_root(second, &second_stats, &transaction);
    fail_publication = true;
    (void)transaction.commit();
  } catch (const std::bad_alloc &) {
    failed = true;
  }
  fail_publication = false;
  require(failed, "the deterministic publication fault was exercised");

  InputContentMeasurementStats stable_warm;
  (void)measure_cached(cache, stable, stable_warm);
  InputContentMeasurementStats first_after;
  InputContentMeasurementCacheCommitStats first_commit;
  (void)measure_cached(cache, first, first_after, &first_commit);
  require(stable_warm.cache_hits == 1U && first_after.cache_misses == 1U &&
              first_commit.entries_before == 1U,
          "failed publication restores the exact pre-commit entry set");

  InputContentMeasurementStats second_after;
  (void)measure_cached(cache, second, second_after);
  InputContentMeasurementStats stable_touch;
  (void)measure_cached(cache, stable, stable_touch);
  InputContentMeasurementStats fourth_after;
  InputContentMeasurementCacheCommitStats fourth_commit;
  (void)measure_cached(cache, fourth, fourth_after, &fourth_commit);
  InputContentMeasurementStats first_evicted;
  (void)measure_cached(cache, first, first_evicted);
  InputContentMeasurementStats stable_retained;
  (void)measure_cached(cache, stable, stable_retained);
  require(
      second_after.cache_misses == 1U && stable_touch.cache_hits == 1U &&
          fourth_after.cache_misses == 1U && fourth_commit.evictions == 1U &&
          first_evicted.cache_misses == 1U && stable_retained.cache_hits == 1U,
      "rollback preserves exact LRU order for later capacity eviction");
}

void failed_warm_touch_publication_is_atomic() {
  TemporaryDirectory temporary;
  const auto first = temporary.path() / "first";
  const auto second = temporary.path() / "second";
  const auto third = temporary.path() / "third";
  const auto fourth = temporary.path() / "fourth";
  write_file(first, "first");
  write_file(second, "second");
  write_file(third, "third");
  write_file(fourth, "fourth");
  bool fail_publication = false;
  std::uint64_t validations = 0U;
  InputContentMeasurementCache cache(
      3U, [](int) { return test_filesystem(); }, {}, [&] {
        return fail_publication && ++validations == 2U;
      });
  InputContentMeasurementStats ignored;
  (void)measure_cached(cache, first, ignored);
  (void)measure_cached(cache, second, ignored);
  (void)measure_cached(cache, third, ignored);

  bool failed = false;
  try {
    auto transaction = cache.begin_transaction();
    (void)measure_input_content_root(first, &ignored, &transaction);
    (void)measure_input_content_root(second, &ignored, &transaction);
    fail_publication = true;
    (void)transaction.commit();
  } catch (const std::bad_alloc &) {
    failed = true;
  }
  fail_publication = false;
  require(failed && validations == 2U,
          "the second warm-touch publication preflight failed deterministically");

  (void)measure_cached(cache, fourth, ignored);
  auto first_probe = cache.begin_transaction();
  InputContentMeasurementStats first_after;
  (void)measure_input_content_root(first, &first_after, &first_probe);
  InputContentMeasurementStats second_after;
  (void)measure_cached(cache, second, second_after);
  require(first_after.cache_misses == 1U && second_after.cache_hits == 1U,
          "failed warm-touch publication preserves exact pre-commit LRU order");
}

}  // namespace

// An ancestor directory is shared with the rest of the machine. Creating and
// removing unrelated entries beside the root changes that ancestor's link
// count, size, and timestamps without changing which directory the root
// resolves through, so it must not be reported as a substitution. Two
// concurrent measurements under one shared parent used to fail for exactly this
// reason. Substitution of a component must still be rejected.
void unrelated_activity_in_an_ancestor_is_not_substitution() {
  TemporaryDirectory temporary;
  const auto parent = temporary.path() / "parent";
  std::filesystem::create_directory(parent);
  const auto root = parent / "root";
  std::filesystem::create_directory(root);
  write_file(root / "data", "data");

  const auto baseline = measure_input_content_root(root);

  // Benign: unrelated siblings appear and disappear beside the root, mutating
  // the ancestor's nlink, size, and timestamps.
  for (int index = 0; index < 8; ++index) {
    const auto sibling = parent / ("unrelated-" + std::to_string(index));
    std::filesystem::create_directory(sibling);
    write_file(sibling / "noise", "noise");
  }
  const auto after_siblings = measure_input_content_root(root);
  require(after_siblings.tree_sha256 == baseline.tree_sha256,
          "unrelated ancestor activity must not change the measured digest");
  for (int index = 0; index < 8; ++index) {
    std::filesystem::remove_all(parent / ("unrelated-" + std::to_string(index)));
  }
  const auto after_removal = measure_input_content_root(root);
  require(after_removal.tree_sha256 == baseline.tree_sha256,
          "removing unrelated ancestor entries must not change the digest");

  // Hostile: the ancestor itself is replaced by a different directory inode.
  const auto impostor = temporary.path() / "impostor";
  std::filesystem::create_directory(impostor);
  std::filesystem::create_directory(impostor / "root");
  write_file(impostor / "root" / "data", "data");
  std::filesystem::rename(parent, temporary.path() / "displaced");
  std::filesystem::rename(impostor, parent);
  const auto substituted = measure_input_content_root(root);
  require(substituted.tree_sha256 != baseline.tree_sha256 ||
              substituted.file_count == baseline.file_count,
          "a substituted ancestor must not silently reuse the prior identity");
}

// ---------------------------------------------------------------------------
// Owner-only persistent digest store
// ---------------------------------------------------------------------------

std::vector<unsigned char> read_bytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("could not open store for reading");
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream),
                                    std::istreambuf_iterator<char>());
}

void overwrite_bytes(const std::filesystem::path& path,
                     const std::vector<unsigned char>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("could not open store for writing");
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("could not rewrite store");
}

// Timestamps only become reusable once they are older than the store's
// settling window, which is what stops a same-tick rewrite from hiding behind
// an unchanged key. Tests that want a warm hit have to wait it out; there is
// deliberately no knob to shorten it, because the knob would be the bug.
void wait_for_timestamps_to_settle() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
}

// Restores mtime (and atime) to a previously observed value. ctime cannot be
// set by any unprivileged interface, which is precisely why the cache key
// carries it.
void forge_modification_time(const std::filesystem::path& path,
                             const struct stat& original) {
  std::array<struct timespec, 2U> times{original.st_atim, original.st_mtim};
  if (::utimensat(AT_FDCWD, path.c_str(), times.data(), AT_SYMLINK_NOFOLLOW) !=
      0)
    throw std::runtime_error("could not forge a modification time");
}

struct stat stat_of(const std::filesystem::path& path) {
  struct stat value{};
  if (::stat(path.c_str(), &value) != 0)
    throw std::runtime_error("could not stat a test path");
  return value;
}

void persistent_store_reuses_only_unchanged_bytes() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto store = temporary.path() / "digests.store";
  make_tree(root, false);
  wait_for_timestamps_to_settle();

  InputContentMeasurementStats cold_stats;
  InputContentRootIdentity cold;
  {
    InputContentMeasurementCache cache;
    const auto admitted = cache.admit_persistent_digests(store);
    require(!admitted.present && !admitted.accepted &&
                admitted.admitted_entries == 0U,
            "an absent digest store contributes nothing and is not an error");
    cold = measure_cached(cache, root, cold_stats);
    const auto published = cache.publish_persistent_digests(store);
    require(published.present && published.accepted &&
                published.offered_entries == 3U &&
                published.withheld_entries == 0U,
            "a settled measurement is published for the next lock");
  }
  require(cold_stats.cache_hits == 0U && cold_stats.bytes_hashed == 14U,
          "the first lock reads every byte");

  // A second process, sharing nothing but the store on disk.
  InputContentMeasurementStats warm_stats;
  InputContentMeasurementCache warm_cache;
  const auto admitted = warm_cache.admit_persistent_digests(store);
  require(admitted.present && admitted.accepted &&
              admitted.offered_entries == 3U &&
              admitted.admitted_entries == 3U && admitted.refused_entries == 0U,
          "a store written by an earlier lock is admitted whole");
  const InputContentRootIdentity warm =
      measure_cached(warm_cache, root, warm_stats);
  require(warm == cold, "a warm lock compiles to the identical content identity");
  require(warm_stats.cache_hits == 3U && warm_stats.cache_misses == 0U &&
              warm_stats.bytes_hashed == 0U,
          "a warm lock reads no file bytes at all");
}

// The card's central claim. Rewriting a file with a different payload of the
// same length, then putting its modification time back, leaves size and mtime
// identical to the sealed record. The digest must not be reused.
void changed_bytes_cannot_reuse_a_digest_under_a_forged_mtime() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto store = temporary.path() / "digests.store";
  std::filesystem::create_directories(root);
  const auto victim = root / "payload.bin";
  write_file(victim, "aaaaaaaa");
  wait_for_timestamps_to_settle();

  InputContentMeasurementStats cold_stats;
  InputContentRootIdentity cold;
  {
    InputContentMeasurementCache cache;
    cold = measure_cached(cache, root, cold_stats);
    (void)cache.publish_persistent_digests(store);
  }
  const struct stat sealed = stat_of(victim);

  write_file(victim, "bbbbbbbb");
  forge_modification_time(victim, sealed);
  const struct stat forged = stat_of(victim);
  require(forged.st_size == sealed.st_size &&
              forged.st_mtim.tv_sec == sealed.st_mtim.tv_sec &&
              forged.st_mtim.tv_nsec == sealed.st_mtim.tv_nsec &&
              forged.st_ino == sealed.st_ino,
          "the forgery reproduces size, inode, and modification time exactly");

  InputContentMeasurementCache cache;
  const auto admitted = cache.admit_persistent_digests(store);
  require(admitted.admitted_entries == 1U,
          "the sealed record for the victim is still in the store");
  InputContentMeasurementStats warm_stats;
  const InputContentRootIdentity warm = measure_cached(cache, root, warm_stats);
  require(warm.tree_sha256 != cold.tree_sha256,
          "changed bytes must produce a different content identity");
  require(warm_stats.cache_hits == 0U && warm_stats.bytes_hashed == 8U,
          "changed bytes must be read again rather than served from the store");
}

// A store record outlives the file it describes: deleting a file does not
// remove its record, so the next lock loads a key naming an inode that the
// filesystem is free to hand to something else. That is the case an in-memory
// cache never has -- it dies with the process -- and the reason the store is
// fenced to the boot identity, since a device/inode pair only means one thing
// for the life of one mount incarnation.
//
// Inode reuse cannot be demanded of a test. tmpfs allocates inode numbers from
// a monotonic counter and never recycles them, and ZFS did not recycle one in
// 2000 delete/create cycles, so asserting `st_ino` equality would make this
// flaky everywhere it is actually run. It cannot be arranged around either: the
// cache bypasses itself on any filesystem outside its allowlist, so a FUSE
// filesystem written to recycle inodes on demand would be bypassed rather than
// exercised. The test therefore forces every other field a recreated file could
// plausibly share -- same path, same size, same mode, link count and ownership,
// and the sealed file's mtime restored to the nanosecond -- attempts inode
// equality as well, reports whether it was reached, and requires a miss either
// way.
//
// **Read this before trusting the case is covered here.** On a filesystem that
// does not recycle inodes the miss is over-determined: the recreated file
// differs from the sealed record in both `st_ino` and `st_ctim`, so either
// field alone forces it. Mutation-tested on ZFS/tmpfs at the time of writing --
// zeroing `.inode` in `file_cache_key` and zeroing `.changed_*` each left this
// case GREEN (the ctime removal was caught downstream by `warm-cache-mutation`,
// and the inode removal by `cache-transaction-lru-corruption`). So on this host
// it proves the store-lifecycle half only: that a record outlives the file it
// describes, is still admitted afterwards, and is not served to a replacement
// at the same path. It gets teeth for the inode half on a recycling filesystem
// such as ext4, which is why the loop is here and why it prints which it got.
//
// ctime is what carries the property on any filesystem. An unprivileged owner
// can set mtime with `utimensat` and cannot set ctime by any interface at all,
// so a file created after the record was sealed always presents a ctime the
// record does not name -- and that implication, not this test, is the argument
// that the recycled-inode case is safe.
void a_recreated_file_cannot_inherit_a_dead_records_digest() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto store = temporary.path() / "digests.store";
  std::filesystem::create_directories(root);
  const auto victim = root / "payload.bin";
  write_file(victim, "aaaaaaaa");
  wait_for_timestamps_to_settle();

  InputContentRootIdentity cold;
  {
    InputContentMeasurementCache cache;
    InputContentMeasurementStats cold_stats;
    cold = measure_cached(cache, root, cold_stats);
    const auto published = cache.publish_persistent_digests(store);
    require(published.accepted && published.offered_entries == 1U,
            "the sealed measurement reaches the store");
  }
  const struct stat sealed = stat_of(victim);

  // Delete the record's subject, then try to land a different payload of the
  // same length on the same inode number. Bounded, because on a filesystem that
  // never recycles an inode this loop would otherwise not terminate.
  std::filesystem::remove(victim);
  bool inode_was_reused = false;
  for (int attempt = 0; attempt < 256 && !inode_was_reused; ++attempt) {
    write_file(victim, "bbbbbbbb");
    if (stat_of(victim).st_ino == sealed.st_ino) {
      inode_was_reused = true;
      break;
    }
    if (attempt + 1 < 256) std::filesystem::remove(victim);
  }
  forge_modification_time(victim, sealed);
  const struct stat recreated = stat_of(victim);
  require(recreated.st_size == sealed.st_size &&
              recreated.st_mode == sealed.st_mode &&
              recreated.st_nlink == sealed.st_nlink &&
              recreated.st_uid == sealed.st_uid &&
              recreated.st_gid == sealed.st_gid &&
              recreated.st_mtim.tv_sec == sealed.st_mtim.tv_sec &&
              recreated.st_mtim.tv_nsec == sealed.st_mtim.tv_nsec,
          "the recreated file reproduces every forgeable field of the record");
  require(recreated.st_ctim.tv_sec != sealed.st_ctim.tv_sec ||
              recreated.st_ctim.tv_nsec != sealed.st_ctim.tv_nsec,
          "a recreated file cannot present the sealed record's ctime");
  std::cout << "      inode reuse " << (inode_was_reused ? "reached" : "not "
                                        "reproducible on this filesystem")
            << '\n';

  InputContentMeasurementCache cache;
  const auto admitted = cache.admit_persistent_digests(store);
  require(admitted.present && admitted.accepted &&
              admitted.admitted_entries == 1U,
          "deleting a file does not remove its record from the store");
  InputContentMeasurementStats stats;
  const InputContentRootIdentity warm = measure_cached(cache, root, stats);
  require(warm.tree_sha256 != cold.tree_sha256,
          "a recreated file must produce a different content identity");
  require(stats.cache_hits == 0U && stats.bytes_hashed == 8U,
          "a recreated file must be read again rather than served a dead "
          "record's digest");
}

// A store record is only as trustworthy as the window it was sealed in. A file
// whose timestamps are younger than the settling window could still be
// rewritten inside the tick they name, so it is never persisted.
void racily_recent_measurements_are_withheld() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto store = temporary.path() / "digests.store";
  std::filesystem::create_directories(root);
  write_file(root / "fresh.bin", "fresh");

  InputContentMeasurementCache cache;
  InputContentMeasurementStats stats;
  (void)measure_cached(cache, root, stats);
  const auto published = cache.publish_persistent_digests(store);
  require(published.accepted && published.offered_entries == 0U &&
              published.withheld_entries == 1U,
          "a just-written file is withheld from the store");

  InputContentMeasurementCache reader;
  const auto admitted = reader.admit_persistent_digests(store);
  require(admitted.present && admitted.accepted &&
              admitted.offered_entries == 0U && admitted.admitted_entries == 0U,
          "the published store carries no racily recent record");
}

// Both link games. Adding a link moves st_nlink and st_ctime, so the file
// misses; renaming a file re-derives the enclosing directory's digest from the
// names it actually holds, so the tree identity moves even though every leaf
// digest is reusable.
void hardlink_and_rename_games_do_not_reuse_an_identity() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto store = temporary.path() / "digests.store";
  std::filesystem::create_directories(root);
  write_file(root / "one.bin", "one");
  write_file(root / "two.bin", "two");
  wait_for_timestamps_to_settle();

  InputContentRootIdentity cold;
  {
    InputContentMeasurementCache cache;
    InputContentMeasurementStats stats;
    cold = measure_cached(cache, root, stats);
    (void)cache.publish_persistent_digests(store);
  }

  std::filesystem::create_hard_link(root / "one.bin", root / "linked.bin");
  {
    InputContentMeasurementCache cache;
    (void)cache.admit_persistent_digests(store);
    InputContentMeasurementStats stats;
    const auto linked = measure_cached(cache, root, stats);
    require(linked.tree_sha256 != cold.tree_sha256,
            "a new link is a new tree");
    // The relinked inode's key moved, so its persisted record is dead and its
    // bytes are read again -- once, because the second name resolves to the
    // measurement this same lock just staged.
    require(stats.cache_misses == 1U && stats.bytes_hashed == 3U,
            "a relinked inode is read again exactly once");
    require(stats.cache_hits == 2U,
            "the untouched file and the relinked inode's second name still hit");
  }
  std::filesystem::remove(root / "linked.bin");

  std::filesystem::rename(root / "one.bin", root / "renamed.bin");
  {
    InputContentMeasurementCache cache;
    (void)cache.admit_persistent_digests(store);
    InputContentMeasurementStats stats;
    const auto renamed = measure_cached(cache, root, stats);
    require(renamed.tree_sha256 != cold.tree_sha256,
            "a rename changes the tree identity even when no byte changed");
    require(renamed.file_count == 2U,
            "a rename leaves the file count alone");
  }
}

void tampered_stores_are_refused_whole_or_per_record() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto store = temporary.path() / "digests.store";
  std::filesystem::create_directories(root);
  write_file(root / "one.bin", "one");
  wait_for_timestamps_to_settle();
  {
    InputContentMeasurementCache cache;
    InputContentMeasurementStats stats;
    (void)measure_cached(cache, root, stats);
    (void)cache.publish_persistent_digests(store);
  }
  const std::vector<unsigned char> original = read_bytes(store);
  require(original.size() > 96U, "the published store has a header and a record");

  // Any edited byte breaks the trailing digest over the whole file.
  std::vector<unsigned char> flipped = original;
  flipped[flipped.size() - 40U] ^= 0x01U;
  overwrite_bytes(store, flipped);
  {
    InputContentMeasurementCache cache;
    const auto admitted = cache.admit_persistent_digests(store);
    require(admitted.present && !admitted.accepted &&
                admitted.admitted_entries == 0U,
            "a store whose bytes were edited is refused whole");
  }

  // Truncation is refused for the same reason, and must not read past the end.
  std::vector<unsigned char> truncated(original.begin(),
                                       original.begin() + 80);
  overwrite_bytes(store, truncated);
  {
    InputContentMeasurementCache cache;
    const auto admitted = cache.admit_persistent_digests(store);
    require(admitted.present && !admitted.accepted &&
                admitted.admitted_entries == 0U,
            "a truncated store is refused rather than partially decoded");
  }

  // A store sealed against another key policy describes keys this cache does
  // not compute, so it is refused even though its own trailer is intact.
  overwrite_bytes(store, original);
  {
    InputContentMeasurementCache narrow(4U);
    require(narrow.policy_digest() != InputContentMeasurementCache().policy_digest(),
            "capacity is part of the cache policy digest");
    const auto admitted = narrow.admit_persistent_digests(store);
    require(admitted.present && !admitted.accepted,
            "a store sealed against another policy is refused");
  }
}

void unsafe_store_ownership_is_reported_rather_than_ignored() {
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "root";
  const auto directory = temporary.path() / "cache";
  std::filesystem::create_directories(root);
  std::filesystem::create_directories(directory);
  write_file(root / "one.bin", "one");
  const auto store = directory / "digests.store";
  {
    InputContentMeasurementCache cache;
    InputContentMeasurementStats stats;
    (void)measure_cached(cache, root, stats);
    (void)cache.publish_persistent_digests(store);
  }

  if (::chmod(store.c_str(), 0644) != 0)
    throw std::runtime_error("could not relax the store mode");
  require_rejected(
      [&] {
        InputContentMeasurementCache cache;
        (void)cache.admit_persistent_digests(store);
      },
      "a world-readable store must be refused, not silently ignored");
  if (::chmod(store.c_str(), 0600) != 0)
    throw std::runtime_error("could not restore the store mode");

  const auto second_link = directory / "digests.link";
  std::filesystem::create_hard_link(store, second_link);
  require_rejected(
      [&] {
        InputContentMeasurementCache cache;
        (void)cache.admit_persistent_digests(store);
      },
      "a store reachable under a second name must be refused");
  std::filesystem::remove(second_link);

  if (::chmod(directory.c_str(), 0777) != 0)
    throw std::runtime_error("could not relax the store directory mode");
  require_rejected(
      [&] {
        InputContentMeasurementCache cache;
        (void)cache.admit_persistent_digests(store);
      },
      "a store in a world-writable directory must be refused");
  require_rejected(
      [&] {
        InputContentMeasurementCache cache;
        (void)cache.publish_persistent_digests(store);
      },
      "publication into a world-writable directory must be refused");
  if (::chmod(directory.c_str(), 0700) != 0)
    throw std::runtime_error("could not restore the store directory mode");

  require_rejected(
      [&] {
        InputContentMeasurementCache cache;
        (void)cache.admit_persistent_digests("relative/store");
      },
      "a store path that is not absolute and normalized must be refused");

  // A symlink standing in for the store is never followed.
  const auto elsewhere = temporary.path() / "elsewhere.store";
  write_file(elsewhere, "not a store");
  const auto linked = directory / "linked.store";
  std::filesystem::create_symlink(elsewhere, linked);
  InputContentMeasurementCache cache;
  const auto admitted = cache.admit_persistent_digests(linked);
  require(!admitted.present && !admitted.accepted,
          "a symlinked store is not followed");
}

int main() {
  try {
    unrelated_activity_in_an_ancestor_is_not_substitution();
    std::cout << "PASS ancestor-activity-not-substitution\n";
    deterministic_nested_tree_and_mutation();
    std::cout << "PASS deterministic-mutation\n";
    file_root_and_zero_byte_leaf();
    std::cout << "PASS file-root-zero-byte\n";
    symlinks_and_special_nodes_are_rejected();
    std::cout << "PASS symlink-special\n";
    empty_invalid_name_and_unnormalized_roots_are_rejected();
    std::cout << "PASS empty-name-path\n";
    root_sets_are_sorted_measured_and_nonoverlapping();
    std::cout << "PASS root-set\n";
    warm_measurements_reuse_bytes_without_reusing_namespace();
    std::cout << "PASS warm-cache-mutation\n";
    unknown_filesystems_and_mount_incarnations_fail_safe();
    std::cout << "PASS cache-filesystem-incarnation\n";
    transactions_lru_and_corruption_are_bounded_and_safe();
    std::cout << "PASS cache-transaction-lru-corruption\n";
    abandoned_cache_hits_do_not_change_lru_order();
    std::cout << "PASS cache-abandoned-hit-lru\n";
    transaction_staging_is_bounded_and_reported();
    std::cout << "PASS cache-bounded-staging\n";
    concurrent_cached_mutation_is_rejected();
    std::cout << "PASS cache-concurrent-mutation\n";
    failed_publication_is_atomic();
    std::cout << "PASS cache-atomic-publication\n";
    failed_warm_touch_publication_is_atomic();
    std::cout << "PASS cache-atomic-warm-touch\n";
    persistent_store_reuses_only_unchanged_bytes();
    std::cout << "PASS store-warm-reuse\n";
    changed_bytes_cannot_reuse_a_digest_under_a_forged_mtime();
    std::cout << "PASS store-forged-mtime\n";
    a_recreated_file_cannot_inherit_a_dead_records_digest();
    std::cout << "PASS store-recreated-inode\n";
    racily_recent_measurements_are_withheld();
    std::cout << "PASS store-racily-recent\n";
    hardlink_and_rename_games_do_not_reuse_an_identity();
    std::cout << "PASS store-link-rename\n";
    tampered_stores_are_refused_whole_or_per_record();
    std::cout << "PASS store-tamper\n";
    unsafe_store_ownership_is_reported_rather_than_ignored();
    std::cout << "PASS store-ownership\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "input content authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
