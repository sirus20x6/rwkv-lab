#include "trainvm/input_content_authority.hpp"
#include "trainvm/reflection_json.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "input content authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
