#include "trainvm/input_content_authority.hpp"
#include "trainvm/reflection_json.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
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

  InputContentMeasurementStats cold_stats;
  const auto cold = measure_input_content_root(root, &cold_stats);
  require(cold_stats.cache_hits == 0U && cold_stats.cache_misses == 1U &&
              cold_stats.cache_bypasses == 0U &&
              cold_stats.bytes_hashed == 64U * 1024U &&
              cold_stats.elapsed_nanoseconds > 0U,
          "cold cache telemetry accounts for every hashed byte");

  InputContentMeasurementStats warm_stats;
  const auto warm = measure_input_content_root(root, &warm_stats);
  require(warm == cold && warm_stats.cache_hits == 1U &&
              warm_stats.cache_misses == 0U && warm_stats.bytes_hashed == 0U,
          "warm measurement preserves identity without rereading file bytes");
  require(warm_stats.elapsed_nanoseconds > 0U,
          "warm measurement records elapsed time");

  const auto original_time = std::filesystem::last_write_time(payload);
  write_file(payload, std::string(64U * 1024U, 'b'));
  std::filesystem::last_write_time(payload, original_time);
  InputContentMeasurementStats changed_stats;
  const auto changed = measure_input_content_root(root, &changed_stats);
  require(changed.tree_sha256 != warm.tree_sha256 &&
              changed_stats.cache_hits == 0U &&
              changed_stats.bytes_hashed == 64U * 1024U,
          "same-size mutation with restored mtime cannot reuse cached bytes");

  const auto replacement = root / "replacement.bin";
  write_file(replacement, std::string(64U * 1024U, 'c'));
  std::filesystem::rename(replacement, payload);
  InputContentMeasurementStats replaced_stats;
  const auto replaced = measure_input_content_root(root, &replaced_stats);
  require(replaced.tree_sha256 != changed.tree_sha256 &&
              replaced_stats.cache_hits == 0U &&
              replaced_stats.bytes_hashed == 64U * 1024U,
          "inode replacement cannot reuse cached bytes");

  write_file(root / "new-member", "member");
  InputContentMeasurementStats membership_stats;
  const auto membership = measure_input_content_root(root, &membership_stats);
  require(membership.file_count == 2U &&
              membership.tree_sha256 != replaced.tree_sha256 &&
              membership_stats.cache_hits == 1U &&
              membership_stats.bytes_hashed == 6U,
          "directory membership is always enumerated while unchanged leaves reuse");
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "input content authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
