#include "trainvm/input_content_authority.hpp"

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

}  // namespace

int main() {
  try {
    deterministic_nested_tree_and_mutation();
    std::cout << "PASS deterministic-mutation\n";
    file_root_and_zero_byte_leaf();
    std::cout << "PASS file-root-zero-byte\n";
    symlinks_and_special_nodes_are_rejected();
    std::cout << "PASS symlink-special\n";
    empty_invalid_name_and_unnormalized_roots_are_rejected();
    std::cout << "PASS empty-name-path\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "input content authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
