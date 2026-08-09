// A native source may not name one machine's directories.
//
// This is the C++ counterpart of the rule PR #103 established for the Python
// trainer defaults, and it exists because the same defect was found twice in a
// day in this suite. `lm_recipe_profiles_tests` hardcoded the deployment host's
// checkout, reached catalog validation, and aborted in hosted CI with
// "training component configuration path is unavailable" while passing locally.
// `recipe_profile_tests` had the identical literal and passed only because
// nothing on its path calls filesystem::exists -- one added assertion away from
// the same failure. Five more sites had the same shape.
//
// That failure mode is the expensive kind: it depends on which machine you are
// on, so it is green for whoever writes it and red in CI on somebody else's
// branch, with a message that points at configuration rather than at the test.
// Catching it at authoring time costs one string search.
//
// What is allowed instead: read the path out of the checked-in fixture that
// declares it. The recipes pin `allowed_read_roots` to the deployment host, so
// a path override cannot simply be redirected -- expansion rejects it before
// anything else runs. Taking the root from the fixture states that requirement
// directly, and the fixture travels with the repository. Where a real file is
// needed, redirect the already-expanded plan and recompile, as
// `checked_in_qwen_example_expands_without_source_changes` does.
//
// The prefixes below are matched anywhere in the file, comments included. A
// comment that needs to discuss one of these directories should describe it
// rather than spell it, so that a search for the literal keeps meaning
// "somewhere that depends on this host".

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

// Assembled from fragments so this file does not trip its own search, and so a
// reader grepping the tree for a host path is not led here first. The skip of
// this file by name below is what actually keeps the guard honest; the split is
// only to keep an ordinary `grep` over the tree readable.
std::vector<std::string> forbidden_prefixes() {
  return {
      std::string("/the") + "array/",
      std::string("/ho") + "me/",
      std::string("/Us") + "ers/",
      std::string("/mnt/") + "hypercard",
      std::string("/me") + "dia/",
  };
}

const std::vector<std::string>& scanned_directories() {
  static const std::vector<std::string> value = {"trainvm/src", "trainvm/include",
                                                 "trainvm/tests"};
  return value;
}

bool is_native_source(const std::filesystem::path& path) {
  const std::filesystem::path extension = path.extension();
  return extension == ".cpp" || extension == ".hpp" || extension == ".h" ||
         extension == ".cc";
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not read " + path.string());
  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
  const std::filesystem::path root(TRAINVM_SOURCE_ROOT);
  const std::filesystem::path self =
      std::filesystem::path(__FILE__).filename();
  const auto prefixes = forbidden_prefixes();

  std::size_t scanned = 0;
  std::vector<std::string> offences;
  for (const std::string& directory : scanned_directories()) {
    const std::filesystem::path base = root / directory;
    check(std::filesystem::is_directory(base),
          "every scanned native source directory must exist");
    if (!std::filesystem::is_directory(base)) continue;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(base)) {
      if (!entry.is_regular_file() || !is_native_source(entry.path())) continue;
      if (entry.path().filename() == self) continue;
      ++scanned;
      const std::string text = read_file(entry.path());
      for (const std::string& prefix : prefixes) {
        if (text.find(prefix) == std::string::npos) continue;
        offences.push_back(
            std::filesystem::relative(entry.path(), root).string() +
            " names a host-specific directory (" + prefix + ")");
      }
    }
  }

  // A guard that scans nothing passes for the wrong reason. This suite is
  // hundreds of files; anything near zero means the walk broke, not that the
  // tree got clean.
  check(scanned > 100U,
        "guard must actually walk the native sources it claims to cover");

  for (const std::string& offence : offences) {
    std::cerr << "FAIL: " << offence << '\n';
    ++failures;
  }
  if (!offences.empty())
    std::cerr << "Read a path out of the checked-in fixture that declares it, "
                 "or redirect the expanded plan before validating it. See the "
                 "comment at the top of "
              << self.string() << ".\n";

  if (failures == 0) {
    std::cout << "native source host path tests passed (" << scanned
              << " sources scanned)\n";
    return 0;
  }
  std::cerr << failures << " native source host path test(s) failed\n";
  return 1;
}
