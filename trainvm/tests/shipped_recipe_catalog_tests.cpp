// Runs the real `trainvm recipe inspect` load path over every recipe-profile
// catalog this repository ships.
//
// Nothing did that before. `scripts/validate_experiment_documents.py` checks
// each recipe's `template_document` against the experiment schema and resolves
// its contract names, but it never evaluates the recipe envelope -- the
// override targets, the content bindings, the compatibility rules -- and it
// cannot, because it runs in CI's schema job where no `trainvm` binary exists.
// The native suite did have recipe-profile tests, over its own fixtures. So a
// shipped catalog could be, and was, refused by the validator on main for as
// long as anyone cared to leave it there.
//
// Two properties of this file are load-bearing and easy to lose in a later
// edit:
//
//   1. Catalogs are discovered by GLOB, never by a list. A hardcoded list is
//      how the fifth catalog gets added unchecked, which is the same defect
//      one layer up.
//   2. Finding no catalogs is its own failure, distinct from "all catalogs
//      valid". A discovery loop over an empty set passes silently and reads
//      exactly like a suite that checked everything -- and this repository has
//      shipped that bug before. A gate written *because* a check was missing
//      is the worst possible place to reintroduce it.
//
// Refusals that are deliberate are recorded in
// docs/experiment-vm/recipe-catalog-exclusions.v1.json with their reason, in
// the style of unresolved-contract-exclusions.v1.json. That list is a
// countdown: a listed catalog that starts loading FAILS here, so the entry is
// deleted rather than left behind after the underlying defect is fixed.

#include "trainvm/recipe_profile.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
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

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open " + path.string());
  return nlohmann::json::parse(input);
}

// The refusal message the validator produces, or nullopt when the catalog
// loads. This is the same call `trainvm recipe inspect` makes; the subcommand
// only formats what comes back.
std::optional<std::string> refusal_for(const std::filesystem::path& catalog) {
  try {
    (void)trainvm::RecipeProfileRegistry::load_file(catalog);
    return std::nullopt;
  } catch (const trainvm::RecipeProfileError& error) {
    return std::string(error.what());
  }
}

}  // namespace

int main() {
  const std::filesystem::path root(TRAINVM_SOURCE_ROOT);
  const std::filesystem::path examples = root / "docs/experiment-vm/examples";

  std::vector<std::filesystem::path> catalogs;
  for (const auto& entry : std::filesystem::directory_iterator(examples)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().ends_with(".recipe-profiles.v1.json"))
      catalogs.push_back(entry.path());
  }
  std::ranges::sort(catalogs);

  // Property 2. Deliberately not folded into the loop below: the loop is
  // vacuously green on an empty set, and that is the failure this whole file
  // exists to not have.
  if (catalogs.empty()) {
    std::cerr << "FAIL: discovered no *.recipe-profiles.v1.json under "
              << examples
              << " -- the glob found nothing, which is not the same as every "
                 "catalog being valid\n";
    return 1;
  }
  std::cout << "discovered " << catalogs.size() << " shipped recipe catalogs\n";
  // A floor, so silently losing catalogs to a renamed suffix reads as a
  // failure rather than as a smaller clean run.
  check(catalogs.size() >= 4U,
        "fewer shipped recipe catalogs than the four on main; a rename or a "
        "deletion is hiding coverage");

  // Absence is a reported failure, not an escaping exception: a missing
  // exclusions document must read as "nothing is excused", so every refused
  // catalog below then fails by name. Fail-closed, and legible.
  const std::filesystem::path exclusions_path =
      root / "docs/experiment-vm/recipe-catalog-exclusions.v1.json";
  nlohmann::json exclusions_document{{"exclusions", nlohmann::json::array()}};
  if (!std::filesystem::exists(exclusions_path)) {
    check(false,
          "docs/experiment-vm/recipe-catalog-exclusions.v1.json is missing; "
          "no catalog refusal can be excused");
  } else {
    exclusions_document = read_json(exclusions_path);
    check(exclusions_document.at("api_version") ==
              "trainvm.recipe-catalog-exclusions/v1",
          "recipe catalog exclusions declare an unexpected api_version");
  }
  std::map<std::string, std::string> excluded;
  for (const auto& entry : exclusions_document.at("exclusions")) {
    const auto catalog = entry.at("catalog").get<std::string>();
    const auto reason = entry.at("reason").get<std::string>();
    check(reason.size() > 200U,
          "recipe catalog exclusion " + catalog +
              " records no substantive reason");
    check(entry.at("card").is_string(),
          "recipe catalog exclusion " + catalog + " cites no card");
    check(excluded.emplace(catalog, entry.at("refusal").get<std::string>())
              .second,
          "recipe catalog exclusion " + catalog + " is listed twice");
  }

  std::set<std::string> seen;
  for (const auto& catalog : catalogs) {
    const std::string name = "examples/" + catalog.filename().string();
    seen.insert(name);
    const auto excluded_entry = excluded.find(name);
    const auto refusal = refusal_for(catalog);
    if (excluded_entry == excluded.end()) {
      check(!refusal.has_value(),
            "shipped recipe catalog " + name +
                " is refused by `trainvm recipe inspect` and is not recorded "
                "in recipe-catalog-exclusions.v1.json: " +
                refusal.value_or(std::string{}));
      continue;
    }
    // The countdown. A listed catalog that loads means the defect was fixed
    // and the entry outlived it.
    check(refusal.has_value(),
          "recipe catalog " + name +
              " now loads but is still listed in "
              "recipe-catalog-exclusions.v1.json -- delete the entry");
    if (refusal)
      check(*refusal == excluded_entry->second,
            "recipe catalog " + name +
                " is refused for a different reason than the one recorded: " +
                *refusal);
  }

  for (const auto& [name, refusal] : excluded) {
    (void)refusal;
    check(seen.contains(name),
          "recipe-catalog-exclusions.v1.json names " + name +
              ", which the glob did not find");
  }

  if (failures != 0) {
    std::cerr << failures << " shipped recipe catalog checks failed\n";
    return 1;
  }
  std::cout << "shipped recipe catalog checks passed\n";
  return 0;
}
