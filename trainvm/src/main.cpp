#include "trainvm/cache_artifact_authority.hpp"
#include "trainvm/compatibility_catalog.hpp"
#include "trainvm/document.hpp"
#include "trainvm/exit_status.hpp"
#include "trainvm/experiment_analysis.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/recipe_profile.hpp"
#include "trainvm/run_authoring.hpp"
#include "trainvm/run_authoring_cli.hpp"
#include "trainvm/rwkv_lab_worker_contract.hpp"
#include "trainvm/input_content_authority.hpp"
#include "trainvm/service.hpp"
#include "trainvm/training_schedules.hpp"
#include "trainvm/training_preflight.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <map>
#include <ranges>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/json.hpp"

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  trainvm validate <experiment.json>\n"
      << "  trainvm plan <experiment.json> [--canonical]\n"
      << "  trainvm preflight <experiment.json> <passive-environment.json>\n"
      << "  trainvm run <author-run.json|yaml> [--dry-run]\n"
      << "  trainvm compile  # read JSON from stdin; emit canonical preview JSON\n"
      << "  trainvm validate-catalog <compatibility.json> <repository-root>\n"
      << "  trainvm print-catalog-digests <compatibility.json> <repository-root>"
         "  # compute the pinned digests; does not check them\n"
      << "  trainvm qualify-evidence"
         "  # read trainvm.cache-qualification-evidence/v1 JSON from stdin;"
         " exit 0 qualified, 3 rejected\n"
      << "  trainvm inspect-training-components <training-components.json>\n"
      << "  trainvm recipe inspect <recipe-profiles.json>\n"
      << "  trainvm recipe expand <recipe-profiles.json> <instance.json>\n"
      << "  trainvm recipe diff <recipe-profiles.json> <left.json> <right.json>\n"
      << "  trainvm inspect-hostd-client <hostd-client.json>\n"
      << "  trainvm inspect-input-content-root <absolute-path>\n"
      << "  trainvm lock-input-content <experiment.json> <root-set.json> "
         "[--content-cache <store>]\n"
         "      # --content-cache reuses digests for files whose identity and "
         "metadata\n"
         "      # are unchanged. The store must live in an owner-only "
         "directory.\n"
      << "  trainvm inspect-rwkv-lab-worker <sha256-code-fingerprint>\n"
      << "  trainvm inspect-rwkv-lab-runtime-requirements\n"
      << "  trainvm inspect-rwkv-lab-deployment"
         "  # read trainvm.rwkv-lab-worker-runtimes/v1 JSON from stdin\n"
      << "  trainvm inspect-training-schedule <implementation-id> "
         "<json-config> <max-step>\n"
      << "  trainvm inspect-registry <experiments.db> [--task <task>] "
         "[--metric <metric>] [--baseline <config>] [--limit <count>]\n"
      << "  trainvm serve --journal <journal.db> --socket <trainvm.sock> "
         "--registry <adapters.json> --host-launch-registry "
         "<host-launches.json> [--training-component-registry "
         "<training-components.json>] [--hostd-client <hostd-client.json>] "
         "[--worker-socket-gid <gid>] "
         "[--recipe-registry <recipe-profiles.json>] "
         "[--cache-evidence-root <receipts/>]\n"
      << "  trainvm simulate <experiment.json> <events.jsonl> [run-id]\n"
      << "  trainvm journal init <journal.db>\n"
      << "  trainvm journal verify <journal.db>\n"
      << "  trainvm journal replay <journal.db>\n"
      << "  trainvm journal show <journal.db> <run-id>\n"
      // A subcommand that the docs name but this list omits is the signature of
      // a stale binary, not a typo — and it reads as a typo, which is why it
      // costs time. `trainvm/build/` is gitignored, so a checkout can carry a
      // months-old binary that `git pull` never refreshes.
      << "\nif a documented subcommand is missing above, this binary predates"
         " it. trainvm/build/ is gitignored and is not refreshed by git pull;"
         " rebuild with:\n"
         "  cmake --build trainvm/build -j \"$(nproc)\" --target trainvm\n"
      // Printed here because a caller that needs the vocabulary is usually a
      // wrapper script, and a wrapper script's author reads the usage dump
      // before it reads a header. The authority is
      // trainvm/include/trainvm/exit_status.hpp; this restates it.
      << "\nexit status, with the same meaning for every subcommand:\n"
         "  0   success\n"
         "  1   uncaught exception; trainvm broke, and this says nothing"
         " about the input\n"
         "  2   malformed input; the document could not be read as what it"
         " claims to be\n"
         "  3   negative verdict; the document was read and the answer is no\n"
         "  4   not found; the document was read and what it named does not"
         " exist\n"
         "  64  usage; the argument vector itself was wrong\n";
}

nlohmann::json read_stdin_json() {
  nlohmann::json value;
  std::cin >> value;
  if (!std::cin) {
    throw std::runtime_error("could not read experiment JSON from stdin");
  }
  return value;
}

void print_diagnostics(const trainvm::CompileResult& result) {
  std::cerr << trainvm::diagnostics_json(result.diagnostics).dump(2) << '\n';
}

int validate_command(const std::filesystem::path& path) {
  auto result = trainvm::compile_document_file(path);
  if (!result.valid()) {
    print_diagnostics(result);
    return trainvm::kExitMalformedInput;
  }
  nlohmann::json output{{"valid", true},
                        {"experiment", result.plan->experiment.metadata.name},
                        {"plan_hash", result.plan->plan_hash},
                        {"warnings", trainvm::diagnostics_json(result.diagnostics)}};
  std::cout << output.dump(2) << '\n';
  return trainvm::kExitSuccess;
}

int plan_command(int argc, char** argv) {
  auto result = trainvm::compile_document_file(argv[2]);
  if (!result.valid()) {
    print_diagnostics(result);
    return trainvm::kExitMalformedInput;
  }
  const bool canonical = argc == 4 && std::string_view(argv[3]) == "--canonical";
  if (argc > 3 && !canonical) {
    usage();
    return trainvm::kExitUsage;
  }
  nlohmann::json output = trainvm::plan_summary(*result.plan);
  if (canonical) {
    output["canonical_plan"] = result.plan->canonical_plan;
  }
  if (!result.diagnostics.empty()) {
    output["diagnostics"] = trainvm::diagnostics_json(result.diagnostics);
  }
  std::cout << output.dump(2) << '\n';
  return trainvm::kExitSuccess;
}

int preflight_command(const std::filesystem::path& experiment_path,
                      const std::filesystem::path& environment_path) {
  const auto compiled = trainvm::compile_document_file(experiment_path);
  if (!compiled.valid() || !compiled.plan) {
    print_diagnostics(compiled);
    return trainvm::kExitMalformedInput;
  }
  const auto environment =
      trainvm::load_training_preflight_environment(environment_path);
  const auto receipt =
      trainvm::run_training_preflight(*compiled.plan, environment);
  std::cout << trainvm::encode_json(receipt).dump(2) << '\n';
  return receipt.passed ? trainvm::kExitSuccess : trainvm::kExitNegativeVerdict;
}

std::string read_bounded_author_run(const std::filesystem::path& path) {
  constexpr std::uintmax_t maximum = 2U * 1024U * 1024U;
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size == 0U || size > maximum) {
    throw std::runtime_error(
        "author-run document must be a readable nonempty file at most 2 MiB");
  }
  std::ifstream input(path, std::ios::binary);
  std::string source(static_cast<std::size_t>(size), '\0');
  input.read(source.data(), static_cast<std::streamsize>(source.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(source.size()))
    throw std::runtime_error("author-run document could not be read exactly");
  return source;
}

std::string author_run_source_format(const std::filesystem::path& path) {
  const std::string extension = path.extension().string();
  if (extension == ".json")
    return "json";
  if (extension == ".yaml" || extension == ".yml")
    return "yaml";
  throw std::runtime_error(
      "author-run document extension must be .json, .yaml, or .yml");
}

int author_run_command(int argc, char** argv) {
  const bool dry_run = argc == 4 && std::string_view(argv[3]) == "--dry-run";
  if ((argc != 3 && argc != 4) || (argc == 4 && !dry_run)) {
    usage();
    return trainvm::kExitUsage;
  }
  const std::filesystem::path path(argv[2]);
  const trainvm::AuthoringClientConfiguration configuration =
      trainvm::load_authoring_client_configuration();
  trainvm::v1::AuthorRunRequest request;
  request.set_request_document(read_bounded_author_run(path));
  request.set_source_format(author_run_source_format(path));

  auto channel = grpc::CreateChannel(configuration.controller_target,
                                     grpc::InsecureChannelCredentials());
  auto stub = trainvm::v1::TrainVM::NewStub(channel);
  const auto invoke = [&](const trainvm::v1::AuthorRunRequest &invocation) {
    grpc::ClientContext context;
    auto stream = stub->AuthorRun(&context, invocation);
    trainvm::AuthorRunStreamValidator validator(
        invocation.dry_run(),
        invocation.expected_plan_hash().empty()
            ? std::nullopt
            : std::optional<std::string>(invocation.expected_plan_hash()));
    trainvm::v1::AuthorRunUpdate update;
    while (stream->Read(&update)) {
      validator.observe(update);
      const nlohmann::json output = trainvm::author_run_update_json(
          update, configuration.dashboard_base_url);
      std::cout << output.dump() << '\n';
    }
    const grpc::Status status = stream->Finish();
    if (!status.ok())
      throw std::runtime_error("controller RPC failed: " +
                               status.error_message());
    return validator.finish();
  };

  // An ordinary launch is deliberately two authority calls. The first is a
  // non-mutating, complete preview; the second is fenced to that exact plan.
  // Static inputs are remeasured on both calls as the fail-closed fallback
  // until the authority-owned Merkle cache can prove a reusable lock.
  request.set_dry_run(true);
  const trainvm::AuthorRunStreamSummary preview = invoke(request);
  if (preview.failed)
    return trainvm::kExitNegativeVerdict;
  if (dry_run)
    return trainvm::kExitSuccess;

  request.set_dry_run(false);
  request.set_expected_plan_hash(preview.plan_hash);
  const trainvm::AuthorRunStreamSummary launch = invoke(request);
  return launch.failed ? trainvm::kExitNegativeVerdict : trainvm::kExitSuccess;
}

int compile_command() {
  const auto result = trainvm::compile_document(read_stdin_json());
  if (!result.valid()) {
    std::cout << nlohmann::json({{"valid", false},
                                 {"diagnostics", trainvm::diagnostics_json(result.diagnostics)}}).dump(2)
              << '\n';
    return trainvm::kExitMalformedInput;
  }
  nlohmann::json output = trainvm::plan_summary(*result.plan);
  output["valid"] = true;
  output["canonical_plan"] = result.plan->canonical_plan;
  output["diagnostics"] = trainvm::diagnostics_json(result.diagnostics);
  std::cout << output.dump(2) << '\n';
  return trainvm::kExitSuccess;
}

// Every way a catalog can be wrong -- unreadable file, unparseable JSON, a
// schema the reflected decoder rejects, an entry that does not match the tree,
// and a catalog digest that differs from the compiled reviewed mapping --
// arrives as std::invalid_argument. Catching it here rather than letting it
// reach main is what separates "the catalog is wrong" from "trainvm crashed";
// they were the same code before, both 1 via the top-level catch.
//
// The message is still written to stderr in the same form main would have used,
// because CLAUDE.md's pin-refresh procedure runs `validate-catalog` expressly to
// read the value to pin out of that message. This changes the status it exits
// with, not whether it fails or what it prints.
int report_invalid_catalog(const std::invalid_argument& error) {
  std::cerr << "trainvm: " << error.what() << '\n';
  return trainvm::kExitMalformedInput;
}

int validate_catalog_command(const std::filesystem::path& catalog_path,
                             const std::filesystem::path& repository_root) {
  std::optional<trainvm::CompatibilityCatalog> loaded;
  try {
    loaded.emplace(trainvm::CompatibilityCatalog::load_file(
        std::filesystem::absolute(catalog_path).lexically_normal(),
        std::filesystem::absolute(repository_root).lexically_normal()));
  } catch (const std::invalid_argument& error) {
    return report_invalid_catalog(error);
  }
  const trainvm::CompatibilityCatalog& catalog = *loaded;
  std::cout << nlohmann::json{
                   {"valid", true},
                   {"authority", trainvm::enum_to_string(catalog.authority())},
                   {"entries", catalog.entries().size()},
                   {"catalog_digest", catalog.catalog_digest()},
                   {"source_tree_digest", catalog.source_tree_digest()},
                   {"classification_surface_digest",
                    catalog.classification_surface_digest()},
                   {"repository_root_identity",
                    catalog.repository_root_identity_display()},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

// The generator the catalog never had. Both pinned digests were maintained by
// hand, so refreshing one after an unrelated source edit meant hashing files
// yourself -- which is how re-pinning became a chore performed without the
// review it was supposed to force.
int print_catalog_digests_command(
    const std::filesystem::path& catalog_path,
    const std::filesystem::path& repository_root) {
  trainvm::CompatibilityCatalogComputedDigests computed;
  try {
    computed = trainvm::CompatibilityCatalog::compute_digests(
        std::filesystem::absolute(catalog_path).lexically_normal(),
        std::filesystem::absolute(repository_root).lexically_normal());
  } catch (const std::invalid_argument& error) {
    return report_invalid_catalog(error);
  }
  std::cout << nlohmann::json{
                   {"source_tree_digest", computed.source_tree_digest},
                   {"classification_surface_digest",
                    computed.classification_surface_digest},
                   {"paths_with_empty_classification_surface",
                    computed.paths_with_empty_classification_surface},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

// Runs the implemented qualification gate over caller-supplied evidence and
// prints the resulting receipt. A benchmark runner is not an authority: it
// measures, and this decides. Exposing the real gate is what stops a runner
// reimplementing the thresholds and quietly disagreeing with the qualify_cache
// node that actually admits an optimization.
//
// Exit status is the verdict: kExitSuccess qualified, kExitNegativeVerdict
// rejected with reasons, kExitMalformedInput for evidence that is not a
// readable trainvm.cache-qualification-evidence/v1 document. A rejection is a
// normal, reportable outcome and must be distinguishable from a broken
// document.
//
// Malformed evidence used to answer 1, which is also what main returns for an
// uncaught exception -- so the one case this comment insists must be
// distinguishable was, for the malformed half, indistinguishable from a crash.
// It is 2 now, matching `validate` and every other command that rejects a
// document it could not read.
int qualify_evidence_command(std::istream& input) {
  nlohmann::json document;
  try {
    input >> document;
  } catch (const std::exception& error) {
    std::cerr << "qualification evidence is not valid JSON: " << error.what()
              << '\n';
    return trainvm::kExitMalformedInput;
  }
  trainvm::CacheQualificationEvidence evidence{};
  std::vector<trainvm::Diagnostic> diagnostics;
  if (!trainvm::decode_json(document, evidence, "", diagnostics) ||
      !diagnostics.empty() || trainvm::encode_json(evidence) != document) {
    std::cerr << "qualification evidence has an invalid reflected schema; "
                 "it must be exactly trainvm.cache-qualification-evidence/v1\n";
    for (const auto& diagnostic : diagnostics) {
      std::cerr << "  " << diagnostic.code << " " << diagnostic.path << " "
                << diagnostic.message << '\n';
    }
    return trainvm::kExitMalformedInput;
  }
  const trainvm::CacheQualificationReceipt receipt =
      trainvm::qualify_cache_artifact(std::move(evidence));
  std::cout << trainvm::cache_qualification_receipt_json(receipt).dump(2)
            << '\n';
  return receipt.qualified ? trainvm::kExitSuccess : trainvm::kExitNegativeVerdict;
}

int inspect_training_components_command(
    const std::filesystem::path& registry_path) {
  const trainvm::TrainingComponentRegistry registry =
      trainvm::TrainingComponentRegistry::load_file(
          std::filesystem::absolute(registry_path).lexically_normal());
  const nlohmann::json document = registry.document_json();
  std::cout << nlohmann::json{
                   {"valid", true},
                   {"components", document.at("components").size()},
                   {"registry_digest", registry.registry_digest()},
                   {"canonical_registry", document},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

int inspect_hostd_client_command(const std::filesystem::path& path) {
  const trainvm::HostdClientConfiguration configuration =
      trainvm::HostdClientConfiguration::load_file(
          std::filesystem::absolute(path).lexically_normal());
  std::cout << nlohmann::json{
                   {"valid", true},
                   {"canonical_configuration",
                    trainvm::encode_json(configuration.document())},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

int inspect_input_content_root_command(int argc, char** argv) {
  if (argc != 3) {
    usage();
    return trainvm::kExitUsage;
  }
  std::cout << trainvm::encode_json(
                   trainvm::measure_input_content_root(argv[2]))
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

nlohmann::json read_bounded_json_file(const std::filesystem::path& path,
                                      std::string_view label) {
  constexpr std::uintmax_t kMaximumAuthoringDocumentBytes = 16U << 20U;
  const auto status = std::filesystem::symlink_status(path);
  if (!std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status))
    throw std::runtime_error(std::string(label) +
                             " must be a regular non-symlink file");
  const auto size = std::filesystem::file_size(path);
  if (size == 0U || size > kMaximumAuthoringDocumentBytes)
    throw std::runtime_error(std::string(label) + " exceeds its byte bound");
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("could not open " + std::string(label));
  nlohmann::json document;
  input >> document;
  if (!input)
    throw std::runtime_error("could not parse " + std::string(label));
  return document;
}

int recipe_command(int argc, char** argv) {
  if (argc < 4) {
    usage();
    return trainvm::kExitUsage;
  }
  const trainvm::RecipeProfileRegistry registry =
      trainvm::RecipeProfileRegistry::load_file(
          std::filesystem::absolute(argv[3]).lexically_normal());
  const std::string_view action(argv[2]);
  if (action == "inspect" && argc == 4) {
    const nlohmann::json document = registry.document_json();
    std::cout << nlohmann::json{
                     {"valid", true},
                     {"recipes", document.at("recipes").size()},
                     {"registry_digest", registry.registry_digest()},
                     {"canonical_registry", document},
                 }
                     .dump(2)
              << '\n';
    return trainvm::kExitSuccess;
  }
  if (action == "expand" && argc == 5) {
    const auto expanded = registry.expand_json(
        read_bounded_json_file(argv[4], "recipe instance"));
    std::cout << trainvm::expanded_recipe_json(expanded).dump(2) << '\n';
    return trainvm::kExitSuccess;
  }
  if (action == "diff" && argc == 6) {
    const auto left = registry.expand_json(
        read_bounded_json_file(argv[4], "left recipe instance"));
    const auto right = registry.expand_json(
        read_bounded_json_file(argv[5], "right recipe instance"));
    const auto differences = trainvm::diff_recipe_plans(left, right);
    std::cout << nlohmann::json{
                     {"api_version", "trainvm.recipe-diff/v1"},
                     {"left_plan_digest", left.expanded_plan_digest},
                     {"right_plan_digest", right.expanded_plan_digest},
                     {"difference_count", differences.size()},
                     {"differences",
                      trainvm::recipe_plan_diff_json(differences)},
                 }
                     .dump(2)
              << '\n';
    return trainvm::kExitSuccess;
  }
  usage();
  return trainvm::kExitUsage;
}

int lock_input_content_command(
    const std::filesystem::path& experiment_path,
    const std::filesystem::path& root_set_path,
    const std::optional<std::filesystem::path>& content_cache_path) {
  nlohmann::json experiment =
      read_bounded_json_file(experiment_path, "experiment document");
  const nlohmann::json root_document =
      read_bounded_json_file(root_set_path, "input content root set");
  trainvm::InputContentRootSet root_set;
  std::vector<trainvm::Diagnostic> diagnostics;
  if (!trainvm::decode_json(root_document, root_set, "", diagnostics) ||
      !diagnostics.empty() || trainvm::encode_json(root_set) != root_document)
    throw std::invalid_argument(
        "input content root set does not match its reflected schema");

  // The cache is loaded before the walk and published only after the document
  // it produced compiles, so a lock that is refused leaves no measurement
  // behind to be reused by the next one.
  std::optional<trainvm::InputContentMeasurementCache> content_cache;
  std::optional<trainvm::InputContentMeasurementTransaction> transaction;
  if (content_cache_path) {
    content_cache.emplace();
    (void)content_cache->admit_persistent_digests(*content_cache_path);
    transaction.emplace(content_cache->begin_transaction());
  }
  const auto identities = trainvm::measure_input_content_root_set(
      root_set, nullptr, transaction ? &*transaction : nullptr);
  if (!experiment.is_object() || !experiment.contains("spec") ||
      !experiment.at("spec").is_object() ||
      !experiment.at("spec").contains("workspace") ||
      !experiment.at("spec").at("workspace").is_object())
    throw std::invalid_argument("experiment workspace is unavailable");
  experiment["spec"]["workspace"]["input_content_roots"] =
      trainvm::encode_json(identities);
  const auto compiled = trainvm::compile_document(experiment);
  if (!compiled.valid()) {
    print_diagnostics(compiled);
    return trainvm::kExitMalformedInput;
  }
  if (content_cache) {
    (void)transaction->commit();
    (void)content_cache->publish_persistent_digests(*content_cache_path);
  }
  std::cout << experiment.dump(2) << '\n';
  return trainvm::kExitSuccess;
}

int inspect_registry_command(int argc, char** argv) {
  trainvm::ExperimentRegistryQuery query;
  for (int index = 3; index < argc; index += 2) {
    if (index + 1 >= argc) {
      usage();
      return trainvm::kExitUsage;
    }
    const std::string_view option(argv[index]);
    if (option == "--task") {
      query.task = argv[index + 1];
    } else if (option == "--metric") {
      query.metric = argv[index + 1];
    } else if (option == "--baseline") {
      query.baseline = argv[index + 1];
    } else if (option == "--limit") {
      const std::string_view text(argv[index + 1]);
      std::size_t value{};
      const auto [end, error] =
          std::from_chars(text.data(), text.data() + text.size(), value);
      if (error != std::errc{} || end != text.data() + text.size()) {
        usage();
        return trainvm::kExitUsage;
      }
      query.campaign_limit = value;
    } else {
      usage();
      return trainvm::kExitUsage;
    }
  }
  const auto snapshot = trainvm::read_experiment_registry(argv[2], query);
  std::cout << trainvm::encode_json(snapshot).dump(2) << '\n';
  return trainvm::kExitSuccess;
}

int simulate_command(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    usage();
    return trainvm::kExitUsage;
  }
  auto compiled = trainvm::compile_document_file(argv[2]);
  if (!compiled.valid()) {
    print_diagnostics(compiled);
    return trainvm::kExitMalformedInput;
  }
  std::ifstream input(argv[3]);
  if (!input) {
    throw std::runtime_error("could not open " + std::string(argv[3]));
  }
  std::string run_id = argc == 5 ? argv[4] : "simulation";
  auto state = trainvm::start_execution(*compiled.plan, run_id);
  nlohmann::json trace = nlohmann::json::array();
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos) {
      continue;
    }
    try {
      auto event = trainvm::event_from_json(nlohmann::json::parse(line));
      const auto result = trainvm::advance_execution(*compiled.plan, state, event);
      trace.push_back({{"event_id", event.event_id},
                       {"event_type", event.event_type},
                       {"source", result.source_node_id},
                       {"target", result.target},
                       {"transition_index", result.transition_index}});
      state = result.state;
    } catch (const std::exception& exception) {
      throw std::runtime_error("simulation line " + std::to_string(line_number) + ": " + exception.what());
    }
  }
  std::cout << nlohmann::json({{"plan_hash", compiled.plan->plan_hash},
                               {"state", trainvm::execution_state_json(state)},
                               {"trace", std::move(trace)}}).dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

int journal_command(int argc, char** argv) {
  if (argc < 4) {
    usage();
    return trainvm::kExitUsage;
  }
  const std::string_view operation(argv[2]);
  const bool valid_operation =
      (operation == "init" && argc == 4) ||
      (operation == "verify" && argc == 4) ||
      (operation == "replay" && argc == 4) ||
      (operation == "show" && argc == 5);
  if (!valid_operation) {
    usage();
    return trainvm::kExitUsage;
  }
  trainvm::AuthorityLock authority_lock(argv[3]);
  trainvm::Journal journal(authority_lock.journal_path(),
                           authority_lock.journal_identity());
  if (operation == "init" && argc == 4) {
    std::cout << nlohmann::json({{"initialized", true}, {"events", journal.event_count()}}).dump(2)
              << '\n';
    return trainvm::kExitSuccess;
  }
  if (operation == "verify" && argc == 4) {
    std::string reason;
    const bool valid = journal.verify_chain(&reason);
    std::cout << nlohmann::json({{"valid", valid}, {"events", journal.event_count()}, {"reason", reason}}).dump(2)
              << '\n';
    return valid ? trainvm::kExitSuccess : trainvm::kExitNegativeVerdict;
  }
  if (operation == "replay" && argc == 4) {
    const auto replayed = journal.rebuild_projections();
    std::cout << nlohmann::json({{"replayed", replayed}, {"valid", true}}).dump(2) << '\n';
    return trainvm::kExitSuccess;
  }
  if (operation == "show" && argc == 5) {
    const auto projection = journal.projection(argv[4]);
    if (!projection) {
      std::cerr << "run not found: " << argv[4] << '\n';
      return trainvm::kExitNotFound;
    }
    std::cout << trainvm::projection_json(*projection).dump(2) << '\n';
    return trainvm::kExitSuccess;
  }
  throw std::logic_error("validated journal operation was not dispatched");
}

int serve_command(int argc, char** argv) {
  // Flag-keyed rather than positional: the previous form pinned each option to
  // a fixed argv index, so adding one meant a new exact argc and every optional
  // combination became its own arm — this branch already carried three arms for
  // two optional flags. Order no longer matters and an unknown or repeated flag
  // is rejected instead of being read as another option's value.
  std::map<std::string_view, std::string_view> options;
  static constexpr std::array<std::string_view, 4> required = {
      "--journal", "--socket", "--registry", "--host-launch-registry"};
  static constexpr std::array<std::string_view, 5> optional = {
      "--training-component-registry", "--hostd-client", "--recipe-registry",
      "--worker-socket-gid", "--cache-evidence-root"};
  if (argc % 2 != 0) {
    usage();
    return trainvm::kExitUsage;
  }
  for (int index = 2; index + 1 < argc; index += 2) {
    const std::string_view flag(argv[index]);
    const bool known =
        std::ranges::find(required, flag) != required.end() ||
        std::ranges::find(optional, flag) != optional.end();
    if (!known || !options.emplace(flag, argv[index + 1]).second) {
      usage();
      return trainvm::kExitUsage;
    }
  }
  if (std::ranges::any_of(required, [&](std::string_view flag) {
        return !options.contains(flag);
      })) {
    usage();
    return trainvm::kExitUsage;
  }

  std::optional<trainvm::HostdClientConfiguration> hostd;
  if (const auto client = options.find("--hostd-client");
      client != options.end()) {
    hostd = trainvm::HostdClientConfiguration::load_file(
        std::filesystem::absolute(client->second).lexically_normal());
  }
  std::optional<std::uint32_t> worker_socket_gid;
  if (const auto group = options.find("--worker-socket-gid");
      group != options.end()) {
    const std::string_view text(group->second);
    std::uint32_t value = 0U;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value == 0U) {
      throw std::invalid_argument(
          "--worker-socket-gid must be one nonzero numeric gid");
    }
    worker_socket_gid = value;
  }
  const auto components = options.find("--training-component-registry");
  const auto recipes = options.find("--recipe-registry");
  // Absolute and lexically normal, resolved here for the same reason
  // --hostd-client is: a relative path read from a command line means
  // whatever the daemon's working directory happened to be. The service
  // attests the rest of the root's provisioning at construction.
  std::optional<std::filesystem::path> cache_evidence_root;
  if (const auto root = options.find("--cache-evidence-root");
      root != options.end()) {
    cache_evidence_root =
        std::filesystem::absolute(root->second).lexically_normal();
  }
  return trainvm::serve(
      options["--journal"], options["--socket"],
      trainvm::AdapterRegistry::load_file(options["--registry"]),
      trainvm::HostLaunchRegistry::load_file(options["--host-launch-registry"]),
      components == options.end()
          ? trainvm::TrainingComponentRegistry({})
          : trainvm::TrainingComponentRegistry::load_file(components->second),
      std::move(hostd), worker_socket_gid,
      recipes == options.end()
          ? std::filesystem::path(
                std::string(trainvm::kInstalledRecipeProfilePath))
          : std::filesystem::path(recipes->second),
      std::move(cache_evidence_root));
}

int inspect_rwkv_lab_worker_command(std::string code_fingerprint) {
  const trainvm::RwkvLabWorkerContract contract =
      trainvm::rwkv_lab_worker_contract(std::move(code_fingerprint));
  std::cout << nlohmann::json{
                   {"adapter_registry",
                    trainvm::encode_json(contract.adapter_registry)},
                   {"provided_capabilities",
                    contract.provided_capabilities},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

int inspect_rwkv_lab_runtime_requirements_command(int argc) {
  if (argc != 2) {
    usage();
    return trainvm::kExitUsage;
  }
  std::cout << trainvm::encode_json(
                   trainvm::rwkv_lab_worker_runtime_requirements())
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

int inspect_rwkv_lab_deployment_command(int argc, char** argv) {
  (void)argv;
  if (argc != 2) {
    usage();
    return trainvm::kExitUsage;
  }
  trainvm::RwkvLabWorkerDeploymentSpec spec;
  std::vector<trainvm::Diagnostic> diagnostics;
  if (!trainvm::decode_json(read_stdin_json(), spec, "", diagnostics)) {
    std::cerr << trainvm::diagnostics_json(diagnostics).dump(2) << '\n';
    return trainvm::kExitMalformedInput;
  }
  const trainvm::RwkvLabWorkerDeploymentContract deployment =
      trainvm::rwkv_lab_worker_deployment(std::move(spec));
  const trainvm::AdapterRegistry adapters(
      deployment.adapter_registry.profiles);
  const trainvm::HostLaunchRegistry launches(
      deployment.host_launch_registry);
  std::cout << nlohmann::json{
                   {"schema", "trainvm.rwkv-lab-worker-deployment/v3"},
                   {"adapter_registry_digest", adapters.registry_digest()},
                   {"host_launch_registry_digest",
                    launches.registry_digest()},
                   {"adapter_registry",
                    trainvm::encode_json(deployment.adapter_registry)},
                   {"host_launch_registry",
                    trainvm::encode_json(deployment.host_launch_registry)},
                   {"provided_capabilities",
                    deployment.provided_capabilities},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

int inspect_training_schedule_command(int argc, char** argv) {
  if (argc != 5) {
    usage();
    return trainvm::kExitUsage;
  }

  const std::string_view max_step_text(argv[4]);
  std::int64_t max_step{};
  const auto [end, error] = std::from_chars(
      max_step_text.data(), max_step_text.data() + max_step_text.size(),
      max_step);
  if (error != std::errc{} ||
      end != max_step_text.data() + max_step_text.size() || max_step < 0 ||
      max_step > 100'000) {
    throw trainvm::TrainingScheduleError(
        "max-step must be an integer in [0, 100000]");
  }

  const std::string implementation_id(argv[2]);
  const nlohmann::json configuration = nlohmann::json::parse(argv[3]);
  std::vector<trainvm::Diagnostic> diagnostics;
  nlohmann::json multipliers = nlohmann::json::array();
  multipliers.get_ref<nlohmann::json::array_t&>().reserve(
      static_cast<std::size_t>(max_step + 1));

  if (implementation_id == "rwkv_lab.schedule.linear_warmup_cosine.v1") {
    trainvm::LinearWarmupCosineSchedule schedule;
    if (!trainvm::decode_json(configuration, schedule, "", diagnostics)) {
      std::cerr << trainvm::diagnostics_json(diagnostics).dump(2) << '\n';
      return trainvm::kExitMalformedInput;
    }
    for (std::int64_t step = 0; step <= max_step; ++step) {
      multipliers.push_back(
          trainvm::linear_warmup_cosine_multiplier(step, schedule));
    }
  } else if (implementation_id == "rwkv_lab.schedule.powercool.v1") {
    trainvm::PowerCoolSchedule schedule;
    if (!trainvm::decode_json(configuration, schedule, "", diagnostics)) {
      std::cerr << trainvm::diagnostics_json(diagnostics).dump(2) << '\n';
      return trainvm::kExitMalformedInput;
    }
    for (std::int64_t step = 0; step <= max_step; ++step) {
      multipliers.push_back(trainvm::powercool_multiplier(step, schedule));
    }
  } else {
    throw trainvm::TrainingScheduleError(
        "unsupported training schedule implementation: " + implementation_id);
  }

  std::cout << nlohmann::json{
                   {"implementation_id", implementation_id},
                   {"configuration", configuration},
                   {"max_step", max_step},
                   {"multipliers", std::move(multipliers)},
               }
                   .dump(2)
            << '\n';
  return trainvm::kExitSuccess;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "validate") {
      return validate_command(argv[2]);
    }
    if (argc >= 3 && std::string_view(argv[1]) == "plan") {
      return plan_command(argc, argv);
    }
    if (argc == 4 && std::string_view(argv[1]) == "preflight") {
      return preflight_command(argv[2], argv[3]);
    }
    if (argc >= 3 && std::string_view(argv[1]) == "run") {
      return author_run_command(argc, argv);
    }
    if (argc == 2 && std::string_view(argv[1]) == "compile") {
      return compile_command();
    }
    if (argc == 2 && std::string_view(argv[1]) == "qualify-evidence") {
      return qualify_evidence_command(std::cin);
    }
    if (argc == 4 && std::string_view(argv[1]) == "validate-catalog") {
      return validate_catalog_command(argv[2], argv[3]);
    }
    if (argc == 4 && std::string_view(argv[1]) == "print-catalog-digests") {
      return print_catalog_digests_command(argv[2], argv[3]);
    }
    if (argc == 3 &&
        std::string_view(argv[1]) == "inspect-training-components") {
      return inspect_training_components_command(argv[2]);
    }
    if (argc == 3 &&
        std::string_view(argv[1]) == "inspect-hostd-client") {
      return inspect_hostd_client_command(argv[2]);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "recipe") {
      return recipe_command(argc, argv);
    }
    if (argc >= 2 &&
        std::string_view(argv[1]) == "inspect-input-content-root") {
      return inspect_input_content_root_command(argc, argv);
    }
    if ((argc == 4 || argc == 6) &&
        std::string_view(argv[1]) == "lock-input-content") {
      std::optional<std::filesystem::path> content_cache;
      if (argc == 6) {
        if (std::string_view(argv[4]) != "--content-cache") {
          usage();
          return trainvm::kExitUsage;
        }
        content_cache = std::filesystem::path(argv[5]);
      }
      return lock_input_content_command(argv[2], argv[3], content_cache);
    }
    if (argc == 3 &&
        std::string_view(argv[1]) == "inspect-rwkv-lab-worker") {
      return inspect_rwkv_lab_worker_command(argv[2]);
    }
    if (argc >= 2 && std::string_view(argv[1]) ==
                         "inspect-rwkv-lab-runtime-requirements") {
      return inspect_rwkv_lab_runtime_requirements_command(argc);
    }
    if (argc >= 2 &&
        std::string_view(argv[1]) == "inspect-rwkv-lab-deployment") {
      return inspect_rwkv_lab_deployment_command(argc, argv);
    }
    if (argc >= 2 &&
        std::string_view(argv[1]) == "inspect-training-schedule") {
      return inspect_training_schedule_command(argc, argv);
    }
    if (argc >= 3 && std::string_view(argv[1]) == "inspect-registry") {
      return inspect_registry_command(argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "serve") {
      return serve_command(argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "simulate") {
      return simulate_command(argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "journal") {
      return journal_command(argc, argv);
    }
    usage();
    return trainvm::kExitUsage;
  } catch (const std::exception& exception) {
    // Nothing below this line decided anything: an exception reached the top,
    // so trainvm has no verdict to report. kExitUncaughtException is reserved
    // for exactly this, which is why a subcommand that *can* recognize a bad
    // document catches it locally and answers kExitMalformedInput instead.
    std::cerr << "trainvm: " << exception.what() << '\n';
    return trainvm::kExitUncaughtException;
  }
}
