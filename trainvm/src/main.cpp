#include "trainvm/cache_artifact_authority.hpp"
#include "trainvm/compatibility_catalog.hpp"
#include "trainvm/document.hpp"
#include "trainvm/experiment_analysis.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/rwkv_lab_worker_contract.hpp"
#include "trainvm/input_content_authority.hpp"
#include "trainvm/service.hpp"
#include "trainvm/training_schedules.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  trainvm validate <experiment.json>\n"
      << "  trainvm plan <experiment.json> [--canonical]\n"
      << "  trainvm compile  # read JSON from stdin; emit canonical preview JSON\n"
      << "  trainvm validate-catalog <compatibility.json> <repository-root>\n"
      << "  trainvm print-catalog-digests <compatibility.json> <repository-root>"
         "  # compute the pinned digests; does not check them\n"
      << "  trainvm qualify-evidence"
         "  # read trainvm.cache-qualification-evidence/v1 JSON from stdin;"
         " exit 0 qualified, 3 rejected\n"
      << "  trainvm inspect-training-components <training-components.json>\n"
      << "  trainvm inspect-hostd-client <hostd-client.json>\n"
      << "  trainvm inspect-input-content-root <absolute-path>\n"
      << "  trainvm lock-input-content <experiment.json> <root-set.json>\n"
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
         "<host-launches.json> --training-component-registry "
         "<training-components.json> [--hostd-client <hostd-client.json>]\n"
      << "  trainvm simulate <experiment.json> <events.jsonl> [run-id]\n"
      << "  trainvm journal init <journal.db>\n"
      << "  trainvm journal verify <journal.db>\n"
      << "  trainvm journal replay <journal.db>\n"
      << "  trainvm journal show <journal.db> <run-id>\n";
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
    return 2;
  }
  nlohmann::json output{{"valid", true},
                        {"experiment", result.plan->experiment.metadata.name},
                        {"plan_hash", result.plan->plan_hash},
                        {"warnings", trainvm::diagnostics_json(result.diagnostics)}};
  std::cout << output.dump(2) << '\n';
  return 0;
}

int plan_command(int argc, char** argv) {
  auto result = trainvm::compile_document_file(argv[2]);
  if (!result.valid()) {
    print_diagnostics(result);
    return 2;
  }
  const bool canonical = argc == 4 && std::string_view(argv[3]) == "--canonical";
  if (argc > 3 && !canonical) {
    usage();
    return 64;
  }
  nlohmann::json output = trainvm::plan_summary(*result.plan);
  if (canonical) {
    output["canonical_plan"] = result.plan->canonical_plan;
  }
  if (!result.diagnostics.empty()) {
    output["diagnostics"] = trainvm::diagnostics_json(result.diagnostics);
  }
  std::cout << output.dump(2) << '\n';
  return 0;
}

int compile_command() {
  const auto result = trainvm::compile_document(read_stdin_json());
  if (!result.valid()) {
    std::cout << nlohmann::json({{"valid", false},
                                 {"diagnostics", trainvm::diagnostics_json(result.diagnostics)}}).dump(2)
              << '\n';
    return 2;
  }
  nlohmann::json output = trainvm::plan_summary(*result.plan);
  output["valid"] = true;
  output["canonical_plan"] = result.plan->canonical_plan;
  output["diagnostics"] = trainvm::diagnostics_json(result.diagnostics);
  std::cout << output.dump(2) << '\n';
  return 0;
}

int validate_catalog_command(const std::filesystem::path& catalog_path,
                             const std::filesystem::path& repository_root) {
  const auto catalog = trainvm::CompatibilityCatalog::load_file(
      std::filesystem::absolute(catalog_path).lexically_normal(),
      std::filesystem::absolute(repository_root).lexically_normal());
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
  return 0;
}

// The generator the catalog never had. Both pinned digests were maintained by
// hand, so refreshing one after an unrelated source edit meant hashing files
// yourself -- which is how re-pinning became a chore performed without the
// review it was supposed to force.
int print_catalog_digests_command(
    const std::filesystem::path& catalog_path,
    const std::filesystem::path& repository_root) {
  const auto computed = trainvm::CompatibilityCatalog::compute_digests(
      std::filesystem::absolute(catalog_path).lexically_normal(),
      std::filesystem::absolute(repository_root).lexically_normal());
  std::cout << nlohmann::json{
                   {"source_tree_digest", computed.source_tree_digest},
                   {"classification_surface_digest",
                    computed.classification_surface_digest},
                   {"paths_with_empty_classification_surface",
                    computed.paths_with_empty_classification_surface},
               }
                   .dump(2)
            << '\n';
  return 0;
}

// Runs the implemented qualification gate over caller-supplied evidence and
// prints the resulting receipt. A benchmark runner is not an authority: it
// measures, and this decides. Exposing the real gate is what stops a runner
// reimplementing the thresholds and quietly disagreeing with the qualify_cache
// node that actually admits an optimization.
//
// Exit status is the verdict: 0 qualified, 3 rejected with reasons, 1 for
// malformed evidence. A rejection is a normal, reportable outcome and must be
// distinguishable from a broken document.
int qualify_evidence_command(std::istream& input) {
  nlohmann::json document;
  try {
    input >> document;
  } catch (const std::exception& error) {
    std::cerr << "qualification evidence is not valid JSON: " << error.what()
              << '\n';
    return 1;
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
    return 1;
  }
  const trainvm::CacheQualificationReceipt receipt =
      trainvm::qualify_cache_artifact(std::move(evidence));
  std::cout << trainvm::cache_qualification_receipt_json(receipt).dump(2)
            << '\n';
  return receipt.qualified ? 0 : 3;
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
  return 0;
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
  return 0;
}

int inspect_input_content_root_command(int argc, char** argv) {
  if (argc != 3) {
    usage();
    return 64;
  }
  std::cout << trainvm::encode_json(
                   trainvm::measure_input_content_root(argv[2]))
                   .dump(2)
            << '\n';
  return 0;
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

int lock_input_content_command(const std::filesystem::path& experiment_path,
                               const std::filesystem::path& root_set_path) {
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
  const auto identities = trainvm::measure_input_content_root_set(root_set);
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
    return 2;
  }
  std::cout << experiment.dump(2) << '\n';
  return 0;
}

int inspect_registry_command(int argc, char** argv) {
  trainvm::ExperimentRegistryQuery query;
  for (int index = 3; index < argc; index += 2) {
    if (index + 1 >= argc) {
      usage();
      return 64;
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
        return 64;
      }
      query.campaign_limit = value;
    } else {
      usage();
      return 64;
    }
  }
  const auto snapshot = trainvm::read_experiment_registry(argv[2], query);
  std::cout << trainvm::encode_json(snapshot).dump(2) << '\n';
  return 0;
}

int simulate_command(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    usage();
    return 64;
  }
  auto compiled = trainvm::compile_document_file(argv[2]);
  if (!compiled.valid()) {
    print_diagnostics(compiled);
    return 2;
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
  return 0;
}

int journal_command(int argc, char** argv) {
  if (argc < 4) {
    usage();
    return 64;
  }
  const std::string_view operation(argv[2]);
  const bool valid_operation =
      (operation == "init" && argc == 4) ||
      (operation == "verify" && argc == 4) ||
      (operation == "replay" && argc == 4) ||
      (operation == "show" && argc == 5);
  if (!valid_operation) {
    usage();
    return 64;
  }
  trainvm::AuthorityLock authority_lock(argv[3]);
  trainvm::Journal journal(authority_lock.journal_path(),
                           authority_lock.journal_identity());
  if (operation == "init" && argc == 4) {
    std::cout << nlohmann::json({{"initialized", true}, {"events", journal.event_count()}}).dump(2)
              << '\n';
    return 0;
  }
  if (operation == "verify" && argc == 4) {
    std::string reason;
    const bool valid = journal.verify_chain(&reason);
    std::cout << nlohmann::json({{"valid", valid}, {"events", journal.event_count()}, {"reason", reason}}).dump(2)
              << '\n';
    return valid ? 0 : 3;
  }
  if (operation == "replay" && argc == 4) {
    const auto replayed = journal.rebuild_projections();
    std::cout << nlohmann::json({{"replayed", replayed}, {"valid", true}}).dump(2) << '\n';
    return 0;
  }
  if (operation == "show" && argc == 5) {
    const auto projection = journal.projection(argv[4]);
    if (!projection) {
      std::cerr << "run not found: " << argv[4] << '\n';
      return 4;
    }
    std::cout << trainvm::projection_json(*projection).dump(2) << '\n';
    return 0;
  }
  throw std::logic_error("validated journal operation was not dispatched");
}

int serve_command(int argc, char** argv) {
  const bool with_hostd = argc == 14;
  if ((argc != 12 && !with_hostd) ||
      std::string_view(argv[2]) != "--journal" ||
      std::string_view(argv[4]) != "--socket" ||
      std::string_view(argv[6]) != "--registry" ||
      std::string_view(argv[8]) != "--host-launch-registry" ||
      std::string_view(argv[10]) != "--training-component-registry" ||
      (with_hostd && std::string_view(argv[12]) != "--hostd-client")) {
    usage();
    return 64;
  }
  std::optional<trainvm::HostdClientConfiguration> hostd;
  if (with_hostd) {
    hostd = trainvm::HostdClientConfiguration::load_file(
        std::filesystem::absolute(argv[13]).lexically_normal());
  }
  return trainvm::serve(argv[3], argv[5],
                        trainvm::AdapterRegistry::load_file(argv[7]),
                        trainvm::HostLaunchRegistry::load_file(argv[9]),
                        trainvm::TrainingComponentRegistry::load_file(
                            argv[11]),
                        std::move(hostd));
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
  return 0;
}

int inspect_rwkv_lab_runtime_requirements_command(int argc) {
  if (argc != 2) {
    usage();
    return 64;
  }
  std::cout << trainvm::encode_json(
                   trainvm::rwkv_lab_worker_runtime_requirements())
                   .dump(2)
            << '\n';
  return 0;
}

int inspect_rwkv_lab_deployment_command(int argc, char** argv) {
  (void)argv;
  if (argc != 2) {
    usage();
    return 64;
  }
  trainvm::RwkvLabWorkerDeploymentSpec spec;
  std::vector<trainvm::Diagnostic> diagnostics;
  if (!trainvm::decode_json(read_stdin_json(), spec, "", diagnostics)) {
    std::cerr << trainvm::diagnostics_json(diagnostics).dump(2) << '\n';
    return 2;
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
  return 0;
}

int inspect_training_schedule_command(int argc, char** argv) {
  if (argc != 5) {
    usage();
    return 64;
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
      return 2;
    }
    for (std::int64_t step = 0; step <= max_step; ++step) {
      multipliers.push_back(
          trainvm::linear_warmup_cosine_multiplier(step, schedule));
    }
  } else if (implementation_id == "rwkv_lab.schedule.powercool.v1") {
    trainvm::PowerCoolSchedule schedule;
    if (!trainvm::decode_json(configuration, schedule, "", diagnostics)) {
      std::cerr << trainvm::diagnostics_json(diagnostics).dump(2) << '\n';
      return 2;
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
  return 0;
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
    if (argc >= 2 &&
        std::string_view(argv[1]) == "inspect-input-content-root") {
      return inspect_input_content_root_command(argc, argv);
    }
    if (argc == 4 && std::string_view(argv[1]) == "lock-input-content") {
      return lock_input_content_command(argv[2], argv[3]);
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
    return 64;
  } catch (const std::exception& exception) {
    std::cerr << "trainvm: " << exception.what() << '\n';
    return 1;
  }
}
