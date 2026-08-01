#include "trainvm/compatibility_catalog.hpp"
#include "trainvm/document.hpp"
#include "trainvm/experiment_analysis.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/service.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
      << "  trainvm inspect-registry <experiments.db> [--task <task>] "
         "[--metric <metric>] [--baseline <config>] [--limit <count>]\n"
      << "  trainvm serve --journal <journal.db> --socket <trainvm.sock> "
         "--registry <adapters.json> --host-launch-registry "
         "<host-launches.json> --training-component-registry "
         "<training-components.json>\n"
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
                   {"repository_root_identity",
                    catalog.repository_root_identity_display()},
               }
                   .dump(2)
            << '\n';
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
  if (argc != 12 || std::string_view(argv[2]) != "--journal" ||
      std::string_view(argv[4]) != "--socket" ||
      std::string_view(argv[6]) != "--registry" ||
      std::string_view(argv[8]) != "--host-launch-registry" ||
      std::string_view(argv[10]) != "--training-component-registry") {
    usage();
    return 64;
  }
  return trainvm::serve(argv[3], argv[5],
                        trainvm::AdapterRegistry::load_file(argv[7]),
                        trainvm::HostLaunchRegistry::load_file(argv[9]),
                        trainvm::TrainingComponentRegistry::load_file(
                            argv[11]));
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
    if (argc == 4 && std::string_view(argv[1]) == "validate-catalog") {
      return validate_catalog_command(argv[2], argv[3]);
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
