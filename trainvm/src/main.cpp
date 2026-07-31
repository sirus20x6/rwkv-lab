#include "trainvm/document.hpp"
#include "trainvm/journal.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  trainvm validate <experiment.json>\n"
      << "  trainvm plan <experiment.json> [--canonical]\n"
      << "  trainvm journal init <journal.db>\n"
      << "  trainvm journal append <journal.db> <event.json>\n"
      << "  trainvm journal verify <journal.db>\n"
      << "  trainvm journal replay <journal.db>\n"
      << "  trainvm journal show <journal.db> <run-id>\n";
}

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open " + path.string());
  }
  nlohmann::json value;
  input >> value;
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

int journal_command(int argc, char** argv) {
  if (argc < 4) {
    usage();
    return 64;
  }
  const std::string_view operation(argv[2]);
  trainvm::Journal journal(argv[3]);
  if (operation == "init" && argc == 4) {
    std::cout << nlohmann::json({{"initialized", true}, {"events", journal.event_count()}}).dump(2)
              << '\n';
    return 0;
  }
  if (operation == "append" && argc == 5) {
    const auto event = trainvm::event_from_json(read_json(argv[4]));
    const auto sequence = journal.append(event);
    std::cout << nlohmann::json({{"journal_sequence", sequence}, {"event_id", event.event_id}}).dump(2)
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
  usage();
  return 64;
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
