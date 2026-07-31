#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {

struct CompiledPlan {
  Experiment experiment;
  nlohmann::json canonical_plan;
  std::string plan_hash;
};

struct CompileResult {
  std::optional<CompiledPlan> plan;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] bool valid() const;
};

CompileResult compile_document(const nlohmann::json& source);
CompileResult compile_document_file(const std::filesystem::path& path);
std::string sha256_hex(std::string_view value);
std::string severity_name(Diagnostic::Severity severity);
nlohmann::json diagnostics_json(const std::vector<Diagnostic>& diagnostics);
nlohmann::json plan_summary(const CompiledPlan& plan);

}  // namespace trainvm
