#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace trainvm {

inline constexpr std::string_view kInputContentRootApiVersion =
    "trainvm.input-content-root/v1";

enum class ContentRootKind {
  file,
  directory,
};

struct InputContentRootIdentity final {
  std::string api_version;
  std::string path;
  ContentRootKind kind{};
  std::uint64_t file_count{};
  std::uint64_t total_bytes{};
  std::string tree_sha256;

  bool operator==(const InputContentRootIdentity&) const = default;
};

[[nodiscard]] InputContentRootIdentity measure_input_content_root(
    const std::filesystem::path& path);

}  // namespace trainvm
