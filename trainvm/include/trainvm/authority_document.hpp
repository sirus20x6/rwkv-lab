#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace trainvm {

// Reads one small authority-owned regular file through a pinned descriptor,
// rejects symlinks and writable/untrusted ownership, and double-attests file
// identity and metadata across the complete read.
[[nodiscard]] std::string read_authority_document(
    const std::filesystem::path& path, std::string_view document_kind,
    std::uintmax_t maximum_bytes);

}  // namespace trainvm
