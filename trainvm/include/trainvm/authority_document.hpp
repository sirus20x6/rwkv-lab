#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <sys/types.h>

namespace trainvm {

// Reads one small authority-owned regular file through a pinned descriptor,
// rejects symlinks and writable/untrusted ownership, and double-attests file
// identity and metadata across the complete read.
[[nodiscard]] std::string read_authority_document(
    const std::filesystem::path& path, std::string_view document_kind,
    std::uintmax_t maximum_bytes);

struct AuthorityDocumentPublicationPolicy final {
  uid_t owner_uid{};
  gid_t owner_gid{};
  mode_t file_mode{0600};
  uid_t parent_owner_uid{};
  gid_t parent_owner_gid{};

  bool operator==(const AuthorityDocumentPublicationPolicy&) const = default;
};

// Atomically publishes one bounded authority document inside an exact,
// securely resolved owner-controlled directory. The temporary inode is fully
// written, owned, permissioned, and fsynced before rename; the parent is then
// fsynced. No path-resolution or durability fallback is permitted.
void publish_authority_document(
    const std::filesystem::path& path, std::string_view document_kind,
    std::string_view contents, AuthorityDocumentPublicationPolicy policy,
    std::uintmax_t maximum_bytes);

}  // namespace trainvm
