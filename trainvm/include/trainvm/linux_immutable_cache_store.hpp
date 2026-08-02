#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <sys/types.h>

#include "trainvm/cache_artifact_authority.hpp"

namespace trainvm {

struct LinuxImmutableCacheStoreConfig {
  std::filesystem::path publication_root;
  std::vector<std::filesystem::path> allowed_source_roots;
  uid_t authority_uid{};
  uid_t source_uid{};
  std::uint64_t maximum_file_count{1'000'000U};
  std::uint64_t maximum_total_bytes{1ULL << 50U};
  std::uint64_t maximum_single_file_bytes{1ULL << 40U};
};

// Linux production store for compiled/JIT cache artifacts. All traversal is
// rooted in constructor-pinned descriptors. Publication rejects symlinks,
// mounts crossed below a root, nonregular entries, mutation during copy, and
// unsafe publication-root ownership; promotion is content-addressed, fsynced,
// read-only, and atomic within the pinned root.
class LinuxImmutableCacheStore final : public ICacheArtifactStore {
 public:
  explicit LinuxImmutableCacheStore(LinuxImmutableCacheStoreConfig config);
  ~LinuxImmutableCacheStore() override;

  LinuxImmutableCacheStore(const LinuxImmutableCacheStore&) = delete;
  LinuxImmutableCacheStore& operator=(const LinuxImmutableCacheStore&) = delete;
  LinuxImmutableCacheStore(LinuxImmutableCacheStore&&) noexcept;
  LinuxImmutableCacheStore& operator=(LinuxImmutableCacheStore&&) noexcept;

  [[nodiscard]] ImmutableCacheTreeReceipt
  publish(const CacheNamespaceAuthorityReceipt& authority,
          const CacheArtifactCandidate& candidate) override;
  [[nodiscard]] ImmutableCacheTreeReceipt
  verify(const std::string& content_address) override;

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace trainvm
