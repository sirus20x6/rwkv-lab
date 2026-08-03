#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

#include <sys/types.h>

#include <nlohmann/json.hpp>

#include "trainvm/cache_artifact_authority.hpp"
#include "trainvm/cache_namespace_authority.hpp"

namespace trainvm {

struct LinuxCacheEvidenceConfig {
  std::filesystem::path receipt_root;
  uid_t authority_uid{};
  std::size_t maximum_receipt_bytes{1U << 20U};
};

// Canonical names and documents are shared with the authority-owned receipt
// publisher. Constructing bytes does not grant authority: readers accept only
// immutable files beneath their constructor-pinned root.
[[nodiscard]] std::string
cache_runtime_probe_receipt_name(const CacheRuntimeProbeContext& context);
[[nodiscard]] nlohmann::json
cache_runtime_probe_receipt_json(const CacheRuntimeProbeContext& context,
                                 const CacheRuntimeProbeSnapshot& snapshot);
[[nodiscard]] std::string cache_qualification_evidence_receipt_name(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact);
[[nodiscard]] nlohmann::json cache_qualification_evidence_receipt_json(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact,
    const CacheQualificationEvidence& evidence);

class LinuxSealedCacheRuntimeProbe final : public ICacheRuntimeProbe {
 public:
  explicit LinuxSealedCacheRuntimeProbe(LinuxCacheEvidenceConfig config);
  ~LinuxSealedCacheRuntimeProbe() override;

  LinuxSealedCacheRuntimeProbe(const LinuxSealedCacheRuntimeProbe&) = delete;
  LinuxSealedCacheRuntimeProbe&
  operator=(const LinuxSealedCacheRuntimeProbe&) = delete;
  LinuxSealedCacheRuntimeProbe(LinuxSealedCacheRuntimeProbe&&) noexcept;
  LinuxSealedCacheRuntimeProbe&
  operator=(LinuxSealedCacheRuntimeProbe&&) noexcept;

  [[nodiscard]] CacheRuntimeProbeSnapshot
  capture(const CacheRuntimeProbeContext& context) override;

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

class LinuxImmutableCacheQualificationSource final
    : public ICacheQualificationEvidenceSource {
 public:
  explicit LinuxImmutableCacheQualificationSource(
      LinuxCacheEvidenceConfig config);
  ~LinuxImmutableCacheQualificationSource() override;

  LinuxImmutableCacheQualificationSource(
      const LinuxImmutableCacheQualificationSource&) = delete;
  LinuxImmutableCacheQualificationSource&
  operator=(const LinuxImmutableCacheQualificationSource&) = delete;
  LinuxImmutableCacheQualificationSource(
      LinuxImmutableCacheQualificationSource&&) noexcept;
  LinuxImmutableCacheQualificationSource&
  operator=(LinuxImmutableCacheQualificationSource&&) noexcept;

  [[nodiscard]] CacheQualificationEvidence
  capture(const CacheNamespaceAuthorityReceipt& authority,
          const ImmutableCacheTreeReceipt& artifact) override;
  void require_trusted(const CacheNamespaceAuthorityReceipt& authority,
                       const ImmutableCacheTreeReceipt& artifact,
                       const CacheQualificationReceipt& qualification) override;

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Authority-side immutable publisher. It accepts typed observations, writes a
// canonical receipt through a private temporary file, fsyncs bytes and mode,
// promotes with renameat2(RENAME_NOREPLACE), and treats an identical existing
// receipt as exact replay rather than replacing history.
class LinuxCacheEvidencePublisher final {
 public:
  explicit LinuxCacheEvidencePublisher(LinuxCacheEvidenceConfig config);
  ~LinuxCacheEvidencePublisher();

  LinuxCacheEvidencePublisher(const LinuxCacheEvidencePublisher&) = delete;
  LinuxCacheEvidencePublisher& operator=(
      const LinuxCacheEvidencePublisher&) = delete;
  LinuxCacheEvidencePublisher(LinuxCacheEvidencePublisher&&) noexcept;
  LinuxCacheEvidencePublisher& operator=(
      LinuxCacheEvidencePublisher&&) noexcept;

  [[nodiscard]] std::string publish_runtime(
      const CacheRuntimeProbeContext& context,
      const CacheRuntimeProbeSnapshot& snapshot);
  [[nodiscard]] std::string publish_qualification(
      const CacheNamespaceAuthorityReceipt& authority,
      const ImmutableCacheTreeReceipt& artifact,
      const CacheQualificationEvidence& evidence);

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace trainvm
