#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace trainvm {

inline constexpr std::string_view kConversionPublicationApiVersion =
    "trainvm.conversion-publication/v1";

class ConversionPublicationError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// How a model was produced from other models. The kind decides what evidence
// publication requires: a distillation without a teacher identity is not a
// weaker record of the same thing, it is an unattributable one.
enum class ConversionKind {
  layer_replacement,
  cross_architecture_distillation,
  attention_alignment,
  checkpoint_consolidation,
};

// One input a conversion consumed. Conversion is not commutative — replacing
// layer 3 then layer 19 is a different model from the reverse — so order is
// part of the binding rather than a presentation detail.
struct ConversionSourceBinding final {
  std::string role;
  std::string artifact_id;
  // Content, not path. A path re-binds silently when the bytes beneath it
  // change; a digest does not.
  std::string content_digest;
  std::uint64_t order{};

  bool operator==(const ConversionSourceBinding&) const = default;
};

// A loss that shaped the produced weights. Recorded because two models with
// identical sources and different objectives are different models, and a
// lineage that cannot tell them apart is not lineage.
struct ConversionLossRecord final {
  std::string name;
  double weight{};

  bool operator==(const ConversionLossRecord&) const = default;
};

struct ConversionQualificationReport final {
  std::string report_artifact_id;
  std::string report_digest;
  bool passed{};

  bool operator==(const ConversionQualificationReport&) const = default;
};

struct ConversionPublicationRequest final {
  std::string api_version{std::string(kConversionPublicationApiVersion)};
  ConversionKind kind{};
  std::vector<ConversionSourceBinding> sources;
  // Required for distillation, refused for every other kind: a layer
  // replacement has no teacher, and naming one would invent provenance.
  std::optional<std::string> teacher_identity;
  std::vector<ConversionLossRecord> losses;
  std::optional<ConversionQualificationReport> qualification;
  // False while the conversion exists only in a process's memory. An
  // in-memory result has no bytes anyone else can verify, so it may describe
  // itself but may never claim publication.
  bool materialized{};
  std::string produced_digest;

  bool operator==(const ConversionPublicationRequest&) const = default;
};

[[nodiscard]] std::string_view conversion_kind_name(ConversionKind kind);
[[nodiscard]] std::optional<ConversionKind> conversion_kind_from_name(
    std::string_view name);
[[nodiscard]] bool conversion_kind_requires_teacher(ConversionKind kind);

// Throws unless the request may be published. Every rule is a refusal, not a
// warning: a conversion that cannot prove its lineage does not get a
// downgraded record, it gets none.
void validate_conversion_publication(const ConversionPublicationRequest& request);

// The lineage record a published model carries. Deriving it re-validates, so
// there is no path that produces a record for a request publication would
// have refused.
[[nodiscard]] nlohmann::json conversion_lineage_record(
    const ConversionPublicationRequest& request);
[[nodiscard]] std::string conversion_lineage_digest(
    const ConversionPublicationRequest& request);

}  // namespace trainvm
