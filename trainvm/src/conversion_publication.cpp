#include "trainvm/conversion_publication.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <memory>
#include <set>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumSources = 64U;
constexpr std::size_t kMaximumLosses = 32U;
constexpr std::size_t kMaximumIdentifierBytes = 256U;

[[noreturn]] void refuse(std::string message) {
  throw ConversionPublicationError(std::move(message));
}

bool valid_digest(std::string_view value) {
  if (value.size() != 71U || !value.starts_with("sha256:")) return false;
  return std::ranges::all_of(value.substr(7U), [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

bool valid_identifier(std::string_view value) {
  if (value.empty() || value.size() > kMaximumIdentifierBytes) return false;
  return std::ranges::all_of(value, [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == '-' || character == ':' ||
           character == '/' || character == '@';
  });
}

std::string hex_digest(std::string_view material) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), material.data(), material.size()) != 1)
    refuse("conversion lineage digest failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> value{};
  unsigned int size = 0U;
  if (EVP_DigestFinal_ex(context.get(), value.data(), &size) != 1 || size != 32U)
    refuse("conversion lineage digest final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  for (unsigned int index = 0U; index < size; ++index) {
    result.push_back(digits[value[index] >> 4U]);
    result.push_back(digits[value[index] & 0x0fU]);
  }
  return result;
}

}  // namespace

std::string_view conversion_kind_name(ConversionKind kind) {
  switch (kind) {
    case ConversionKind::layer_replacement:
      return "layer_replacement";
    case ConversionKind::cross_architecture_distillation:
      return "cross_architecture_distillation";
    case ConversionKind::attention_alignment:
      return "attention_alignment";
    case ConversionKind::checkpoint_consolidation:
      return "checkpoint_consolidation";
  }
  refuse("unknown conversion kind");
}

std::optional<ConversionKind> conversion_kind_from_name(std::string_view name) {
  for (const ConversionKind kind :
       {ConversionKind::layer_replacement,
        ConversionKind::cross_architecture_distillation,
        ConversionKind::attention_alignment,
        ConversionKind::checkpoint_consolidation}) {
    if (conversion_kind_name(kind) == name) return kind;
  }
  return std::nullopt;
}

bool conversion_kind_requires_teacher(ConversionKind kind) {
  return kind == ConversionKind::cross_architecture_distillation;
}

void validate_conversion_publication(
    const ConversionPublicationRequest& request) {
  if (request.api_version != kConversionPublicationApiVersion)
    refuse("conversion publication api version is unknown");

  // The card's first rule, and the one worth stating plainly: a conversion
  // held only in a process's memory has no bytes another party can verify.
  // It may be described, inspected, and evaluated; it may not be published.
  if (!request.materialized)
    refuse(
        "an in-memory conversion cannot claim publication: no durable bytes "
        "exist for anyone to verify");

  if (!valid_digest(request.produced_digest))
    refuse("a published conversion must carry a sha256 content digest");

  if (request.sources.empty() || request.sources.size() > kMaximumSources)
    refuse("a published conversion must bind between 1 and 64 ordered sources");

  // Order must be exactly 0..n-1, each once. Conversion is not commutative,
  // so a gap or a duplicate means the recorded order cannot reproduce the
  // model and is therefore not a lineage.
  std::set<std::uint64_t> orders;
  std::set<std::string> artifact_ids;
  for (const ConversionSourceBinding& source : request.sources) {
    if (!valid_identifier(source.role) ||
        !valid_identifier(source.artifact_id))
      refuse("conversion source role and artifact id must be identifiers");
    if (!valid_digest(source.content_digest))
      refuse(
          "conversion source " + source.artifact_id +
          " must bind a content digest, not a path: a path re-binds silently "
          "when the bytes beneath it change");
    if (!orders.insert(source.order).second)
      refuse("conversion sources share an order position");
    if (!artifact_ids.insert(source.artifact_id).second)
      refuse("conversion binds the same source artifact twice");
  }
  if (*orders.begin() != 0U || *orders.rbegin() != request.sources.size() - 1U)
    refuse("conversion source order must be contiguous from zero");

  const bool needs_teacher = conversion_kind_requires_teacher(request.kind);
  if (needs_teacher) {
    if (!request.teacher_identity ||
        !valid_identifier(*request.teacher_identity))
      refuse(
          "a distillation must name its teacher: an unattributable student "
          "cannot be published");
    if (std::ranges::none_of(
            request.sources, [](const ConversionSourceBinding& source) {
              return source.role == "teacher";
            }))
      refuse("a distillation must bind its teacher as a source");
  } else if (request.teacher_identity) {
    // Naming a teacher for a conversion that has none invents provenance.
    refuse(std::string("a ") + std::string(conversion_kind_name(request.kind)) +
           " conversion has no teacher and may not name one");
  }

  if (request.losses.size() > kMaximumLosses)
    refuse("a published conversion may record at most 32 losses");
  std::set<std::string> loss_names;
  for (const ConversionLossRecord& loss : request.losses) {
    if (!valid_identifier(loss.name))
      refuse("conversion loss name must be an identifier");
    if (!std::isfinite(loss.weight) || loss.weight < 0.0)
      refuse("conversion loss weight must be finite and non-negative");
    if (!loss_names.insert(loss.name).second)
      refuse("conversion records the same loss twice");
  }
  // A conversion that trained anything shaped its weights with an objective.
  // Only a pure remap can honestly record none.
  if (request.losses.empty() &&
      request.kind != ConversionKind::layer_replacement &&
      request.kind != ConversionKind::checkpoint_consolidation)
    refuse(std::string("a ") + std::string(conversion_kind_name(request.kind)) +
           " conversion must record the losses that shaped it");

  if (!request.qualification)
    refuse("a published conversion must carry a qualification report");
  if (!valid_identifier(request.qualification->report_artifact_id) ||
      !valid_digest(request.qualification->report_digest))
    refuse("conversion qualification report must be an identified artifact");
  if (!request.qualification->passed)
    refuse(
        "a conversion whose qualification did not pass cannot be published");
}

nlohmann::json conversion_lineage_record(
    const ConversionPublicationRequest& request) {
  // Re-validate rather than trust the caller: there must be no path that
  // yields a lineage record for a request publication would refuse.
  validate_conversion_publication(request);

  std::vector<ConversionSourceBinding> ordered = request.sources;
  std::ranges::sort(ordered, {}, &ConversionSourceBinding::order);

  nlohmann::json sources = nlohmann::json::array();
  for (const ConversionSourceBinding& source : ordered) {
    sources.push_back({{"order", source.order},
                       {"role", source.role},
                       {"artifact_id", source.artifact_id},
                       {"content_digest", source.content_digest}});
  }
  nlohmann::json losses = nlohmann::json::array();
  std::vector<ConversionLossRecord> sorted_losses = request.losses;
  std::ranges::sort(sorted_losses, {}, &ConversionLossRecord::name);
  for (const ConversionLossRecord& loss : sorted_losses)
    losses.push_back({{"name", loss.name}, {"weight", loss.weight}});

  nlohmann::json record{
      {"api_version", std::string(kConversionPublicationApiVersion)},
      {"kind", std::string(conversion_kind_name(request.kind))},
      {"sources", std::move(sources)},
      {"losses", std::move(losses)},
      {"produced_digest", request.produced_digest},
      {"qualification",
       {{"report_artifact_id", request.qualification->report_artifact_id},
        {"report_digest", request.qualification->report_digest},
        {"passed", request.qualification->passed}}},
  };
  if (request.teacher_identity)
    record["teacher_identity"] = *request.teacher_identity;
  return record;
}

std::string conversion_lineage_digest(
    const ConversionPublicationRequest& request) {
  return hex_digest(conversion_lineage_record(request).dump());
}

}  // namespace trainvm
