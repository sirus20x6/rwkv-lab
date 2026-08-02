#include "trainvm/conversion_publication.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_refused(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const ConversionPublicationError&) {
    return;
  }
  throw std::runtime_error(message);
}

std::string digest(char digit) { return "sha256:" + std::string(64U, digit); }

// A publishable layer replacement: the GDN-to-RWKV shape, where a base model
// and an ordered set of converted layers produce a new model by remap.
ConversionPublicationRequest replacement() {
  return {
      .api_version = std::string(kConversionPublicationApiVersion),
      .kind = ConversionKind::layer_replacement,
      .sources = {{.role = "base",
                   .artifact_id = "model.qwen35-9b-base",
                   .content_digest = digest('1'),
                   .order = 0U},
                  {.role = "patch",
                   .artifact_id = "patch.gdn-layers",
                   .content_digest = digest('2'),
                   .order = 1U}},
      .teacher_identity = std::nullopt,
      .losses = {},
      .qualification = ConversionQualificationReport{
          .report_artifact_id = "report.conversion-qualification",
          .report_digest = digest('3'),
          .passed = true},
      .materialized = true,
      .produced_digest = digest('4'),
  };
}

ConversionPublicationRequest distillation() {
  auto request = replacement();
  request.kind = ConversionKind::cross_architecture_distillation;
  request.sources.push_back({.role = "teacher",
                             .artifact_id = "model.teacher-35b",
                             .content_digest = digest('5'),
                             .order = 2U});
  request.teacher_identity = "model.teacher-35b@revision-1";
  request.losses = {{.name = "logit_kl", .weight = 1.0},
                    {.name = "block_relative_l2", .weight = 0.5}};
  return request;
}

void a_publishable_conversion_is_accepted() {
  validate_conversion_publication(replacement());
  validate_conversion_publication(distillation());
  // Without this, every refusal below could be passing for an unrelated
  // reason and the suite would prove nothing.
}

// The card's headline rule.
void an_in_memory_conversion_cannot_claim_publication() {
  auto request = replacement();
  request.materialized = false;
  require_refused([&] { validate_conversion_publication(request); },
                  "an in-memory conversion was allowed to publish");
  require_refused([&] { (void)conversion_lineage_record(request); },
                  "an in-memory conversion produced a lineage record");
}

// Sources are ordered because conversion is not commutative: replacing layer
// 3 then layer 19 is a different model from the reverse.
void source_order_must_be_exact_and_content_bound() {
  auto gap = replacement();
  gap.sources[1].order = 5U;
  require_refused([&] { validate_conversion_publication(gap); },
                  "a gap in source order was accepted");

  auto duplicate = replacement();
  duplicate.sources[1].order = 0U;
  require_refused([&] { validate_conversion_publication(duplicate); },
                  "duplicate source order positions were accepted");

  auto repeated = replacement();
  repeated.sources[1].artifact_id = repeated.sources[0].artifact_id;
  require_refused([&] { validate_conversion_publication(repeated); },
                  "the same source artifact was bound twice");

  auto pathlike = replacement();
  pathlike.sources[0].content_digest = "/thearray/git/moe-mla/Qwen35-9B-Base";
  require_refused([&] { validate_conversion_publication(pathlike); },
                  "a path was accepted where a content digest is required");

  auto empty = replacement();
  empty.sources.clear();
  require_refused([&] { validate_conversion_publication(empty); },
                  "a conversion with no bound sources was accepted");
}

// A student nobody can attribute is not publishable, and a conversion with no
// teacher may not invent one.
void teacher_identity_is_required_exactly_where_it_exists() {
  auto anonymous = distillation();
  anonymous.teacher_identity = std::nullopt;
  require_refused([&] { validate_conversion_publication(anonymous); },
                  "a distillation published without naming its teacher");

  auto unbound = distillation();
  unbound.sources.erase(unbound.sources.begin() + 2);
  unbound.sources[0].order = 0U;
  unbound.sources[1].order = 1U;
  require_refused([&] { validate_conversion_publication(unbound); },
                  "a distillation named a teacher it did not bind as a source");

  auto invented = replacement();
  invented.teacher_identity = "model.teacher-35b@revision-1";
  require_refused([&] { validate_conversion_publication(invented); },
                  "a layer replacement invented a teacher");
}

// Two models with the same sources and different objectives are different
// models. A lineage that cannot distinguish them is not lineage.
void trained_conversions_must_record_their_losses() {
  auto lossless = distillation();
  lossless.losses.clear();
  require_refused([&] { validate_conversion_publication(lossless); },
                  "a distillation published without recording any loss");

  auto repeated = distillation();
  repeated.losses.push_back(repeated.losses.front());
  require_refused([&] { validate_conversion_publication(repeated); },
                  "the same loss was recorded twice");

  auto negative = distillation();
  negative.losses[0].weight = -1.0;
  require_refused([&] { validate_conversion_publication(negative); },
                  "a negative loss weight was accepted");

  // A pure remap trains nothing and may honestly record no loss.
  validate_conversion_publication(replacement());
}

void qualification_must_exist_and_must_have_passed() {
  auto missing = replacement();
  missing.qualification = std::nullopt;
  require_refused([&] { validate_conversion_publication(missing); },
                  "a conversion published with no qualification report");

  auto failed = replacement();
  failed.qualification->passed = false;
  require_refused([&] { validate_conversion_publication(failed); },
                  "a conversion published a failed qualification");

  auto unidentified = replacement();
  unidentified.qualification->report_digest = "not-a-digest";
  require_refused([&] { validate_conversion_publication(unidentified); },
                  "a qualification report with no content digest was accepted");
}

// The record is order-normalized and content-bound, so two publications of the
// same lineage agree byte for byte and any changed input changes the digest.
void the_lineage_record_is_canonical_and_binding() {
  const auto record = conversion_lineage_record(distillation());
  require(record.at("sources").size() == 3U,
          "the lineage record carries every bound source");
  require(record.at("sources")[0].at("order") == 0U &&
              record.at("sources")[2].at("order") == 2U,
          "the lineage record is emitted in source order");
  require(record.contains("teacher_identity"),
          "a distillation lineage names its teacher");
  require(!conversion_lineage_record(replacement()).contains("teacher_identity"),
          "a replacement lineage names no teacher");

  // Caller ordering must not change the bytes.
  auto shuffled = distillation();
  std::swap(shuffled.sources[0], shuffled.sources[2]);
  require(conversion_lineage_digest(shuffled) ==
              conversion_lineage_digest(distillation()),
          "the lineage digest depends on caller ordering");

  // Any changed input must change the digest.
  auto changed_source = distillation();
  changed_source.sources[0].content_digest = digest('9');
  require(conversion_lineage_digest(changed_source) !=
              conversion_lineage_digest(distillation()),
          "a changed source digest left the lineage digest unchanged");

  auto changed_loss = distillation();
  changed_loss.losses[0].weight = 0.25;
  require(conversion_lineage_digest(changed_loss) !=
              conversion_lineage_digest(distillation()),
          "a changed loss weight left the lineage digest unchanged");
}

}  // namespace

int main() {
  try {
    a_publishable_conversion_is_accepted();
    an_in_memory_conversion_cannot_claim_publication();
    source_order_must_be_exact_and_content_bound();
    teacher_identity_is_required_exactly_where_it_exists();
    trained_conversions_must_record_their_losses();
    qualification_must_exist_and_must_have_passed();
    the_lineage_record_is_canonical_and_binding();
    std::cout << "conversion publication tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "conversion publication test failure: " << error.what()
              << '\n';
    return 1;
  }
}
