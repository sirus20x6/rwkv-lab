#include "trainvm/run_authoring_cli.hpp"

#include "trainvm/document.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace trainvm {
namespace {

nlohmann::json parsed_authority_object(std::string_view source,
                                       std::string_view description) {
  constexpr std::size_t maximum = 2U * 1024U * 1024U;
  if (source.empty() || source.size() > maximum)
    throw std::runtime_error(std::string(description) +
                             " is empty or exceeds 2 MiB");
  auto value = nlohmann::json::parse(source);
  if (!value.is_object())
    throw std::runtime_error(std::string(description) + " is not an object");
  return value;
}

bool canonical_plan_hash(std::string_view value) {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](const unsigned char character) {
           return (character >= static_cast<unsigned char>('0') &&
                   character <= static_cast<unsigned char>('9')) ||
                  (character >= static_cast<unsigned char>('a') &&
                   character <= static_cast<unsigned char>('f'));
         });
}

} // namespace

AuthorRunStreamValidator::AuthorRunStreamValidator(
    const bool dry_run, std::optional<std::string> expected_plan_hash)
    : dry_run_(dry_run), expected_plan_hash_(std::move(expected_plan_hash)) {
  if (expected_plan_hash_ && !canonical_plan_hash(*expected_plan_hash_))
    throw std::runtime_error("expected author-run plan hash is not canonical");
}

void AuthorRunStreamValidator::observe(const v1::AuthorRunUpdate &update) {
  if (saw_terminal_)
    throw std::runtime_error("authority emitted an update after its terminal");
  if (update.stage() <= v1::AUTHOR_RUN_STAGE_UNSPECIFIED ||
      update.stage() > v1::AUTHOR_RUN_STAGE_FAILED)
    throw std::runtime_error("authority emitted an unknown author-run stage");
  if (last_stage_ == v1::AUTHOR_RUN_STAGE_UNSPECIFIED &&
      update.stage() != v1::AUTHOR_RUN_STAGE_VALIDATING)
    throw std::runtime_error("authority stream did not begin with validation");
  if (update.stage() != v1::AUTHOR_RUN_STAGE_FAILED &&
      update.stage() != v1::AUTHOR_RUN_STAGE_COMPLETE &&
      last_stage_ != v1::AUTHOR_RUN_STAGE_UNSPECIFIED &&
      (update.stage() < last_stage_ ||
       update.stage() > last_stage_ + 1))
    throw std::runtime_error("authority stream violated its stage graph");
  if ((update.stage() == v1::AUTHOR_RUN_STAGE_FAILED ||
       update.stage() == v1::AUTHOR_RUN_STAGE_COMPLETE) != update.terminal())
    throw std::runtime_error(
        "authority terminal flag disagrees with its terminal stage");
  if (update.stage() == v1::AUTHOR_RUN_STAGE_COMPLETE &&
      last_stage_ != (dry_run_ ? v1::AUTHOR_RUN_STAGE_PREFLIGHT
                              : v1::AUTHOR_RUN_STAGE_SUBMITTING))
    throw std::runtime_error(
        "authority completed before the required final stage");
  if (update.dry_run() != dry_run_)
    throw std::runtime_error("authority changed the requested dry-run identity");

  if (!update.plan_hash().empty()) {
    if (!canonical_plan_hash(update.plan_hash()))
      throw std::runtime_error("authority emitted a noncanonical plan hash");
    if (expected_plan_hash_ && update.plan_hash() != *expected_plan_hash_)
      throw std::runtime_error("authority stream crossed the preview plan fence");
    if (!observed_plan_hash_.empty() &&
        update.plan_hash() != observed_plan_hash_)
      throw std::runtime_error("authority changed plan identity within one stream");
    observed_plan_hash_ = update.plan_hash();
  }
  if (update.has_run()) {
    if (dry_run_)
      throw std::runtime_error("dry-run authority stream unexpectedly created a run");
    if (!canonical_plan_hash(update.run().plan_hash()) ||
        (!observed_plan_hash_.empty() &&
         update.run().plan_hash() != observed_plan_hash_) ||
        (expected_plan_hash_ &&
         update.run().plan_hash() != *expected_plan_hash_))
      throw std::runtime_error("created run does not match the preview plan fence");
  }

  if (!update.terminal())
  {
    last_stage_ = update.stage();
    return;
  }
  saw_terminal_ = true;
  last_stage_ = update.stage();
  failed_ = update.stage() == v1::AUTHOR_RUN_STAGE_FAILED;
  if (!failed_ && update.stage() != v1::AUTHOR_RUN_STAGE_COMPLETE)
    throw std::runtime_error("authority terminal has an invalid stage");
  if (failed_)
    return;

  if (observed_plan_hash_.empty())
    throw std::runtime_error("successful authority terminal omitted plan identity");
  const auto plan = parsed_authority_object(
      update.canonical_plan_json(), "authority canonical plan");
  const auto receipt = parsed_authority_object(
      update.preflight_receipt_json(), "authority preflight receipt");
  if (sha256_hex(plan.dump()) != observed_plan_hash_)
    throw std::runtime_error(
        "authority canonical plan does not match its declared plan hash");
  if (!receipt.contains("passed") || !receipt.at("passed").is_boolean() ||
      !receipt.at("passed").get<bool>() ||
      !receipt.contains("plan_hash") ||
      !receipt.at("plan_hash").is_string() ||
      receipt.at("plan_hash").get<std::string>() != observed_plan_hash_)
    throw std::runtime_error(
        "successful authority terminal omitted a matching passing preflight");
  if (dry_run_) {
    if (update.has_run() || !update.dashboard_url().empty())
      throw std::runtime_error("dry-run authority terminal mutated launch state");
  } else if (!update.has_run() || update.run().run_id().empty() ||
             update.dashboard_url().empty()) {
    throw std::runtime_error(
        "launch authority terminal omitted run or dashboard identity");
  }
}

AuthorRunStreamSummary AuthorRunStreamValidator::finish() const {
  if (!saw_terminal_)
    throw std::runtime_error(
        "author-run stream ended without an authority terminal update");
  return {.failed = failed_, .plan_hash = observed_plan_hash_};
}

nlohmann::json author_run_update_json(
    const v1::AuthorRunUpdate &update, std::string_view dashboard_base_url) {
  nlohmann::json diagnostics = nlohmann::json::array();
  for (const auto &item : update.diagnostics()) {
    diagnostics.push_back({{"severity", item.severity()},
                           {"code", item.code()},
                           {"path", item.document_path()},
                           {"message", item.message()},
                           {"help", item.help()}});
  }
  nlohmann::json output{
      {"stage", v1::AuthorRunStage_Name(update.stage())},
      {"detail", update.detail()},
      {"terminal", update.terminal()},
      {"dry_run", update.dry_run()},
      {"content_lock_reused", update.content_lock_reused()},
      {"diagnostics", std::move(diagnostics)},
  };
  if (!update.plan_hash().empty())
    output["plan_hash"] = update.plan_hash();
  if (update.has_run()) {
    output["run"] = {{"run_id", update.run().run_id()},
                     {"revision", update.run().revision()},
                     {"plan_hash", update.run().plan_hash()}};
  }
  if (!update.dashboard_url().empty()) {
    output["dashboard_url"] =
        update.dashboard_url().starts_with('/')
            ? std::string(dashboard_base_url) + update.dashboard_url()
            : update.dashboard_url();
  }
  if (!update.recipe_expansion_json().empty())
    output["recipe_expansion"] = parsed_authority_object(
        update.recipe_expansion_json(), "authority recipe expansion");
  if (!update.canonical_plan_json().empty())
    output["canonical_plan"] = parsed_authority_object(
        update.canonical_plan_json(), "authority canonical plan");
  if (!update.preflight_receipt_json().empty())
    output["preflight_receipt"] = parsed_authority_object(
        update.preflight_receipt_json(), "authority preflight receipt");
  return output;
}

} // namespace trainvm
