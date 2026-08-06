#include "trainvm/eval_examples_contract.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "trainvm/document.hpp"
#include "trainvm/journal.hpp"

namespace {

namespace fs = std::filesystem;
int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Callable>
void rejects(Callable &&callable, std::string_view message) {
  try {
    callable();
    check(false, message);
  } catch (const std::invalid_argument &) {
  }
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::string pattern = "/tmp/trainvm-eval-examples-XXXXXX";
    char *created = ::mkdtemp(pattern.data());
    if (created == nullptr)
      throw std::runtime_error("mkdtemp failed");
    path_ = created;
  }
  ~TemporaryDirectory() { fs::remove_all(path_); }
  const fs::path &path() const { return path_; }

private:
  fs::path path_;
};

trainvm::Json document(std::string media_path, std::string media_digest,
                       std::uint64_t media_size) {
  const std::string digest = "sha256:" + std::string(64U, 'a');
  trainvm::Json body{
      {"api_version", trainvm::kEvalExamplesSchema},
      {"run_id", "run-1"},
      {"node_id", "train"},
      {"attempt_id", "train@1"},
      {"optimizer_step", 0U},
      {"step_domain", "optimizer_step"},
      {"series_id", "fixed-validation"},
      {"heldout",
       {{"identity_field", "sample_id"},
        {"identities_digest", digest},
        {"selector_digest", digest}}},
      {"evaluator",
       {{"component_digest", digest}, {"metric_names", {"eval.loss"}}}},
      {"checkpoint",
       {{"artifact_id", "checkpoint-0"}, {"manifest_digest", digest}}},
      {"policy_digest", digest},
      {"examples",
       {{{"example_id", "sample-1"},
         {"heldout_item_id", "row-1"},
         {"heldout_item_digest", digest},
         {"input", {{{"kind", "text"}, {"text", "prompt"}}}},
         {"target", {{{"kind", "text"}, {"text", "target"}}}},
         {"prediction",
          {{{"kind", "image"},
            {"path", std::move(media_path)},
            {"media_type", "image/png"},
            {"sha256", std::move(media_digest)},
            {"size_bytes", media_size}}}}}}}};
  body["canonical_manifest_digest"] =
      "sha256:" + trainvm::sha256_hex(body.dump());
  return body;
}

void write_file(const fs::path &path, std::string_view content) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  output.close();
}

} // namespace

int main() {
  TemporaryDirectory temporary;
  const std::string media = "immutable media bytes";
  const std::string media_digest = "sha256:" + trainvm::sha256_hex(media);
  const fs::path run_directory = temporary.path() / "run";
  const std::string artifact_id = "eval-examples-test";
  const fs::path revision = run_directory / "trainvm_artifacts" /
                            "eval_examples" / "revisions" / artifact_id;
  const fs::path media_path = revision / "objects" / "media";
  write_file(media_path, media);
  trainvm::Json manifest_document =
      document("objects/media", media_digest, media.size());
  const std::string manifest_bytes = manifest_document.dump();
  const fs::path manifest_path = revision / "manifest.json";
  write_file(manifest_path, manifest_bytes);
  const trainvm::EvalExamplesManifest manifest =
      trainvm::validate_eval_examples_manifest(manifest_document);
  const std::string authority_digest = "sha256:" + std::string(64U, 'a');
  const trainvm::Json resolved_training{
      {"components",
       {{"evaluator",
         {{"configuration", {{"metrics", {"eval.loss"}}}},
          {"descriptor_digest", authority_digest},
          {"descriptor", {{"key", {{"category", "evaluator"}}}}}}}}}};
  trainvm::Event checkpoint{};
  checkpoint.run_id = "run-1";
  checkpoint.node_id = "train";
  checkpoint.event_type = "artifact.published";
  checkpoint.optimizer_step = 0U;
  checkpoint.payload = {{"artifact_id", "checkpoint-0"},
                        {"kind", "checkpoint"},
                        {"complete", true},
                        {"fingerprint_algorithm", "manifest_sha256"},
                        {"fingerprint", authority_digest}};
  trainvm::Event metric{};
  metric.run_id = "run-1";
  metric.node_id = "train";
  metric.event_type = "metric.sampled";
  metric.optimizer_step = 0U;
  metric.payload = {{"name", "eval.loss"}, {"step_domain", "optimizer_step"}};
  trainvm::validate_eval_examples_gate_provenance(manifest, resolved_training,
                                                  {checkpoint, metric});
  trainvm::Event examples_event{};
  examples_event.run_id = "run-1";
  examples_event.node_id = "train";
  examples_event.attempt_id = "train@2";
  examples_event.event_type = "artifact.published";
  examples_event.optimizer_step = 0U;
  examples_event.payload = {
      {"kind", "eval_examples"},
      {"schema", trainvm::kEvalExamplesSchema},
      {"complete", true},
      {"eval_examples_manifest", manifest_document},
  };
  check(!trainvm::durable_step_zero_eval_gate_satisfied({examples_event},
                                                        "run-1", "train"),
        "examples without prior scalar do not satisfy replayed gate");
  check(trainvm::durable_step_zero_eval_gate_satisfied({metric, examples_event},
                                                       "run-1", "train"),
        "ordered durable scalar and examples satisfy replayed gate across "
        "attempts");
  trainvm::Event wrong_step = examples_event;
  wrong_step.optimizer_step = 1U;
  check(!trainvm::durable_step_zero_eval_gate_satisfied({metric, wrong_step},
                                                        "run-1", "train"),
        "wrong-step examples do not satisfy replayed gate");
  trainvm::Event malformed = examples_event;
  malformed.payload["eval_examples_manifest"]["examples"] =
      trainvm::Json::array();
  check(!trainvm::durable_step_zero_eval_gate_satisfied({metric, malformed},
                                                        "run-1", "train"),
        "malformed examples do not satisfy replayed gate");
  const trainvm::Json gated_publication{
      {"eval_examples",
       {{"declaration",
         {{"required", true},
          {"type", "eval_examples"},
          {"schema", trainvm::kEvalExamplesSchema}}}}}};
  check(trainvm::invocation_requires_step_zero_eval_gate(gated_publication) &&
            !trainvm::invocation_requires_step_zero_eval_gate(
                trainvm::Json::object()),
        "only required eval-examples publication activates the gate");
  trainvm::validate_eval_examples_payload(
      manifest, "file://" + manifest_path.string(), manifest_bytes,
      "sha256:" + trainvm::sha256_hex(manifest_bytes), run_directory.string(),
      artifact_id);
  check(manifest.examples.size() == 1U && manifest.optimizer_step == 0U,
        "valid reflected manifest and pinned payload are accepted");
  rejects(
      [&] {
        trainvm::validate_eval_examples_payload(
            manifest, "file://" + manifest_path.string(), manifest_bytes,
            "sha256:" + trainvm::sha256_hex(manifest_bytes),
            (temporary.path() / "other-run").string(), artifact_id);
      },
      "payload outside the invocation run directory is rejected");

  trainvm::Event noncheckpoint = checkpoint;
  noncheckpoint.payload["kind"] = "report";
  rejects(
      [&] {
        trainvm::validate_eval_examples_gate_provenance(
            manifest, resolved_training, {noncheckpoint, metric});
      },
      "non-checkpoint parent provenance is rejected");
  trainvm::Event wrong_digest = checkpoint;
  wrong_digest.payload["fingerprint"] = "sha256:" + std::string(64U, 'b');
  rejects(
      [&] {
        trainvm::validate_eval_examples_gate_provenance(
            manifest, resolved_training, {wrong_digest, metric});
      },
      "wrong checkpoint manifest digest is rejected");
  trainvm::Event wrong_checkpoint_step = checkpoint;
  wrong_checkpoint_step.optimizer_step = 1U;
  rejects(
      [&] {
        trainvm::validate_eval_examples_gate_provenance(
            manifest, resolved_training, {wrong_checkpoint_step, metric});
      },
      "post-mutation checkpoint parent provenance is rejected");
  rejects(
      [&] {
        trainvm::validate_eval_examples_gate_provenance(
            manifest, resolved_training, {metric});
      },
      "missing checkpoint parent is rejected");
  trainvm::Json undeclared = resolved_training;
  undeclared["components"]["evaluator"]["configuration"]["metrics"] = {
      "other.loss"};
  rejects(
      [&] {
        trainvm::validate_eval_examples_gate_provenance(manifest, undeclared,
                                                        {checkpoint, metric});
      },
      "undeclared scalar metric is rejected");

  write_file(media_path, "tampered");
  rejects(
      [&] {
        trainvm::validate_eval_examples_payload(
            manifest, "file://" + manifest_path.string(), manifest_bytes,
            "sha256:" + trainvm::sha256_hex(manifest_bytes),
            run_directory.string(), artifact_id);
      },
      "tampered media is rejected");

  trainvm::Json traversal = document("../escape", media_digest, media.size());
  rejects([&] { (void)trainvm::validate_eval_examples_manifest(traversal); },
          "relative traversal is rejected");
  trainvm::Json empty = manifest_document;
  empty["examples"] = trainvm::Json::array();
  empty.erase("canonical_manifest_digest");
  empty["canonical_manifest_digest"] =
      "sha256:" + trainvm::sha256_hex(empty.dump());
  rejects([&] { (void)trainvm::validate_eval_examples_manifest(empty); },
          "empty evidence is rejected");

  if (failures != 0)
    return 1;
  std::cout << "eval examples contract tests passed\n";
  return 0;
}
