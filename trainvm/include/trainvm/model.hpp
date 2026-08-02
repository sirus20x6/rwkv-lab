#pragma once

#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/input_content_authority.hpp"

namespace trainvm {

using Json = nlohmann::json;

enum class AcceleratorVendor { nvidia, amd, intel, none };
enum class ParameterType { string, integer, number, boolean, path, duration };
enum class ArtifactType { path, checkpoint, dataset, image_gallery, metrics, report, opaque };
enum class Immutability { immutable, append_only, mutable_until_publish };
enum class Fingerprint { sha256, manifest_sha256, adapter, none };
enum class ComponentRuntime { builtin, python_worker, native_worker, external_worker };
enum class Idempotency { replay_safe, receipt_required, at_most_once };
enum class Effect { read_only, workspace_write, process, resource, external };
enum class Backoff { none, fixed, exponential };
enum class ProgressDirection { increasing, decreasing };
enum class ControlType { number, integer, boolean, string, enumeration };
enum class ApplyPoint {
  immediate,
  next_microbatch,
  next_optimizer_step,
  next_eval,
  next_checkpoint,
  restart,
};
enum class MetricType { counter, gauge, histogram };
enum class StepDomain { microbatch, optimizer_step, sample, token, epoch, wall_time };
enum class Aggregation { last, sum, mean, weighted_mean, min, max, histogram };
enum class TrainingComponentCategory {
  optimizer,
  parameter_router,
  learning_rate_schedule,
  weight_decay_schedule,
  activation,
  normalization,
  objective,
  precision,
  gradient_clipping,
  gradient_accumulation,
  curriculum,
  metric_reducer,
};
enum class ReconcilePolicy { fail_closed, adopt_if_fingerprint_matches, restart_from_checkpoint };
enum class OrphanPolicy { leave_and_block, adopt_if_identity_matches, terminate_and_recover };
enum class ProfilerBackend { torch, nsys, ncu };
enum class ProfilerActivity { cpu, accelerator };

struct Metadata {
  std::string name;
  std::optional<std::string> description;
  std::optional<std::map<std::string, std::string>> labels;
};

struct Workspace {
  std::string root;
  std::string run_directory;
  std::string concurrency_key;
  std::optional<std::vector<std::string>> allowed_read_roots;
  std::optional<std::vector<InputContentRootIdentity>> input_content_roots;
  std::optional<std::vector<std::string>> allowed_write_roots;
};

struct Accelerators {
  AcceleratorVendor vendor{};
  std::int64_t count{};
  std::optional<double> minimum_memory_gib;
  bool exclusive{};
  std::optional<std::map<std::string, std::string>> selector;
};

struct CpuIoPolicy {
  std::optional<std::string> cpuset;
  std::optional<std::vector<std::int64_t>> cpus;
  std::optional<std::int64_t> cpu_weight;
  std::optional<std::int64_t> io_weight;
  std::optional<std::int64_t> omp_threads;
  std::optional<std::int64_t> preprocessing_workers;
  std::optional<std::int64_t> nice;
};

struct Resources {
  Accelerators accelerators;
  std::optional<double> minimum_host_memory_gib;
  std::optional<std::int64_t> cpu_threads;
  std::optional<std::int64_t> lease_timeout_seconds;
  std::optional<CpuIoPolicy> cpu_io_policy;
};

struct CompilePhase {
  bool enabled{};
};

struct WarmupPhase {
  bool enabled{};
  std::optional<std::int64_t> steps;
};

struct QualifyPhase {
  bool enabled{};
  std::optional<std::int64_t> steps;
};

struct GpuTraceCapture {
  bool enabled{};
  std::optional<ProfilerBackend> backend;
  std::optional<std::int64_t> warmup_steps;
  std::optional<std::int64_t> skip_steps;
  std::optional<std::int64_t> capture_steps;
  std::optional<std::string> output_artifact;
  std::optional<std::vector<ProfilerActivity>> activities;
  std::optional<bool> record_shapes;
  std::optional<bool> profile_memory;
  std::optional<bool> with_stack;
};

struct ExecutionPhases {
  std::string component;
  std::string operation;
  std::optional<CompilePhase> compile;
  std::optional<WarmupPhase> warmup;
  std::optional<QualifyPhase> qualify;
  std::optional<GpuTraceCapture> gpu_trace;
};

struct Parameter {
  ParameterType type{};
  Json value;
  std::optional<std::string> description;
  std::optional<bool> secret_reference;
};

struct Artifact {
  ArtifactType type{};
  std::optional<std::string> schema;
  Immutability immutability{};
  Fingerprint fingerprint{};
  std::optional<bool> required;
  std::optional<std::string> description;
};

struct Operation {
  std::string contract;
  std::optional<std::string> description;
};

struct Component {
  std::string adapter;
  std::string version;
  ComponentRuntime runtime{};
  std::map<std::string, Operation> operations;
};

struct NodeOutputReference {
  std::string node;
  std::string name;
};

struct TrainingComponentKey {
  TrainingComponentCategory category{};
  std::string name;
  std::string version;

  auto operator<=>(const TrainingComponentKey&) const = default;
};

struct TrainingComponentSelection {
  TrainingComponentKey key;
  Json configuration = Json::object();
};

// One research topology attached to a composition, with only the parameters
// that topology declares. Closed per model family: the rwkv family resolves
// these against the scratch-RWKV profile registry.
struct TrainingTopologySelection {
  std::string topology;
  Json parameters = Json::object();
};

struct TrainingComposition {
  std::string model_family;
  std::map<std::string, TrainingComponentSelection> components;
  // Optional so a document without topologies encodes exactly as before.
  std::optional<std::vector<TrainingTopologySelection>> topologies;
};

struct Binding {
  std::optional<Json> literal;
  std::optional<std::string> parameter;
  std::optional<std::string> artifact;
  std::optional<std::string> control;
  std::optional<std::string> context;
  std::optional<NodeOutputReference> node_output;
};

struct Invocation {
  std::string component;
  std::string operation;
  std::map<std::string, Binding> inputs;
  std::optional<TrainingComposition> training;
};

struct Transition {
  std::string on;
  std::optional<Json> where;
  std::string target;
  std::optional<std::string> description;
};

struct Retry {
  std::int64_t maximum_attempts{};
  Backoff backoff{};
  std::optional<double> initial_delay_seconds;
  std::optional<double> maximum_delay_seconds;
  std::optional<std::vector<std::string>> retryable_events;
};

struct LoopGuard {
  std::int64_t max_visits{};
  std::string progress_field;
  ProgressDirection direction{};
};

struct Node {
  std::optional<std::string> description;
  Invocation invoke;
  std::optional<std::map<std::string, std::string>> publishes;
  Idempotency idempotency{};
  Effect effect{};
  std::optional<std::int64_t> timeout_seconds;
  std::optional<Retry> retry;
  std::optional<LoopGuard> loop_guard;
  std::vector<Transition> transitions;
};

struct Workflow {
  std::string entrypoint;
  std::map<std::string, Node> nodes;
};

struct Control {
  ControlType type{};
  Json default_value;
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::optional<std::vector<Json>> values;
  ApplyPoint apply{};
  bool mutable_after_start{};
  std::optional<bool> requires_pause;
  std::vector<std::string> targets;
  std::optional<std::string> description;
  std::optional<std::string> unit;
};

struct Controls {
  std::map<std::string, Control> catalog;
};

struct Metric {
  std::string name;
  MetricType type{};
  std::string unit;
  StepDomain step_domain{};
  Aggregation aggregation{};
  std::optional<std::string> description;
};

struct Observability {
  std::int64_t heartbeat_seconds{};
  std::vector<Metric> metrics;
  std::int64_t retain_raw_metrics_days{};
  std::optional<std::string> eval_gallery_artifact;
  std::optional<std::string> log_artifact;
};

struct Recovery {
  bool exact_resume{};
  std::optional<std::string> checkpoint_artifact;
  ReconcilePolicy reconcile{};
  OrphanPolicy orphan_policy{};
  std::int64_t graceful_stop_seconds{};
  std::optional<bool> release_accelerators_when_paused;
};

struct Spec {
  Workspace workspace;
  Resources resources;
  std::optional<ExecutionPhases> execution;
  std::map<std::string, Parameter> parameters;
  std::map<std::string, Artifact> artifacts;
  std::map<std::string, Component> components;
  Workflow workflow;
  Controls controls;
  Observability observability;
  Recovery recovery;
};

struct Experiment {
  std::string api_version;
  std::string kind;
  Metadata metadata;
  Spec spec;
};

}  // namespace trainvm
