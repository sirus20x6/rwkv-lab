#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include "trainvm/document.hpp"
#include "trainvm/dispatch.hpp"
#include "trainvm/command.hpp"
#include "trainvm/lease.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

class Controller;

// A durable command was valid when issued but has lost the active run/resource
// fence required to apply it. Boundary services map this typed condition to
// FAILED_PRECONDITION; untyped runtime failures remain authority corruption.
class OperationPreconditionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Event {
  std::string event_id;
  std::string run_id;
  std::uint64_t run_revision{};
  std::uint64_t plan_revision{};
  std::string node_id;
  std::string attempt_id;
  std::uint64_t worker_sequence{};
  std::string event_type;
  std::uint32_t event_version{1};
  std::int64_t wall_time_ns{};
  std::uint64_t monotonic_time_ns{};
  std::optional<std::uint64_t> optimizer_step;
  nlohmann::json payload = nlohmann::json::object();
};

struct RunProjection {
  std::string run_id;
  std::string experiment_name;
  std::string plan_hash;
  std::string desired_state;
  std::string observed_state;
  std::string current_node_id;
  std::string current_attempt_id;
  std::uint64_t run_revision{};
  std::uint64_t optimizer_step{};
  std::int64_t last_heartbeat_ns{};
  std::uint64_t last_event_sequence{};
  std::string failure_summary;

  bool operator==(const RunProjection&) const = default;
};

enum class RunCreationDisposition { inserted, replayed };

class RunCreationConflict final : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

struct RunCreationResult {
  RunCreationDisposition disposition{};
  Event created_event;

  bool operator==(const RunCreationResult&) const = default;
};

class Journal {
public:
  explicit Journal(const std::filesystem::path& path);
  ~Journal();

  Journal(const Journal&) = delete;
  Journal& operator=(const Journal&) = delete;
  Journal(Journal&&) = delete;
  Journal& operator=(Journal&&) = delete;

  [[nodiscard]] std::optional<Event> event(const std::string& event_id) const;
  [[nodiscard]] std::vector<Event> events_for_run(const std::string& run_id) const;
  [[nodiscard]] std::optional<RunProjection> projection(const std::string& run_id) const;
  [[nodiscard]] std::optional<CompiledPlan> compiled_plan(const std::string& plan_hash) const;
  [[nodiscard]] std::optional<Dispatch> dispatch(const std::string& dispatch_id) const;
  [[nodiscard]] std::optional<ControlCommand> control_command(
      const std::string& command_id) const;
  [[nodiscard]] std::uint64_t latest_control_revision(const std::string& run_id) const;
  [[nodiscard]] std::uint64_t latest_effective_control_revision(
      const std::string& run_id) const;
  LeaseAcquireResult acquire_lease(const std::string& concurrency_key,
                                   const std::string& owner_run_id,
                                   const std::string& lease_id, std::int64_t now_ns,
                                   std::int64_t timeout_ns);
  bool renew_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                   const std::string& lease_id, std::uint64_t fencing_token,
                   std::int64_t now_ns, std::int64_t timeout_ns);
  bool release_lease(const std::string& concurrency_key, const std::string& owner_run_id,
                     const std::string& lease_id, std::uint64_t fencing_token,
                     std::int64_t now_ns);
  [[nodiscard]] std::optional<ResourceLease> active_lease(
      const std::string& concurrency_key, std::int64_t now_ns) const;
  [[nodiscard]] std::uint64_t event_count() const;
  [[nodiscard]] std::string journal_id() const;
  [[nodiscard]] bool verify_chain(std::string* reason = nullptr) const;
  std::uint64_t rebuild_projections();

 private:
  friend class Controller;

  class ReadSnapshot {
   public:
    ReadSnapshot(ReadSnapshot&& other) noexcept;
    ReadSnapshot& operator=(ReadSnapshot&&) = delete;
    ~ReadSnapshot();
    ReadSnapshot(const ReadSnapshot&) = delete;
    ReadSnapshot& operator=(const ReadSnapshot&) = delete;

   private:
    friend class Journal;
    explicit ReadSnapshot(sqlite3* database);

    sqlite3* database_{};
  };

  sqlite3* database_{};

  [[nodiscard]] ReadSnapshot read_snapshot() const;
  std::uint64_t append(const Event& event);
  std::vector<std::uint64_t> append_batch(const std::vector<Event>& events);
  Dispatch prepare_dispatch(const Dispatch& dispatch,
                            const Event& prepared_event);
  void complete_dispatch(const std::string& dispatch_id,
                         const std::string& result_event_id,
                         const std::vector<Event>& events);
  void initialize();
  std::uint64_t append_uncommitted(const Event& event);
  RunCreationResult create_run(const CompiledPlan& plan, const std::vector<Event>& events);
  LeaseAcquireResult acquire_lease_with_events(
      const std::string& concurrency_key, const std::string& owner_run_id,
      const std::string& lease_id, std::int64_t now_ns, std::int64_t timeout_ns,
      const std::vector<Event>& events);
  bool complete_builtin_admission(const ResourceLease& lease, std::int64_t now_ns,
                                  const std::vector<Event>& events);
  bool prepare_worker_launch(const WorkerLaunchTicket& launch, std::int64_t now_ns,
                             const Event& event);
  WorkerReadinessDisposition accept_worker_ready(
      const WorkerLaunchTicket& launch, const WorkerHelloEvidence& hello,
      std::int64_t now_ns, const std::vector<Event>& events);
  Dispatch prepare_fenced_dispatch(const Dispatch& dispatch,
                                   const Event& prepared_event,
                                   const WorkerLaunchTicket& launch,
                                   std::int64_t now_ns);
  Dispatch prepare_dispatch_impl(
      const Dispatch& dispatch, const Event& prepared_event,
      const std::optional<WorkerLaunchTicket>& launch,
      std::optional<std::int64_t> now_ns);
  void complete_fenced_dispatch(const std::string& dispatch_id,
                                const std::string& result_event_id,
                                const std::vector<Event>& events,
                                const WorkerSessionIdentity& identity,
                                std::int64_t now_ns);
  void complete_dispatch_impl(
      const std::string& dispatch_id, const std::string& result_event_id,
      const std::vector<Event>& events,
      const std::optional<WorkerSessionIdentity>& identity,
      std::optional<std::int64_t> now_ns);
  void complete_managed_builtin_dispatch(
      const Dispatch& dispatch, const ResourceLease& lease,
      std::int64_t now_ns, bool release_lease,
      const std::vector<Event>& events);
  [[nodiscard]] bool has_lease_release_receipt(
      const std::string& concurrency_key, const std::string& owner_run_id,
      const std::string& lease_id, std::uint64_t fencing_token,
      std::int64_t released_at_ns) const;
  ControlSubmission submit_control_command(ControlCommand command);
  ControlCommand acknowledge_control_command(const std::string& run_id,
                                              const std::string& command_id,
                                              const ControlAcknowledgementIdentity& identity,
                                              ControlCommandStatus status,
                                              std::optional<std::uint64_t> effective_step,
                                              nlohmann::json effective_values,
                                              nlohmann::json diagnostics);
};

nlohmann::json event_json(const Event& event);
Event event_from_json(const nlohmann::json& input);
nlohmann::json projection_json(const RunProjection& projection);

}  // namespace trainvm
