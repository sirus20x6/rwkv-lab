#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include "trainvm/dispatch.hpp"
#include "trainvm/lease.hpp"

namespace trainvm {

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

class Journal {
 public:
  explicit Journal(const std::filesystem::path& path);
  ~Journal();

  Journal(const Journal&) = delete;
  Journal& operator=(const Journal&) = delete;
  Journal(Journal&&) = delete;
  Journal& operator=(Journal&&) = delete;

  std::uint64_t append(const Event& event);
  std::vector<std::uint64_t> append_batch(const std::vector<Event>& events);
  [[nodiscard]] std::optional<Event> event(const std::string& event_id) const;
  [[nodiscard]] std::vector<Event> events_for_run(const std::string& run_id) const;
  [[nodiscard]] std::optional<RunProjection> projection(const std::string& run_id) const;
  Dispatch prepare_dispatch(const Dispatch& dispatch, const Event& prepared_event);
  void complete_dispatch(const std::string& dispatch_id, const std::string& result_event_id,
                         const std::vector<Event>& events);
  [[nodiscard]] std::optional<Dispatch> dispatch(const std::string& dispatch_id) const;
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
  [[nodiscard]] bool verify_chain(std::string* reason = nullptr) const;
  std::uint64_t rebuild_projections();

 private:
  sqlite3* database_{};

  void initialize();
  std::uint64_t append_uncommitted(const Event& event);
};

nlohmann::json event_json(const Event& event);
Event event_from_json(const nlohmann::json& input);
nlohmann::json projection_json(const RunProjection& projection);

}  // namespace trainvm
