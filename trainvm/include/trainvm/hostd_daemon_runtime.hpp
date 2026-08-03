#pragma once

#include <memory>
#include <stdexcept>

#include "trainvm/hostd_daemon_configuration.hpp"
#include "trainvm/hostd_startup_controller.hpp"
#include "trainvm/hostd_transport.hpp"

namespace trainvm {

class HostdDaemonRuntimeError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Single-threaded production object graph for hostd. Construction pins all
// authority inputs but exposes no socket. advance_startup() performs one
// bounded recovery/audit transition and binds the shared endpoint only after
// the controller reaches admitting. The same owner thread must serve it.
//
// Process launch is exposed only for a root host authority with a distinct,
// non-root no-new-privileges worker identity. Every launch carries a durable
// cgroup-device policy intent, exact kernel installation receipt, and stopped
// credential observation. Restart reconciliation owns the same process and
// cgroup authority before the socket is admitted.
class HostdDaemonRuntime final {
public:
  explicit HostdDaemonRuntime(HostdDaemonConfiguration configuration);
  ~HostdDaemonRuntime();

  HostdDaemonRuntime(const HostdDaemonRuntime &) = delete;
  HostdDaemonRuntime &operator=(const HostdDaemonRuntime &) = delete;
  HostdDaemonRuntime(HostdDaemonRuntime &&) = delete;
  HostdDaemonRuntime &operator=(HostdDaemonRuntime &&) = delete;

  [[nodiscard]] HostdStartupControllerStatus advance_startup();
  [[nodiscard]] HostdStartupControllerStatus startup_status() const;
  [[nodiscard]] bool ready() const;
  [[nodiscard]] bool process_launch_enabled() const noexcept;
  [[nodiscard]] HostdCoordinatorStatus coordinator_status() const;
  [[nodiscard]] HostdSocketIdentity socket_identity();
  [[nodiscard]] HostdServeResult serve_one();

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace trainvm
