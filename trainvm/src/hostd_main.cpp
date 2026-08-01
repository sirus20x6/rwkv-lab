#include <poll.h>
#include <signal.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "trainvm/hostd_daemon_configuration.hpp"
#include "trainvm/hostd_daemon_runtime.hpp"

namespace {

volatile sig_atomic_t stop_requested = 0;

void request_stop(int) { stop_requested = 1; }

void install_signal_handlers() {
  struct sigaction action{};
  action.sa_handler = request_stop;
  if (::sigemptyset(&action.sa_mask) != 0 ||
      ::sigaction(SIGINT, &action, nullptr) != 0 ||
      ::sigaction(SIGTERM, &action, nullptr) != 0)
    throw std::runtime_error("could not install hostd signal handlers");
}

void wait_for_startup_wake(std::int64_t nanoseconds) {
  const std::int64_t milliseconds =
      std::max<std::int64_t>(1, nanoseconds / 1'000'000LL);
  const int bounded =
      static_cast<int>(std::min<std::int64_t>(milliseconds, 1000LL));
  while (::poll(nullptr, 0, bounded) < 0 && errno == EINTR &&
         stop_requested == 0) {
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool run = argc == 3 && std::string_view(argv[1]) == "--config";
    const bool validate =
        argc == 3 && std::string_view(argv[1]) == "--validate-config";
    if (!run && !validate) {
      std::cerr << "usage: trainvm-hostd (--config|--validate-config) "
                   "/absolute/hostd.json\n";
      return 2;
    }
    auto configuration = trainvm::HostdDaemonConfiguration::load_file(argv[2]);
    if (validate) {
      std::cout << "valid " << trainvm::kHostdDaemonConfigurationApiVersion
                << '\n';
      return 0;
    }
    install_signal_handlers();
    const std::int64_t wake = configuration.serve_wake_interval_ns();
    trainvm::HostdDaemonRuntime runtime(std::move(configuration));
    while (!runtime.ready() && stop_requested == 0) {
      const auto status = runtime.advance_startup();
      if (status.phase == trainvm::HostdStartupPhase::reconciling)
        wait_for_startup_wake(wake);
    }
    while (stop_requested == 0)
      (void)runtime.serve_one();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "trainvm-hostd: " << error.what() << '\n';
    return 1;
  }
}
