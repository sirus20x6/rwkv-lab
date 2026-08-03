#include "trainvm/gpu_fault_observer.hpp"

#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace trainvm;

struct Arguments final {
  std::filesystem::path state_path;
  std::filesystem::path boot_id_path{"/proc/sys/kernel/random/boot_id"};
  bool initialize{};
};

Arguments arguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--initialize") {
      result.initialize = true;
    } else if (argument == "--state" && index + 1 < argc) {
      result.state_path = argv[++index];
    } else if (argument == "--boot-id-file" && index + 1 < argc) {
      result.boot_id_path = argv[++index];
    } else {
      throw std::invalid_argument("unknown GPU fault observer argument");
    }
  }
  if (result.state_path.empty() || !result.state_path.is_absolute() ||
      result.state_path.lexically_normal() != result.state_path ||
      !result.boot_id_path.is_absolute() ||
      result.boot_id_path.lexically_normal() != result.boot_id_path)
    throw std::invalid_argument("GPU fault observer paths must be absolute");
  return result;
}

std::string boot_id(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open Linux boot ID");
  std::string value;
  std::getline(input, value);
  std::string extra;
  if (value.empty() || std::getline(input, extra))
    throw std::runtime_error("Linux boot ID file is malformed");
  return value;
}

GpuFaultObserverState initial_state(const Arguments& config,
                                    const std::string& current_boot_id) {
  const std::uint64_t now = linux_boottime_now_ns();
  struct stat status {};
  if (::lstat(config.state_path.c_str(), &status) != 0) {
    if (errno != ENOENT)
      throw std::runtime_error(std::string("could not inspect prior GPU fault ") +
                               "state: " + std::strerror(errno));
    return make_gpu_fault_observer_state(current_boot_id, now);
  }
  GpuFaultObserverState state =
      read_gpu_fault_observer_state(config.state_path, ::geteuid());
  if (state.boot_id != current_boot_id)
    return make_gpu_fault_observer_state(current_boot_id, now);
  return update_gpu_fault_observer_state(std::move(state), now);
}

void publish(const Arguments& config, GpuFaultObserverState& state,
             std::optional<NvidiaXidEvent> event = std::nullopt) {
  state = update_gpu_fault_observer_state(
      std::move(state), linux_boottime_now_ns(), std::move(event));
  write_gpu_fault_observer_state(config.state_path, state, ::geteuid());
}

void observe_stream(const Arguments& config, GpuFaultObserverState state) {
  constexpr int heartbeat_ms = 2000;
  constexpr std::size_t maximum_line_bytes = 64U << 10U;
  std::string pending;
  pending.reserve(4096U);
  std::array<char, 4096U> bytes{};
  while (true) {
    struct pollfd input {
      .fd = STDIN_FILENO, .events = POLLIN, .revents = 0
    };
    int ready = 0;
    do {
      ready = ::poll(&input, 1U, heartbeat_ms);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0)
      throw std::runtime_error(std::string("GPU fault observer poll failed: ") +
                               std::strerror(errno));
    if (ready == 0) {
      publish(config, state);
      continue;
    }
    if ((input.revents & (POLLERR | POLLNVAL)) != 0)
      throw std::runtime_error("GPU fault observer input failed");
    if ((input.revents & (POLLIN | POLLHUP)) == 0) continue;
    ssize_t count = 0;
    do {
      count = ::read(STDIN_FILENO, bytes.data(), bytes.size());
    } while (count < 0 && errno == EINTR);
    if (count < 0)
      throw std::runtime_error(std::string("GPU fault observer read failed: ") +
                               std::strerror(errno));
    if (count == 0)
      throw std::runtime_error("GPU fault observer input ended");
    pending.append(bytes.data(), static_cast<std::size_t>(count));
    if (pending.size() > maximum_line_bytes && pending.find('\n') == pending.npos) {
      pending.clear();
      publish(config, state);
      continue;
    }
    std::size_t newline = 0U;
    while ((newline = pending.find('\n')) != std::string::npos) {
      std::string line = pending.substr(0U, newline);
      pending.erase(0U, newline + 1U);
      if (const auto xid = parse_nvidia_xid_line(line)) {
        publish(config, state, xid);
        std::cerr << "trainvm GPU fault observer blocked new grants after "
                  << "NVIDIA Xid " << xid->code << '\n';
      }
    }
    publish(config, state);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments config = arguments(argc, argv);
    GpuFaultObserverState state = initial_state(config, boot_id(config.boot_id_path));
    write_gpu_fault_observer_state(config.state_path, state, ::geteuid());
    if (config.initialize) return 0;
    observe_stream(config, std::move(state));
  } catch (const std::exception& error) {
    std::cerr << "trainvm GPU fault observer: " << error.what() << '\n';
    return 2;
  }
}
