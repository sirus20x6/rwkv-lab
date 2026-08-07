#include "trainvm/gpu_fault_observer.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace trainvm;

constexpr std::string_view kBoot =
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Callable>
void require_blocked(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const HostdStateError&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    constexpr std::string_view source = "/tmp/trainvm-gpu-fault-XXXXXX";
    std::ranges::copy(source, pattern.begin());
    const char* created = ::mkdtemp(pattern.data());
    if (created == nullptr || ::chmod(created, 0700) != 0)
      throw std::runtime_error("could not create protected temporary directory");
    path_ = created;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void exact_xid_parsing_never_matches_later_numbers() {
  const auto xid31 = parse_nvidia_xid_line(
      "NVRM: Xid (PCI:0000:41:00): 31, pid=5179, name=kwin_wayland");
  require(xid31 && xid31->code == 31U &&
              xid31->line_digest.starts_with("sha256:"),
          "the exact numeric field is parsed even when a later PID contains 79");
  const auto xid79 = parse_nvidia_xid_line(
      "kernel: NVRM: Xid (PCI:0000:41:00): 79, GPU has fallen off the bus");
  require(xid79 && xid79->code == 79U,
          "Xid 79 is recognized only in the exact Xid-code field");
  require(!parse_nvidia_xid_line(
              "NVRM: Xid report for pid 5179 without an Xid code") &&
              !parse_nvidia_xid_line(
                  "NVRM: Xid (PCI:0000:41:00): 31x, later=79") &&
              !parse_nvidia_xid_line("ordinary kernel line containing 79"),
          "near matches and unrelated 79 values are ignored");
}

void state_is_canonical_persistent_and_fail_closed() {
  TemporaryDirectory directory;
  const auto path = directory.path() / "state.json";
  const std::uint64_t now = linux_boottime_now_ns();
  auto clear = make_gpu_fault_observer_state(std::string(kBoot), now);
  write_gpu_fault_observer_state(path, clear, ::geteuid());
  require(read_gpu_fault_observer_state(path, ::geteuid()) == clear,
          "atomic state publication round-trips exact canonical evidence");

  LinuxGpuFaultAdmissionGuard allowed(path, std::string(kBoot),
                                      10'000'000'000ULL, ::geteuid());
  allowed.require_new_grant_allowed();

  const auto event = parse_nvidia_xid_line(
      "NVRM: Xid (PCI:0000:41:00): 31, pid=5179");
  require(event.has_value(), "test Xid event parses");
  auto blocked = update_gpu_fault_observer_state(
      clear, linux_boottime_now_ns(), event);
  write_gpu_fault_observer_state(path, blocked, ::geteuid());
  require_blocked([&] { allowed.require_new_grant_allowed(); },
                  "a recorded Xid latches new admissions closed");

  auto heartbeat = update_gpu_fault_observer_state(
      blocked, linux_boottime_now_ns());
  require(heartbeat.blocked && heartbeat.event_count == 1U &&
              heartbeat.last_xid == 31U,
          "heartbeat refresh cannot clear a same-boot fault latch");

  auto stale = make_gpu_fault_observer_state(std::string(kBoot), 1U);
  write_gpu_fault_observer_state(path, stale, ::geteuid());
  require_blocked([&] { allowed.require_new_grant_allowed(); },
                  "stale observer evidence blocks new admissions");

  std::string tampered = gpu_fault_observer_state_json(clear);
  const auto marker = tampered.find("\"blocked\":false");
  require(marker != std::string::npos, "canonical state contains blocked field");
  tampered.replace(marker, std::string("\"blocked\":false").size(),
                   "\"blocked\":true ");
  require_blocked(
      [&] { (void)gpu_fault_observer_state_from_json(tampered); },
      "tampered state cannot pass its canonical digest");

  const std::string canonical = gpu_fault_observer_state_json(clear);
  const std::string duplicate = canonical.substr(0U, canonical.size() - 2U) +
                                ",\"blocked\":false}\n";
  require_blocked(
      [&] { (void)gpu_fault_observer_state_from_json(duplicate); },
      "duplicate JSON fields cannot reinterpret observer authority");
}

}  // namespace

int main() {
  try {
    exact_xid_parsing_never_matches_later_numbers();
    state_is_canonical_persistent_and_fail_closed();
    std::cout << "GPU fault observer tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GPU fault observer test failure: " << error.what() << '\n';
    return 1;
  }
}
