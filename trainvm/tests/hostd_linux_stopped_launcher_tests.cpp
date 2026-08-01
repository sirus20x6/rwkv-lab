#include "trainvm/hostd_linux_stopped_launcher.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace trainvm;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_rejected(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const LinuxStoppedLauncherError&) {
    return;
  }
  throw std::runtime_error(message);
}

void proc_parsers_are_strict_and_comm_safe() {
  using hostd_linux_stopped_launcher_test_seam::parse_proc_starttime;
  using hostd_linux_stopped_launcher_test_seam::parse_unified_cgroup;
  require(parse_proc_starttime(
              "123 (worker ) with spaces) S 1 2 3 4 5 6 7 8 9 10 11 "
              "12 13 14 15 16 17 18 987654 20\n") == 987654U,
          "proc stat parser finds field 22 after the final comm boundary");
  require(parse_unified_cgroup("0::/trainvm/launch-abc\n") ==
              "/trainvm/launch-abc" &&
              parse_unified_cgroup("0::/\n") == "/",
          "unified cgroup parser accepts one canonical v2 path");
  for (const std::string value : {
           "", "0::", "1::/trainvm", "0::relative", "0::/a//b",
           "0::/a/../b", "0::/a\n0::/b\n", "0:name=/legacy:/x\n"}) {
    require_rejected([&] { (void)parse_unified_cgroup(value); },
                     "ambiguous cgroup membership must be rejected");
  }
  for (const std::string value : {
           "", "123 worker S 1 2 3", "123 (worker) S 1 2 3",
           "123 (worker) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 0"}) {
    require_rejected([&] { (void)parse_proc_starttime(value); },
                     "truncated or zero-starttime proc stat must be rejected");
  }
}

void malformed_launch_fails_before_clone() {
  LinuxStoppedLauncherKernel launcher;
  require_rejected(
      [&] { (void)launcher.spawn_stopped({}); },
      "empty launch specification fails before any clone side effect");
  LinuxStoppedLaunchSpec malformed{
      .launch_id = "launch",
      .cgroup_fd = -1,
      .expected_cgroup_path = "/trainvm/launch",
      .expected_cgroup_device = 1,
      .expected_cgroup_inode = 2,
      .executable_fd = -1,
      .executable_name = "worker",
      .executable_digest = "sha256:" + std::string(64U, 'g'),
      .working_directory_fd = -1,
      .arguments = {},
  };
  require_rejected(
      [&] { (void)launcher.spawn_stopped(malformed); },
      "non-hex executable authority fails before descriptor or clone use");
}

}  // namespace

int main() {
  try {
    proc_parsers_are_strict_and_comm_safe();
    malformed_launch_fails_before_clone();
    std::cout << "Linux stopped launcher tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Linux stopped launcher test failure: " << error.what()
              << '\n';
    return 1;
  }
}
