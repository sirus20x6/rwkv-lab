#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/worker_bootstrap.hpp"

#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

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
      .code_fd = std::nullopt,
      .worker_bootstrap_fd = -1,
      .executable_name = "worker",
      .executable_digest = "sha256:" + std::string(64U, 'g'),
      .working_directory_fd = -1,
      .arguments = {},
  };
  require_rejected(
      [&] { (void)launcher.spawn_stopped(malformed); },
      "non-hex executable authority fails before descriptor or clone use");
}

void inherited_descriptor_abi_is_exact_and_exec_surviving() {
  auto bootstrap = create_sealed_worker_bootstrap({
      .api_version = std::string(kWorkerBootstrapApiVersion),
      .controller_target = "unix:/run/trainvm/test.sock",
      .run_id = "run-abi",
      .node_id = "train",
      .attempt_id = "attempt-abi",
      .launch_nonce = "launch-abi",
      .adapter = "rwkv-lab.mageflow",
      .adapter_version = "1.0.0",
      .code_fingerprint = "sha256:" + std::string(64U, 'a'),
      .capabilities = {},
      .last_acked_controller_sequence = 0U,
      .concurrency_key = "gpu:0",
      .lease_id = "lease-abi",
      .fencing_token = 1U,
      .bootstrap_digest = {},
  });
  const int bootstrap_copy = bootstrap.duplicate_fd();
  const int code_copy = bootstrap.duplicate_fd();
  const int bootstrap_high = ::fcntl(bootstrap_copy, F_DUPFD_CLOEXEC, 64);
  const int code_high = ::fcntl(code_copy, F_DUPFD_CLOEXEC, 64);
  (void)::close(bootstrap_copy);
  (void)::close(code_copy);
  require(bootstrap_high >= 64 && code_high >= 64,
          "test inherited descriptors are outside the fixed ABI range");
  int result_pipe[2]{-1, -1};
  require(::pipe2(result_pipe, O_CLOEXEC) == 0,
          "create inherited descriptor result pipe");
  const pid_t child = ::fork();
  require(child >= 0, "fork inherited descriptor ABI observer");
  if (child == 0) {
    (void)::close(result_pipe[0]);
    bool valid =
        hostd_linux_stopped_launcher_test_seam::
            install_inherited_worker_descriptors(code_high, bootstrap_high);
    valid = valid &&
            (::fcntl(kLinuxWorkerCodeDescriptor, F_GETFD) & FD_CLOEXEC) == 0 &&
            (::fcntl(kLinuxWorkerBootstrapDescriptor, F_GETFD) & FD_CLOEXEC) ==
                0;
    try {
      valid = valid &&
              worker_bootstrap_from_sealed_fd(kLinuxWorkerCodeDescriptor) ==
                  bootstrap.spec() &&
              worker_bootstrap_from_sealed_fd(
                  kLinuxWorkerBootstrapDescriptor) == bootstrap.spec();
    } catch (...) {
      valid = false;
    }
    const char result = valid ? 'Y' : 'N';
    (void)::write(result_pipe[1], &result, 1U);
    ::_exit(valid ? 0 : 1);
  }
  (void)::close(result_pipe[1]);
  char result = 'N';
  require(::read(result_pipe[0], &result, 1U) == 1,
          "read inherited descriptor ABI observation");
  int status = 0;
  require(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0 && result == 'Y',
          "sealed code/bootstrap descriptors occupy fixed exec-surviving fds");
  (void)::close(result_pipe[0]);
  (void)::close(bootstrap_high);
  (void)::close(code_high);
}

}  // namespace

int main() {
  try {
    proc_parsers_are_strict_and_comm_safe();
    malformed_launch_fails_before_clone();
    inherited_descriptor_abi_is_exact_and_exec_surviving();
    std::cout << "Linux stopped launcher tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Linux stopped launcher test failure: " << error.what()
              << '\n';
    return 1;
  }
}
