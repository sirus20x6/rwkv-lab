#include "trainvm/hostd_linux_process_policy_kernel.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class FakeKernel final : public trainvm::ILinuxProcessPolicyKernel {
 public:
  std::string read(int cgroup_fd, std::string_view control) override {
    require(cgroup_fd >= 0, "installer must pass a duplicated descriptor");
    return controls.at(std::string(control)) + "\n";
  }

  void write(int cgroup_fd, std::string_view control,
             std::string_view value) override {
    require(cgroup_fd >= 0, "installer must pass a duplicated descriptor");
    controls[std::string(control)] = std::string(value);
  }

  std::map<std::string, std::string> controls{
      {"cpuset.cpus", ""},
      {"cpuset.mems", ""},
      {"cpuset.mems.effective", "0-1"},
      {"cpu.weight", "100"},
      {"io.weight", "default 100"},
  };
};

}  // namespace

int main() {
  try {
    int descriptors[2]{};
    require(::pipe(descriptors) == 0, "create harmless descriptor fixture");
    trainvm::LinuxAllocationCgroup cgroup(
        {.unified_path = "/trainvm/test", .device = 31U, .inode = 41U},
        descriptors[0], descriptors[1], {}, false);
    FakeKernel kernel;
    trainvm::LinuxProcessPolicyInstaller installer(kernel);
    trainvm::CpuIoPolicy source;
    source.cpuset = "2-5";
    source.cpu_weight = 300;
    source.io_weight = 70;
    source.omp_threads = 4;
    source.nice = 2;
    const auto policy = trainvm::compile_linux_process_policy(source);
    const auto installed = installer.install(policy, cgroup);
    require(installed.cpuset == "2-5" && installed.cpuset_mems == "0-1" &&
                installed.cpu_weight == 300 && installed.io_weight == 70 &&
                installed.installation_digest.starts_with("sha256:") &&
                kernel.controls["cpuset.mems"] == "0-1" &&
                kernel.controls["cpuset.cpus"] == "2-5" &&
                kernel.controls["cpu.weight"] == "300" &&
                kernel.controls["io.weight"] == "default 70",
            "installer must lower and re-read every declared cgroup control");
    installer.verify(policy, installed, cgroup);

    kernel.controls["cpu.weight"] = "301";
    bool drift_rejected = false;
    try {
      installer.verify(policy, installed, cgroup);
    } catch (const trainvm::LinuxProcessPolicyKernelError&) {
      drift_rejected = true;
    }
    require(drift_rejected, "recovery verification must reject control drift");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
