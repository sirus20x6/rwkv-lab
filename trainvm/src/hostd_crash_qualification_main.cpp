#include <unistd.h>
#include <signal.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "trainvm/hostd_crash_qualification.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/worker_bootstrap.hpp"

namespace {

void usage() {
  std::cerr
      << "usage: trainvm-hostd-crash-qualification --workspace /disposable/dir"
         " [--cgroup-parent /sys/fs/cgroup/...] [--receipt /out.json]\n"
         "\n"
         "Destructive. It forks and SIGKILLs real processes and writes a\n"
         "disposable host ledger and cgroup subtree beneath --workspace.\n"
         "Point it only at a host you are willing to damage.\n"
         "Exit status: 0 when the gate is open, 3 when any declared crash\n"
         "point is unqualified, 1 on harness failure.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool qualification_worker = false;
    bool bootstrap_descriptor_declared = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      qualification_worker =
          qualification_worker || argument == "--qualification-worker";
      bootstrap_descriptor_declared =
          bootstrap_descriptor_declared ||
          argument == "--trainvm-bootstrap-fd=4";
    }
    if (qualification_worker) {
      if (!bootstrap_descriptor_declared)
        throw std::runtime_error(
            "qualification worker has no fixed bootstrap descriptor");
      const auto bootstrap = trainvm::worker_bootstrap_from_sealed_fd(
          trainvm::kLinuxWorkerBootstrapDescriptor);
      if (bootstrap.adapter != "trainvm.hostd-crash-qualification" ||
          bootstrap.capabilities !=
              std::vector<std::string>{"qualification.hostd-crash"}) {
        throw std::runtime_error(
            "qualification worker bootstrap has the wrong authority");
      }
      const pid_t daemon = ::getppid();
      if (daemon <= 1 || ::kill(daemon, SIGUSR1) != 0)
        throw std::runtime_error(
            "qualification worker could not attest exec readiness");
      for (;;)
        (void)::pause();
    }

    trainvm::HostdCrashQualificationConfig config;
    std::filesystem::path receipt_path;
    for (int index = 1; index < argc; ++index) {
      const std::string_view flag(argv[index]);
      const bool has_value = index + 1 < argc;
      if (flag == "--workspace" && has_value) {
        config.workspace = argv[++index];
      } else if (flag == "--cgroup-parent" && has_value) {
        config.cgroup_parent = argv[++index];
      } else if (flag == "--receipt" && has_value) {
        receipt_path = argv[++index];
      } else {
        usage();
        return 2;
      }
    }
    if (config.workspace.empty()) {
      usage();
      return 2;
    }

    const auto receipt = trainvm::qualify_hostd_crash_recovery(config);
    const std::string document =
        trainvm::hostd_crash_qualification_receipt_json(receipt).dump(2);
    if (receipt_path.empty()) {
      std::cout << document << '\n';
    } else {
      std::ofstream output(receipt_path, std::ios::binary | std::ios::trunc);
      if (!output)
        throw std::runtime_error("could not write the qualification receipt");
      output << document << '\n';
      if (!output)
        throw std::runtime_error("qualification receipt write failed");
      std::cout << receipt_path.string() << '\n';
    }
    return receipt.gate_open ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "trainvm-hostd-crash-qualification: " << error.what() << '\n';
    return 1;
  }
}
