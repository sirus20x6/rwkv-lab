#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "trainvm/hostd_crash_qualification.hpp"

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
