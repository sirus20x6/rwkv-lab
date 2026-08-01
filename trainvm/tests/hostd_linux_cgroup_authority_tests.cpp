#include "trainvm/hostd_linux_cgroup_authority.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

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
  } catch (const LinuxCgroupAuthorityError&) {
    return;
  }
  throw std::runtime_error(message);
}

void names_are_bounded_deterministic_and_domain_separated() {
  using hostd_linux_cgroup_authority_test_seam::allocation_cgroup_name;
  const std::string first = allocation_cgroup_name("allocation-a", "launch-a");
  require(first == allocation_cgroup_name("allocation-a", "launch-a") &&
              first.starts_with("launch-") && first.size() == 39U,
          "allocation cgroup name is deterministic and bounded");
  require(first != allocation_cgroup_name("allocation-b", "launch-a") &&
              first != allocation_cgroup_name("allocation-a", "launch-b"),
          "allocation and launch IDs both bind the cgroup name");
  require_rejected([&] { (void)allocation_cgroup_name("", "launch"); },
                   "empty allocation ID is rejected before filesystem use");
}

void non_cgroup_filesystems_fail_closed() {
  require_rejected(
      [&] {
        LinuxCgroupAuthority authority({
            .root_path = std::filesystem::temp_directory_path(),
            .root_unified_path = "/trainvm-test",
            .expected_owner_uid = ::geteuid(),
            .expected_owner_gid = ::getegid(),
        });
      },
      "ordinary filesystems cannot impersonate cgroup-v2 authority");
  require_rejected(
      [&] {
        LinuxCgroupAuthority authority({
            .root_path = "/sys/fs/cgroup/../cgroup",
            .root_unified_path = "/trainvm-test",
            .expected_owner_uid = 0,
            .expected_owner_gid = 0,
        });
      },
      "noncanonical cgroup root paths fail before open");
}

}  // namespace

int main() {
  try {
    names_are_bounded_deterministic_and_domain_separated();
    non_cgroup_filesystems_fail_closed();
    std::cout << "Linux cgroup authority tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Linux cgroup authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
