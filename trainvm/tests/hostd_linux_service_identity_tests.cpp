#include "trainvm/hostd_linux_service_identity.hpp"

#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

std::string read_text(const std::string &path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("could not read live Linux identity fixture");
  std::ostringstream result;
  result << stream.rdbuf();
  return result.str();
}

void parser_is_strictly_unified_and_canonical() {
  using hostd_linux_service_identity_test_seam::parse_unified_cgroup_path;
  require(parse_unified_cgroup_path("0::/system.slice/trainvm.service\n") ==
              "/system.slice/trainvm.service" &&
              parse_unified_cgroup_path("0::/\n") == "/",
          "unified cgroup parser accepts canonical root and service paths");
  const std::string_view invalid[]{
      "",          "0::",       "1:name:/x\n", "0::relative\n",
      "0::/a//b\n", "0::/a/../b\n", "0::/a/\n",   "0::/a\n0::/b\n",
  };
  for (const auto value : invalid)
    require(!parse_unified_cgroup_path(value),
            "unified cgroup parser rejects ambiguous membership text");
  require(!parse_unified_cgroup_path("0::/service\n", 4U),
          "unified cgroup parser enforces its byte bound");
}

void live_authority_binds_process_instance_uid_and_cgroup() {
  using hostd_linux_service_identity_test_seam::parse_unified_cgroup_path;
  const auto cgroup = parse_unified_cgroup_path(read_text("/proc/self/cgroup"));
  const auto start = hostd_linux_session_test_seam::parse_proc_stat_starttime(
      read_text("/proc/self/stat"), ::getpid());
  require(cgroup && start, "live process has canonical cgroup-v2 evidence");
  HostdLinuxServiceIdentityAuthority authority({
      .api_version = std::string(kHostdLinuxServiceIdentityApiVersion),
      .roles = {{.cgroup_path = *cgroup,
                 .service_identity = "trainvm.test.service",
                 .expected_uid = ::geteuid(),
                 .expected_gid = ::getegid(),
                 .access = HostdSessionAccess::grant_release}},
      .maximum_cgroup_file_bytes = 4096U,
  });
  const HostdSocketPeerInstance peer{
      .uid = ::geteuid(),
      .gid = ::getegid(),
      .pid = ::getpid(),
      .process_starttime_ticks = *start,
  };
  const auto accepted = authority.authorize(peer);
  require(accepted.service_identity == "trainvm.test.service" &&
              accepted.access == HostdSessionAccess::grant_release &&
              accepted.service_identity_enforced,
          "live authority returns only the pinned exact service role");
  auto wrong = peer;
  ++wrong.process_starttime_ticks;
  require_throws<HostdLinuxServiceIdentityError>(
      [&] { (void)authority.authorize(wrong); },
      "reused or forged process starttime is rejected");
  wrong = peer;
  wrong.uid = static_cast<uid_t>(wrong.uid + 1U);
  require_throws<HostdLinuxServiceIdentityError>(
      [&] { (void)authority.authorize(wrong); },
      "peer credentials must match the configured service role");
  bool thread_rejected = false;
  std::thread sibling([&] {
    try {
      (void)authority.authorize(peer);
    } catch (const HostdLinuxServiceIdentityError &) {
      thread_rejected = true;
    }
  });
  sibling.join();
  require(thread_rejected && authority.authorize(peer) == accepted,
          "authority cannot cross its creator task and remains usable there");
}

} // namespace

int main() {
  try {
    parser_is_strictly_unified_and_canonical();
    std::cout << "PASS parser\n";
    live_authority_binds_process_instance_uid_and_cgroup();
    std::cout << "PASS live-authority\n";
    std::cout << "hostd Linux service identity tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd Linux service identity test failure: " << error.what()
              << '\n';
    return 1;
  }
}
