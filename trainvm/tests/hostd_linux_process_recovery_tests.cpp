#include "trainvm/hostd_linux_process_recovery.hpp"

#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("test proc read failed");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string executable_digest() {
  const int descriptor = ::open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) throw std::runtime_error("test exe open failed");
  struct Close final {
    int value;
    ~Close() { (void)::close(value); }
  } close{descriptor};
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("test digest init failed");
  std::array<unsigned char, 1U << 16U> buffer{};
  while (true) {
    const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
    if (count < 0 ||
        (count > 0 &&
         EVP_DigestUpdate(context.get(), buffer.data(),
                          static_cast<std::size_t>(count)) != 1))
      throw std::runtime_error("test digest update failed");
    if (count == 0) break;
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 ||
      size != 32U)
    throw std::runtime_error("test digest final failed");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "sha256:";
  for (std::size_t index = 0U; index < size; ++index) {
    result.push_back(digits[digest[index] >> 4U]);
    result.push_back(digits[digest[index] & 0x0fU]);
  }
  return result;
}

std::string boot_id() {
  std::string result = read_text("/proc/sys/kernel/random/boot_id");
  while (!result.empty() &&
         (result.back() == '\n' || result.back() == '\r'))
    result.pop_back();
  return result;
}

HostProcessSpawnRequest current_identity(pid_t pid = ::getpid()) {
  const std::string process = "/proc/" + std::to_string(pid);
  const std::uint64_t starttime =
      hostd_linux_process_recovery_test_seam::parse_proc_starttime(
          read_text(process + "/stat"));
  const std::string cgroup =
      hostd_linux_process_recovery_test_seam::parse_unified_cgroup(
          read_text(process + "/cgroup"));
  const std::filesystem::path cgroup_path =
      std::filesystem::path("/sys/fs/cgroup") / cgroup.substr(1U);
  struct stat status {};
  if (::stat(cgroup_path.c_str(), &status) != 0)
    throw std::runtime_error("test cgroup stat failed");
  return seal_host_process_spawn_request({
      .api_version = std::string(kHostProcessSpawnRequestApiVersion),
      .launch_id = "recovery-probe-launch",
      .launch_intent_digest = "sha256:" + std::string(64U, 'a'),
      .host_pid = pid,
      .process_starttime_ticks = starttime,
      .boot_id = boot_id(),
      .cgroup_path = cgroup,
      .cgroup_device = static_cast<std::uint64_t>(status.st_dev),
      .cgroup_inode = static_cast<std::uint64_t>(status.st_ino),
      .executable_digest = executable_digest(),
      .canonical_request_digest = {},
  });
}

void exact_current_process_is_pinned_without_mutation() {
  const HostProcessSpawnRequest expected = current_identity();
  LinuxProcessRecoveryProbe probe;
  auto recovered = probe.observe(expected);
  require(recovered.disposition ==
                  LinuxProcessRecoveryDisposition::exact_live_process &&
              recovered.process && recovered.process->alive() &&
              recovered.process->identity() == expected,
          "probe returns one pidfd-pinned exact live process");

  auto forged = expected;
  ++forged.process_starttime_ticks;
  forged.canonical_request_digest.clear();
  forged = seal_host_process_spawn_request(std::move(forged));
  auto mismatch = probe.observe(forged);
  require(mismatch.disposition ==
                  LinuxProcessRecoveryDisposition::identity_mismatch &&
              !mismatch.process,
          "PID reuse/starttime mismatch never yields authority");
}

void exited_pid_is_classified_without_signalling() {
  const pid_t child = ::fork();
  if (child < 0) throw std::runtime_error("test fork failed");
  if (child == 0) ::_exit(0);
  int status = 0;
  if (::waitpid(child, &status, 0) != child)
    throw std::runtime_error("test child wait failed");
  auto expected = current_identity();
  expected.host_pid = child;
  expected.canonical_request_digest.clear();
  expected = seal_host_process_spawn_request(std::move(expected));
  LinuxProcessRecoveryProbe probe;
  auto gone = probe.observe(expected);
  require(gone.disposition == LinuxProcessRecoveryDisposition::already_gone &&
              !gone.process,
          "a terminal recorded PID is reported gone without mutation");
}

void parser_seams_reject_ambiguous_proc_values() {
  bool rejected = false;
  try {
    (void)hostd_linux_process_recovery_test_seam::parse_unified_cgroup(
        "0::/one\n0::/two\n");
  } catch (const HostLedgerError&) {
    rejected = true;
  }
  require(rejected, "multiple cgroup memberships are rejected");
}

}  // namespace

int main() {
  try {
    exact_current_process_is_pinned_without_mutation();
    exited_pid_is_classified_without_signalling();
    parser_seams_reject_ambiguous_proc_values();
    std::cout << "hostd Linux process recovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd Linux process recovery test failure: " << error.what()
              << '\n';
    return 1;
  }
}
