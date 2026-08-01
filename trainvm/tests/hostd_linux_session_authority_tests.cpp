#include "trainvm/hostd_linux_session_authority.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

constexpr std::string_view kBootId = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

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

struct RandomAction final {
  ssize_t count{};
  int error_number{};
};

class FakeLinuxKernel final : public IHostdLinuxSessionKernel {
public:
  HostdLinuxSessionEnforcementGrade enforcement_grade() const override {
    return grade;
  }

  HostdLinuxRandomRead getrandom_bytes(void *buffer,
                                       std::size_t count) override {
    std::scoped_lock lock(mutex);
    if (throw_random)
      throw std::runtime_error("injected getrandom seam failure");
    RandomAction action{.count = static_cast<ssize_t>(count),
                        .error_number = 0};
    if (!random_actions.empty()) {
      action = random_actions.front();
      random_actions.pop_front();
    }
    if (action.count > 0 && static_cast<std::size_t>(action.count) <= count) {
      auto *bytes = static_cast<unsigned char *>(buffer);
      for (ssize_t index = 0; index < action.count; ++index)
        bytes[index] = static_cast<unsigned char>(random_byte++);
    }
    ++random_calls;
    return {.count = action.count, .error_number = action.error_number};
  }

  HostdLinuxClockRead clock_boottime() override {
    std::scoped_lock lock(mutex);
    ++clock_calls;
    if (clock_fail)
      return {.success = false, .value = {}, .error_number = EIO};
    const std::int64_t value = clock_ns;
    if (advance_clock)
      clock_ns += 1'000'000LL;
    return {.success = true,
            .value = {.tv_sec = static_cast<time_t>(value / 1'000'000'000LL),
                      .tv_nsec = static_cast<long>(value % 1'000'000'000LL)},
            .error_number = 0};
  }

  HostdLinuxBootIdRead read_boot_id() override {
    std::scoped_lock lock(mutex);
    ++boot_calls;
    if (boot_fail)
      return {.success = false, .value = {}, .error_number = EIO};
    std::string result = boot_id;
    if (!boot_reads.empty()) {
      result = std::move(boot_reads.front());
      boot_reads.pop_front();
    }
    return {.success = true, .value = std::move(result), .error_number = 0};
  }

  HostdLinuxPeerKernelObservation observe_process(pid_t) override {
    std::scoped_lock lock(mutex);
    ++process_calls;
    if (throw_process)
      throw std::runtime_error("injected process observation failure");
    return process;
  }

  std::mutex mutex;
  std::deque<RandomAction> random_actions;
  unsigned int random_byte{1U};
  std::size_t random_calls{};
  std::size_t clock_calls{};
  std::size_t boot_calls{};
  std::size_t process_calls{};
  std::string boot_id{std::string(kBootId)};
  std::deque<std::string> boot_reads;
  std::int64_t clock_ns{1'000'000'000LL};
  bool advance_clock{};
  bool clock_fail{};
  bool boot_fail{};
  bool throw_process{};
  bool throw_random{};
  HostdLinuxSessionEnforcementGrade grade{
      HostdLinuxSessionEnforcementGrade::cooperative_namespace_observation};
  HostdLinuxPeerKernelObservation process{
      .pid = 4242,
      .effective_uid_before = 1001U,
      .effective_gid_before = 1002U,
      .effective_uid_after = 1001U,
      .effective_gid_after = 1002U,
      .process_starttime_ticks_before = 9001U,
      .process_starttime_ticks_after = 9001U,
      .process_directory_device_before = 11U,
      .process_directory_inode_before = 12U,
      .process_directory_device_after = 11U,
      .process_directory_inode_after = 12U,
      .pidfd_available = true,
      .pidfd_alive_before = true,
      .pidfd_alive_after = true,
      .complete = true,
      .error_number = 0};
};

HostdLinuxSocketPeerCredentials credentials() {
  return {.uid = 1001U, .gid = 1002U, .pid = 4242};
}

HostdLinuxNamespaceIdentity namespace_identity(const char *path) {
  const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC);
  require(descriptor >= 0, "open live namespace identity");
  struct stat status{};
  require(::fstat(descriptor, &status) == 0 && ::close(descriptor) == 0,
          "inspect and close live namespace identity");
  return {.device = static_cast<std::uint64_t>(status.st_dev),
          .inode = static_cast<std::uint64_t>(status.st_ino)};
}

HostdLinuxHostNamespacePolicy current_namespace_policy() {
  return {.mount_namespace = namespace_identity("/proc/self/ns/mnt"),
          .pid_namespace = namespace_identity("/proc/self/ns/pid"),
          .cgroup_namespace = namespace_identity("/proc/self/ns/cgroup"),
          .time_namespace = namespace_identity("/proc/self/ns/time"),
          .time_for_children_namespace =
              namespace_identity("/proc/self/ns/time_for_children")};
}

void nonce_handles_eintr_short_reads_domains_and_poison() {
  auto kernel = std::make_shared<FakeLinuxKernel>();
  kernel->random_actions = {{.count = -1, .error_number = EINTR},
                            {.count = 5, .error_number = 0},
                            {.count = 27, .error_number = 0}};
  HostdLinuxCSPRNGNonceSource nonce(kernel, 32U);
  const std::string challenge = nonce.next_hex_256("challenge_id");
  kernel->random_byte = 1U;
  const std::string session = nonce.next_hex_256("session_nonce");
  require(
      challenge.size() == 64U && session.size() == 64U &&
          challenge != session &&
          std::ranges::all_of(challenge,
                              [](unsigned char value) {
                                return (value >= '0' && value <= '9') ||
                                       (value >= 'a' && value <= 'f');
                              }) &&
          kernel->random_calls == 4U,
      "getrandom EINTR/short reads are completed and purpose domains differ");

  auto failure_kernel = std::make_shared<FakeLinuxKernel>();
  failure_kernel->random_actions.push_back({.count = 0, .error_number = 0});
  HostdLinuxCSPRNGNonceSource failed(failure_kernel, 4U);
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)failed.next_hex_256("challenge_id"); },
      "zero-length getrandom fails closed");
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)failed.next_hex_256("challenge_id"); },
      "entropy failure permanently poisons the source");
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)nonce.next_hex_256("INVALID PURPOSE"); },
      "nonce purpose must be a bounded canonical domain");

  auto overread_kernel = std::make_shared<FakeLinuxKernel>();
  overread_kernel->random_actions.push_back({.count = 33, .error_number = 0});
  HostdLinuxCSPRNGNonceSource overread(overread_kernel, 4U);
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)overread.next_hex_256("challenge_id"); },
      "impossible getrandom over-read fails closed");

  auto throwing_kernel = std::make_shared<FakeLinuxKernel>();
  throwing_kernel->throw_random = true;
  HostdLinuxCSPRNGNonceSource throwing(throwing_kernel, 4U);
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)throwing.next_hex_256("challenge_id"); },
      "kernel seam exceptions are normalized and poison entropy");
}

void nonce_is_creator_thread_bound_before_kernel_or_mutex_state() {
  constexpr std::size_t count = 16U;
  auto kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxCSPRNGNonceSource nonce(kernel, 2U);
  std::atomic<std::size_t> rejected{};
  std::vector<std::thread> threads;
  for (std::size_t index = 0U; index < count; ++index) {
    threads.emplace_back([&] {
      try {
        (void)nonce.next_hex_256("challenge_id");
      } catch (const HostdSessionChallengeRejected &) {
        rejected.fetch_add(1U, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  require(rejected.load(std::memory_order_relaxed) == count &&
              kernel->random_calls == 0U,
          "sibling threads reject before entropy or nonce state is touched");
  const auto first = nonce.next_hex_256("challenge_id");
  const auto second = nonce.next_hex_256("session_nonce");
  require(first != second && kernel->random_calls == 2U,
          "creator thread retains intact nonce state after thread rejection");
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)nonce.next_hex_256("challenge_id"); },
      "bounded nonce lifetime fails closed at exhaustion");
}

void boottime_is_boot_bound_high_water_poisoned_and_thread_bound() {
  auto kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxBoottimeSource time(std::string(kBootId), kernel);
  require(time.now().boottime_ns == 1'000'000'000LL,
          "CLOCK_BOOTTIME is returned in nanoseconds");
  kernel->clock_ns = 1'100'000'000LL;
  require(time.now().boottime_ns == 1'100'000'000LL,
          "boottime may advance monotonically");
  kernel->clock_ns = 1'050'000'000LL;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)time.now(); }, "boottime regression is rejected");
  kernel->clock_ns = 1'200'000'000LL;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)time.now(); },
      "boottime regression permanently poisons the authority");

  auto torn_kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxBoottimeSource torn(std::string(kBootId), torn_kernel);
  torn_kernel->boot_reads = {std::string(kBootId),
                             "ffffffff-ffff-ffff-ffff-ffffffffffff"};
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)torn.now(); }, "boot identity tear poisons observation");

  auto failed_kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxBoottimeSource failed_time(std::string(kBootId), failed_kernel);
  failed_kernel->clock_fail = true;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)failed_time.now(); },
      "CLOCK_BOOTTIME syscall failure is normalized and poisoned");

  auto thread_kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxBoottimeSource thread_bound(std::string(kBootId), thread_kernel);
  const std::size_t boot_calls_before = thread_kernel->boot_calls;
  const std::size_t clock_calls_before = thread_kernel->clock_calls;
  bool thread_rejected = false;
  bool unexpected_thread_error = false;
  std::thread sibling([&] {
    try {
      (void)thread_bound.now();
    } catch (const HostdSessionChallengeRejected &) {
      thread_rejected = true;
    } catch (...) {
      unexpected_thread_error = true;
    }
  });
  sibling.join();
  require(thread_rejected && !unexpected_thread_error &&
              thread_kernel->boot_calls == boot_calls_before &&
              thread_kernel->clock_calls == clock_calls_before,
          "sibling thread rejects before boot identity, clock, or mutex state");
  require(thread_bound.now().boottime_ns == 1'000'000'000LL,
          "creator thread retains boottime authority after thread rejection");
}

void proc_parsers_handle_comm_and_effective_credentials() {
  std::string stat = "4242 (worker ) name) S";
  for (std::uint64_t field = 4U; field <= 21U; ++field)
    stat += " " + std::to_string(field);
  stat += " 9001 23 24\n";
  const auto starttime =
      hostd_linux_session_test_seam::parse_proc_stat_starttime(stat, 4242);
  const auto ids =
      hostd_linux_session_test_seam::parse_proc_status_effective_credentials(
          "Name:\tworker\nUid:\t1000\t1001\t1002\t1003\n"
          "Gid:\t2000\t2001\t2002\t2003\n");
  require(starttime == 9001U && ids && ids->first == 1001U &&
              ids->second == 2001U,
          "proc parsers handle parenthesized comm and bind effective UID/GID");
  require(!hostd_linux_session_test_seam::parse_proc_stat_starttime(stat, 7) &&
              !hostd_linux_session_test_seam::
                  parse_proc_status_effective_credentials(
                      "Uid:\t1\t2\t3\t4\nUid:\t1\t2\t3\t4\n"
                      "Gid:\t1\t2\t3\t4\n"),
          "wrong PID and duplicate status identity fields fail closed");
}

void peer_observer_binds_credentials_and_rejects_torn_or_reused_pid() {
  auto kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxPeerProcessObserver observer(kernel);
  const HostdSocketPeerInstance first = observer.observe(credentials());
  require(first == HostdSocketPeerInstance{.uid = 1001U,
                                           .gid = 1002U,
                                           .pid = 4242,
                                           .process_starttime_ticks = 9001U} &&
              observer.reobserve(credentials(), first) == first,
          "peer credentials bind one stable procfs process instance");

  const std::size_t calls_before = kernel->process_calls;
  bool thread_rejected = false;
  bool unexpected_thread_error = false;
  std::thread sibling([&] {
    try {
      (void)observer.observe(credentials());
    } catch (const HostdSessionChallengeRejected &) {
      thread_rejected = true;
    } catch (...) {
      unexpected_thread_error = true;
    }
  });
  sibling.join();
  require(thread_rejected && !unexpected_thread_error &&
              kernel->process_calls == calls_before,
          "sibling peer observation rejects before procfs or observer mutex");

  kernel->process.process_starttime_ticks_after = 9002U;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)observer.observe(credentials()); },
      "torn starttime rejects PID reuse");
  kernel->process.process_starttime_ticks_after = 9001U;
  kernel->process.effective_uid_after = 1003U;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)observer.observe(credentials()); },
      "torn effective credentials fail closed");
  kernel->process.effective_uid_after = 1001U;
  kernel->process.process_directory_inode_after = 99U;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)observer.observe(credentials()); },
      "changed pinned proc directory fails closed");
  kernel->process.process_directory_inode_after = 12U;
  kernel->process.pidfd_alive_after = false;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)observer.observe(credentials()); },
      "pidfd-observed disappearance fails closed");
  kernel->process.pidfd_alive_after = true;
  kernel->process.process_starttime_ticks_before = 9002U;
  kernel->process.process_starttime_ticks_after = 9002U;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)observer.reobserve(credentials(), first); },
      "stable but reused PID does not match the challenged instance");
  kernel->throw_process = true;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)observer.observe(credentials()); },
      "peer kernel seam exceptions are normalized");
}

void socket_factory_binds_kernel_credentials_and_explicit_grade() {
  std::array<int, 2U> sockets{-1, -1};
  require(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0,
                       sockets.data()) == 0,
          "create connected Unix socket peer fixture");
  auto kernel = std::make_shared<FakeLinuxKernel>();
  kernel->process.pid = ::getpid();
  kernel->process.effective_uid_before = ::geteuid();
  kernel->process.effective_uid_after = ::geteuid();
  kernel->process.effective_gid_before = ::getegid();
  kernel->process.effective_gid_after = ::getegid();
  auto peer = make_hostd_linux_bound_socket_peer(
      sockets[0], kernel,
      HostdLinuxSessionEnforcementGrade::cooperative_namespace_observation);
  require(peer.instance().pid == ::getpid() &&
              peer.instance().uid == ::geteuid() &&
              peer.instance().gid == ::getegid() &&
              peer.reobserve() == peer.instance() &&
              peer.enforcement_grade() ==
                  HostdLinuxSessionEnforcementGrade::
                      cooperative_namespace_observation,
          "socket factory derives credentials and binds the proc instance");
  bool thread_rejected = false;
  bool unexpected_thread_error = false;
  std::thread sibling([&] {
    try {
      (void)peer.reobserve();
    } catch (const HostdSessionChallengeRejected &) {
      thread_rejected = true;
    } catch (...) {
      unexpected_thread_error = true;
    }
  });
  sibling.join();
  require(thread_rejected && !unexpected_thread_error &&
              peer.reobserve() == peer.instance(),
          "bound socket peer rejects sibling thread before mutable state");
  require(::close(sockets[0]) == 0 && ::close(sockets[1]) == 0,
          "close original socket peer fixture descriptors");
  require(peer.reobserve() == peer.instance(),
          "bound observer owns its accepted-socket duplicate");

  std::array<int, 2U> strict_sockets{-1, -1};
  require(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0,
                       strict_sockets.data()) == 0,
          "create strict socket pidfd probe fixture");
  require_throws<HostdSessionChallengeRejected>(
      [&] {
        (void)make_hostd_linux_bound_socket_peer(
            strict_sockets[0], kernel,
            HostdLinuxSessionEnforcementGrade::
                strict_host_namespaces_and_socket_pidfd);
      },
      "strict socket grade refuses a cooperative namespace kernel");
  kernel->grade = HostdLinuxSessionEnforcementGrade::
      strict_host_namespaces_and_socket_pidfd;
  try {
    auto strict = make_hostd_linux_bound_socket_peer(
        strict_sockets[0], kernel,
        HostdLinuxSessionEnforcementGrade::
            strict_host_namespaces_and_socket_pidfd);
    require(strict.reobserve() == strict.instance(),
            "runtime SO_PEERPIDFD probe binds a strict live peer");
  } catch (const HostdSessionChallengeRejected &) {
    // An older running kernel may not implement the compile-visible option.
    // Strict grade must reject; it must never silently downgrade.
  }
  kernel->grade =
      HostdLinuxSessionEnforcementGrade::cooperative_namespace_observation;
  kernel->throw_process = true;
  require_throws<HostdSessionChallengeRejected>(
      [&] {
        (void)make_hostd_linux_bound_socket_peer(
            strict_sockets[0], kernel,
            HostdLinuxSessionEnforcementGrade::
                cooperative_namespace_observation);
      },
      "socket factory normalizes process-observer seam exceptions");
  kernel->throw_process = false;
  require(::close(strict_sockets[0]) == 0 &&
              ::close(strict_sockets[1]) == 0,
          "close strict socket pidfd probe descriptors");
}

void inherited_state_fails_before_touching_fork_unsafe_mutexes() {
  auto kernel = std::make_shared<FakeLinuxKernel>();
  HostdLinuxCSPRNGNonceSource nonce(kernel, 4U);
  const pid_t child = ::fork();
  require(child >= 0, "fork inherited-authority rejection fixture");
  if (child == 0) {
    try {
      (void)nonce.next_hex_256("challenge_id");
      ::_exit(90);
    } catch (const HostdSessionChallengeRejected &) {
      ::_exit(0);
    } catch (...) {
      ::_exit(91);
    }
  }
  int status = 0;
  require(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0,
          "fork child rejects inherited nonce state before locking");
  require(nonce.next_hex_256("challenge_id").size() == 64U,
          "fork rejection does not poison the parent authority copy");
}

void strict_namespace_grade_rejects_unproven_host_identity() {
  require_throws<HostdSessionChallengeError>(
      [&] {
        (void)make_hostd_linux_session_kernel(
            {.api_version =
                 std::string(kHostdLinuxSessionAuthorityApiVersion),
             .enforcement_grade =
                 HostdLinuxSessionEnforcementGrade::
                     strict_host_namespaces_and_socket_pidfd,
             .expected_host_namespaces = std::nullopt});
      },
      "strict namespace grade rejects a self-asserted unproven host context");
  const auto policy = current_namespace_policy();
  auto strict = make_hostd_linux_session_kernel(
      {.api_version = std::string(kHostdLinuxSessionAuthorityApiVersion),
       .enforcement_grade = HostdLinuxSessionEnforcementGrade::
           strict_host_namespaces_and_socket_pidfd,
       .expected_host_namespaces = policy});
  require(strict->enforcement_grade() ==
                  HostdLinuxSessionEnforcementGrade::
                      strict_host_namespaces_and_socket_pidfd &&
              strict->read_boot_id().success,
          "strict factory accepts and reattests exact guarded host namespaces");
  auto wrong = policy;
  ++wrong.time_namespace.inode;
  require_throws<HostdSessionChallengeError>(
      [&] {
        (void)make_hostd_linux_session_kernel(
            {.api_version =
                 std::string(kHostdLinuxSessionAuthorityApiVersion),
             .enforcement_grade =
                 HostdLinuxSessionEnforcementGrade::
                     strict_host_namespaces_and_socket_pidfd,
             .expected_host_namespaces = wrong});
      },
      "strict factory rejects a mismatched host time namespace");
  wrong = policy;
  ++wrong.cgroup_namespace.inode;
  require_throws<HostdSessionChallengeError>(
      [&] {
        (void)make_hostd_linux_session_kernel(
            {.api_version =
                 std::string(kHostdLinuxSessionAuthorityApiVersion),
             .enforcement_grade =
                 HostdLinuxSessionEnforcementGrade::
                     strict_host_namespaces_and_socket_pidfd,
             .expected_host_namespaces = wrong});
      },
      "strict factory rejects a mismatched host cgroup namespace");
}

void real_kernel_rejects_a_sibling_linux_task() {
  auto kernel = make_hostd_linux_session_kernel(
      {.api_version = std::string(kHostdLinuxSessionAuthorityApiVersion),
       .enforcement_grade = HostdLinuxSessionEnforcementGrade::
           cooperative_namespace_observation,
       .expected_host_namespaces = std::nullopt});
  require(kernel->read_boot_id().success,
          "real creator task can use its pinned procfs authority");
  bool thread_rejected = false;
  bool unexpected_thread_error = false;
  std::thread sibling([&] {
    try {
      (void)kernel->clock_boottime();
    } catch (const HostdSessionChallengeError &) {
      thread_rejected = true;
    } catch (...) {
      unexpected_thread_error = true;
    }
  });
  sibling.join();
  require(thread_rejected && !unexpected_thread_error,
          "real sibling Linux TID cannot borrow creator namespace authority");
  require(kernel->clock_boottime().success,
          "real creator authority remains usable after sibling rejection");
}

bool optional_live_self_smoke() {
  if (std::getenv("TRAINVM_RUN_LIVE_SESSION_AUTHORITY_SMOKE") == nullptr)
    return false;
  auto kernel = make_hostd_linux_session_kernel(
      {.api_version = std::string(kHostdLinuxSessionAuthorityApiVersion),
       .enforcement_grade = HostdLinuxSessionEnforcementGrade::
           cooperative_namespace_observation,
       .expected_host_namespaces = std::nullopt});
  const auto boot = kernel->read_boot_id();
  require(boot.success, "live pinned procfs boot identity is readable");
  HostdLinuxCSPRNGNonceSource nonce(kernel, 16U);
  HostdLinuxBoottimeSource time(boot.value, kernel);
  HostdLinuxPeerProcessObserver peers(kernel);
  const HostdLinuxSocketPeerCredentials self{
      .uid = ::geteuid(), .gid = ::getegid(), .pid = ::getpid()};
  const auto observed = peers.observe(self);
  require(peers.reobserve(self, observed) == observed,
          "live self process instance remains pinned");
  const auto challenge_nonce = nonce.next_hex_256("challenge_id");
  const auto session_nonce = nonce.next_hex_256("session_nonce");
  const auto now = time.now();
  require(challenge_nonce != session_nonce && now.boot_id == boot.value,
          "live entropy domains and boot clock remain exact");
  std::array<int, 2U> sockets{-1, -1};
  require(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0,
                       sockets.data()) == 0,
          "live socket-derived peer fixture is available");
  auto bound = make_hostd_linux_bound_socket_peer(
      sockets[0], kernel,
      HostdLinuxSessionEnforcementGrade::cooperative_namespace_observation);
  require(bound.reobserve() == bound.instance(),
          "live socket-derived credentials remain bound to proc identity");
  require(::close(sockets[0]) == 0 && ::close(sockets[1]) == 0,
          "close live socket-derived peer fixture");
  std::cout << "LIVE linux-session pid=" << observed.pid
            << " uid=" << observed.uid << " gid=" << observed.gid
            << " starttime=" << observed.process_starttime_ticks
            << " boottime_ns=" << now.boottime_ns
            << " nonce-prefix=" << challenge_nonce.substr(0U, 8U) << '\n';
  return true;
}

} // namespace

int main() {
  try {
    nonce_handles_eintr_short_reads_domains_and_poison();
    std::cout << "PASS nonce-faults\n";
    nonce_is_creator_thread_bound_before_kernel_or_mutex_state();
    std::cout << "PASS nonce-thread-binding\n";
    boottime_is_boot_bound_high_water_poisoned_and_thread_bound();
    std::cout << "PASS boottime\n";
    proc_parsers_handle_comm_and_effective_credentials();
    std::cout << "PASS proc-parsers\n";
    peer_observer_binds_credentials_and_rejects_torn_or_reused_pid();
    std::cout << "PASS peer-instance\n";
    socket_factory_binds_kernel_credentials_and_explicit_grade();
    std::cout << "PASS socket-peer\n";
    inherited_state_fails_before_touching_fork_unsafe_mutexes();
    std::cout << "PASS fork-poison\n";
    strict_namespace_grade_rejects_unproven_host_identity();
    std::cout << "PASS strict-namespace-policy\n";
    real_kernel_rejects_a_sibling_linux_task();
    std::cout << "PASS real-thread-binding\n";
    if (optional_live_self_smoke())
      std::cout << "PASS optional-live-self\n";
    else
      std::cout << "SKIP optional-live-self\n";
    std::cout << "hostd Linux session authority tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd Linux session authority test failure: " << error.what()
              << '\n';
    return 1;
  }
}
