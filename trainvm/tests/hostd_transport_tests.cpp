#include "trainvm/hostd_transport.hpp"

#include <fcntl.h>
#include <openssl/sha.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using namespace trainvm;

constexpr std::string_view kDigest =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

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

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    constexpr std::string_view source = "/tmp/trainvm-hostd-transport-XXXXXX";
    std::ranges::copy(source, pattern.begin());
    char *created = ::mkdtemp(pattern.data());
    require(created != nullptr, "create transport temporary directory");
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0,
            "protect transport temporary directory");
    parent_fd_ =
        ::open(path_.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    require(parent_fd_ >= 0, "pin transport temporary directory");
  }

  ~TemporaryDirectory() {
    if (parent_fd_ >= 0)
      (void)::close(parent_fd_);
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }
  [[nodiscard]] int parent_fd() const { return parent_fd_; }

private:
  std::filesystem::path path_;
  int parent_fd_{-1};
};

class HeldToken final : public IHostdSingletonToken {
public:
  bool attest_held() const override { return held; }
  bool held{true};
};

class ThrowAtBindCheckpoint final : public IHostdSocketBindFaultInjector {
public:
  explicit ThrowAtBindCheckpoint(HostdSocketBindCheckpoint selected)
      : selected_(selected) {}

  void checkpoint(HostdSocketBindCheckpoint checkpoint) override {
    if (checkpoint == selected_)
      throw std::runtime_error("injected bind checkpoint failure");
  }

private:
  HostdSocketBindCheckpoint selected_;
};

class ObserveBlockedSignals final : public IHostdSocketBindFaultInjector {
public:
  void checkpoint(HostdSocketBindCheckpoint checkpoint) override {
    if (checkpoint != HostdSocketBindCheckpoint::signals_blocked)
      return;
    sigset_t current{};
    if (::pthread_sigmask(SIG_SETMASK, nullptr, &current) != 0)
      throw std::runtime_error("could not observe checkpoint signal mask");
    observed = ::sigismember(&current, SIGUSR1) == 1 &&
               ::sigismember(&current, SIGTERM) == 1;
  }

  bool observed{};
};

HostdSocketAuthorityConfig socket_config(const TemporaryDirectory &directory,
                                         std::string name = "hostd.sock") {
  return {.api_version = std::string(kHostdStatusTransportApiVersion),
          .socket_path = directory.path() / std::move(name),
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .expected_parent_mode = 0700U,
          .expected_socket_mode = 0600U,
          .listen_backlog = 8U,
          .enforcement_grade =
              HostdSocketEnforcementGrade::cooperative_test,
          .fault_injector = nullptr};
}

HostResourceId mutex_id() {
  return {.kind = HostResourceKind::host_mutex,
          .vendor = std::nullopt,
          .stable_id = "host-mutex:transport",
          .parent_id = std::nullopt};
}

HostInventoryReceipt inventory() {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-transport",
      .boot_id = "boot-transport",
      .broker_epoch = "broker-transport",
      .begin_revision = "revision-transport",
      .end_revision = "revision-transport",
      .probes = {},
      .resources = {{.id = mutex_id(),
                     .disposition =
                         ResourceObservationDisposition::audited_eligible,
                     .compute_contexts = ResourceContextDisposition::absent,
                     .graphics_contexts = ResourceContextDisposition::absent,
                     .pci_bdf = std::nullopt,
                     .device_major = std::nullopt,
                     .device_minor = std::nullopt,
                     .numa_node = std::nullopt,
                     .pcie_root_id = std::nullopt,
                     .fabric_clique_id = std::nullopt,
                     .total_memory_bytes = 0U,
                     .labels = {{"scope", "transport-test"}}}},
  };
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

class Auditor final : public IHostdStartupAuditor {
public:
  explicit Auditor(const HostInventoryReceipt &observed)
      : receipt{.api_version = std::string(kHostdStartupAuditApiVersion),
                .audit_id = "transport-audit",
                .host_id = observed.host_id,
                .boot_id = observed.boot_id,
                .broker_epoch = observed.broker_epoch,
                .inventory_digest = observed.inventory_digest,
                .disposition = HostdStartupAuditDisposition::passed,
                .observed_orphans = 0U,
                .unresolved_orphans = 0U,
                .evidence_digest = std::string(kDigest)} {}

  HostdStartupAuditReceipt audit(const HostInventoryReceipt &) override {
    return receipt;
  }
  HostdStartupAuditReceipt receipt;
};

class BlockingAuditor final : public IHostdStartupAuditor {
public:
  explicit BlockingAuditor(const HostInventoryReceipt &observed)
      : receipt_(Auditor(observed).receipt) {}

  HostdStartupAuditReceipt audit(const HostInventoryReceipt &) override {
    std::unique_lock lock(mutex_);
    entered_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] { return released_; });
    return receipt_;
  }

  void wait_until_entered() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return entered_; });
  }

  void release() {
    std::scoped_lock lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

private:
  HostdStartupAuditReceipt receipt_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool entered_{};
  bool released_{};
};

class InvalidTextAuditor final : public IHostdStartupAuditor {
public:
  HostdStartupAuditReceipt audit(const HostInventoryReceipt &) override {
    throw std::runtime_error(std::string("invalid-") + char(0x01));
  }
};

struct CoordinatorFixture final {
  explicit CoordinatorFixture(const TemporaryDirectory &directory)
      : observed(inventory()),
        authority(std::make_shared<HostLedgerFilesystemAuthority>(
            HostLedgerFilesystemAuthority::acquire(
                {.api_version = std::string(kHostLedgerAuthorityApiVersion),
                 .ledger_path = directory.path() / "ledger.db",
                 .expected_owner_uid = ::geteuid(),
                 .expected_owner_gid = ::getegid(),
                 .enforcement_grade =
                     HostLedgerEnforcementGrade::cooperative_test}))),
        ledger(std::make_shared<SQLiteHostLedger>(authority, observed)),
        coordinator(std::make_shared<HostGrantCoordinator>(
            HostdCoordinatorConfig{
                .api_version = std::string(kHostdCoordinatorApiVersion),
                .host_id = observed.host_id,
                .boot_id = observed.boot_id,
                .broker_epoch = observed.broker_epoch,
                .maximum_live_sessions = 8U,
                .maximum_logical_scopes = 8U},
            ledger)) {}

  void admit() {
    Auditor auditor(observed);
    (void)coordinator->run_startup_audit(auditor);
  }

  HostInventoryReceipt observed;
  std::shared_ptr<HostLedgerFilesystemAuthority> authority;
  std::shared_ptr<SQLiteHostLedger> ledger;
  std::shared_ptr<HostGrantCoordinator> coordinator;
};

HostdStatusClientConfig client_config(HostdSocketAuthority &authority) {
  return {.socket_path = authority.socket_path(),
          .expected_endpoint = authority.reattest(),
          .expected_server_uid = ::geteuid(),
          .expected_server_gid = ::getegid(),
          .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes};
}

std::int64_t deadline(std::int64_t delta_ns = 2'000'000'000LL) {
  return hostd_monotonic_now_ns() + delta_ns;
}

HostdStatusPeerPolicy peer_policy() {
  return {.allowed_uid = ::geteuid(), .allowed_gid = ::getegid()};
}

int create_raw_listener(const HostdSocketAuthorityConfig &config) {
  const int descriptor =
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  require(descriptor >= 0, "create raw replacement listener");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string path = config.socket_path.string();
  require(path.size() < sizeof(address.sun_path), "raw listener path fits");
  std::memcpy(address.sun_path, path.data(), path.size());
  address.sun_path[path.size()] = '\0';
  const socklen_t length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + path.size() + 1U);
  require(::bind(descriptor, reinterpret_cast<const sockaddr *>(&address),
                 length) == 0,
          "bind raw replacement listener");
  require(::chmod(path.c_str(), config.expected_socket_mode) == 0,
          "protect raw listener path");
  require(::listen(descriptor, static_cast<int>(config.listen_backlog)) == 0,
          "listen on raw replacement listener");
  return descriptor;
}

void put_u16(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint16_t value) {
  bytes.at(offset) = std::byte((value >> 8U) & 0xffU);
  bytes.at(offset + 1U) = std::byte(value & 0xffU);
}

void put_u32(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint32_t value) {
  for (std::size_t index = 0U; index < 4U; ++index)
    bytes.at(offset + index) =
        std::byte((value >> (24U - static_cast<unsigned int>(index) * 8U)) &
                  0xffU);
}

void put_u64(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint64_t value) {
  for (std::size_t index = 0U; index < 8U; ++index)
    bytes.at(offset + index) =
        std::byte((value >> (56U - static_cast<unsigned int>(index) * 8U)) &
                  0xffU);
}

std::vector<std::byte> packet_with_payload(std::uint16_t opcode,
                                           std::uint64_t correlation,
                                           nlohmann::json payload) {
  const std::string canonical = payload.dump();
  std::vector<std::byte> bytes(kHostdStatusWireHeaderBytes + canonical.size());
  bytes[0] = std::byte{'T'};
  bytes[1] = std::byte{'V'};
  bytes[2] = std::byte{'H'};
  bytes[3] = std::byte{'D'};
  put_u16(bytes, 4U, kHostdStatusWireVersion);
  put_u16(bytes, 6U,
          static_cast<std::uint16_t>(kHostdStatusWireHeaderBytes));
  put_u16(bytes, 8U, opcode);
  put_u16(bytes, 10U, 0U);
  put_u32(bytes, 12U, static_cast<std::uint32_t>(canonical.size()));
  put_u64(bytes, 16U, correlation);
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  (void)::SHA256(reinterpret_cast<const unsigned char *>(canonical.data()),
                 canonical.size(), digest.data());
  for (std::size_t index = 0U; index < digest.size(); ++index)
    bytes[24U + index] = std::byte(digest[index]);
  std::memcpy(bytes.data() + kHostdStatusWireHeaderBytes, canonical.data(),
              canonical.size());
  return bytes;
}

nlohmann::json status_payload(const HostdCoordinatorStatus &status) {
  nlohmann::json audit = nullptr;
  if (status.startup_audit) {
    const auto &value = *status.startup_audit;
    audit = {{"api_version", value.api_version},
             {"audit_id", value.audit_id},
             {"boot_id", value.boot_id},
             {"broker_epoch", value.broker_epoch},
             {"disposition", static_cast<unsigned int>(value.disposition)},
             {"evidence_digest", value.evidence_digest},
             {"host_id", value.host_id},
             {"inventory_digest", value.inventory_digest},
             {"observed_orphans", value.observed_orphans},
             {"unresolved_orphans", value.unresolved_orphans}};
  }
  const auto lifecycle = [&] {
    switch (status.lifecycle) {
    case HostdLifecycle::sealed:
      return "sealed";
    case HostdLifecycle::startup_auditing:
      return "startup_auditing";
    case HostdLifecycle::admitting:
      return "admitting";
    case HostdLifecycle::poisoned:
      return "poisoned";
    }
    return "invalid";
  }();
  return {{"admission_counts_are_cached_evidence",
           status.admission_counts_are_cached_evidence},
          {"admission_sessions", status.admission_sessions},
          {"api_version", status.api_version},
          {"boot_id", status.boot_id},
          {"broker_epoch", status.broker_epoch},
          {"host_id", status.host_id},
          {"inventory_digest", status.inventory_digest},
          {"lifecycle", lifecycle},
          {"live_sessions", status.live_sessions},
          {"poison_reason", status.poison_reason},
          {"release_only_sessions", status.release_only_sessions},
          {"stale_admission_sessions", status.stale_admission_sessions},
          {"startup_audit", std::move(audit)}};
}

void serve_fake_response(const HostdSocketAuthority &authority,
                         const std::vector<std::byte> &response,
                         std::optional<int> passed_fd = std::nullopt) {
  pollfd readiness{.fd = authority.listener_fd(),
                   .events = POLLIN,
                   .revents = 0};
  require(::poll(&readiness, 1U, 2000) == 1 &&
              (readiness.revents & POLLIN) != 0,
          "fake server observes client connection");
  const int accepted = ::accept4(authority.listener_fd(), nullptr, nullptr,
                                 SOCK_CLOEXEC);
  require(accepted >= 0, "fake server accepts client");
  std::array<std::byte, 4096U> request{};
  require(::recv(accepted, request.data(), request.size(), 0) > 0,
          "fake server receives request");
  iovec vector{.iov_base = const_cast<std::byte *>(response.data()),
               .iov_len = response.size()};
  alignas(cmsghdr)
      std::array<std::byte,
                 CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int))>
          control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control.data();
  message.msg_controllen = passed_fd ? control.size()
                                     : CMSG_SPACE(sizeof(ucred));
  cmsghdr *credentials_header = CMSG_FIRSTHDR(&message);
  require(credentials_header != nullptr, "construct response credentials");
  credentials_header->cmsg_level = SOL_SOCKET;
  credentials_header->cmsg_type = SCM_CREDENTIALS;
  credentials_header->cmsg_len = CMSG_LEN(sizeof(ucred));
  const ucred credentials{.pid = ::getpid(),
                          .uid = ::geteuid(),
                          .gid = ::getegid()};
  std::memcpy(CMSG_DATA(credentials_header), &credentials,
              sizeof(credentials));
  if (passed_fd) {
    cmsghdr *rights = CMSG_NXTHDR(&message, credentials_header);
    require(rights != nullptr, "construct response rights");
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(rights), &*passed_fd, sizeof(int));
  }
  require(::sendmsg(accepted, &message, MSG_NOSIGNAL) ==
              static_cast<ssize_t>(response.size()),
          "fake server sends response");
  require(::close(accepted) == 0, "close fake accepted socket");
}

int connect_raw(const std::filesystem::path &path) {
  const int descriptor =
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  require(descriptor >= 0, "create raw transport client");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string native = path.string();
  std::memcpy(address.sun_path, native.data(), native.size());
  address.sun_path[native.size()] = '\0';
  const socklen_t length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + native.size() + 1U);
  require(::connect(descriptor, reinterpret_cast<const sockaddr *>(&address),
                    length) == 0,
          "connect raw transport client");
  return descriptor;
}

HostdServeResult send_raw(HostdStatusServer &server,
                          const std::filesystem::path &path,
                          const std::vector<std::byte> &packet,
                          std::optional<int> passed_fd = std::nullopt,
                          bool expect_silence = false) {
  HostdServeResult outcome = HostdServeResult::timed_out;
  std::jthread serving([&] { outcome = server.serve_one(deadline()); });
  const int client = connect_raw(path);
  alignas(cmsghdr)
      std::array<std::byte,
                 CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int))>
          control{};
  iovec vector{.iov_base = const_cast<std::byte *>(packet.data()),
               .iov_len = packet.size()};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1U;
  message.msg_control = control.data();
  message.msg_controllen = passed_fd ? control.size()
                                     : CMSG_SPACE(sizeof(ucred));
  cmsghdr *header = CMSG_FIRSTHDR(&message);
  require(header != nullptr, "construct SCM_CREDENTIALS header");
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_CREDENTIALS;
  header->cmsg_len = CMSG_LEN(sizeof(ucred));
  const ucred credentials{.pid = ::getpid(),
                          .uid = ::geteuid(),
                          .gid = ::getegid()};
  std::memcpy(CMSG_DATA(header), &credentials, sizeof(credentials));
  if (passed_fd) {
    cmsghdr *rights = CMSG_NXTHDR(&message, header);
    require(rights != nullptr, "construct SCM_RIGHTS header");
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(rights), &*passed_fd, sizeof(int));
  }
  require(::sendmsg(client, &message, MSG_NOSIGNAL) ==
              static_cast<ssize_t>(packet.size()),
          "send raw transport packet");
  serving.join();
  if (expect_silence) {
    std::array<std::byte, 256U> response{};
    const ssize_t received =
        ::recv(client, response.data(), response.size(), MSG_DONTWAIT);
    require(received == 0 ||
                (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)),
            "configured response bound suppresses an oversized typed error");
  }
  require(::close(client) == 0, "close raw transport client");
  return outcome;
}

std::size_t open_fd_count() {
  std::size_t count = 0U;
  for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd")) {
    (void)entry;
    ++count;
  }
  return count;
}

void authority_requires_external_singleton_and_pins_path() {
  TemporaryDirectory directory;
  auto missing = std::make_shared<HeldToken>();
  missing->held = false;
  require_throws<HostdTransportError>(
      [&] {
        (void)HostdSocketAuthority::self_bind(
            socket_config(directory), directory.parent_fd(), missing);
      },
      "self-bind refuses a missing external singleton token");

  auto held = std::make_shared<HeldToken>();
  auto authority = HostdSocketAuthority::self_bind(
      socket_config(directory), directory.parent_fd(), held);
  const auto identity = authority.reattest();
  require(identity.path_inode != 0U && !authority.poisoned(),
          "self-bound listener retains pinned parent/path identity");
  const int fd_flags = ::fcntl(authority.listener_fd(), F_GETFD);
  const int status_flags = ::fcntl(authority.listener_fd(), F_GETFL);
  require((fd_flags & FD_CLOEXEC) != 0 &&
              (status_flags & O_NONBLOCK) != 0,
          "listener is atomically CLOEXEC and nonblocking");
  require_throws<HostdTransportError>(
      [&] {
        (void)HostdSocketAuthority::self_bind(
            socket_config(directory), directory.parent_fd(), held);
      },
      "second live bind never unlinks the first socket pathname");
  require(authority.reattest() == identity,
          "failed second bind leaves original authority intact");
  held->held = false;
  require_throws<HostdTransportError>(
      [&] { (void)authority.reattest(); },
      "authority poisons immediately when its retained singleton is lost");
  require(authority.poisoned(),
          "singleton loss remains observable through authority status");
}

void startup_faults_rollback_and_restore_process_state() {
  TemporaryDirectory directory;
  struct stat cwd_before{};
  require(::stat(".", &cwd_before) == 0, "capture cwd before bind faults");
  sigset_t mask_before{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr, &mask_before) == 0,
          "capture signal mask before bind faults");

  const std::array checkpoints{
      HostdSocketBindCheckpoint::identity_captured,
      HostdSocketBindCheckpoint::before_socket_protection,
      HostdSocketBindCheckpoint::before_listen};
  std::size_t sequence = 0U;
  for (const auto checkpoint : checkpoints) {
    const std::string name = "fault-" + std::to_string(sequence++) + ".sock";
    auto config = socket_config(directory, name);
    config.fault_injector =
        std::make_shared<ThrowAtBindCheckpoint>(checkpoint);
    require_throws<HostdTransportError>(
        [&] {
          (void)HostdSocketAuthority::self_bind(
              config, directory.parent_fd(), std::make_shared<HeldToken>());
        },
        "post-bind checkpoint failure is normalized");
    require(!std::filesystem::exists(config.socket_path),
            "captured post-bind failure exact-unlinks its own pathname");
  }

  struct stat cwd_after{};
  require(::stat(".", &cwd_after) == 0 &&
              cwd_after.st_dev == cwd_before.st_dev &&
              cwd_after.st_ino == cwd_before.st_ino,
          "post-bind failure restores the exact process cwd");
  sigset_t mask_after{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr, &mask_after) == 0,
          "capture signal mask after bind faults");
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    require(::sigismember(&mask_before, signal_number) ==
                ::sigismember(&mask_after, signal_number),
            "post-bind failure restores the process signal mask");

  auto mask_observer = std::make_shared<ObserveBlockedSignals>();
  auto mask_config = socket_config(directory, "mask-checkpoint.sock");
  mask_config.fault_injector = mask_observer;
  {
    auto mask_authority = HostdSocketAuthority::self_bind(
        mask_config, directory.parent_fd(), std::make_shared<HeldToken>());
    require(mask_observer->observed,
            "signals are blocked before startup task enumeration");
    (void)mask_authority.reattest();
  }
  sigset_t mask_after_success{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr, &mask_after_success) == 0,
          "capture signal mask after successful bind");
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    require(::sigismember(&mask_before, signal_number) ==
                ::sigismember(&mask_after_success, signal_number),
            "successful bind restores the process signal mask");

  std::mutex thread_mutex;
  std::condition_variable thread_changed;
  bool thread_ready = false;
  bool thread_stop = false;
  std::jthread extra_thread([&] {
    std::unique_lock lock(thread_mutex);
    thread_ready = true;
    thread_changed.notify_all();
    thread_changed.wait(lock, [&] { return thread_stop; });
  });
  {
    std::unique_lock lock(thread_mutex);
    thread_changed.wait(lock, [&] { return thread_ready; });
  }
  const auto threaded_config = socket_config(directory, "threaded.sock");
  require_throws<HostdTransportError>(
      [&] {
        (void)HostdSocketAuthority::self_bind(
            threaded_config, directory.parent_fd(),
            std::make_shared<HeldToken>());
      },
      "self-bind refuses to change cwd after another thread exists");
  require(!std::filesystem::exists(threaded_config.socket_path),
          "multi-thread rejection occurs before creating a pathname");
  sigset_t mask_after_thread_rejection{};
  require(::pthread_sigmask(SIG_SETMASK, nullptr,
                            &mask_after_thread_rejection) == 0,
          "capture signal mask after multi-thread rejection");
  for (int signal_number = 1; signal_number < NSIG; ++signal_number)
    require(::sigismember(&mask_before, signal_number) ==
                ::sigismember(&mask_after_thread_rejection, signal_number),
            "multi-thread rejection restores the process signal mask");
  {
    std::scoped_lock lock(thread_mutex);
    thread_stop = true;
    thread_changed.notify_all();
  }
  extra_thread.join();

  auto capture_failstop_config =
      socket_config(directory, "capture-failstop.sock");
  capture_failstop_config.fault_injector =
      std::make_shared<ThrowAtBindCheckpoint>(
          HostdSocketBindCheckpoint::before_identity_capture);
  const pid_t capture_child = ::fork();
  require(capture_child >= 0, "fork capture fail-stop test child");
  if (capture_child == 0) {
    std::set_terminate([] { ::_exit(85); });
    (void)HostdSocketAuthority::self_bind(
        capture_failstop_config, directory.parent_fd(),
        std::make_shared<HeldToken>());
    ::_exit(87);
  }
  int capture_child_status = 0;
  require(::waitpid(capture_child, &capture_child_status, 0) == capture_child &&
              WIFEXITED(capture_child_status) &&
              WEXITSTATUS(capture_child_status) == 85,
          "pre-capture failure terminates instead of losing rollback identity");
  require(::unlink(capture_failstop_config.socket_path.c_str()) == 0,
          "remove capture fail-stop child's intentionally stale test path");

  auto failstop_config = socket_config(directory, "cwd-failstop.sock");
  failstop_config.fault_injector =
      std::make_shared<ThrowAtBindCheckpoint>(
          HostdSocketBindCheckpoint::before_cwd_restore);
  const pid_t child = ::fork();
  require(child >= 0, "fork cwd fail-stop test child");
  if (child == 0) {
    std::set_terminate([] { ::_exit(86); });
    (void)HostdSocketAuthority::self_bind(
        failstop_config, directory.parent_fd(),
        std::make_shared<HeldToken>());
    ::_exit(87);
  }
  int child_status = 0;
  require(::waitpid(child, &child_status, 0) == child &&
              WIFEXITED(child_status) && WEXITSTATUS(child_status) == 86,
          "cwd restoration failure terminates instead of continuing unsafely");
  require(::unlink(failstop_config.socket_path.c_str()) == 0,
          "remove fail-stop child's intentionally stale test pathname");
}

void path_replacement_poison_and_guarded_move_cleanup() {
  TemporaryDirectory directory;
  const auto config = socket_config(directory);
  auto held = std::make_shared<HeldToken>();
  auto authority = HostdSocketAuthority::self_bind(
      config, directory.parent_fd(), held);
  const auto original = authority.reattest();
  const auto displaced = directory.path() / "displaced.sock";
  require(::rename(config.socket_path.c_str(), displaced.c_str()) == 0,
          "replace visible socket pathname");
  const int replacement = create_raw_listener(config);
  require_throws<HostdTransportError>(
      [&] { (void)authority.reattest(); },
      "listener/path replacement poisons the authority");
  require(authority.poisoned() && !authority.poison_reason().empty(),
          "path replacement poison remains observable");
  require(::close(replacement) == 0, "close replacement listener");
  require(original.path_inode != 0U, "original endpoint was pinned");

  require(std::filesystem::exists(config.socket_path),
          "poisoned owner does not remove a replacement pathname");

  TemporaryDirectory move_directory;
  const auto first_path = move_directory.path() / "first.sock";
  const auto second_path = move_directory.path() / "second.sock";
  auto first = HostdSocketAuthority::self_bind(
      socket_config(move_directory, "first.sock"),
      move_directory.parent_fd(), std::make_shared<HeldToken>());
  auto second = HostdSocketAuthority::self_bind(
      socket_config(move_directory, "second.sock"),
      move_directory.parent_fd(), std::make_shared<HeldToken>());
  second = std::move(first);
  require(std::filesystem::exists(first_path) &&
              !std::filesystem::exists(second_path),
          "move assignment guarded-unlinks only the destination's old path");
  (void)second.reattest();
}

void status_only_lifecycle_and_endpoint_identity() {
  TemporaryDirectory directory;
  CoordinatorFixture fixture(directory);
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  HostdStatusServer server(authority, fixture.coordinator, peer_policy());
  const auto client = client_config(*authority);

  HostdServeResult sealed_result = HostdServeResult::timed_out;
  std::jthread sealed_server(
      [&] { sealed_result = server.serve_one(deadline()); });
  const auto sealed = hostd_request_status(client, 11U, deadline());
  sealed_server.join();
  require(sealed_result == HostdServeResult::served &&
              sealed.kind == HostdStatusReplyKind::status && sealed.status &&
              sealed.status->lifecycle == HostdLifecycle::sealed,
          "status transport truthfully exposes the sealed lifecycle");

  BlockingAuditor auditor(fixture.observed);
  std::jthread auditing(
      [&] { (void)fixture.coordinator->run_startup_audit(auditor); });
  auditor.wait_until_entered();
  HostdServeResult auditing_result = HostdServeResult::timed_out;
  std::jthread auditing_server(
      [&] { auditing_result = server.serve_one(deadline()); });
  const auto during_audit = hostd_request_status(client, 12U, deadline());
  auditing_server.join();
  require(auditing_result == HostdServeResult::served &&
              during_audit.status &&
              during_audit.status->lifecycle ==
                  HostdLifecycle::startup_auditing,
          "status transport truthfully exposes startup auditing");
  auditor.release();
  auditing.join();
  HostdServeResult admitted_result = HostdServeResult::timed_out;
  std::jthread admitted_server(
      [&] { admitted_result = server.serve_one(deadline()); });
  const auto admitted = hostd_request_status(client, 13U, deadline());
  admitted_server.join();
  require(admitted_result == HostdServeResult::served &&
              admitted.kind == HostdStatusReplyKind::status &&
              admitted.status &&
              admitted.status->lifecycle == HostdLifecycle::admitting &&
              admitted.correlation_id == 13U,
          "admitting coordinator returns an exact typed status response");

  HostdStatusServer unauthorized_server(
      authority, fixture.coordinator,
      {.allowed_uid = static_cast<uid_t>(::geteuid() + 1U),
       .allowed_gid = ::getegid()});
  HostdServeResult unauthorized_result = HostdServeResult::served;
  std::jthread unauthorized_thread([&] {
    unauthorized_result = unauthorized_server.serve_one(deadline());
  });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 14U, deadline()); },
      "status server closes a peer outside explicit UID/GID policy");
  unauthorized_thread.join();
  require(unauthorized_result == HostdServeResult::rejected,
          "wrong peer credentials are rejected before packet authority");

  auto wrong_endpoint = client;
  ++wrong_endpoint.expected_endpoint.path_inode;
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(wrong_endpoint, 15U, deadline()); },
      "client refuses a path whose pinned endpoint identity is inexact");

  TemporaryDirectory poisoned_directory;
  CoordinatorFixture poisoned_fixture(poisoned_directory);
  auto poisoned_authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(poisoned_directory),
                                      poisoned_directory.parent_fd(),
                                      std::make_shared<HeldToken>()));
  Auditor failed_auditor(poisoned_fixture.observed);
  failed_auditor.receipt.disposition = HostdStartupAuditDisposition::failed;
  failed_auditor.receipt.observed_orphans = 1U;
  failed_auditor.receipt.unresolved_orphans = 1U;
  require_throws<HostdStateError>(
      [&] {
        (void)poisoned_fixture.coordinator->run_startup_audit(failed_auditor);
      },
      "failed startup audit poisons coordinator");
  HostdStatusServer poisoned_server(poisoned_authority,
                                    poisoned_fixture.coordinator,
                                    peer_policy());
  HostdServeResult poisoned_result = HostdServeResult::timed_out;
  std::jthread poisoned_thread(
      [&] { poisoned_result = poisoned_server.serve_one(deadline()); });
  const auto poisoned = hostd_request_status(
      client_config(*poisoned_authority), 16U, deadline());
  poisoned_thread.join();
  require(poisoned_result == HostdServeResult::served && poisoned.status &&
              poisoned.status->lifecycle == HostdLifecycle::poisoned &&
              !poisoned.status->poison_reason.empty(),
          "status transport truthfully exposes poisoned lifecycle evidence");

  TemporaryDirectory invalid_text_directory;
  CoordinatorFixture invalid_text_fixture(invalid_text_directory);
  auto invalid_text_authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(invalid_text_directory),
                                      invalid_text_directory.parent_fd(),
                                      std::make_shared<HeldToken>()));
  InvalidTextAuditor invalid_text_auditor;
  require_throws<HostdStateError>(
      [&] {
        (void)invalid_text_fixture.coordinator->run_startup_audit(
            invalid_text_auditor);
      },
      "invalid-text auditor poisons coordinator");
  HostdStatusServer invalid_text_server(invalid_text_authority,
                                        invalid_text_fixture.coordinator,
                                        peer_policy());
  HostdServeResult invalid_text_result = HostdServeResult::served;
  std::jthread invalid_text_thread(
      [&] { invalid_text_result = invalid_text_server.serve_one(deadline()); });
  require_throws<HostdTransportError>(
      [&] {
        (void)hostd_request_status(client_config(*invalid_text_authority), 17U,
                                   deadline());
      },
      "server refuses unsafe status text before JSON serialization");
  invalid_text_thread.join();
  require(invalid_text_result == HostdServeResult::rejected,
          "server normalizes invalid status serialization state to rejection");
}

void malformed_packets_rights_and_deadlines_are_bounded() {
  TemporaryDirectory directory;
  CoordinatorFixture fixture(directory);
  fixture.admit();
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  HostdStatusServer server(
      authority, fixture.coordinator, peer_policy(),
      {.maximum_payload_bytes = 1024U,
       .per_session_timeout_ns = 30'000'000LL});

  std::vector<std::vector<std::byte>> malformed;
  auto bad_version = hostd_encode_status_request(21U);
  put_u16(bad_version, 4U, 999U);
  malformed.push_back(std::move(bad_version));
  auto bad_opcode = hostd_encode_status_request(22U);
  put_u16(bad_opcode, 8U, 999U);
  malformed.push_back(std::move(bad_opcode));
  auto bad_flags = hostd_encode_status_request(23U);
  put_u16(bad_flags, 10U, 1U);
  malformed.push_back(std::move(bad_flags));
  auto bad_digest = hostd_encode_status_request(24U);
  bad_digest[24U] ^= std::byte{1U};
  malformed.push_back(std::move(bad_digest));
  auto trailing = hostd_encode_status_request(25U);
  trailing.push_back(std::byte{0U});
  malformed.push_back(std::move(trailing));
  malformed.push_back(packet_with_payload(
      1U, 26U,
      {{"api_version", kHostdStatusTransportApiVersion}, {"extra", true}}));
  for (const auto &packet : malformed)
    require(send_raw(server, authority->socket_path(), packet) ==
                HostdServeResult::rejected,
            "version/op/flags/fields/digest/trailing packet is rejected");

  HostdStatusServer tiny_response_server(
      authority, fixture.coordinator, peer_policy(),
      {.maximum_payload_bytes = 1U,
       .per_session_timeout_ns = 30'000'000LL});
  require(send_raw(tiny_response_server, authority->socket_path(),
                   hostd_encode_status_request(29U), std::nullopt, true) ==
              HostdServeResult::rejected,
          "typed errors never exceed the configured response payload bound");

  auto oversized = hostd_encode_status_request(27U);
  oversized.resize(kHostdStatusWireHeaderBytes + 2048U, std::byte{0U});
  require(send_raw(server, authority->socket_path(), oversized) ==
              HostdServeResult::rejected,
          "MSG_TRUNC oversized packet is rejected without partial decode");

  const int harmless = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  require(harmless >= 0, "open harmless descriptor for SCM_RIGHTS test");
  const std::size_t before = open_fd_count();
  require(send_raw(server, authority->socket_path(),
                   hostd_encode_status_request(28U), harmless) ==
              HostdServeResult::rejected,
          "status-only transport rejects every SCM_RIGHTS packet");
  require(open_fd_count() == before,
          "rejected SCM_RIGHTS packet leaks no received descriptor");
  require(::close(harmless) == 0, "close harmless sender descriptor");

  require(server.serve_one(deadline(15'000'000LL)) ==
              HostdServeResult::timed_out,
          "absolute monotonic accept deadline is bounded");
  require(server.serve_one(std::numeric_limits<std::int64_t>::min()) ==
              HostdServeResult::timed_out,
          "an extreme expired deadline cannot underflow into a long wait");
  require_throws<HostdTransportError>(
      [&] {
        HostdStatusServer invalid(
            authority, fixture.coordinator, peer_policy(),
            {.maximum_payload_bytes = 1024U,
             .per_session_timeout_ns =
                 std::numeric_limits<std::int64_t>::max()});
        (void)invalid;
      },
      "an extreme session timeout is rejected before deadline arithmetic");
  HostdServeResult idle = HostdServeResult::served;
  std::jthread idle_server([&] { idle = server.serve_one(deadline()); });
  const int idle_client = connect_raw(authority->socket_path());
  idle_server.join();
  require(idle == HostdServeResult::rejected,
          "per-session receive deadline rejects an idle peer");
  require(::close(idle_client) == 0, "close idle client");

  std::vector<int> backlog_fillers;
  sockaddr_un saturated_address{};
  saturated_address.sun_family = AF_UNIX;
  const std::string saturated_path = authority->socket_path().string();
  std::memcpy(saturated_address.sun_path, saturated_path.data(),
              saturated_path.size());
  saturated_address.sun_path[saturated_path.size()] = '\0';
  const socklen_t saturated_length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + saturated_path.size() + 1U);
  bool observed_eagain = false;
  for (std::size_t attempt = 0U; attempt < 128U; ++attempt) {
    const int filler =
        ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    require(filler >= 0, "create backlog saturation client");
    if (::connect(filler,
                  reinterpret_cast<const sockaddr *>(&saturated_address),
                  saturated_length) == 0 || errno == EINPROGRESS ||
        errno == EALREADY) {
      backlog_fillers.push_back(filler);
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      observed_eagain = true;
      require(::close(filler) == 0, "close EAGAIN saturation client");
      break;
    }
    require(false, "unexpected backlog saturation connect outcome");
  }
  require(observed_eagain,
          "AF_UNIX listener backlog reaches deterministic EAGAIN");
  require_throws<HostdTransportError>(
      [&] {
        (void)hostd_request_status(client_config(*authority), 30U,
                                   deadline(30'000'000LL));
      },
      "AF_UNIX EAGAIN connect retries stop at the absolute deadline");
  for (const int filler : backlog_fillers)
    require(::close(filler) == 0, "close backlog saturation client");
}

void client_rejects_corruption_delegation_and_no_children_exist() {
  TemporaryDirectory directory;
  CoordinatorFixture fixture(directory);
  fixture.admit();
  auto held = std::make_shared<HeldToken>();
  auto authority = std::make_shared<HostdSocketAuthority>(
      HostdSocketAuthority::self_bind(socket_config(directory),
                                      directory.parent_fd(), held));
  const auto client = client_config(*authority);
  std::jthread fake_server([&] {
    pollfd readiness{.fd = authority->listener_fd(),
                     .events = POLLIN,
                     .revents = 0};
    require(::poll(&readiness, 1U, 2000) == 1 &&
                (readiness.revents & POLLIN) != 0,
            "fake server observes correlation-test connection");
    const int accepted = ::accept4(authority->listener_fd(), nullptr, nullptr,
                                   SOCK_CLOEXEC);
    require(accepted >= 0, "fake server accepts correlation test");
    std::array<std::byte, 4096U> request{};
    require(::recv(accepted, request.data(), request.size(), 0) > 0,
            "fake server receives status request");
    auto mismatched = hostd_encode_status_request(999U);
    put_u16(mismatched, 8U, 2U);
    require(::send(accepted, mismatched.data(), mismatched.size(), MSG_NOSIGNAL) ==
                static_cast<ssize_t>(mismatched.size()),
            "fake server sends mismatched correlation");
    require(::close(accepted) == 0, "close fake accepted socket");
  });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 31U, deadline()); },
      "client rejects a response with the wrong correlation ID");
  fake_server.join();

  auto invalid_status = status_payload(fixture.coordinator->status());
  invalid_status["startup_audit"]["host_id"] = "different-host";
  invalid_status["live_sessions"] = 0U;
  invalid_status["admission_sessions"] = 1U;
  const auto semantic_response = packet_with_payload(
      2U, 32U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"status", invalid_status}});
  std::jthread semantic_server(
      [&] { serve_fake_response(*authority, semantic_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 32U, deadline()); },
      "client rejects semantically contradictory status and audit evidence");
  semantic_server.join();

  const auto wrong_type_response = packet_with_payload(
      2U, 34U,
      {{"api_version", 7},
       {"status", status_payload(fixture.coordinator->status())}});
  std::jthread wrong_type_server(
      [&] { serve_fake_response(*authority, wrong_type_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 34U, deadline()); },
      "client normalizes untrusted JSON type exceptions");
  wrong_type_server.join();

  const auto invalid_error_response = packet_with_payload(
      3U, 35U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"code", "bad_error"},
       {"message", std::string("control\x01text", 12U)}});
  std::jthread invalid_error_server(
      [&] { serve_fake_response(*authority, invalid_error_response); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 35U, deadline()); },
      "client rejects typed-error strings outside the text policy");
  invalid_error_server.join();

  const auto valid_response = packet_with_payload(
      2U, 33U,
      {{"api_version", kHostdStatusTransportApiVersion},
       {"status", status_payload(fixture.coordinator->status())}});
  const int harmless = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  require(harmless >= 0, "open delegated response descriptor");
  const std::size_t before = open_fd_count();
  std::jthread rights_server(
      [&] { serve_fake_response(*authority, valid_response, harmless); });
  require_throws<HostdTransportError>(
      [&] { (void)hostd_request_status(client, 33U, deadline()); },
      "client rejects SCM_RIGHTS delegation from an otherwise valid server");
  rights_server.join();
  require(open_fd_count() == before,
          "client closes every rejected response-side descriptor");
  require(::close(harmless) == 0, "close delegated response descriptor");

  int status = 0;
  errno = 0;
  require(::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
          "status transport creates no child process");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string_view, void (*)()>> tests{
      {"authority", authority_requires_external_singleton_and_pins_path},
      {"startup-faults", startup_faults_rollback_and_restore_process_state},
      {"replacement", path_replacement_poison_and_guarded_move_cleanup},
      {"status", status_only_lifecycle_and_endpoint_identity},
      {"framing", malformed_packets_rights_and_deadlines_are_bounded},
      {"client-hardening",
       client_rejects_corruption_delegation_and_no_children_exist},
  };
  try {
    for (const auto &[name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
    std::cout << "hostd transport tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd transport test failure: " << error.what() << '\n';
    return 1;
  }
}
