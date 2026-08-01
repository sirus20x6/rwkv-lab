#include "trainvm/hostd_client_bootstrap.hpp"
#include "trainvm/service.hpp"

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace trainvm;

constexpr std::string_view kBootId =
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Callable>
void require_rejected(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const HostdClientConfigurationError&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

HostdClientConfiguration configuration() {
  return HostdClientConfiguration({
      .api_version = std::string(kHostdClientConfigurationApiVersion),
      .socket_path = "/run/trainvm/hostd.sock",
      .expected_endpoint = {
          .parent_device = 1U,
          .parent_inode = 2U,
          .parent_mode = 0700U,
          .parent_owner_uid = static_cast<std::uint32_t>(::geteuid()),
          .parent_owner_gid = static_cast<std::uint32_t>(::getegid()),
          .path_device = 1U,
          .path_inode = 3U,
          .path_mode = 0600U,
          .owner_uid = static_cast<std::uint32_t>(::geteuid()),
          .owner_gid = static_cast<std::uint32_t>(::getegid()),
          .link_count = 1U,
      },
      .expected_server_uid = static_cast<std::uint32_t>(::geteuid()),
      .expected_server_gid = static_cast<std::uint32_t>(::getegid()),
      .request_timeout_ns = 5'000'000'000LL,
  });
}

HostdStatusReply status_reply(const HostIdentity& host,
                              HostdLifecycle lifecycle =
                                  HostdLifecycle::sealed) {
  return {
      .kind = HostdStatusReplyKind::status,
      .correlation_id = 1U,
      .status = HostdCoordinatorStatus{
          .api_version = std::string(kHostdCoordinatorApiVersion),
          .lifecycle = lifecycle,
          .host_id = host.host_id,
          .boot_id = host.boot_id,
          .broker_epoch = "broker-bootstrap-001",
          .inventory_digest = "sha256:" + std::string(64U, 'a'),
          .live_sessions = 0U,
          .admission_sessions = 0U,
          .stale_admission_sessions = 0U,
          .release_only_sessions = 0U,
          .admission_counts_are_cached_evidence = true,
          .startup_audit = std::nullopt,
          .poison_reason = {},
      },
      .error = std::nullopt,
  };
}

void bootstrap_pins_status_and_stays_observational() {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-hostd-bootstrap-" +
                          std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const HostIdentity host{.host_id = "host-bootstrap-001",
                          .boot_id = std::string(kBootId)};
  AuthorityLock authority(directory / "journal.db");
  Journal journal(authority.journal_path(), authority.journal_identity(),
                  HostGrantEnforcement::required, host);
  bool queried = false;
  const auto config = configuration();
  const auto bundle = bootstrap_hostd_clients(
      journal, host, config,
      [] {
        return AuthorityTimeSample{.wall = {.nanoseconds = 1},
                                   .boot = {.nanoseconds = 1},
                                   .boot_id = std::string(kBootId)};
      },
      [&](const HostdStatusClientConfig& observed, std::uint64_t correlation,
          std::int64_t deadline) {
        const auto expected = config.status();
        queried = observed.socket_path == expected.socket_path &&
                  observed.expected_endpoint == expected.expected_endpoint &&
                  observed.expected_server_uid == expected.expected_server_uid &&
                  observed.expected_server_gid == expected.expected_server_gid &&
                  observed.maximum_payload_bytes ==
                      expected.maximum_payload_bytes &&
                  correlation == 1U && deadline > 0;
        return status_reply(host);
      });
  require(queried && bundle.broker_epoch == "broker-bootstrap-001" &&
              bundle.claim_provider && bundle.resource_client &&
              bundle.process_client && journal.event_count() == 0U &&
              !journal.current_hostd_controller_fence("gpu:0"),
          "bootstrap pins status and defers controller mutation until use");

  HostIdentity other = host;
  other.host_id = "host-bootstrap-other";
  require_rejected(
      [&] {
        (void)bootstrap_hostd_clients(
            journal, host, config,
            [] {
              return AuthorityTimeSample{.wall = {.nanoseconds = 1},
                                         .boot = {.nanoseconds = 1},
                                         .boot_id = std::string(kBootId)};
            },
            [&](const auto&, auto, auto) { return status_reply(other); });
      },
      "status from another host cannot bootstrap mutation clients");
  require_rejected(
      [&] {
        (void)bootstrap_hostd_clients(
            journal, host, config,
            [] {
              return AuthorityTimeSample{.wall = {.nanoseconds = 1},
                                         .boot = {.nanoseconds = 1},
                                         .boot_id = std::string(kBootId)};
            },
            [&](const auto&, auto, auto) {
              auto result = status_reply(host, HostdLifecycle::poisoned);
              result.status->poison_reason = "poisoned";
              return result;
            });
      },
      "poisoned hostd cannot bootstrap mutation clients");
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  try {
    bootstrap_pins_status_and_stays_observational();
    std::cout << "hostd client bootstrap tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd client bootstrap test failure: " << error.what()
              << '\n';
    return 1;
  }
}
