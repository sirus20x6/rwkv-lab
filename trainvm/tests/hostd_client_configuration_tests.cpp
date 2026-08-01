#include "trainvm/hostd_client_configuration.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace {

using namespace trainvm;

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

HostdClientConfigurationDocument document() {
  return {
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
  };
}

void direct_configuration_is_exact() {
  const auto source = document();
  const HostdClientConfiguration config(source);
  require(config.document() == source &&
              config.mutation().socket_path == source.socket_path &&
              config.mutation().expected_endpoint ==
                  source.expected_endpoint &&
              config.status().expected_endpoint == source.expected_endpoint &&
              config.request_timeout_ns() == source.request_timeout_ns,
          "validated configuration yields matching status and mutation policy");

  auto mismatched_owner = source;
  ++mismatched_owner.expected_server_uid;
  require_rejected(
      [&] { (void)HostdClientConfiguration(mismatched_owner); },
      "server credentials cannot disagree with socket authority");
  auto relative = source;
  relative.socket_path = "hostd.sock";
  require_rejected([&] { (void)HostdClientConfiguration(relative); },
                   "relative socket paths are rejected");
  auto unbounded = source;
  unbounded.request_timeout_ns = 31'000'000'000LL;
  require_rejected([&] { (void)HostdClientConfiguration(unbounded); },
                   "unbounded request timeouts are rejected");
}

void secure_file_loader_rejects_ambiguous_documents() {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-hostd-client-config-" +
                          std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto path = directory / "hostd-client.json";
  {
    std::ofstream output(path);
    output << encode_json(document()).dump();
  }
  (void)::chmod(path.c_str(), 0600);
  const auto loaded = HostdClientConfiguration::load_file(path);
  require(loaded.document() == document(),
          "secure loader preserves the exact declarative document");

  const auto duplicate = directory / "duplicate.json";
  {
    std::ofstream output(duplicate);
    output << R"json({"api_version":"trainvm.hostd-client/v1","api_version":"trainvm.hostd-client/v1"})json";
  }
  (void)::chmod(duplicate.c_str(), 0600);
  require_rejected([&] { (void)HostdClientConfiguration::load_file(duplicate); },
                   "duplicate JSON keys are rejected before schema decoding");

  const auto writable = directory / "writable.json";
  std::filesystem::copy_file(path, writable);
  (void)::chmod(writable.c_str(), 0666);
  require_rejected([&] { (void)HostdClientConfiguration::load_file(writable); },
                   "group/world-writable configuration has no authority");
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  try {
    direct_configuration_is_exact();
    secure_file_loader_rejects_ambiguous_documents();
    std::cout << "hostd client configuration tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd client configuration test failure: " << error.what()
              << '\n';
    return 1;
  }
}
