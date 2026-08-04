#include <poll.h>
#include <signal.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/authority_document.hpp"
#include "trainvm/hostd_boot_provisioning.hpp"
#include "trainvm/hostd_daemon_configuration.hpp"
#include "trainvm/hostd_daemon_runtime.hpp"
#include "trainvm/hostd_gpu_authorization.hpp"

namespace {

volatile sig_atomic_t stop_requested = 0;

void request_stop(int) { stop_requested = 1; }

void install_signal_handlers() {
  struct sigaction action{};
  action.sa_handler = request_stop;
  if (::sigemptyset(&action.sa_mask) != 0 ||
      ::sigaction(SIGINT, &action, nullptr) != 0 ||
      ::sigaction(SIGTERM, &action, nullptr) != 0)
    throw std::runtime_error("could not install hostd signal handlers");
}

void wait_for_startup_wake(std::int64_t nanoseconds) {
  const std::int64_t milliseconds =
      std::max<std::int64_t>(1, nanoseconds / 1'000'000LL);
  const int bounded =
      static_cast<int>(std::min<std::int64_t>(milliseconds, 1000LL));
  while (::poll(nullptr, 0, bounded) < 0 && errno == EINTR &&
         stop_requested == 0) {
  }
}

void require_publication_identity(
    const trainvm::HostdDaemonConfigurationDocument& configuration) {
  if (::geteuid() != static_cast<uid_t>(configuration.authority_uid)) {
    throw std::runtime_error(
        "hostd boot publication requires the configured authority uid");
  }
}

void publish_document(const std::filesystem::path& path,
                      std::string_view kind, std::string_view contents,
                      const trainvm::HostdDaemonConfigurationDocument& config,
                      std::uintmax_t maximum_bytes) {
  require_publication_identity(config);
  trainvm::publish_authority_document(
      path, kind, contents,
      {.owner_uid = static_cast<uid_t>(config.authority_uid),
       .owner_gid = static_cast<gid_t>(config.authority_gid),
       .file_mode = 0640,
       .parent_owner_uid = static_cast<uid_t>(config.authority_uid),
       .parent_owner_gid = static_cast<gid_t>(config.authority_gid)},
      maximum_bytes);
}

int materialize_configuration(const std::filesystem::path& source,
                              const std::filesystem::path& destination) {
  const trainvm::HostdDaemonConfiguration configuration_template =
      trainvm::HostdDaemonConfiguration::load_file(source);
  const std::filesystem::path socket_parent = std::filesystem::path(
      configuration_template.document().socket.path).parent_path();
  if (destination.parent_path() != socket_parent ||
      destination.filename() ==
          std::filesystem::path(
              configuration_template.document().socket.path).filename()) {
    throw std::runtime_error(
        "materialized hostd configuration must be a distinct file in the protected socket directory");
  }
  const trainvm::HostdDaemonConfiguration configuration =
      trainvm::materialize_hostd_daemon_boot(
          configuration_template,
          trainvm::observe_hostd_linux_boot_authority());
  publish_document(destination, "materialized hostd configuration",
                   trainvm::hostd_daemon_configuration_json(configuration),
                   configuration.document(),
                   trainvm::kHostdDaemonConfigurationMaximumBytes);
  std::cout << "materialized "
            << trainvm::kHostdDaemonConfigurationApiVersion << '\n';
  return 0;
}

int authorize_gpu_start(int argc, char** argv) {
  const std::filesystem::path source(argv[2]);
  const std::filesystem::path destination(argv[3]);
  const trainvm::HostdDaemonConfiguration configuration_template =
      trainvm::HostdDaemonConfiguration::load_file(source);
  require_publication_identity(configuration_template.document());
  if (!destination.is_absolute() ||
      destination.lexically_normal() != destination ||
      destination.parent_path() != source.parent_path() ||
      destination == source) {
    throw std::runtime_error(
        "hostd GPU authorization must be a distinct canonical file beside the protected template");
  }
  trainvm::HostdDisplayGpuPolicy policy{};
  if (std::string_view(argv[4]) == "deny") {
    policy = trainvm::HostdDisplayGpuPolicy::deny;
    if (argc != 5)
      throw std::runtime_error("deny display policy cannot carry GPU IDs");
  } else if (std::string_view(argv[4]) == "cooperative_allowlist") {
    policy = trainvm::HostdDisplayGpuPolicy::cooperative_allowlist;
    if (argc < 6 || argc > 21)
      throw std::runtime_error(
          "cooperative display policy requires 1 through 16 exact GPU IDs");
  } else {
    throw std::runtime_error(
        "GPU authorization policy must be deny or cooperative_allowlist");
  }
  std::vector<std::string> allowed;
  for (int index = 5; index < argc; ++index)
    allowed.emplace_back(argv[index]);
  const trainvm::HostdGpuAuthorization authorization =
      trainvm::make_hostd_gpu_authorization(
          configuration_template,
          trainvm::observe_hostd_linux_boot_authority(), policy,
          std::move(allowed));
  trainvm::publish_authority_document(
      destination, "hostd GPU authorization",
      trainvm::hostd_gpu_authorization_json(authorization),
      {.owner_uid = static_cast<uid_t>(
           configuration_template.document().authority_uid),
       .owner_gid = static_cast<gid_t>(
           configuration_template.document().authority_gid),
       .file_mode = 0600,
       .parent_owner_uid = static_cast<uid_t>(
           configuration_template.document().authority_uid),
       .parent_owner_gid = static_cast<gid_t>(
           configuration_template.document().authority_gid)},
      trainvm::kHostdGpuAuthorizationMaximumBytes);
  std::cout << "authorized " << trainvm::kHostdGpuAuthorizationApiVersion
            << " for boot " << authorization.document().boot_id << '\n';
  return 0;
}

int check_gpu_authorization(const std::filesystem::path& source,
                            const std::filesystem::path& authorization_path) {
  const trainvm::HostdDaemonConfiguration configuration_template =
      trainvm::HostdDaemonConfiguration::load_file(source);
  const trainvm::HostdDaemonConfiguration boot_configuration =
      trainvm::materialize_hostd_daemon_boot(
          configuration_template,
          trainvm::observe_hostd_linux_boot_authority());
  const trainvm::HostdGpuAuthorization authorization =
      trainvm::HostdGpuAuthorization::load_file(authorization_path);
  authorization.require_matches(boot_configuration);
  std::cout << "authorized " << trainvm::kHostdGpuAuthorizationApiVersion
            << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool run = argc == 7 &&
                     std::string_view(argv[1]) == "--config" &&
                     std::string_view(argv[3]) == "--gpu-authorization" &&
                     std::string_view(argv[5]) == "--publish-client-config";
    const bool validate =
        argc == 3 && std::string_view(argv[1]) == "--validate-config";
    const bool materialize =
        argc == 4 && std::string_view(argv[1]) == "--materialize-config";
    const bool check_authorization =
        argc == 4 &&
        std::string_view(argv[1]) == "--check-gpu-authorization";
    const bool authorize =
        argc >= 5 && std::string_view(argv[1]) == "--authorize-gpu-start";
    if (!run && !validate && !materialize && !check_authorization &&
        !authorize) {
      std::cerr
          << "usage:\n"
          << "  trainvm-hostd --validate-config /absolute/hostd.json\n"
          << "  trainvm-hostd --materialize-config /absolute/template.json /absolute/runtime.json\n"
          << "  trainvm-hostd --authorize-gpu-start /absolute/template.json /absolute/authorization.json (deny|cooperative_allowlist) [GPU-uuid ...]\n"
          << "  trainvm-hostd --check-gpu-authorization /absolute/template.json /absolute/authorization.json\n"
          << "  trainvm-hostd --config /absolute/runtime.json --gpu-authorization /absolute/authorization.json --publish-client-config /absolute/client.json\n";
      return 2;
    }
    if (authorize)
      return authorize_gpu_start(argc, argv);
    if (check_authorization)
      return check_gpu_authorization(argv[2], argv[3]);
    if (materialize)
      return materialize_configuration(argv[2], argv[3]);
    auto configuration = trainvm::HostdDaemonConfiguration::load_file(argv[2]);
    if (validate) {
      std::cout << "valid " << trainvm::kHostdDaemonConfigurationApiVersion
                << '\n';
      return 0;
    }
    trainvm::HostdGpuAuthorization gpu_authorization =
        trainvm::HostdGpuAuthorization::load_file(argv[4]);
    gpu_authorization.require_matches(configuration);
    install_signal_handlers();
    const std::int64_t wake = configuration.serve_wake_interval_ns();
    trainvm::HostdDaemonRuntime runtime(configuration,
                                        std::move(gpu_authorization));
    while (!runtime.ready() && stop_requested == 0) {
      const auto status = runtime.advance_startup();
      if (status.phase == trainvm::HostdStartupPhase::reconciling)
        wait_for_startup_wake(wake);
    }
    if (stop_requested != 0)
      return 0;
    {
      const std::filesystem::path client_path(argv[6]);
      const auto& document = configuration.document();
      const std::filesystem::path socket_path(document.socket.path);
      if (!client_path.is_absolute() ||
          client_path.lexically_normal() != client_path ||
          client_path.parent_path() != socket_path.parent_path() ||
          client_path.filename() == socket_path.filename() ||
          client_path == std::filesystem::path(argv[2])) {
        throw std::runtime_error(
            "hostd client publication must be a distinct canonical file in the protected socket directory");
      }
      const trainvm::HostdClientConfiguration client =
          trainvm::make_hostd_client_configuration(
              configuration, runtime.socket_identity(),
              document.transport.mutation_session_timeout_ns);
      publish_document(client_path, "hostd client configuration",
                       trainvm::hostd_client_configuration_json(client),
                       document,
                       trainvm::kHostdClientConfigurationMaximumBytes);
    }
    while (stop_requested == 0)
      (void)runtime.serve_one();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "trainvm-hostd: " << error.what() << '\n';
    return 1;
  }
}
