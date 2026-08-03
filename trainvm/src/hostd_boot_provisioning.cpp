#include "trainvm/hostd_boot_provisioning.hpp"

#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {

HostdDaemonConfiguration materialize_hostd_daemon_boot(
    const HostdDaemonConfiguration& configuration_template,
    const HostdLinuxBootAuthoritySnapshot& boot_authority) {
  HostdDaemonConfigurationDocument document =
      configuration_template.document();
  document.boot_id = boot_authority.boot_id;
  document.host_namespaces = boot_authority.host_namespaces;
  return HostdDaemonConfiguration(std::move(document));
}

std::string hostd_daemon_configuration_json(
    const HostdDaemonConfiguration& configuration) {
  return encode_json(configuration.document()).dump(2) + "\n";
}

HostdClientConfiguration make_hostd_client_configuration(
    const HostdDaemonConfiguration& daemon_configuration,
    const HostdSocketIdentity& endpoint, std::int64_t request_timeout_ns) {
  const HostdDaemonConfigurationDocument& daemon =
      daemon_configuration.document();
  return HostdClientConfiguration({
      .api_version = std::string(kHostdClientConfigurationApiVersion),
      .socket_path = daemon.socket.path,
      .expected_endpoint = endpoint,
      .expected_server_uid = daemon.authority_uid,
      .expected_server_gid = daemon.authority_gid,
      .request_timeout_ns = request_timeout_ns,
  });
}

std::string hostd_client_configuration_json(
    const HostdClientConfiguration& configuration) {
  return encode_json(configuration.document()).dump(2) + "\n";
}

}  // namespace trainvm
