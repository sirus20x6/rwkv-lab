#pragma once

#include <functional>
#include <string_view>

#include "trainvm/hostd_mutation_client.hpp"
#include "trainvm/reconciler.hpp"

namespace trainvm {

using HostdResourceOpenFactory =
    std::function<HostdMutationOpen(std::string_view request_id)>;

struct HostdResourceClientOptions final {
  HostdMutationClientChannelOptions channel;
  HostdResourceOpenFactory open_for_request;
};

class HostdResourceClient final : public IHostGrantClient {
 public:
  explicit HostdResourceClient(HostdResourceClientOptions options);

  [[nodiscard]] BundleRequestResult request_bundle(
      const ResourceBundleRequest& request) override;
  [[nodiscard]] BundleReleaseResult release_bundle(
      const ResourceReleaseRequest& request) override;

 private:
  [[nodiscard]] HostdMutationOpen open_for(
      std::string_view request_id, std::string_view journal_id,
      std::string_view run_id, std::string_view logical_lease_id,
      std::uint64_t logical_fencing_token) const;

  HostdResourceOpenFactory open_for_request_;
  HostdMutationClientChannel channel_;
};

}  // namespace trainvm
