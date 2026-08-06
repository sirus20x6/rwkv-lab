#include "trainvm/hostd_client_configuration.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::uintmax_t kMaximumConfigurationBytes = 64U << 10U;
constexpr std::int64_t kMinimumRequestTimeoutNs = 1'000'000LL;
constexpr std::int64_t kMaximumRequestTimeoutNs = 30'000'000'000LL;

class FileDescriptor final {
 public:
  explicit FileDescriptor(int value) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

nlohmann::json read_configuration(const std::filesystem::path& path) {
  if (path.empty() || !path.is_absolute() || path.lexically_normal() != path) {
    throw HostdClientConfigurationError(
        "hostd client configuration path must be canonical and absolute");
  }
  const int opened =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (opened < 0) {
    throw HostdClientConfigurationError(
        "could not securely open hostd client configuration: " +
        std::string(std::strerror(errno)));
  }
  FileDescriptor descriptor(opened);
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) >
          kMaximumConfigurationBytes ||
      (before.st_uid != 0U && before.st_uid != ::geteuid()) ||
      (before.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw HostdClientConfigurationError(
        "hostd client configuration must be a bounded owner/root-owned regular file that is not group/world-writable");
  }
  std::string text(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0U;
  while (offset < text.size()) {
    const ssize_t count =
        ::read(descriptor.get(), text.data() + offset, text.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      throw HostdClientConfigurationError(
          "hostd client configuration changed while being read");
    }
    offset += static_cast<std::size_t>(count);
  }
  char extra{};
  ssize_t trailing{};
  do {
    trailing = ::read(descriptor.get(), &extra, 1U);
  } while (trailing < 0 && errno == EINTR);
  struct stat after {};
  if (trailing != 0 || ::fstat(descriptor.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    throw HostdClientConfigurationError(
        "hostd client configuration changed while being read");
  }

  bool duplicate_key = false;
  std::vector<std::set<std::string>> keys;
  try {
    const nlohmann::json::parser_callback_t callback =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (keys.size() <= index + 1U) keys.resize(index + 2U);
            keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (keys.size() <= index) keys.resize(index + 1U);
            if (!keys[index].insert(parsed.get<std::string>()).second)
              duplicate_key = true;
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     keys.size() > index + 1U) {
            keys[index + 1U].clear();
          }
          return true;
        };
    nlohmann::json result = nlohmann::json::parse(text, callback);
    if (duplicate_key) {
      throw HostdClientConfigurationError(
          "hostd client configuration contains a duplicate object key");
    }
    return result;
  } catch (const HostdClientConfigurationError&) {
    throw;
  } catch (const nlohmann::json::exception& error) {
    throw HostdClientConfigurationError(
        "hostd client configuration is not valid JSON: " +
        std::string(error.what()));
  }
}

bool canonical_socket_path(std::string_view value) {
  if (value.empty() || value.size() > 107U || value.find('\0') != value.npos)
    return false;
  const std::filesystem::path path(value);
  return path.is_absolute() && path.lexically_normal() == path &&
         path.filename() != "." && path.filename() != "..";
}

void validate_endpoint(const HostdSocketIdentity& endpoint) {
  if (endpoint.parent_device == 0U || endpoint.parent_inode == 0U ||
      endpoint.path_device == 0U || endpoint.path_inode == 0U ||
      endpoint.link_count != 1U ||
      (endpoint.parent_mode != 0700U && endpoint.parent_mode != 0750U) ||
      (endpoint.path_mode != 0600U && endpoint.path_mode != 0660U) ||
      endpoint.owner_uid != endpoint.parent_owner_uid ||
      endpoint.owner_gid != endpoint.parent_owner_gid) {
    throw HostdClientConfigurationError(
        "hostd endpoint identity is not a canonical protected socket");
  }
}

}  // namespace

HostdClientConfiguration::HostdClientConfiguration(
    HostdClientConfigurationDocument document)
    : document_(std::move(document)) {
  if (document_.api_version != kHostdClientConfigurationApiVersion ||
      !canonical_socket_path(document_.socket_path) ||
      document_.request_timeout_ns < kMinimumRequestTimeoutNs ||
      document_.request_timeout_ns > kMaximumRequestTimeoutNs ||
      document_.expected_server_uid >
          static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max()) ||
      document_.expected_server_gid >
          static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max())) {
    throw HostdClientConfigurationError(
        "hostd client configuration is noncanonical or outside its bounds");
  }
  validate_endpoint(document_.expected_endpoint);
  if (document_.expected_server_uid != document_.expected_endpoint.owner_uid ||
      document_.expected_server_gid != document_.expected_endpoint.owner_gid) {
    throw HostdClientConfigurationError(
        "hostd server credentials disagree with the pinned endpoint owner");
  }
  mutation_ = {
      .socket_path = document_.socket_path,
      .expected_endpoint = document_.expected_endpoint,
      .expected_server_uid = static_cast<uid_t>(document_.expected_server_uid),
      .expected_server_gid = static_cast<gid_t>(document_.expected_server_gid),
      .maximum_payload_bytes = kHostdStatusMaximumPayloadBytes,
  };
}

HostdClientConfiguration HostdClientConfiguration::load_file(
    const std::filesystem::path& path) {
  HostdClientConfigurationDocument document;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(read_configuration(path), document, "", diagnostics)) {
    throw HostdClientConfigurationError(
        "hostd client configuration schema validation failed: " +
        diagnostics_json(diagnostics).dump());
  }
  return HostdClientConfiguration(std::move(document));
}

const HostdMutationClientConfig& HostdClientConfiguration::mutation() const
    noexcept {
  return mutation_;
}

HostdStatusClientConfig HostdClientConfiguration::status() const {
  return {.socket_path = mutation_.socket_path,
          .expected_endpoint = mutation_.expected_endpoint,
          .expected_server_uid = mutation_.expected_server_uid,
          .expected_server_gid = mutation_.expected_server_gid,
          .maximum_payload_bytes = mutation_.maximum_payload_bytes};
}

std::int64_t HostdClientConfiguration::request_timeout_ns() const noexcept {
  return document_.request_timeout_ns;
}

const HostdClientConfigurationDocument& HostdClientConfiguration::document()
    const noexcept {
  return document_;
}

}  // namespace trainvm
