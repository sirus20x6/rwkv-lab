#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <sys/types.h>

namespace trainvm {

struct LinuxCgroupAuthorityConfig final {
  std::filesystem::path root_path;
  std::string root_unified_path;
  uid_t expected_owner_uid{};
  gid_t expected_owner_gid{};

  bool operator==(const LinuxCgroupAuthorityConfig&) const = default;
};

struct LinuxAllocationCgroupIdentity final {
  std::string unified_path;
  std::uint64_t device{};
  std::uint64_t inode{};

  bool operator==(const LinuxAllocationCgroupIdentity&) const = default;
};

class LinuxCgroupAuthorityError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class LinuxAllocationCgroup final {
 public:
  LinuxAllocationCgroup(LinuxAllocationCgroup&& other) noexcept;
  LinuxAllocationCgroup& operator=(LinuxAllocationCgroup&& other) noexcept;
  ~LinuxAllocationCgroup();

  LinuxAllocationCgroup(const LinuxAllocationCgroup&) = delete;
  LinuxAllocationCgroup& operator=(const LinuxAllocationCgroup&) = delete;

  [[nodiscard]] const LinuxAllocationCgroupIdentity& identity() const;
  [[nodiscard]] int duplicate_fd() const;
  [[nodiscard]] bool empty() const;
  void remove_if_empty();
  // Once a launch intent is durable, the cgroup must survive errors for exact
  // retry/startup audit. Before that point destruction removes a new empty dir.
  void retain_for_durable_intent() noexcept;

 private:
  friend class LinuxCgroupAuthority;
  LinuxAllocationCgroup(LinuxAllocationCgroupIdentity identity, int descriptor,
                        int parent_descriptor, std::string name,
                        bool remove_on_destroy) noexcept;
  void close_and_maybe_remove() noexcept;

  LinuxAllocationCgroupIdentity identity_;
  int descriptor_{-1};
  int parent_descriptor_{-1};
  std::string name_;
  bool remove_on_destroy_{};
};

class LinuxCgroupAuthority final {
 public:
  explicit LinuxCgroupAuthority(LinuxCgroupAuthorityConfig config);
  ~LinuxCgroupAuthority();

  LinuxCgroupAuthority(const LinuxCgroupAuthority&) = delete;
  LinuxCgroupAuthority& operator=(const LinuxCgroupAuthority&) = delete;

  [[nodiscard]] LinuxAllocationCgroup open_or_create(
      const std::string& allocation_id, const std::string& launch_id) const;
  // Opens only the deterministic existing directory and requires its pinned
  // device/inode/path to match the durable spawn receipt. Unlike launch-time
  // open_or_create(), a nonempty cgroup is expected while a recovered worker
  // is still live.
  [[nodiscard]] LinuxAllocationCgroup open_existing_for_recovery(
      const std::string& allocation_id, const std::string& launch_id,
      const LinuxAllocationCgroupIdentity& expected) const;

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

namespace hostd_linux_cgroup_authority_test_seam {

[[nodiscard]] std::string allocation_cgroup_name(
    const std::string& allocation_id, const std::string& launch_id);

}  // namespace hostd_linux_cgroup_authority_test_seam

}  // namespace trainvm
