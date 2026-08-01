#pragma once

#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"

namespace trainvm {

class LinuxPreparedLaunch final {
 public:
  LinuxPreparedLaunch(LinuxPreparedLaunch&&) noexcept = default;
  LinuxPreparedLaunch& operator=(LinuxPreparedLaunch&&) noexcept = default;

  LinuxPreparedLaunch(const LinuxPreparedLaunch&) = delete;
  LinuxPreparedLaunch& operator=(const LinuxPreparedLaunch&) = delete;

  [[nodiscard]] const HostProcessLaunchIntent& intent() const;
  [[nodiscard]] const HostProcessSpawnReceipt& spawn_receipt() const;
  [[nodiscard]] const LinuxStoppedChildIdentity& child_identity() const;
  void release_to_exec();

 private:
  friend class LinuxProcessAuthority;
  LinuxPreparedLaunch(HostProcessLaunchIntent intent,
                      HostProcessSpawnReceipt spawn_receipt,
                      LinuxAllocationCgroup cgroup,
                      LinuxStoppedChild child) noexcept;

  HostProcessLaunchIntent intent_;
  HostProcessSpawnReceipt spawn_receipt_;
  // Destruction is reverse declaration order: child is killed/reaped before
  // the retained cgroup descriptor is closed.
  LinuxAllocationCgroup cgroup_;
  LinuxStoppedChild child_;
};

class LinuxProcessAuthority final {
 public:
  LinuxProcessAuthority(SQLiteHostLedger& ledger, AuthorityClock& clock,
                        LinuxCgroupAuthority& cgroups,
                        LinuxStoppedLauncherKernel& launcher);

  // Returns only after both intent and stopped-child identity are durable.
  // The returned gate remains closed until the caller has durably copied the
  // spawn receipt into its journal and calls release_to_exec().
  [[nodiscard]] LinuxPreparedLaunch prepare(
      const ResolvedLaunch& resolved, const ResourceBundleGrant& grant);

 private:
  SQLiteHostLedger& ledger_;
  AuthorityClock& clock_;
  LinuxCgroupAuthority& cgroups_;
  LinuxStoppedLauncherKernel& launcher_;
};

}  // namespace trainvm
