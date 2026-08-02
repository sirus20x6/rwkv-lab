#pragma once

#include <memory>

#include "trainvm/sqlite_filesystem_authority.hpp"
#include "trainvm/hostd_transport.hpp"

namespace trainvm {

// Bridges the durable, host-global ledger flock into the socket authority.
// The token retains the same filesystem authority object used by
// SQLiteHostLedger; it never creates a second lock or trusts the socket path as
// singleton evidence.
class HostdLedgerSingletonToken final : public IHostdSingletonToken {
public:
  explicit HostdLedgerSingletonToken(
      std::shared_ptr<SqliteFilesystemAuthority> authority);

  [[nodiscard]] bool attest_held() const override;

private:
  std::shared_ptr<SqliteFilesystemAuthority> authority_;
};

} // namespace trainvm
