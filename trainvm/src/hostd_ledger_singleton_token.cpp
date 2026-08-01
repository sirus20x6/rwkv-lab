#include "trainvm/hostd_ledger_singleton_token.hpp"

#include <utility>

namespace trainvm {

HostdLedgerSingletonToken::HostdLedgerSingletonToken(
    std::shared_ptr<HostLedgerFilesystemAuthority> authority)
    : authority_(std::move(authority)) {
  if (!authority_)
    throw HostLedgerAuthorityError(
        "hostd singleton token requires ledger authority");
}

bool HostdLedgerSingletonToken::attest_held() const {
  try {
    (void)authority_->attest_after_open();
    (void)authority_->validate_auxiliary_files();
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace trainvm
