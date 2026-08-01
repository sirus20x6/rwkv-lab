#include "trainvm/hostd_ledger_singleton_token.hpp"

#include <unistd.h>

#include <sys/stat.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    const std::string source = "/tmp/trainvm-hostd-token-XXXXXX";
    std::copy(source.begin(), source.end(), pattern.begin());
    char *created = ::mkdtemp(pattern.data());
    if (created == nullptr)
      throw std::runtime_error("mkdtemp failed");
    path_ = created;
    if (::chmod(path_.c_str(), 0700) != 0)
      throw std::runtime_error("chmod failed");
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

HostLedgerAuthorityConfig config(const std::filesystem::path &path) {
  return {.api_version = std::string(kHostLedgerAuthorityApiVersion),
          .ledger_path = path,
          .expected_owner_uid = ::geteuid(),
          .expected_owner_gid = ::getegid(),
          .enforcement_grade = HostLedgerEnforcementGrade::cooperative_test};
}

void token_retains_and_reattests_the_exact_ledger_authority() {
  TemporaryDirectory temporary;
  const auto ledger = temporary.path() / "ledger.sqlite3";
  auto authority = std::make_shared<HostLedgerFilesystemAuthority>(
      HostLedgerFilesystemAuthority::acquire(config(ledger)));
  auto token = std::make_shared<HostdLedgerSingletonToken>(authority);
  authority.reset();

  require(token->attest_held(), "live token reattests its retained authority");
  bool second_rejected = false;
  try {
    auto second = HostLedgerFilesystemAuthority::acquire(config(ledger));
    (void)second;
  } catch (const HostLedgerAuthorityError &) {
    second_rejected = true;
  }
  require(second_rejected,
          "token lifetime retains the host-global nonblocking flock");

  token.reset();
  auto reacquired = HostLedgerFilesystemAuthority::acquire(config(ledger));
  require(reacquired.attest_after_open().database_file.inode != 0U,
          "authority is reacquirable only after the token releases it");
}

void null_authority_is_rejected() {
  bool rejected = false;
  try {
    HostdLedgerSingletonToken token(nullptr);
    (void)token;
  } catch (const HostLedgerAuthorityError &) {
    rejected = true;
  }
  require(rejected, "a token cannot manufacture singleton evidence");
}

} // namespace

int main() {
  try {
    token_retains_and_reattests_the_exact_ledger_authority();
    null_authority_is_rejected();
    std::cout << "hostd ledger singleton token tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd ledger singleton token test failure: " << error.what()
              << '\n';
    return 1;
  }
}
