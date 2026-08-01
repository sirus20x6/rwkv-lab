#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "trainvm/hostd_session_challenge.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdJournalFenceAttestorApiVersion =
    "trainvm.hostd-journal-fence-attestor/v1";

class HostdJournalFenceAttestorError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Requested controller bootstrap identity. Construction registers it into the
// durable Journal controller authority and accepts only the next monotonic
// generation with a never-before-used controller ID.
struct HostdJournalFenceAttestorConfig final {
  std::string api_version{
      std::string(kHostdJournalFenceAttestorApiVersion)};
  std::string broker_epoch;
  HostdJournalControllerFence controller;

  bool operator==(const HostdJournalFenceAttestorConfig&) const = default;
};

// Production read-only attestor rooted in an already-open Journal that retained
// exact filesystem and host authority. Construction performs the one explicit
// durable controller-generation registration. attest() is read-only: it never
// opens a caller-provided path, advances a fence, or turns serialized evidence
// into authority. Journal and the time source must outlive this object.
class HostdJournalFenceAttestor final : public IHostdJournalFenceAttestor {
 public:
  HostdJournalFenceAttestor(
      Journal& journal, HostdJournalFenceAttestorConfig config,
      std::shared_ptr<IHostdSessionChallengeTimeSource> time_source);

  [[nodiscard]] const HostdSessionChallengeClaim& claim() const noexcept;
  [[nodiscard]] HostdJournalFenceEvidence
  attest(const HostdJournalFenceQuery& query) override;

 private:
  Journal& journal_;
  const HostdJournalFenceAttestorConfig config_;
  const std::shared_ptr<IHostdSessionChallengeTimeSource> time_source_;
  HostdSessionChallengeClaim claim_;
};

}  // namespace trainvm
