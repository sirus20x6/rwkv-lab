#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "trainvm/hostd.hpp"
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
// durable Journal controller authority and accepts only the next generation
// with a never-before-used controller ID in the exact concurrency-key scope.
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

struct HostdDynamicJournalFenceAttestorConfig final {
  std::string api_version{
      std::string(kHostdJournalFenceAttestorApiVersion)};
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;

  bool operator==(const HostdDynamicJournalFenceAttestorConfig&) const =
      default;
};

class IHostdJournalChallengeBoundary {
 public:
  virtual ~IHostdJournalChallengeBoundary() = default;
  [[nodiscard]] virtual JournalAuthoritySnapshot authority() = 0;
  virtual void require_current(const JournalControllerFence& controller) = 0;
  [[nodiscard]] virtual JournalLogicalFenceSnapshot logical_fence(
      const HostdSessionAttribution& attribution,
      const AuthorityTimeSample& now) = 0;
};

class HostdJournalChallengeBoundary final
    : public IHostdJournalChallengeBoundary {
 public:
  explicit HostdJournalChallengeBoundary(Journal& journal);
  [[nodiscard]] JournalAuthoritySnapshot authority() override;
  void require_current(const JournalControllerFence& controller) override;
  [[nodiscard]] JournalLogicalFenceSnapshot logical_fence(
      const HostdSessionAttribution& attribution,
      const AuthorityTimeSample& now) override;

 private:
  Journal& journal_;
};

// Read-only daemon-side verifier for all current controller scopes in one
// already pinned TrainVM journal. Unlike HostdJournalFenceAttestor, this class
// never registers a controller and is not fixed to one claim at construction.
class HostdDynamicJournalFenceAttestor final
    : public IHostdJournalFenceAttestor {
 public:
  HostdDynamicJournalFenceAttestor(
      std::shared_ptr<IHostdJournalChallengeBoundary> journal,
      HostdDynamicJournalFenceAttestorConfig config,
      std::shared_ptr<IHostdSessionChallengeTimeSource> time_source);
  HostdDynamicJournalFenceAttestor(
      Journal& journal, HostdDynamicJournalFenceAttestorConfig config,
      std::shared_ptr<IHostdSessionChallengeTimeSource> time_source);

  [[nodiscard]] HostdJournalFenceEvidence attest(
      const HostdJournalFenceQuery& query) override;

 private:
  std::shared_ptr<IHostdJournalChallengeBoundary> journal_;
  HostdDynamicJournalFenceAttestorConfig config_;
  std::shared_ptr<IHostdSessionChallengeTimeSource> time_source_;
};

}  // namespace trainvm
