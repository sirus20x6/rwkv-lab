#pragma once

#include <memory>

#include "trainvm/authority_time.hpp"
#include "trainvm/hostd.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

class IHostdJournalLogicalFenceBoundary {
 public:
  virtual ~IHostdJournalLogicalFenceBoundary() = default;
  [[nodiscard]] virtual JournalLogicalFenceSnapshot snapshot(
      const HostdSessionAttribution& attribution,
      const AuthorityTimeSample& now) = 0;
};

class HostdJournalLogicalFenceBoundary final
    : public IHostdJournalLogicalFenceBoundary {
 public:
  explicit HostdJournalLogicalFenceBoundary(Journal& journal);
  [[nodiscard]] JournalLogicalFenceSnapshot snapshot(
      const HostdSessionAttribution& attribution,
      const AuthorityTimeSample& now) override;

 private:
  Journal& journal_;
};

// Fresh grant-time logical authority rooted in an already pinned Journal.
// Serialized request claims select no path and mint no authority: the exact
// attribution is used only as a key into the retained journal boundary.
class JournalHostdLogicalFenceEvidenceSource final
    : public IHostdLogicalFenceEvidenceSource {
 public:
  JournalHostdLogicalFenceEvidenceSource(
      std::shared_ptr<IHostdJournalLogicalFenceBoundary> journal,
      AuthorityClock& clock);
  JournalHostdLogicalFenceEvidenceSource(Journal& journal,
                                         AuthorityClock& clock);

  [[nodiscard]] HostdLogicalFenceEvidence attest(
      const HostdSessionAttribution& attribution) override;

 private:
  std::shared_ptr<IHostdJournalLogicalFenceBoundary> journal_;
  AuthorityClock& clock_;
};

}  // namespace trainvm
