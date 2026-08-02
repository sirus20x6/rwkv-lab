#include "trainvm/lifecycle_admission.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/rwkv_lab_worker_contract.hpp"

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

constexpr std::string_view kFingerprint =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";

const std::vector<AdapterProfile>& registered_profiles() {
  static const std::vector<AdapterProfile> value =
      rwkv_lab_worker_contract(std::string(kFingerprint))
          .adapter_registry.profiles;
  return value;
}

constexpr LifecycleControlVerb kVerbs[] = {
    LifecycleControlVerb::cancel,
    LifecycleControlVerb::pause_keep_resources,
    LifecycleControlVerb::pause_release_resources,
    LifecycleControlVerb::resume,
    LifecycleControlVerb::checkpoint_now,
};

// The declaration, restated independently of the gate. If the gate ever
// answers from something other than the declared capability, these disagree.
bool declared(const OperationLifecycleCapabilities& lifecycle,
              LifecycleControlVerb verb, bool checkpoint_first) {
  switch (verb) {
    case LifecycleControlVerb::cancel:
      return lifecycle.graceful_stop;
    case LifecycleControlVerb::checkpoint_now:
      return lifecycle.checkpoint_now;
    case LifecycleControlVerb::pause_keep_resources:
      return lifecycle.pause_keep_resources &&
             (!checkpoint_first || lifecycle.checkpoint_now);
    case LifecycleControlVerb::pause_release_resources:
      return lifecycle.pause_release_resources &&
             (!checkpoint_first || lifecycle.checkpoint_now);
    case LifecycleControlVerb::resume:
      return lifecycle.pause_keep_resources;
  }
  return false;
}

// Every declared safe point, for every registered adapter, in both the
// checkpoint-first and no-checkpoint form. A verb the operation does not
// declare must be refused with a stable code; one it declares must be
// admitted. There is no third answer.
void every_adapter_admits_exactly_what_it_declares() {
  const auto& profiles = registered_profiles();
  require(profiles.size() == 17U,
          "the contract registers the complete seventeen-adapter catalog");
  std::size_t admitted = 0U;
  std::size_t refused = 0U;
  for (const AdapterProfile& profile : profiles) {
    for (const LifecycleControlVerb verb : kVerbs) {
      for (const bool checkpoint_first : {false, true}) {
        const auto outcome = admit_lifecycle_control(profile.lifecycle, verb,
                                                     checkpoint_first);
        const bool expected =
            declared(profile.lifecycle, verb, checkpoint_first);
        require(outcome.has_value() != expected,
                std::string("adapter ") + profile.key.adapter + " verb " +
                    std::string(lifecycle_control_verb_name(verb)) +
                    " admission disagrees with its declaration");
        if (outcome) {
          require(!outcome->code.empty() && !outcome->message.empty(),
                  "a refusal must carry a stable code and message");
          ++refused;
        } else {
          ++admitted;
        }
      }
    }
  }
  require(admitted > 0U && refused > 0U,
          "the matrix must exercise both admission and refusal");
}

// Terminal-checkpoint adapters never overclaim resumability. Graceful stop is
// admitted only when that individual trainer has an implemented signal path.
void terminal_checkpoint_adapter_never_overclaims() {
  const auto& profiles = registered_profiles();
  std::set<std::string> terminal_adapters;
  for (const AdapterProfile& profile : profiles) {
    if (profile.lifecycle.resume_grade != ResumeGrade::terminal_checkpoint) {
      continue;
    }
    terminal_adapters.insert(profile.key.adapter);
    require(!resume_preserves_trajectory(profile.lifecycle.resume_grade) &&
                !resume_from_checkpoint_supported(profile.lifecycle.resume_grade),
            "a terminal-checkpoint operation may not claim a resumable "
            "checkpoint or a preserved trajectory");
    for (const bool checkpoint_first : {false, true}) {
      require(
          admit_lifecycle_control(profile.lifecycle,
                                  LifecycleControlVerb::cancel,
                                  checkpoint_first)
                  .has_value() != profile.lifecycle.graceful_stop,
          "terminal checkpoint cancellation must match its declaration");
      for (const LifecycleControlVerb verb :
           {LifecycleControlVerb::pause_keep_resources,
            LifecycleControlVerb::pause_release_resources,
            LifecycleControlVerb::resume,
            LifecycleControlVerb::checkpoint_now}) {
        require(
            admit_lifecycle_control(profile.lifecycle, verb, checkpoint_first)
                .has_value(),
            "a terminal-checkpoint operation refuses pause, resume, and "
            "checkpoint-now");
      }
    }
  }
  require(terminal_adapters ==
              std::set<std::string>{"rwkv-lab.rwkv-rlvr",
                                    "rwkv-lab.rwkv-scratch"},
          "only the two honest terminal-checkpoint trainers use this grade");
}

// The resumable adapters declare compatible, not exact. They may pause,
// release, and checkpoint, but nothing may present them as trajectory
// preserving.
void compatible_adapters_do_not_claim_exact_resume() {
  std::size_t compatible = 0U;
  for (const AdapterProfile& profile : registered_profiles()) {
    if (profile.lifecycle.resume_grade != ResumeGrade::compatible) continue;
    ++compatible;
    require(resume_from_checkpoint_supported(profile.lifecycle.resume_grade),
            "a compatible operation resumes from its checkpoint");
    require(!resume_preserves_trajectory(profile.lifecycle.resume_grade),
            "a compatible operation must not claim a preserved trajectory");
    for (const LifecycleControlVerb verb :
         {LifecycleControlVerb::pause_keep_resources,
          LifecycleControlVerb::pause_release_resources,
          LifecycleControlVerb::checkpoint_now,
          LifecycleControlVerb::resume}) {
      require(!admit_lifecycle_control(profile.lifecycle, verb, true),
              "a compatible operation admits its declared controls with a "
              "checkpoint-first pause");
    }
  }
  require(compatible == 14U,
          "fourteen registered adapters resume from a compatible checkpoint");
}

// A resource-releasing pause hands the accelerator to someone else, so the
// replacement worker can only come back through a durable checkpoint. The
// registry must refuse any profile that claims otherwise, whichever field is
// made incoherent.
void incoherent_lifecycle_declarations_are_refused_by_the_registry() {
  const AdapterProfile base = registered_profiles().front();

  auto with = [&base](auto&& mutate) {
    AdapterProfile profile = base;
    mutate(profile.lifecycle);
    return std::vector<AdapterProfile>{std::move(profile)};
  };

  require_throws(
      [&] {
        (void)AdapterRegistry(with([](OperationLifecycleCapabilities& value) {
          value.checkpoint_now = false;
        }));
      },
      "pause_release_resources without checkpoint_now must be refused");
  require_throws(
      [&] {
        (void)AdapterRegistry(with([](OperationLifecycleCapabilities& value) {
          value.resume_grade = ResumeGrade::restart_only;
        }));
      },
      "pause_release_resources without a resumable grade must be refused");
  require_throws(
      [&] {
        (void)AdapterRegistry(with([](OperationLifecycleCapabilities& value) {
          value.stateful = false;
        }));
      },
      "a stateless operation cannot declare pause, checkpoint, or resume");
  require_throws(
      [&] {
        (void)AdapterRegistry(with([](OperationLifecycleCapabilities& value) {
          value.resume_grade = ResumeGrade::exact;
          value.checkpoint_now = false;
          value.pause_release_resources = false;
        }));
      },
      "an exact-resume operation must support checkpoint_now");
  require_throws(
      [&] {
        (void)AdapterRegistry(with([](OperationLifecycleCapabilities& value) {
          value.resume_grade = ResumeGrade::terminal_checkpoint;
          value.pause_release_resources = false;
        }));
      },
      "a terminal-checkpoint operation cannot support checkpoint_now");

  // The unmodified profile must still be accepted, so the cases above fail
  // for the reason under test and not because the fixture drifted.
  (void)AdapterRegistry({base});
}

// The gate answers from the declaration alone. A profile that declares a
// capability is admitted no matter which adapter, operation, or component it
// belongs to, and one that does not is refused for every identity.
void admission_depends_only_on_declared_capabilities() {
  OperationLifecycleCapabilities permissive{
      .stateful = true,
      .graceful_stop = true,
      .checkpoint_now = true,
      .pause_keep_resources = true,
      .pause_release_resources = true,
      .compile = false,
      .warmup = false,
      .qualify = false,
      .profile = false,
      .resume_grade = ResumeGrade::exact,
  };
  for (const LifecycleControlVerb verb : kVerbs) {
    for (const bool checkpoint_first : {false, true}) {
      require(!admit_lifecycle_control(permissive, verb, checkpoint_first),
              "a fully declared operation admits every control");
    }
  }
  require(resume_preserves_trajectory(permissive.resume_grade),
          "only an exact grade preserves the trajectory");

  const OperationLifecycleCapabilities silent{};
  for (const LifecycleControlVerb verb : kVerbs) {
    for (const bool checkpoint_first : {false, true}) {
      require(admit_lifecycle_control(silent, verb, checkpoint_first)
                  .has_value(),
              "an operation that declares nothing admits nothing");
    }
  }

  // Removing exactly one capability must refuse exactly the controls that
  // depend on it, and no others.
  OperationLifecycleCapabilities without_checkpoint = permissive;
  without_checkpoint.checkpoint_now = false;
  require(!admit_lifecycle_control(without_checkpoint,
                                   LifecycleControlVerb::pause_keep_resources,
                                   false) &&
              admit_lifecycle_control(without_checkpoint,
                                      LifecycleControlVerb::pause_keep_resources,
                                      true)
                  .has_value() &&
              admit_lifecycle_control(without_checkpoint,
                                      LifecycleControlVerb::checkpoint_now,
                                      false)
                  .has_value() &&
              !admit_lifecycle_control(without_checkpoint,
                                       LifecycleControlVerb::cancel, false),
          "losing checkpoint_now refuses only the checkpoint-dependent "
          "controls");

  OperationLifecycleCapabilities without_release = permissive;
  without_release.pause_release_resources = false;
  require(admit_lifecycle_control(
              without_release,
              LifecycleControlVerb::pause_release_resources, true)
                  .has_value() &&
              !admit_lifecycle_control(
                  without_release,
                  LifecycleControlVerb::pause_keep_resources, true),
          "losing pause_release_resources leaves resource-retaining pause "
          "available");
}

// Refusal codes are operator-visible contract, not free text.
void refusal_codes_are_stable_per_control_family() {
  const OperationLifecycleCapabilities silent{};
  const std::map<LifecycleControlVerb, std::string> expected{
      {LifecycleControlVerb::cancel, "cancel.unsupported_by_operation"},
      {LifecycleControlVerb::checkpoint_now,
       "checkpoint.unsupported_by_operation"},
      {LifecycleControlVerb::pause_keep_resources,
       "lifecycle.unsupported_by_operation"},
      {LifecycleControlVerb::pause_release_resources,
       "lifecycle.unsupported_by_operation"},
      {LifecycleControlVerb::resume, "lifecycle.unsupported_by_operation"},
  };
  for (const auto& [verb, code] : expected) {
    const auto refused = admit_lifecycle_control(silent, verb, false);
    require(refused && refused->code == code,
            "refusal code is stable for " +
                std::string(lifecycle_control_verb_name(verb)));
  }
}

}  // namespace

int main() {
  try {
    every_adapter_admits_exactly_what_it_declares();
    terminal_checkpoint_adapter_never_overclaims();
    compatible_adapters_do_not_claim_exact_resume();
    incoherent_lifecycle_declarations_are_refused_by_the_registry();
    admission_depends_only_on_declared_capabilities();
    refusal_codes_are_stable_per_control_family();
    std::cout << "lifecycle equivalence tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "lifecycle equivalence test failure: " << error.what() << '\n';
    return 1;
  }
}
