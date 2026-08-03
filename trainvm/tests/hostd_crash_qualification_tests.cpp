#include "trainvm/hostd_crash_qualification.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const HostdCrashQualificationError&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern = "/tmp/trainvm-crash-qualification-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = ::mkdtemp(writable.data());
    if (created == nullptr)
      throw std::runtime_error("mkdtemp failed");
    path_ = created;
    require(::chmod(path_.c_str(), 0700) == 0,
            "qualification fixture directory is protected");
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void every_declared_point_has_exactly_one_executor() {
  const auto points = declared_hostd_crash_points();
  require(!points.empty(), "the crash contract declares at least one point");
  auto sorted = points;
  std::ranges::sort(sorted, {}, [](HostdCrashPoint point) {
    return static_cast<int>(point);
  });
  require(std::ranges::adjacent_find(sorted) == sorted.end(),
          "the crash contract declares each point exactly once");
  for (const HostdCrashPoint point : points)
    (void)hostd_crash_point_executor(point);
  const auto privileged = std::ranges::count_if(points, [](HostdCrashPoint p) {
    return hostd_crash_point_executor(p) ==
           HostdCrashExecutor::privileged_launch;
  });
  require(privileged > 0,
          "the contract keeps privileged launch windows visible");
}

HostdCrashQualificationReceipt sealed(HostdCrashQualificationReceipt receipt) {
  receipt.declared_points = receipt.cases.size();
  receipt.qualified_points = 0U;
  receipt.unqualified_points = 0U;
  receipt.blocking_points.clear();
  for (const HostdCrashCaseReceipt& value : receipt.cases) {
    if (value.status == HostdCrashCaseStatus::qualified) {
      ++receipt.qualified_points;
    } else {
      ++receipt.unqualified_points;
      receipt.blocking_points.push_back(value.crash_point);
    }
  }
  receipt.gate_open = receipt.blocking_points.empty();
  receipt.receipt_digest = hostd_crash_qualification_receipt_digest(receipt);
  return receipt;
}

HostdCrashQualificationReceipt all_unqualified_receipt() {
  HostdCrashQualificationReceipt receipt;
  receipt.qualification_binary_digest =
      "sha256:" + std::string(64U, 'b');
  for (const HostdCrashPoint point : declared_hostd_crash_points()) {
    receipt.cases.push_back(
        {.crash_point = point,
         .executor = hostd_crash_point_executor(point),
         .status = HostdCrashCaseStatus::unqualified,
         .unqualified_reason =
             HostdCrashUnqualifiedReason::privilege_unavailable,
         .crash_delivered = false,
         .crashed_pid = 0,
         .detail = "not executed",
         .invariants = {},
         .evidence = {}});
  }
  return sealed(std::move(receipt));
}

void receipt_validation_rejects_incomplete_or_overclaiming_evidence() {
  const auto baseline = all_unqualified_receipt();
  validate_hostd_crash_qualification_receipt(baseline);
  require(!baseline.gate_open,
          "an entirely unqualified receipt keeps the gate closed");

  auto unidentified_binary = baseline;
  unidentified_binary.qualification_binary_digest.clear();
  require_throws(
      [&] {
        validate_hostd_crash_qualification_receipt(
            sealed(unidentified_binary));
      },
      "a crash receipt without exact qualification binary identity must be rejected");

  auto missing = baseline;
  missing.cases.pop_back();
  require_throws([&] { validate_hostd_crash_qualification_receipt(sealed(missing)); },
                 "a receipt that omits a declared point must be rejected");

  auto reordered = baseline;
  std::swap(reordered.cases.front(), reordered.cases.back());
  require_throws(
      [&] { validate_hostd_crash_qualification_receipt(sealed(reordered)); },
      "a receipt whose cases leave declared order must be rejected");

  auto overclaiming = baseline;
  overclaiming.cases.front().invariants.push_back(
      HostdCrashInvariant::no_double_launch);
  require_throws(
      [&] { validate_hostd_crash_qualification_receipt(sealed(overclaiming)); },
      "an unqualified case cannot claim a proven invariant");

  auto silent = baseline;
  silent.cases.front().status = HostdCrashCaseStatus::qualified;
  silent.cases.front().unqualified_reason = HostdCrashUnqualifiedReason::none;
  require_throws(
      [&] { validate_hostd_crash_qualification_receipt(sealed(silent)); },
      "a qualified case must name the invariants it proved");

  auto forged_gate = baseline;
  forged_gate.gate_open = true;
  require_throws(
      [&] { validate_hostd_crash_qualification_receipt(forged_gate); },
      "the gate cannot be opened while blocking points remain");

  auto forged_digest = baseline;
  forged_digest.receipt_digest = "sha256:" + std::string(64U, '0');
  require_throws(
      [&] { validate_hostd_crash_qualification_receipt(forged_digest); },
      "the receipt digest must bind its content");

  auto forged_executor = baseline;
  forged_executor.cases.front().executor =
      HostdCrashExecutor::privileged_launch ==
              forged_executor.cases.front().executor
          ? HostdCrashExecutor::durable_ledger
          : HostdCrashExecutor::privileged_launch;
  require_throws(
      [&] {
        validate_hostd_crash_qualification_receipt(sealed(forged_executor));
      },
      "a case cannot report an undeclared executor");
}

void receipt_json_round_trips_every_declared_point() {
  const auto receipt = all_unqualified_receipt();
  const auto document = hostd_crash_qualification_receipt_json(receipt);
  require(document.at("api_version") == kHostdCrashQualificationApiVersion,
          "the receipt document carries its api version");
  require(document.at("cases").size() == declared_hostd_crash_points().size(),
          "the receipt document covers every declared point");
  require(document.at("gate_open") == false,
          "the receipt document reports its closed gate");
  require(document.at("cases").front().at("status") == "unqualified",
          "case status is reflected as a stable name");
}

void host_probe_reports_the_current_authority_grade() {
  TemporaryDirectory workspace;
  HostdCrashQualificationConfig config;
  config.workspace = workspace.path();
  const auto host = probe_hostd_crash_qualification_host(config);
  require(host.effective_uid == static_cast<std::uint32_t>(::geteuid()),
          "the host probe reports the live effective uid");
  require(host.root_authority == (::geteuid() == 0U),
          "the host probe reports root authority honestly");
  require(!host.kernel_release.empty(),
          "the host probe reports its kernel release");
  if (host.cgroup_delegation)
    require(!host.cgroup_root_unified_path.empty(),
            "a delegated cgroup subtree reports its unified path");
}

// The destructive matrix itself. It is opt-in because it kills processes and
// writes a disposable host ledger; CI enables it through the environment.
void destructive_matrix_gate_is_honest() {
  if (::getenv("TRAINVM_HOSTD_CRASH_QUALIFICATION") == nullptr) {
    std::cout << "hostd crash qualification: destructive matrix skipped "
                 "(set TRAINVM_HOSTD_CRASH_QUALIFICATION=1 to run)\n";
    return;
  }
  TemporaryDirectory workspace;
  HostdCrashQualificationConfig config;
  config.workspace = workspace.path();
  const auto receipt = qualify_hostd_crash_recovery(config);
  validate_hostd_crash_qualification_receipt(receipt);
  require(receipt.cases.size() == declared_hostd_crash_points().size(),
          "the destructive matrix covers every declared point");
  for (const HostdCrashCaseReceipt& value : receipt.cases) {
    if (value.status != HostdCrashCaseStatus::qualified) {
      require(value.unqualified_reason ==
                      HostdCrashUnqualifiedReason::cgroup_delegation_unavailable ||
                  value.unqualified_reason ==
                      HostdCrashUnqualifiedReason::privilege_unavailable,
              "a crash window failed for a reason other than unavailable "
              "host authority: " + value.detail);
      continue;
    }
    if (value.executor == HostdCrashExecutor::durable_ledger ||
        value.executor == HostdCrashExecutor::real_process ||
        value.executor == HostdCrashExecutor::privileged_launch)
      require(value.crash_delivered && value.crashed_pid > 0,
              "a qualified crash window must record a real process death");
  }
  require(receipt.gate_open ==
              (receipt.host.root_authority &&
               receipt.host.cgroup_delegation && receipt.findings.empty()),
          "the deployment gate opens exactly when the required host authority "
          "and every destructive invariant are present");
  std::cout << "hostd crash qualification: "
            << receipt.qualified_points << " qualified, "
            << receipt.unqualified_points << " unqualified\n";
}

}  // namespace

int main() {
  try {
    every_declared_point_has_exactly_one_executor();
    receipt_validation_rejects_incomplete_or_overclaiming_evidence();
    receipt_json_round_trips_every_declared_point();
    host_probe_reports_the_current_authority_grade();
    destructive_matrix_gate_is_honest();
    std::cout << "hostd crash qualification tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd crash qualification test failure: " << error.what()
              << '\n';
    return 1;
  }
}
