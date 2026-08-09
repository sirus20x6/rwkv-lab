#include "trainvm/service.hpp"

#include "trainvm/lifecycle_admission.hpp"

#include "trainvm/controller.hpp"
#include "trainvm/document.hpp"
#include "trainvm/eval_examples_contract.hpp"
#include "trainvm/external_profiler_artifact.hpp"
#include "trainvm/final_evaluation.hpp"
#include "trainvm/reflection_json.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/security/server_credentials.h>

namespace trainvm {

AuthorityLock::AuthorityLock(
    const std::filesystem::path& journal_path,
    SqliteAuthorityEnforcementGrade enforcement_grade) {
  const auto absolute_journal =
      std::filesystem::absolute(journal_path).lexically_normal();
  const std::string filename = absolute_journal.filename().string();
  if (filename.empty() || filename == "." || filename == ".." ||
      filename.find('/') != std::string::npos) {
    throw std::runtime_error("authority journal requires a plain filename");
  }
  const auto close_all = [&] {
    if (kernel_namespace_descriptor_ >= 0) {
      (void)::close(kernel_namespace_descriptor_);
      kernel_namespace_descriptor_ = -1;
    }
    filesystem_authority_.reset();
  };
  const auto fail = [&](std::string message) -> void {
    close_all();
    throw std::runtime_error(std::move(message));
  };

  // The abstract Unix-socket name is a kernel-resident lock for the configured
  // absolute namespace. Unlike the co-located sidecar, it cannot be renamed
  // together with a replaced journal and therefore prevents two cooperating
  // authorities from splitting across old and replacement inodes.
  kernel_namespace_descriptor_ =
      ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (kernel_namespace_descriptor_ < 0) {
    fail("could not create journal namespace lock socket: " +
         std::string(std::strerror(errno)));
  }
  const std::string kernel_name =
      "trainvm-journal-" + sha256_hex(absolute_journal.string());
  sockaddr_un kernel_address {};
  if (kernel_name.size() + 1U > sizeof(kernel_address.sun_path)) {
    fail("journal namespace lock identity exceeds the Linux abstract socket limit");
  }
  kernel_address.sun_family = AF_UNIX;
  std::memcpy(kernel_address.sun_path + 1, kernel_name.data(), kernel_name.size());
  const socklen_t kernel_address_size = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + 1U + kernel_name.size());
  if (::bind(kernel_namespace_descriptor_,
             reinterpret_cast<const sockaddr*>(&kernel_address),
             kernel_address_size) != 0) {
    fail("another TrainVM authority owns journal namespace " +
         absolute_journal.string() + ": " + std::strerror(errno));
  }

  try {
    filesystem_authority_ = std::make_shared<SqliteFilesystemAuthority>(
        SqliteFilesystemAuthority::acquire({
            .api_version = std::string(kSqliteAuthorityApiVersion),
            .ledger_path = absolute_journal,
            .expected_owner_uid = ::geteuid(),
            .expected_owner_gid = ::getegid(),
            .enforcement_grade = enforcement_grade,
        }));
  } catch (const std::exception& exception) {
    fail(exception.what());
  }
  const auto attestation = filesystem_authority_->attest_before_open();
  (void)filesystem_authority_->validate_auxiliary_files();
  const std::string lock_name = filename + ".authority.lock";
  journal_identity_ = {
      .directory_path = absolute_journal.parent_path().string(),
      .journal_name = filename,
      .authority_name = lock_name,
      .directory_device = attestation.authority_directory.device,
      .directory_inode = attestation.authority_directory.inode,
      .device = attestation.database_file.device,
      .inode = attestation.database_file.inode,
      .authority_device = attestation.lock_file.device,
      .authority_inode = attestation.lock_file.inode,
      .owner_uid = attestation.authority_directory.owner_uid,
  };
  journal_path_ = absolute_journal;
}

AuthorityLock::~AuthorityLock() {
  if (kernel_namespace_descriptor_ >= 0) {
    ::close(kernel_namespace_descriptor_);
  }
}

const std::filesystem::path& AuthorityLock::journal_path() const noexcept {
  return journal_path_;
}

const JournalFileIdentity& AuthorityLock::journal_identity() const noexcept {
  return journal_identity_;
}

const std::shared_ptr<SqliteFilesystemAuthority>&
AuthorityLock::filesystem_authority() const noexcept {
  return filesystem_authority_;
}

namespace {

constexpr std::size_t kMaximumCommandBytes = 64U * 1024U;
constexpr std::size_t kMaximumSubmissionBytes = 2U * 1024U * 1024U;
constexpr std::size_t kMaximumWorkerMessageBytes = 64U * 1024U;

class SignalMaskGuard {
 public:
  SignalMaskGuard() {
    ::sigemptyset(&signals_);
    ::sigaddset(&signals_, SIGINT);
    ::sigaddset(&signals_, SIGTERM);
    const int result = ::pthread_sigmask(SIG_BLOCK, &signals_, &previous_);
    if (result != 0) {
      throw std::runtime_error("could not block authority shutdown signals: " +
                               std::string(std::strerror(result)));
    }
    active_ = true;
  }

  ~SignalMaskGuard() {
    if (active_) {
      (void)::pthread_sigmask(SIG_SETMASK, &previous_, nullptr);
    }
  }

  SignalMaskGuard(const SignalMaskGuard&) = delete;
  SignalMaskGuard& operator=(const SignalMaskGuard&) = delete;

  [[nodiscard]] const sigset_t* signals() const { return &signals_; }

 private:
  sigset_t signals_{};
  sigset_t previous_{};
  bool active_{};
};

class UmaskGuard {
 public:
  explicit UmaskGuard(mode_t mask) : previous_(::umask(mask)) {}
  ~UmaskGuard() { (void)::umask(previous_); }

  UmaskGuard(const UmaskGuard&) = delete;
  UmaskGuard& operator=(const UmaskGuard&) = delete;

 private:
  mode_t previous_{};
};

class SocketCleanupGuard {
 public:
  explicit SocketCleanupGuard(std::filesystem::path path) : path_(std::move(path)) {}
  ~SocketCleanupGuard() {
    if (!preserved_path_.empty()) {
      struct stat current {};
      if (::lstat(path_.c_str(), &current) != 0 && errno == ENOENT) {
        (void)::rename(preserved_path_.c_str(), path_.c_str());
      }
    }
    if (!claimed_) return;
    struct stat status {};
    if (::lstat(path_.c_str(), &status) == 0 && S_ISSOCK(status.st_mode) &&
        status.st_dev == device_ && status.st_ino == inode_) {
      (void)::unlink(path_.c_str());
    }
  }

  void claim() {
    struct stat status {};
    if (::lstat(path_.c_str(), &status) != 0 || !S_ISSOCK(status.st_mode)) {
      throw std::runtime_error("authority did not create its Unix socket " + path_.string());
    }
    device_ = status.st_dev;
    inode_ = status.st_ino;
    claimed_ = true;
  }

  void preserve_replacement() {
    struct stat status {};
    if (::lstat(path_.c_str(), &status) != 0) {
      if (errno == ENOENT) return;
      throw std::runtime_error("could not inspect authority socket before shutdown: " +
                               std::string(std::strerror(errno)));
    }
    if (claimed_ && status.st_dev == device_ && status.st_ino == inode_) {
      return;
    }
    preserved_path_ = path_.string() + ".preserved." +
                      std::to_string(static_cast<long long>(::getpid())) + "." +
                      std::to_string(static_cast<unsigned long long>(status.st_ino));
    struct stat existing {};
    if (::lstat(preserved_path_.c_str(), &existing) == 0 || errno != ENOENT) {
      throw std::runtime_error("could not reserve replacement-socket preservation path");
    }
    if (::rename(path_.c_str(), preserved_path_.c_str()) != 0) {
      preserved_path_.clear();
      throw std::runtime_error("could not preserve replacement socket before shutdown: " +
                               std::string(std::strerror(errno)));
    }
  }

  void restore_replacement() {
    if (preserved_path_.empty()) return;
    struct stat current {};
    if (::lstat(path_.c_str(), &current) == 0 || errno != ENOENT) {
      throw std::runtime_error("authority socket path was unexpectedly occupied during shutdown");
    }
    if (::rename(preserved_path_.c_str(), path_.c_str()) != 0) {
      throw std::runtime_error("could not restore replacement socket after shutdown: " +
                               std::string(std::strerror(errno)));
    }
    preserved_path_.clear();
  }

  SocketCleanupGuard(const SocketCleanupGuard&) = delete;
  SocketCleanupGuard& operator=(const SocketCleanupGuard&) = delete;

 private:
  std::filesystem::path path_;
  dev_t device_{};
  ino_t inode_{};
  bool claimed_{};
  std::filesystem::path preserved_path_;
};

class SocketAuthorityLock {
 public:
  explicit SocketAuthorityLock(const std::filesystem::path& socket_path) {
    const auto parent = std::filesystem::weakly_canonical(socket_path.parent_path());
    const auto path = parent / (socket_path.filename().string() + ".authority.lock");
    descriptor_ = ::open(path.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, S_IRUSR | S_IWUSR);
    if (descriptor_ < 0) {
      throw std::runtime_error("could not open socket authority lock " + path.string() + ": " +
                               std::strerror(errno));
    }
    if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
      const std::string message = std::strerror(errno);
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("another TrainVM authority owns socket " +
                               socket_path.string() + ": " + message);
    }
    if (::fchmod(descriptor_, S_IRUSR | S_IWUSR) != 0) {
      const std::string message = std::strerror(errno);
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("could not restrict socket authority lock " + path.string() +
                               ": " + message);
    }
  }

  ~SocketAuthorityLock() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  SocketAuthorityLock(const SocketAuthorityLock&) = delete;
  SocketAuthorityLock& operator=(const SocketAuthorityLock&) = delete;

 private:
  int descriptor_{-1};
};

bool cancelled(const grpc::ServerContext* context) {
  return context != nullptr && context->IsCancelled();
}

grpc::Status cancellation_status() {
  return {grpc::StatusCode::CANCELLED, "command was cancelled before persistence"};
}

v1::Diagnostic::Severity wire_severity(Diagnostic::Severity severity) {
  switch (severity) {
    case Diagnostic::Severity::info:
      return v1::Diagnostic::SEVERITY_INFO;
    case Diagnostic::Severity::warning:
      return v1::Diagnostic::SEVERITY_WARNING;
    case Diagnostic::Severity::error:
      return v1::Diagnostic::SEVERITY_ERROR;
  }
  return v1::Diagnostic::SEVERITY_UNSPECIFIED;
}

v1::AuthorRunStage wire_author_run_stage(AuthorRunStage stage) {
  switch (stage) {
    case AuthorRunStage::validating:
      return v1::AUTHOR_RUN_STAGE_VALIDATING;
    case AuthorRunStage::resolving:
      return v1::AUTHOR_RUN_STAGE_RESOLVING;
    case AuthorRunStage::locking_inputs:
      return v1::AUTHOR_RUN_STAGE_LOCKING_INPUTS;
    case AuthorRunStage::preflight:
      return v1::AUTHOR_RUN_STAGE_PREFLIGHT;
    case AuthorRunStage::provisioning:
      return v1::AUTHOR_RUN_STAGE_PROVISIONING;
    case AuthorRunStage::submitting:
      return v1::AUTHOR_RUN_STAGE_SUBMITTING;
    case AuthorRunStage::complete:
      return v1::AUTHOR_RUN_STAGE_COMPLETE;
    case AuthorRunStage::failed:
      return v1::AUTHOR_RUN_STAGE_FAILED;
  }
  return v1::AUTHOR_RUN_STAGE_FAILED;
}

void add_diagnostic(v1::RunCommandResponse& response, const Diagnostic& diagnostic) {
  auto* output = response.add_diagnostics();
  output->set_severity(wire_severity(diagnostic.severity));
  output->set_code(diagnostic.code);
  output->set_document_path(diagnostic.path);
  output->set_message(diagnostic.message);
}

void add_diagnostic(v1::SubmitExperimentResponse& response, const Diagnostic& diagnostic) {
  auto* output = response.add_diagnostics();
  output->set_severity(wire_severity(diagnostic.severity));
  output->set_code(diagnostic.code);
  output->set_document_path(diagnostic.path);
  output->set_message(diagnostic.message);
}

void add_diagnostic(v1::AuthorRunUpdate& response,
                    const TrainingPreflightDiagnostic& diagnostic) {
  auto* output = response.add_diagnostics();
  output->set_severity(wire_severity(diagnostic.severity));
  output->set_code(diagnostic.code);
  output->set_document_path(diagnostic.path);
  output->set_message(diagnostic.message);
  output->set_help(diagnostic.help);
}

void add_diagnostic(v1::PlanDiffResponse& response,
                    const Diagnostic& diagnostic) {
  auto* output = response.add_diagnostics();
  output->set_severity(wire_severity(diagnostic.severity));
  output->set_code(diagnostic.code);
  output->set_document_path(diagnostic.path);
  output->set_message(diagnostic.message);
}

void fill_stored_diagnostic(const nlohmann::json& diagnostic,
                            v1::Diagnostic& output) {
  if (!diagnostic.is_object()) {
    throw std::runtime_error("stored control diagnostic is not an object");
  }
  const auto severity = diagnostic.find("severity");
  if (severity != diagnostic.end() && severity->is_string()) {
    const auto value = severity->get<std::string>();
    if (value == "info") output.set_severity(v1::Diagnostic::SEVERITY_INFO);
    if (value == "warning")
      output.set_severity(v1::Diagnostic::SEVERITY_WARNING);
    if (value == "error") output.set_severity(v1::Diagnostic::SEVERITY_ERROR);
  }
  const auto string_value = [&](std::string_view key) -> std::string {
    const auto value = diagnostic.find(std::string(key));
    if (value != diagnostic.end() && value->is_string()) {
      return value->get<std::string>();
    }
    return {};
  };
  output.set_code(string_value("code"));
  output.set_document_path(string_value("document_path"));
  if (output.document_path().empty()) {
    output.set_document_path(string_value("path"));
  }
  output.set_message(string_value("message"));
  output.set_help(string_value("help"));
}

void add_stored_diagnostics(v1::RunCommandResponse& response,
                            const nlohmann::json& diagnostics) {
  if (!diagnostics.is_array()) return;
  for (const auto& diagnostic : diagnostics) {
    if (!diagnostic.is_object()) continue;
    fill_stored_diagnostic(diagnostic, *response.add_diagnostics());
  }
}

v1::ApplyPoint wire_apply_point(ApplyPoint point) {
  switch (point) {
    case ApplyPoint::immediate:
      return v1::APPLY_POINT_IMMEDIATE;
    case ApplyPoint::next_microbatch:
      return v1::APPLY_POINT_NEXT_MICROBATCH;
    case ApplyPoint::next_optimizer_step:
      return v1::APPLY_POINT_NEXT_OPTIMIZER_STEP;
    case ApplyPoint::next_eval:
      return v1::APPLY_POINT_NEXT_EVAL;
    case ApplyPoint::next_checkpoint:
      return v1::APPLY_POINT_NEXT_CHECKPOINT;
    case ApplyPoint::restart:
      return v1::APPLY_POINT_RESTART;
  }
  return v1::APPLY_POINT_UNSPECIFIED;
}

v1::ControlType wire_control_type(ControlType type) {
  switch (type) {
    case ControlType::number:
      return v1::CONTROL_TYPE_NUMBER;
    case ControlType::integer:
      return v1::CONTROL_TYPE_INTEGER;
    case ControlType::boolean:
      return v1::CONTROL_TYPE_BOOLEAN;
    case ControlType::string:
      return v1::CONTROL_TYPE_STRING;
    case ControlType::enumeration:
      return v1::CONTROL_TYPE_ENUMERATION;
  }
  return v1::CONTROL_TYPE_UNSPECIFIED;
}

v1::ControlCommandResult::Status wire_command_status(ControlCommandStatus status) {
  switch (status) {
    case ControlCommandStatus::requested:
      return v1::ControlCommandResult::STATUS_REQUESTED;
    case ControlCommandStatus::applied:
      return v1::ControlCommandResult::STATUS_APPLIED;
    case ControlCommandStatus::rejected:
      return v1::ControlCommandResult::STATUS_REJECTED;
    case ControlCommandStatus::restart_required:
      return v1::ControlCommandResult::STATUS_RESTART_REQUIRED;
  }
  return v1::ControlCommandResult::STATUS_UNSPECIFIED;
}

v1::HostdLifecycle wire_hostd_lifecycle(HostdLifecycle lifecycle) {
  switch (lifecycle) {
    case HostdLifecycle::sealed:
      return v1::HOSTD_LIFECYCLE_SEALED;
    case HostdLifecycle::startup_auditing:
      return v1::HOSTD_LIFECYCLE_STARTUP_AUDITING;
    case HostdLifecycle::startup_blocked:
      return v1::HOSTD_LIFECYCLE_STARTUP_BLOCKED;
    case HostdLifecycle::admitting:
      return v1::HOSTD_LIFECYCLE_ADMITTING;
    case HostdLifecycle::poisoned:
      return v1::HOSTD_LIFECYCLE_POISONED;
  }
  return v1::HOSTD_LIFECYCLE_UNSPECIFIED;
}

v1::HostdStartupPhase wire_hostd_startup_phase(HostdStartupPhase phase) {
  switch (phase) {
    case HostdStartupPhase::reconciling:
      return v1::HOSTD_STARTUP_PHASE_RECONCILING;
    case HostdStartupPhase::auditing:
      return v1::HOSTD_STARTUP_PHASE_AUDITING;
    case HostdStartupPhase::admitting:
      return v1::HOSTD_STARTUP_PHASE_ADMITTING;
    case HostdStartupPhase::exhausted:
      return v1::HOSTD_STARTUP_PHASE_EXHAUSTED;
    case HostdStartupPhase::failed:
      return v1::HOSTD_STARTUP_PHASE_FAILED;
  }
  return v1::HOSTD_STARTUP_PHASE_UNSPECIFIED;
}

v1::HostdProcessPhase wire_hostd_process_phase(
    HostdProcessAuthorityPhase phase) {
  switch (phase) {
    case HostdProcessAuthorityPhase::launch_intent:
      return v1::HOSTD_PROCESS_PHASE_LAUNCH_INTENT;
    case HostdProcessAuthorityPhase::spawned:
      return v1::HOSTD_PROCESS_PHASE_SPAWNED;
    case HostdProcessAuthorityPhase::terminal_pending_release:
      return v1::HOSTD_PROCESS_PHASE_TERMINAL_PENDING_RELEASE;
  }
  return v1::HOSTD_PROCESS_PHASE_UNSPECIFIED;
}

v1::HostResourceKind wire_host_resource_kind(HostResourceKind kind) {
  switch (kind) {
    case HostResourceKind::accelerator:
      return v1::HOST_RESOURCE_KIND_ACCELERATOR;
    case HostResourceKind::accelerator_partition:
      return v1::HOST_RESOURCE_KIND_ACCELERATOR_PARTITION;
    case HostResourceKind::host_mutex:
      return v1::HOST_RESOURCE_KIND_HOST_MUTEX;
  }
  return v1::HOST_RESOURCE_KIND_UNSPECIFIED;
}

v1::HostAcceleratorVendor wire_host_accelerator_vendor(
    const std::optional<HostAcceleratorVendor>& vendor) {
  if (!vendor) return v1::HOST_ACCELERATOR_VENDOR_UNSPECIFIED;
  switch (*vendor) {
    case HostAcceleratorVendor::nvidia:
      return v1::HOST_ACCELERATOR_VENDOR_NVIDIA;
    case HostAcceleratorVendor::amd:
      return v1::HOST_ACCELERATOR_VENDOR_AMD;
    case HostAcceleratorVendor::intel:
      return v1::HOST_ACCELERATOR_VENDOR_INTEL;
    case HostAcceleratorVendor::other:
      return v1::HOST_ACCELERATOR_VENDOR_OTHER;
  }
  return v1::HOST_ACCELERATOR_VENDOR_UNSPECIFIED;
}

nlohmann::json assignment_value(const v1::ScalarValue& input) {
  switch (input.value_case()) {
    case v1::ScalarValue::kNumberValue:
      return input.number_value();
    case v1::ScalarValue::kIntegerValue:
      return input.integer_value();
    case v1::ScalarValue::kBooleanValue:
      return input.boolean_value();
    case v1::ScalarValue::kStringValue:
      return input.string_value();
    case v1::ScalarValue::VALUE_NOT_SET:
      throw std::invalid_argument("control assignment has no scalar value");
  }
  throw std::invalid_argument("control assignment has an unsupported scalar value");
}

nlohmann::json assignments_json(const v1::ControlPatchCommand& input) {
  nlohmann::json output = nlohmann::json::object();
  for (const auto& assignment : input.assignments()) {
    if (assignment.key().empty()) {
      throw std::invalid_argument("control assignment key must not be empty");
    }
    if (output.contains(assignment.key())) {
      throw std::invalid_argument("control patch contains a duplicate assignment key");
    }
    output[assignment.key()] = assignment_value(assignment.value());
  }
  return output;
}

void set_wire_scalar(const nlohmann::json& value, v1::ScalarValue& output) {
  if (value.is_number_float()) {
    output.set_number_value(value.get<double>());
  } else if (value.is_number_integer()) {
    output.set_integer_value(value.get<std::int64_t>());
  } else if (value.is_number_unsigned()) {
    const auto unsigned_value = value.get<std::uint64_t>();
    if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::overflow_error("control integer exceeds the wire signed integer range");
    }
    output.set_integer_value(static_cast<std::int64_t>(unsigned_value));
  } else if (value.is_boolean()) {
    output.set_boolean_value(value.get<bool>());
  } else if (value.is_string()) {
    output.set_string_value(value.get<std::string>());
  } else {
    throw std::runtime_error("validated control command contains a non-scalar value");
  }
}

template <typename AddAssignment>
void fill_wire_assignments(const nlohmann::json& values,
                           AddAssignment&& add_assignment) {
  if (!values.is_object()) {
    throw std::runtime_error("stored control values are not an object");
  }
  for (auto iterator = values.begin(); iterator != values.end(); ++iterator) {
    auto* assignment = add_assignment();
    assignment->set_key(iterator.key());
    set_wire_scalar(iterator.value(), *assignment->mutable_value());
  }
}

void fill_control_result(const ControlCommand& command, v1::RunCommandResponse& response) {
  response.set_command_sequence(command.control_revision);
  auto* output = response.mutable_control();
  output->set_command_id(command.command_id);
  output->set_control_revision(command.control_revision);
  output->set_apply_point(wire_apply_point(command.apply_point));
  output->set_requires_pause(command.requires_pause);
  output->set_status(wire_command_status(command.status));
  fill_wire_assignments(command.assignments,
                        [&] { return output->add_assignments(); });
  add_stored_diagnostics(response, command.diagnostics);
}

v1::WorkerCommand worker_control_command(const ControlCommand& command) {
  if (command.status != ControlCommandStatus::requested ||
      command.command_id.empty() || command.control_revision == 0U) {
    throw std::invalid_argument(
        "only a durable requested control can be sent to a worker");
  }
  v1::WorkerCommand output;
  // The stream sequence is assigned by the caller from the durable request
  // event. Control revision remains the version of the effective-value state.
  output.set_command_id(command.command_id);
  auto* controls = output.mutable_controls();
  controls->set_expected_control_revision(command.expected_control_revision);
  controls->set_control_revision(command.control_revision);
  controls->set_apply_point(wire_apply_point(command.apply_point));
  controls->set_requires_pause(command.requires_pause);
  for (auto iterator = command.assignments.begin();
       iterator != command.assignments.end(); ++iterator) {
    auto* assignment = controls->add_assignments();
    assignment->set_key(iterator.key());
    set_wire_scalar(iterator.value(), *assignment->mutable_value());
  }
  return output;
}

v1::WorkerCommand worker_checkpoint_command(const CheckpointCommand& command) {
  if (command.status != CheckpointCommandStatus::requested ||
      command.command_id.empty() || command.controller_sequence == 0U) {
    throw std::invalid_argument(
        "only a durable requested checkpoint can be sent to a worker");
  }
  v1::WorkerCommand output;
  output.set_controller_sequence(command.controller_sequence);
  output.set_command_id(command.command_id);
  output.mutable_checkpoint()->set_reason(command.reason);
  return output;
}

void fill_checkpoint_result(const CheckpointCommand& command,
                            v1::RunCommandResponse& response) {
  response.set_command_sequence(command.controller_sequence);
  auto* output = response.mutable_checkpoint();
  output->set_command_id(command.command_id);
  output->set_controller_sequence(command.controller_sequence);
  output->set_reason(command.reason);
  switch (command.status) {
    case CheckpointCommandStatus::requested:
      output->set_status(v1::CheckpointCommandResult::STATUS_REQUESTED);
      break;
    case CheckpointCommandStatus::applied:
      output->set_status(v1::CheckpointCommandResult::STATUS_APPLIED);
      break;
    case CheckpointCommandStatus::rejected:
      output->set_status(v1::CheckpointCommandResult::STATUS_REJECTED);
      break;
  }
  if (command.optimizer_step) output->set_optimizer_step(*command.optimizer_step);
  output->set_artifact_id(command.artifact_id);
  add_stored_diagnostics(response, command.diagnostics);
}

v1::RunCommandResponse::Disposition replay_disposition(
    const CheckpointCommand& command) {
  switch (command.status) {
    case CheckpointCommandStatus::requested:
      return v1::RunCommandResponse::DISPOSITION_ACCEPTED;
    case CheckpointCommandStatus::applied:
      return v1::RunCommandResponse::DISPOSITION_ALREADY_APPLIED;
    case CheckpointCommandStatus::rejected:
      return v1::RunCommandResponse::DISPOSITION_REJECTED;
  }
  throw std::invalid_argument("invalid checkpoint command status");
}

v1::WorkerCommand worker_lifecycle_command(const LifecycleCommand& command) {
  if (command.status != LifecycleCommandStatus::requested ||
      command.command_id.empty() || command.controller_sequence == 0U) {
    throw std::invalid_argument(
        "only a durable requested lifecycle command can be sent to a worker");
  }
  v1::WorkerCommand output;
  output.set_controller_sequence(command.controller_sequence);
  output.set_command_id(command.command_id);
  switch (command.kind) {
    case LifecycleCommandKind::pause: {
      auto* pause = output.mutable_pause();
      pause->set_checkpoint_first(command.checkpoint_first);
      pause->set_release_resources(command.release_resources);
      break;
    }
    case LifecycleCommandKind::resume:
      output.mutable_resume();
      break;
    case LifecycleCommandKind::cancel: {
      auto* cancel = output.mutable_cancel();
      cancel->set_reason(command.cancel_reason);
      cancel->mutable_graceful_timeout()->set_seconds(
          command.graceful_timeout_ns / 1'000'000'000LL);
      cancel->mutable_graceful_timeout()->set_nanos(
          static_cast<std::int32_t>(command.graceful_timeout_ns %
                                    1'000'000'000LL));
      break;
    }
  }
  return output;
}

void fill_lifecycle_result(const LifecycleCommand& command,
                           v1::RunCommandResponse& response) {
  response.set_command_sequence(command.controller_sequence);
  auto* output = response.mutable_lifecycle();
  output->set_command_id(command.command_id);
  output->set_controller_sequence(command.controller_sequence);
  switch (command.kind) {
    case LifecycleCommandKind::pause:
      output->set_kind(v1::LifecycleCommandResult::KIND_PAUSE);
      break;
    case LifecycleCommandKind::resume:
      output->set_kind(v1::LifecycleCommandResult::KIND_RESUME);
      break;
    case LifecycleCommandKind::cancel:
      output->set_kind(v1::LifecycleCommandResult::KIND_CANCEL);
      break;
  }
  switch (command.status) {
    case LifecycleCommandStatus::requested:
      output->set_status(v1::LifecycleCommandResult::STATUS_REQUESTED);
      break;
    case LifecycleCommandStatus::applied:
      output->set_status(v1::LifecycleCommandResult::STATUS_APPLIED);
      break;
    case LifecycleCommandStatus::rejected:
      output->set_status(v1::LifecycleCommandResult::STATUS_REJECTED);
      break;
  }
  output->set_checkpoint_first(command.checkpoint_first);
  output->set_release_resources(command.release_resources);
  if (command.optimizer_step) output->set_optimizer_step(*command.optimizer_step);
  output->set_artifact_id(command.artifact_id);
  output->set_reason(command.cancel_reason);
  output->mutable_graceful_timeout()->set_seconds(
      command.graceful_timeout_ns / 1'000'000'000LL);
  output->mutable_graceful_timeout()->set_nanos(
      static_cast<std::int32_t>(command.graceful_timeout_ns %
                                1'000'000'000LL));
  add_stored_diagnostics(response, command.diagnostics);
}

v1::RunCommandResponse::Disposition replay_disposition(
    const LifecycleCommand& command) {
  switch (command.status) {
    case LifecycleCommandStatus::requested:
      return v1::RunCommandResponse::DISPOSITION_ACCEPTED;
    case LifecycleCommandStatus::applied:
      return v1::RunCommandResponse::DISPOSITION_ALREADY_APPLIED;
    case LifecycleCommandStatus::rejected:
      return v1::RunCommandResponse::DISPOSITION_REJECTED;
  }
  throw std::invalid_argument("invalid lifecycle command status");
}

v1::RunCommandResponse::Disposition replay_disposition(const ControlCommand& command) {
  switch (command.status) {
    case ControlCommandStatus::requested:
      return v1::RunCommandResponse::DISPOSITION_ACCEPTED;
    case ControlCommandStatus::applied:
      return v1::RunCommandResponse::DISPOSITION_ALREADY_APPLIED;
    case ControlCommandStatus::rejected:
    case ControlCommandStatus::restart_required:
      return v1::RunCommandResponse::DISPOSITION_REJECTED;
  }
  return v1::RunCommandResponse::DISPOSITION_UNSPECIFIED;
}

v1::DesiredState desired_state(std::string_view state) {
  if (state == "queued") return v1::DESIRED_STATE_QUEUED;
  if (state == "running") return v1::DESIRED_STATE_RUNNING;
  if (state == "paused") return v1::DESIRED_STATE_PAUSED;
  if (state == "cancelled") return v1::DESIRED_STATE_CANCELLED;
  if (state == "completed") return v1::DESIRED_STATE_COMPLETED;
  return v1::DESIRED_STATE_UNSPECIFIED;
}

v1::ObservedState observed_state(std::string_view state) {
  if (state == "draft") return v1::OBSERVED_STATE_DRAFT;
  if (state == "validated") return v1::OBSERVED_STATE_VALIDATED;
  if (state == "queued") return v1::OBSERVED_STATE_QUEUED;
  if (state == "acquiring") return v1::OBSERVED_STATE_ACQUIRING;
  if (state == "running") return v1::OBSERVED_STATE_RUNNING;
  if (state == "pausing") return v1::OBSERVED_STATE_PAUSING;
  if (state == "paused") return v1::OBSERVED_STATE_PAUSED;
  if (state == "recovering") return v1::OBSERVED_STATE_RECOVERING;
  if (state == "completing") return v1::OBSERVED_STATE_COMPLETING;
  if (state == "completed") return v1::OBSERVED_STATE_COMPLETED;
  if (state == "cancelling") return v1::OBSERVED_STATE_CANCELLING;
  if (state == "cancelled") return v1::OBSERVED_STATE_CANCELLED;
  if (state == "failing") return v1::OBSERVED_STATE_FAILING;
  if (state == "failed") return v1::OBSERVED_STATE_FAILED;
  if (state == "blocked") return v1::OBSERVED_STATE_BLOCKED;
  return v1::OBSERVED_STATE_UNSPECIFIED;
}

std::string observed_state_name(v1::ObservedState state) {
  switch (state) {
    case v1::OBSERVED_STATE_DRAFT:
      return "draft";
    case v1::OBSERVED_STATE_VALIDATED:
      return "validated";
    case v1::OBSERVED_STATE_QUEUED:
      return "queued";
    case v1::OBSERVED_STATE_ACQUIRING:
      return "acquiring";
    case v1::OBSERVED_STATE_RUNNING:
      return "running";
    case v1::OBSERVED_STATE_PAUSING:
      return "pausing";
    case v1::OBSERVED_STATE_PAUSED:
      return "paused";
    case v1::OBSERVED_STATE_RECOVERING:
      return "recovering";
    case v1::OBSERVED_STATE_COMPLETING:
      return "completing";
    case v1::OBSERVED_STATE_COMPLETED:
      return "completed";
    case v1::OBSERVED_STATE_CANCELLING:
      return "cancelling";
    case v1::OBSERVED_STATE_CANCELLED:
      return "cancelled";
    case v1::OBSERVED_STATE_FAILING:
      return "failing";
    case v1::OBSERVED_STATE_FAILED:
      return "failed";
    case v1::OBSERVED_STATE_BLOCKED:
      return "blocked";
    case v1::OBSERVED_STATE_UNSPECIFIED:
      break;
    default:
      break;
  }
  throw std::invalid_argument("observed-state filter is unspecified");
}

void set_timestamp_ns(std::int64_t nanoseconds,
                      google::protobuf::Timestamp& output) {
  constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
  if (nanoseconds < 0) {
    throw std::runtime_error("durable event contains a negative wall timestamp");
  }
  output.set_seconds(nanoseconds / kNanosecondsPerSecond);
  output.set_nanos(static_cast<std::int32_t>(
      nanoseconds % kNanosecondsPerSecond));
}

void fill_run_summary(const RunProjection& projection, const Journal& journal,
                      v1::RunSummary& output) {
  output.mutable_identity()->set_run_id(projection.run_id);
  output.mutable_identity()->set_revision(projection.run_revision);
  output.mutable_identity()->set_plan_hash(projection.plan_hash);
  output.set_experiment_name(projection.experiment_name);
  output.set_desired_state(desired_state(projection.desired_state));
  output.set_observed_state(observed_state(projection.observed_state));
  output.set_current_node_id(projection.current_node_id);
  output.set_current_attempt_id(projection.current_attempt_id);
  output.set_optimizer_step(projection.optimizer_step);
  output.set_failure_summary(projection.failure_summary);
  const auto requested = journal.latest_control_revision(projection.run_id);
  const auto effective = journal.latest_effective_control_revision(projection.run_id);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  output.set_effective_control_revision(effective);
#pragma GCC diagnostic pop
  output.set_latest_requested_control_revision(requested);
  output.set_latest_effective_control_revision(effective);
  output.set_last_event_sequence(projection.last_event_sequence);
  // Fork provenance is authority-owned history: it is read back from the
  // durable run.created submission rather than carried on the mutable
  // projection, so it cannot drift from what was actually fenced at creation.
  if (const auto created = journal.event(projection.run_id + ":created");
      created && created->payload.contains("submission")) {
    const auto& submission = created->payload.at("submission");
    if (submission.is_object() && submission.contains("forked_from")) {
      const auto& forked = submission.at("forked_from");
      if (forked.is_object() && forked.contains("run_id") &&
          forked.contains("run_revision") && forked.contains("plan_hash") &&
          forked.at("run_id").is_string() &&
          forked.at("run_revision").is_number_unsigned() &&
          forked.at("plan_hash").is_string()) {
        output.set_forked_from_run_id(
            forked.at("run_id").get<std::string>());
        output.set_forked_from_run_revision(
            forked.at("run_revision").get<std::uint64_t>());
        output.set_forked_from_plan_hash(
            forked.at("plan_hash").get<std::string>());
      }
    }
  }
  const auto times = journal.run_wall_time_bounds(projection.run_id);
  if (times) {
    set_timestamp_ns(times->created_wall_time_ns, *output.mutable_created_at());
    set_timestamp_ns(times->updated_wall_time_ns, *output.mutable_updated_at());
  }
  if (projection.last_heartbeat_ns > 0) {
    set_timestamp_ns(projection.last_heartbeat_ns,
                     *output.mutable_last_heartbeat_at());
  }
  if (projection.observed_state == "queued") {
    output.set_wait_reason("waiting for resource admission");
  } else if (projection.observed_state == "acquiring") {
    output.set_wait_reason("waiting for host grant or worker readiness");
  } else if (projection.observed_state == "paused") {
    output.set_wait_reason("run is paused by desired state");
  } else if (projection.observed_state == "blocked") {
    output.set_wait_reason("run requires operator input");
  }
}

void fill_run_summary(const RunProjection& projection, const Journal& journal,
                      v1::RunCommandResponse& response) {
  fill_run_summary(projection, journal, *response.mutable_run());
}

v1::EventEnvelope wire_event(const SequencedEvent& input) {
  v1::EventEnvelope output;
  const Event& event = input.event;
  output.set_journal_sequence(input.journal_sequence);
  output.set_event_id(event.event_id);
  output.set_run_id(event.run_id);
  output.set_run_revision(event.run_revision);
  output.set_plan_revision(event.plan_revision);
  output.set_node_id(event.node_id);
  output.set_attempt_id(event.attempt_id);
  output.set_worker_sequence(event.worker_sequence);
  output.set_event_type(event.event_type);
  output.set_event_version(event.event_version);
  set_timestamp_ns(event.wall_time_ns, *output.mutable_wall_time());
  output.set_monotonic_time_ns(event.monotonic_time_ns);
  if (event.optimizer_step) output.set_optimizer_step(*event.optimizer_step);
  output.set_canonical_json_payload(event.payload.dump());
  return output;
}

std::string run_list_filter_digest(
    std::string_view journal_id,
    const std::set<std::string, std::less<>>& observed_states,
    const std::map<std::string, std::string, std::less<>>& labels) {
  return "sha256:" + sha256_hex(nlohmann::json{
      {"api_version", "trainvm.run-list-filter/v1"},
      {"journal_id", journal_id},
      {"labels", labels},
      {"observed_states", observed_states},
  }.dump());
}

std::string encode_run_page_token(const RunProjection& projection,
                                  std::string_view filter_digest) {
  return nlohmann::json{
      {"api_version", "trainvm.run-page/v1"},
      {"filter_digest", filter_digest},
      {"last_event_sequence", projection.last_event_sequence},
      {"run_id", projection.run_id},
  }.dump();
}

RunProjectionCursor decode_run_page_token(std::string_view encoded,
                                          std::string_view filter_digest) {
  if (encoded.empty() || encoded.size() > 4'096U) {
    throw std::invalid_argument("run page token exceeds its bound");
  }
  const nlohmann::json token = nlohmann::json::parse(encoded);
  static const std::set<std::string> kAllowed{
      "api_version", "filter_digest", "last_event_sequence", "run_id"};
  if (!token.is_object() || token.size() != kAllowed.size()) {
    throw std::invalid_argument("run page token is not canonical");
  }
  for (auto field = token.begin(); field != token.end(); ++field) {
    if (!kAllowed.contains(field.key())) {
      throw std::invalid_argument("run page token has an unknown field");
    }
  }
  if (token.dump() != encoded ||
      token.value("api_version", std::string{}) != "trainvm.run-page/v1" ||
      token.value("filter_digest", std::string{}) != filter_digest ||
      !token.contains("last_event_sequence") ||
      !token.at("last_event_sequence").is_number_unsigned() ||
      !token.contains("run_id") || !token.at("run_id").is_string()) {
    throw std::invalid_argument(
        "run page token is malformed or belongs to another query");
  }
  RunProjectionCursor cursor{
      .last_event_sequence =
          token.at("last_event_sequence").get<std::uint64_t>(),
      .run_id = token.at("run_id").get<std::string>(),
  };
  if (cursor.last_event_sequence == 0U || cursor.run_id.empty() ||
      cursor.run_id.size() > 256U) {
    throw std::invalid_argument("run page token cursor is malformed");
  }
  return cursor;
}

void remove_stale_socket(const std::filesystem::path& path) {
  struct stat status {};
  if (::lstat(path.c_str(), &status) != 0) {
    if (errno == ENOENT) return;
    throw std::runtime_error("could not inspect authority socket " + path.string() + ": " +
                             std::strerror(errno));
  }
  if (!S_ISSOCK(status.st_mode)) {
    throw std::runtime_error("refusing to replace non-socket path " + path.string());
  }
  const std::string encoded = path.string();
  sockaddr_un address{};
  if (encoded.size() >= sizeof(address.sun_path)) {
    throw std::runtime_error("authority socket path exceeds the Unix socket limit");
  }
  const int probe = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (probe < 0) {
    throw std::runtime_error("could not probe authority socket " + path.string() + ": " +
                             std::strerror(errno));
  }
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, encoded.c_str(), encoded.size() + 1U);
  const auto address_size = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + encoded.size() + 1U);
  const int connection = ::connect(probe, reinterpret_cast<sockaddr*>(&address), address_size);
  const int connection_error = errno;
  ::close(probe);
  if (connection == 0 || connection_error == EINPROGRESS || connection_error == EAGAIN) {
    throw std::runtime_error("refusing to replace active authority socket " + path.string());
  }
  if (connection_error != ECONNREFUSED && connection_error != ENOENT) {
    throw std::runtime_error("could not determine whether authority socket is stale: " +
                             std::string(std::strerror(connection_error)));
  }
  if (::unlink(path.c_str()) != 0) {
    throw std::runtime_error("could not remove stale authority socket " + path.string() + ": " +
                             std::strerror(errno));
  }
}

std::vector<std::string> canonical_worker_capabilities(
    const google::protobuf::RepeatedPtrField<std::string>& input) {
  std::vector<std::string> capabilities(input.begin(), input.end());
  if (capabilities.size() > 256U ||
      std::ranges::any_of(capabilities, [](const std::string& capability) {
        return capability.empty() || capability.size() > 256U;
      })) {
    throw std::invalid_argument("worker hello contains invalid capabilities");
  }
  std::ranges::sort(capabilities);
  if (std::ranges::adjacent_find(capabilities) != capabilities.end()) {
    throw std::invalid_argument("worker hello capabilities must be unique");
  }
  return capabilities;
}

WorkerHelloEvidence worker_hello_evidence(const v1::WorkerHello& hello) {
  if (hello.run_id().empty() || hello.node_id().empty() ||
      hello.attempt_id().empty() || hello.launch_nonce().empty() ||
      hello.adapter().empty() || hello.adapter_version().empty() ||
      hello.code_fingerprint().empty() || hello.concurrency_key().empty() ||
      hello.lease_id().empty() || hello.fencing_token() == 0U) {
    throw std::invalid_argument("worker hello identity fields must not be empty");
  }
  return WorkerHelloEvidence{
      .run_id = hello.run_id(),
      .node_id = hello.node_id(),
      .attempt_id = hello.attempt_id(),
      .launch_nonce = hello.launch_nonce(),
      .adapter = hello.adapter(),
      .adapter_version = hello.adapter_version(),
      .code_fingerprint = hello.code_fingerprint(),
      .capabilities = canonical_worker_capabilities(hello.capabilities()),
      .last_acked_controller_sequence =
          hello.last_acked_controller_sequence(),
      .concurrency_key = hello.concurrency_key(),
      .lease_id = hello.lease_id(),
      .fencing_token = hello.fencing_token(),
  };
}

WorkerSessionIdentity worker_session(const WorkerHelloEvidence& hello) {
  return WorkerSessionIdentity{
      .run_id = hello.run_id,
      .node_id = hello.node_id,
      .attempt_id = hello.attempt_id,
      .launch_nonce = hello.launch_nonce,
      .concurrency_key = hello.concurrency_key,
      .lease_id = hello.lease_id,
      .fencing_token = hello.fencing_token,
  };
}

std::int64_t timestamp_ns(const google::protobuf::Timestamp& timestamp) {
  constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
  constexpr std::int64_t kMaximumSeconds =
      std::numeric_limits<std::int64_t>::max() / kNanosecondsPerSecond;
  constexpr std::int32_t kMaximumRemainder = static_cast<std::int32_t>(
      std::numeric_limits<std::int64_t>::max() % kNanosecondsPerSecond);
  if (timestamp.seconds() < 0 || timestamp.nanos() < 0 ||
      timestamp.nanos() >= kNanosecondsPerSecond ||
      timestamp.seconds() > kMaximumSeconds ||
      (timestamp.seconds() == kMaximumSeconds &&
       timestamp.nanos() > kMaximumRemainder)) {
    throw std::invalid_argument("worker event wall timestamp is out of range");
  }
  return timestamp.seconds() * kNanosecondsPerSecond + timestamp.nanos();
}

std::int64_t duration_ns(const google::protobuf::Duration& duration,
                         std::int64_t maximum_seconds) {
  constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
  if (maximum_seconds < 0 || duration.seconds() < 0 || duration.nanos() < 0 ||
      duration.nanos() >= kNanosecondsPerSecond ||
      duration.seconds() > maximum_seconds) {
    throw std::invalid_argument("graceful timeout is out of range");
  }
  return duration.seconds() * kNanosecondsPerSecond + duration.nanos();
}

std::string artifact_kind_name(v1::ArtifactKind kind) {
  switch (kind) {
    case v1::ARTIFACT_KIND_PATH:
      return "path";
    case v1::ARTIFACT_KIND_CHECKPOINT:
      return "checkpoint";
    case v1::ARTIFACT_KIND_DATASET:
      return "dataset";
    case v1::ARTIFACT_KIND_IMAGE_GALLERY:
      return "image_gallery";
    case v1::ARTIFACT_KIND_METRICS:
      return "metrics";
    case v1::ARTIFACT_KIND_REPORT:
      return "report";
    case v1::ARTIFACT_KIND_OPAQUE:
      return "opaque";
    case v1::ARTIFACT_KIND_EVAL_EXAMPLES:
      return "eval_examples";
    case v1::ARTIFACT_KIND_UNSPECIFIED:
      break;
    default:
      break;
  }
  throw std::invalid_argument("artifact manifest kind is unspecified");
}

std::string diagnostic_severity_name(v1::Diagnostic::Severity severity) {
  switch (severity) {
    case v1::Diagnostic::SEVERITY_INFO:
      return "info";
    case v1::Diagnostic::SEVERITY_WARNING:
      return "warning";
    case v1::Diagnostic::SEVERITY_ERROR:
      return "error";
    case v1::Diagnostic::SEVERITY_UNSPECIFIED:
      break;
    default:
      break;
  }
  throw std::invalid_argument("control acknowledgement diagnostic severity is unspecified");
}

nlohmann::json acknowledgement_assignments(
    const v1::ControlPatchAcknowledgement& acknowledgement) {
  nlohmann::json output = nlohmann::json::object();
  for (const auto& assignment : acknowledgement.effective_values()) {
    if (assignment.key().empty() || output.contains(assignment.key())) {
      throw std::invalid_argument(
          "control acknowledgement has an empty or duplicate assignment key");
    }
    if (assignment.value().value_case() == v1::ScalarValue::kNumberValue &&
        !std::isfinite(assignment.value().number_value())) {
      throw std::invalid_argument(
          "control acknowledgement assignment must be finite");
    }
    output[assignment.key()] = assignment_value(assignment.value());
  }
  return output;
}

template <typename Acknowledgement>
nlohmann::json acknowledgement_diagnostics(
    const Acknowledgement& acknowledgement) {
  nlohmann::json output = nlohmann::json::array();
  for (const auto& diagnostic : acknowledgement.diagnostics()) {
    if (diagnostic.code().empty() || diagnostic.message().empty()) {
      throw std::invalid_argument(
          "control acknowledgement diagnostic requires code and message");
    }
    output.push_back({{"severity", diagnostic_severity_name(diagnostic.severity())},
                      {"code", diagnostic.code()},
                      {"document_path", diagnostic.document_path()},
                      {"message", diagnostic.message()},
                      {"help", diagnostic.help()}});
  }
  return output;
}

grpc::Status worker_failure(const std::exception& exception) {
  if (dynamic_cast<const std::invalid_argument*>(&exception) != nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  }
  if (dynamic_cast<const OperationPreconditionError*>(&exception) != nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, exception.what()};
  }
  if (dynamic_cast<const AdapterResolutionError*>(&exception) != nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, exception.what()};
  }
  if (dynamic_cast<const std::logic_error*>(&exception) != nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, exception.what()};
  }
  return {grpc::StatusCode::DATA_LOSS, exception.what()};
}

std::set<std::string, std::less<>> referenced_artifact_names(
    const CompiledPlan& plan, const std::string& node_id) {
  const Spec& spec = plan.experiment.spec;
  const Node& node = spec.workflow.nodes.at(node_id);
  std::set<std::string, std::less<>> names;
  for (const auto& [input_name, binding] : node.invoke.inputs) {
    (void)input_name;
    if (binding.artifact) {
      names.insert(*binding.artifact);
    } else if (binding.node_output) {
      const Node& producer =
          spec.workflow.nodes.at(binding.node_output->node);
      names.insert(producer.publishes->at(binding.node_output->name));
    }
  }
  return names;
}

std::map<std::string, nlohmann::json, std::less<>> invocation_artifacts(
    const Journal& journal, const CompiledPlan& plan,
    const std::string& run_id, const std::string& node_id) {
  const auto required = referenced_artifact_names(plan, node_id);
  std::map<std::string, nlohmann::json, std::less<>> result;
  if (required.empty()) return result;
  for (const Event& event : journal.events_for_run(run_id)) {
    if (event.event_type != "artifact.published" ||
        !event.payload.is_object() ||
        !event.payload.value("complete", false))
      continue;
    const std::string logical_name =
        event.payload.value("logical_name", std::string{});
    if (!required.contains(logical_name)) continue;
    if (event.payload.value("producer_node_id", std::string{}) !=
            event.node_id ||
        event.payload.value("producer_attempt_id", std::string{}) !=
            event.attempt_id) {
      throw std::runtime_error(
          "artifact publication disagrees with its event envelope");
    }
    result[logical_name] = event.payload;
  }
  return result;
}

nlohmann::json invocation_resume_checkpoint(
    const Journal& journal, const std::string& run_id,
    const std::string& node_id, const std::string& attempt_id) {
  std::optional<Event> restart;
  std::optional<Event> checkpoint;
  const std::vector<Event> events = journal.events_for_run(run_id);
  for (const Event& event : events) {
    if (event.event_type == "node.attempt_restarted" &&
        event.node_id == node_id && event.attempt_id == attempt_id) {
      if (restart) {
        throw std::runtime_error(
            "worker attempt has ambiguous restart authority");
      }
      restart = event;
    }
  }
  if (!restart) return nullptr;
  const std::string resume_command_id =
      restart->payload.value("cause_command_id", std::string{});
  const std::string pause_command_id =
      restart->payload.value("pause_command_id", std::string{});
  const std::string artifact_id =
      restart->payload.value("checkpoint_artifact_id", std::string{});
  const auto resume = journal.lifecycle_command(resume_command_id);
  const auto pause = journal.lifecycle_command(pause_command_id);
  if (resume_command_id.empty() || pause_command_id.empty() ||
      artifact_id.empty() || !resume || !pause ||
      resume->kind != LifecycleCommandKind::resume ||
      resume->status != LifecycleCommandStatus::applied ||
      !resume->acknowledgement ||
      resume->acknowledgement->worker_sequence != 0U ||
      pause->kind != LifecycleCommandKind::pause ||
      !pause->checkpoint_first || !pause->release_resources ||
      pause->status != LifecycleCommandStatus::applied ||
      !pause->optimizer_step || pause->artifact_id != artifact_id ||
      resume->optimizer_step != pause->optimizer_step ||
      resume->artifact_id != artifact_id ||
      resume->node_id != node_id || pause->node_id != node_id ||
      resume->attempt_id != pause->attempt_id ||
      resume->attempt_id == attempt_id) {
    throw std::runtime_error(
        "worker attempt has invalid resume checkpoint lineage");
  }
  for (const Event& event : events) {
    if (event.event_type != "artifact.published" ||
        event.payload.value("artifact_id", std::string{}) != artifact_id) {
      continue;
    }
    if (checkpoint) {
      throw std::runtime_error(
          "worker resume checkpoint identity is ambiguous");
    }
    checkpoint = event;
  }
  if (!checkpoint || checkpoint->node_id != node_id ||
      checkpoint->attempt_id != resume->attempt_id ||
      checkpoint->optimizer_step != pause->optimizer_step ||
      checkpoint->payload.value("kind", std::string{}) != "checkpoint" ||
      !checkpoint->payload.value("complete", false) ||
      checkpoint->payload.value("producer_node_id", std::string{}) != node_id ||
      checkpoint->payload.value("producer_attempt_id", std::string{}) !=
          resume->attempt_id) {
    throw std::runtime_error(
        "worker resume checkpoint artifact is missing or invalid");
  }
  return {{"api_version", "trainvm.resume-checkpoint/v1"},
          {"checkpoint", checkpoint->payload},
          {"optimizer_step", *pause->optimizer_step},
          {"pause_command_id", pause_command_id},
          {"resume_command_id", resume_command_id}};
}

std::uint64_t invocation_attempt_baseline_optimizer_step(
    const WorkerInvocationSpec& invocation) {
  if (invocation.resume.is_null()) return 0U;
  if (!invocation.resume.is_object() ||
      invocation.resume.value("api_version", std::string{}) !=
          "trainvm.resume-checkpoint/v1" ||
      !invocation.resume.contains("optimizer_step") ||
      !invocation.resume.at("optimizer_step").is_number_unsigned()) {
    throw std::runtime_error(
        "worker invocation has malformed attempt-baseline authority");
  }
  return invocation.resume.at("optimizer_step").get<std::uint64_t>();
}

// The journal name of a typed execution phase, or nullopt when the value is
// PHASE_UNSPECIFIED or an unknown enumerator from a newer peer. Callers that
// require a phase reject on nullopt; the heartbeat path separates the two
// cases itself, because "outside any phase" is legal there and an unknown
// enumerator is not.
[[nodiscard]] std::optional<std::string_view> execution_phase_name(
    v1::WorkerExecutionPhaseRequest::Phase phase) {
  switch (phase) {
    case v1::WorkerExecutionPhaseRequest::PHASE_COMPILE:
      return "compile";
    case v1::WorkerExecutionPhaseRequest::PHASE_WARMUP:
      return "warmup";
    case v1::WorkerExecutionPhaseRequest::PHASE_UNSPECIFIED:
    default:
      return std::nullopt;
  }
}

// The immutable request for one phase on this attempt, or nullptr when the
// authority never issued it. A duplicate is reported as absent so a caller
// cannot pick whichever copy authorizes it; the receipt path separately
// distinguishes duplicates for its own DATA_LOSS diagnosis.
[[nodiscard]] const v1::WorkerExecutionPhaseRequest* find_phase_request(
    const v1::WorkerWelcome& welcome,
    v1::WorkerExecutionPhaseRequest::Phase phase) {
  const v1::WorkerExecutionPhaseRequest* found = nullptr;
  for (const auto& candidate : welcome.execution_phase_requests()) {
    if (candidate.phase() != phase) continue;
    if (found != nullptr) return nullptr;
    found = &candidate;
  }
  return found;
}

void populate_invocation(v1::WorkerWelcome& welcome,
                         const WorkerInvocationSpec& invocation) {
  const std::string canonical =
      worker_invocation_canonical_json(invocation);
  welcome.set_canonical_invocation_json(canonical);
  welcome.set_invocation_digest(invocation.invocation_digest);

  if (!invocation.execution.is_object()) return;
  const auto append_phase = [&](std::string_view name,
                                v1::WorkerExecutionPhaseRequest::Phase phase) {
    const auto declaration = invocation.execution.find(name);
    if (declaration == invocation.execution.end() ||
        !declaration->is_object()) {
      return;
    }
    const bool enabled = declaration->value("enabled", false);
    nlohmann::json digest_body{{"api_version",
                                "trainvm.worker-execution-phase-request/v1"},
                               {"enabled", enabled},
                               {"invocation_digest",
                                invocation.invocation_digest},
                               {"phase", name}};
    auto* request = welcome.add_execution_phase_requests();
    request->set_phase(phase);
    request->set_enabled(enabled);
    if (name == "warmup" && declaration->contains("steps")) {
      const auto steps = declaration->at("steps").get<std::uint64_t>();
      request->set_steps(steps);
      digest_body["steps"] = steps;
    }
    request->set_request_digest("sha256:" + sha256_hex(digest_body.dump()));
  };
  append_phase("compile",
               v1::WorkerExecutionPhaseRequest::PHASE_COMPILE);
  append_phase("warmup",
               v1::WorkerExecutionPhaseRequest::PHASE_WARMUP);
}

// Why the controller may not accept hostd's passive inventory as canonical GPU
// authoring evidence, or nullopt when it may. Every condition is admission
// evidence — the caller must reject on any of them — but they are enumerated
// individually so a rejection names the one that fired. Collapsed into a single
// disjunction, this check reported fifteen distinct causes with one sentence,
// and a rejection could not be acted on without a debugger.
[[nodiscard]] std::optional<std::string>
passive_inventory_rejection(const HostdCoordinatorStatus &coordinator,
                            const HostdAuthorityStatus &authority,
                            const HostIdentity &expected_host,
                            std::int64_t now_ns, std::uint64_t maximum_age_ns) {
  const auto observed = static_cast<std::uint64_t>(now_ns);
  if (coordinator.host_id != expected_host.host_id)
    return "hostd names host " + coordinator.host_id + "; this controller is "
           "bound to " + expected_host.host_id;
  if (coordinator.boot_id != expected_host.boot_id)
    return "hostd names boot " + coordinator.boot_id + "; this controller is "
           "bound to " + expected_host.boot_id + " (the host rebooted under a "
           "running controller)";
  if (!authority.resource_inventory_observed)
    return "hostd has not yet observed its resource inventory";
  if (authority.resource_inventory_observation_age_ns > maximum_age_ns)
    return "hostd resource inventory is " +
           std::to_string(authority.resource_inventory_observation_age_ns /
                          1'000'000ULL) +
           "ms old; the admission bound is " +
           std::to_string(maximum_age_ns / 1'000'000ULL) + "ms";
  if (authority.current_inventory_digest.empty())
    return "hostd reported no current inventory digest";
  if (authority.current_inventory_receipt_digest.empty())
    return "hostd reported no current inventory receipt digest";
  if (authority.passive_memory_host_id != coordinator.host_id)
    return "hostd passive memory names host " +
           authority.passive_memory_host_id + "; its own status names " +
           coordinator.host_id;
  if (authority.passive_memory_boot_id != coordinator.boot_id)
    return "hostd passive memory names boot " +
           authority.passive_memory_boot_id + "; its own status names " +
           coordinator.boot_id;
  if (authority.passive_memory_inventory_digest !=
      authority.current_inventory_digest)
    return "hostd passive memory was observed against inventory " +
           authority.passive_memory_inventory_digest + "; the current "
           "inventory is " + authority.current_inventory_digest;
  if (authority.passive_memory_inventory_receipt_digest !=
      authority.current_inventory_receipt_digest)
    return "hostd passive memory carries inventory receipt " +
           authority.passive_memory_inventory_receipt_digest +
           "; the current receipt is " +
           authority.current_inventory_receipt_digest;
  if (authority.passive_memory_observed_monotonic_ns == 0U)
    return "hostd has not yet observed passive accelerator memory";
  if (authority.passive_memory_observed_monotonic_ns > observed)
    return "hostd observed passive memory " +
           std::to_string((authority.passive_memory_observed_monotonic_ns -
                           observed) / 1'000'000ULL) +
           "ms after this controller received its reply; both read "
           "CLOCK_MONOTONIC on this host, so they no longer share a clock "
           "origin";
  if (observed - authority.passive_memory_observed_monotonic_ns >
      maximum_age_ns)
    return "hostd passive memory is " +
           std::to_string((observed -
                           authority.passive_memory_observed_monotonic_ns) /
                          1'000'000ULL) +
           "ms old; the admission bound is " +
           std::to_string(maximum_age_ns / 1'000'000ULL) + "ms";
  if (authority.passive_memory_observation_digest.empty())
    return "hostd reported no passive memory observation digest";
  if (authority.passive_accelerator_memory.empty())
    return "hostd observed passive memory but reported no accelerators";
  if (authority.passive_accelerator_memory_truncated)
    return "hostd truncated its passive accelerator memory report";
  if (authority.passive_accelerator_memory_count !=
      authority.passive_accelerator_memory.size())
    return "hostd observed " +
           std::to_string(authority.passive_accelerator_memory_count) +
           " accelerators but carried " +
           std::to_string(authority.passive_accelerator_memory.size());
  return std::nullopt;
}

// Turns a configured receipt root into the publisher configuration the
// deployment holds, and refuses at construction if the root is not actually
// provisioned for one.
//
// The path shape is checked here for the same reason `recipe_registry_path_`
// is -- an absolute, lexically normal, bounded path is the only kind this
// authority accepts from a command line. The rest of the check is delegated by
// *constructing a publisher and throwing it away*: `LinuxCacheEvidencePublisher`
// already refuses a root that does not exist, is not owned by the effective
// uid, is group- or other-writable, lacks an owner-writable `runtime/` or
// `qualification/` subdirectory, or whose subdirectories live on another
// device. Re-stating those rules here would give a deployment two answers to
// "is this root usable", and the publisher's answer is the one that decides at
// publication time. Paying it at startup means a misprovisioned root fails the
// daemon rather than the first worker message that arrives hours later.
[[nodiscard]] std::optional<LinuxCacheEvidenceConfig> attested_cache_evidence(
    std::optional<std::filesystem::path> receipt_root) {
  if (!receipt_root) return std::nullopt;
  if (!receipt_root->is_absolute() ||
      receipt_root->lexically_normal() != *receipt_root ||
      receipt_root->native().size() > 4'096U) {
    throw std::invalid_argument(
        "cache evidence receipt root must be canonical, absolute, and bounded");
  }
  LinuxCacheEvidenceConfig config{
      .receipt_root = std::move(*receipt_root),
      // Not configurable: the publisher refuses any uid but the effective one,
      // so a second answer here could only ever be a wrong one.
      .authority_uid = ::geteuid(),
      .maximum_receipt_bytes = 1U << 20U,
  };
  (void)LinuxCacheEvidencePublisher(config);
  return config;
}

}  // namespace

TrainVMService::TrainVMService(
    const std::filesystem::path& journal_path,
    AdapterRegistry adapter_registry,
    HostLaunchRegistry host_launch_registry,
    std::function<AuthorityTimeSample()> authority_clock,
    TrainingComponentRegistry training_components,
    std::optional<HostdClientConfiguration> hostd_configuration,
    std::string controller_target,
    ICacheQualificationEvidenceResolver* cache_qualification,
    SqliteAuthorityEnforcementGrade filesystem_enforcement_grade,
    std::shared_ptr<ITrainingPreflightEvidenceProvider> preflight_evidence,
    std::filesystem::path recipe_registry_path,
    IWorkerRuntimeEvidenceAuthority* worker_runtime_evidence,
    std::optional<std::filesystem::path> cache_evidence_root)
    : TrainVMService(journal_path, std::move(adapter_registry),
                     std::move(host_launch_registry),
                     HostLaunchResolver::local_host_identity(),
                     std::move(authority_clock),
                     HostGrantEnforcement::required,
                     std::move(training_components), {}, {}, {},
                     cache_qualification, filesystem_enforcement_grade,
                     std::move(preflight_evidence),
                     std::move(recipe_registry_path),
                     worker_runtime_evidence,
                     std::move(cache_evidence_root)) {
  if (hostd_configuration) {
    configure_hostd(*hostd_configuration, std::move(controller_target));
  }
  // The gRPC authority remains useful in launch-disabled mode for compilation,
  // immutable submission, replay, and dashboard reads. The controller target
  // is a worker-launch bootstrap input, so it is intentionally ignored unless
  // hostd supplies both resource and process authority.
}

TrainVMService::TrainVMService(
    const std::filesystem::path& journal_path,
    AdapterRegistry adapter_registry,
    HostLaunchRegistry host_launch_registry,
    HostIdentity authority_host,
    std::function<AuthorityTimeSample()> authority_clock,
    HostGrantEnforcement host_grant_enforcement,
    TrainingComponentRegistry training_components,
    std::shared_ptr<IHostGrantClient> host_grant_client,
    std::shared_ptr<IHostProcessClient> host_process_client,
    std::string controller_target,
    ICacheQualificationEvidenceResolver* cache_qualification,
    SqliteAuthorityEnforcementGrade filesystem_enforcement_grade,
    std::shared_ptr<ITrainingPreflightEvidenceProvider> preflight_evidence,
    std::filesystem::path recipe_registry_path,
    IWorkerRuntimeEvidenceAuthority* worker_runtime_evidence,
    std::optional<std::filesystem::path> cache_evidence_root)
    : authority_lock_(std::make_unique<AuthorityLock>(
          journal_path, filesystem_enforcement_grade)),
      journal_(authority_lock_->journal_path(),
               authority_lock_->journal_identity(),
               host_grant_enforcement, authority_host,
               authority_lock_->filesystem_authority()),
      authority_clock_(
          authority_clock
              ? std::make_shared<AuthorityClock>(std::move(authority_clock))
              : std::make_shared<AuthorityClock>()),
      lease_renewals_(journal_, authority_clock_),
      adapter_registry_(std::move(adapter_registry)),
      host_launch_registry_(std::move(host_launch_registry)),
      training_components_(std::move(training_components)),
      preflight_evidence_(std::move(preflight_evidence)),
      recipe_registry_path_(std::move(recipe_registry_path)),
      authority_host_(std::move(authority_host)),
      host_launch_resolver_(host_launch_registry_, authority_host_),
      host_grant_client_(std::move(host_grant_client)),
      host_grant_saga_(
          host_grant_client_
              ? std::make_unique<HostGrantSagaReconciler>(
                    journal_, *host_grant_client_)
              : nullptr),
      host_process_client_(std::move(host_process_client)),
      controller_target_(std::move(controller_target)),
      host_process_saga_(
          host_process_client_
              ? std::make_unique<HostProcessSagaReconciler>(
                    journal_, *host_process_client_)
              : nullptr),
      reconciler_(journal_, adapter_registry_, training_components_,
                  command_mutex_,
                  [this] { return authority_now(); }, cache_qualification),
      worker_runtime_evidence_(worker_runtime_evidence),
      cache_evidence_(attested_cache_evidence(std::move(cache_evidence_root))) {
  if (!recipe_registry_path_.is_absolute() ||
      recipe_registry_path_.lexically_normal() != recipe_registry_path_ ||
      recipe_registry_path_.native().size() > 4'096U) {
    throw std::invalid_argument(
        "recipe registry path must be canonical, absolute, and bounded");
  }
  if (!preflight_evidence_) {
    PassiveAcceleratorSnapshotSource hostd_accelerators =
        [this](const CompiledPlan &) {
          if (!hostd_status_client_ || hostd_status_timeout_ns_ <= 0)
            throw std::runtime_error(
                "canonical GPU authoring requires configured hostd passive "
                "inventory evidence");
          const std::int64_t now = hostd_monotonic_now_ns();
          if (now <= 0 || hostd_status_timeout_ns_ >
              std::numeric_limits<std::int64_t>::max() - now)
            throw std::runtime_error(
                "hostd passive inventory deadline overflowed");
          std::uint64_t correlation = hostd_status_correlation_.fetch_add(
              1U, std::memory_order_relaxed);
          if (correlation == 0U)
            correlation = hostd_status_correlation_.fetch_add(
                1U, std::memory_order_relaxed);
          const HostdStatusReply reply = hostd_request_status(
              *hostd_status_client_, correlation,
              now + hostd_status_timeout_ns_);
          if (reply.kind != HostdStatusReplyKind::status || !reply.status ||
              !reply.authority_status)
            throw std::runtime_error(
                "hostd omitted its passive authority inventory status");
          const auto &coordinator = *reply.status;
          const auto &authority = *reply.authority_status;
          // Age is measured from the reply, not from the deadline computed
          // before the request. hostd refreshes a stale inventory while serving
          // the status call and stamps the observation once the capture returns,
          // so an observation made during this very request carries a timestamp
          // later than `now` — on this host NVML takes about a second, and the
          // observation landed ~972ms "in the future". Both processes read
          // CLOCK_MONOTONIC on the same machine and cannot actually disagree
          // about time; the comparison instant was simply taken too early.
          const std::int64_t observed_at = hostd_monotonic_now_ns();
          // hostd serves a cached inventory whenever it is younger than its own
          // refresh interval (inventory_refresh_interval_ns in
          // hostd_daemon_runtime.cpp), so that interval must stay below this
          // bound or a well-behaved authority is refused for most of every
          // cycle. Raising this without lowering that one reintroduces the
          // mismatch in the other direction.
          constexpr std::uint64_t maximum_age_ns = 5'000'000'000ULL;
          if (const std::optional<std::string> rejection =
                  passive_inventory_rejection(coordinator, authority,
                                              authority_host_, observed_at,
                                              maximum_age_ns))
            throw std::runtime_error(
                "hostd passive inventory is not admissible: " + *rejection);
          std::vector<PassiveAcceleratorMemoryEvidence> result;
          result.reserve(authority.passive_accelerator_memory.size());
          for (const auto &memory :
               authority.passive_accelerator_memory) {
            // Every observed device is carried through with its disposition.
            // Filtering on `audited_eligible` here decided a policy question —
            // may this plan use this device — with an observation that cannot
            // express the answer, and did it before the plan was in scope: a
            // GPU driving a display is observed occupied forever, so on this
            // host preflight was handed an empty accelerator list and reported
            // insufficient VRAM. Selection is preflight's to make, against the
            // access mode the plan actually declares.
            AcceleratorVendor vendor{};
            switch (memory.vendor) {
            case HostAcceleratorVendor::nvidia:
              vendor = AcceleratorVendor::nvidia;
              break;
            case HostAcceleratorVendor::amd:
              vendor = AcceleratorVendor::amd;
              break;
            case HostAcceleratorVendor::intel:
              vendor = AcceleratorVendor::intel;
              break;
            case HostAcceleratorVendor::other:
              throw std::runtime_error(
                  "hostd passive inventory names an unsupported accelerator "
                  "vendor");
            }
            result.push_back({
                .vendor = vendor,
                .stable_id = memory.stable_id,
                .total_memory_bytes = memory.total_memory_bytes,
                .free_memory_bytes = memory.free_memory_bytes,
                .selector_labels = memory.selector_labels,
                .observation_digest =
                    authority.passive_memory_observation_digest,
                .disposition = memory.disposition,
            });
          }
          return result;
        };
    auto snapshot = make_local_passive_host_snapshot_source(
        authority_host_.host_id, authority_host_.boot_id,
        [this] {
          const auto sample = authority_now();
          if (sample.boot.nanoseconds < 0)
            throw std::runtime_error("authority boot clock is negative");
          return static_cast<std::uint64_t>(sample.boot.nanoseconds);
        }, std::move(hostd_accelerators));
    preflight_evidence_ =
        std::make_shared<RegisteredTrainingPreflightEvidenceProvider>(
            adapter_registry_, std::move(snapshot),
            std::vector<RegisteredTrainingNodeProbe>{
                {.key = {.adapter = "rwkv-lab.hf-multimodal-sft",
                         .version = "1.0.0",
                         .runtime = ComponentRuntime::python_worker,
                         .operation = "train",
                         .contract =
                             "rwkv_lab.hf_multimodal_sft.v1.Train"},
                 .probe = make_hf_multimodal_sft_training_node_probe(
                     [this](const AdapterProfile& profile) {
                       const auto& launch = host_launch_registry_.resolve(
                           profile.key, profile.code_fingerprint);
                       return PassiveRuntimeProfileEvidence{
                           .profile_digest =
                               host_launch_registry_.profile_digest(
                                   profile.key, profile.code_fingerprint),
                           .provided_capabilities =
                               launch.provided_capabilities,
                       };
                     })},
            });
  }
  if (static_cast<bool>(host_process_client_) != !controller_target_.empty() ||
      static_cast<bool>(host_grant_client_) !=
          static_cast<bool>(host_process_client_)) {
    throw std::invalid_argument(
        "host grant/process clients and controller target must be configured together");
  }
}

TrainVMService::~TrainVMService() { stop_reconciliation_supervisor(); }

AuthorityTimeSample TrainVMService::authority_now() const {
  return authority_clock_->sample();
}

void TrainVMService::configure_hostd(
    const HostdClientConfiguration& configuration,
    std::string controller_target) {
  if (hostd_claim_provider_ || host_grant_client_ || host_process_client_ ||
      host_grant_saga_ || host_process_saga_ ||
      controller_target.empty() || controller_target.size() > 4'096U ||
      !controller_target.starts_with("unix:/")) {
    throw std::invalid_argument(
        "hostd clients require one canonical Unix controller target");
  }
  HostdClientBundle bundle = bootstrap_hostd_clients(
      journal_, authority_host_, configuration,
      [this] { return authority_now(); });
  hostd_claim_provider_ = std::move(bundle.claim_provider);
  host_grant_client_ = std::move(bundle.resource_client);
  host_process_client_ = std::move(bundle.process_client);
  hostd_status_client_ = configuration.status();
  hostd_status_timeout_ns_ = configuration.request_timeout_ns();
  controller_target_ = std::move(controller_target);
  host_grant_saga_ = std::make_unique<HostGrantSagaReconciler>(
      journal_, *host_grant_client_);
  host_process_saga_ = std::make_unique<HostProcessSagaReconciler>(
      journal_, *host_process_client_);
}

ReconcileResult TrainVMService::reconcile_once(const std::string& run_id) {
  if (const auto disposition = reconcile_resource_releasing_pause(run_id)) {
    return {.disposition = *disposition,
            .run_id = run_id,
            .launch = std::nullopt};
  }
  if (const auto disposition = reconcile_cancellation(run_id)) {
    return {.disposition = *disposition,
            .run_id = run_id,
            .launch = std::nullopt};
  }
  if (host_grant_saga_) {
    if (const auto disposition = reconcile_host_release(run_id)) {
      return {.disposition = *disposition,
              .run_id = run_id,
              .launch = std::nullopt};
    }
    if (const auto disposition = reconcile_host_grant(run_id)) {
      return {.disposition = *disposition,
              .run_id = run_id,
              .launch = std::nullopt};
    }
  }
  ReconcileResult result = reconciler_.step(run_id);
  if (result.launch && host_process_saga_) {
    const std::string launch_id = run_id + ":worker-launch:" +
                                  result.launch->node_id + ":" +
                                  result.launch->attempt_id;
    {
      std::scoped_lock lock(command_mutex_);
      if (const auto process = journal_.host_process_saga(launch_id);
          process && process->committed) {
        if (process->exited) {
          throw OperationPreconditionError(
              "active worker launch already has terminal host evidence");
        }
        result.disposition = ReconcileDisposition::awaiting_worker;
        result.launch.reset();
        return result;
      }
    }
    const ResolvedLaunchSpec binding = bind_worker_launch(*result.launch);
    (void)launch_worker_process(binding.identity.launch_event_id);
  }
  return result;
}

void TrainVMService::start_reconciliation_supervisor() {
  std::scoped_lock lock(reconciliation_mutex_);
  if (reconciliation_started_ || reconciliation_thread_.joinable()) {
    throw std::logic_error("TrainVM reconciliation supervisor is already running");
  }
  reconciliation_failures_.clear();
  reconciliation_scan_cursor_.clear();
  reconciliation_wake_runs_.clear();
  reconciliation_schedules_.clear();
  reconciliation_metrics_ = {};
  reconciliation_started_ = true;
  try {
    reconciliation_thread_ = std::jthread(
        [this](std::stop_token stop) { reconciliation_loop(stop); });
  } catch (...) {
    reconciliation_started_ = false;
    throw;
  }
}

void TrainVMService::stop_reconciliation_supervisor() noexcept {
  std::jthread stopping;
  {
    std::scoped_lock lock(reconciliation_mutex_);
    if (!reconciliation_started_ && !reconciliation_thread_.joinable()) return;
    reconciliation_started_ = false;
    reconciliation_thread_.request_stop();
    stopping = std::move(reconciliation_thread_);
  }
  reconciliation_condition_.notify_all();
  // The local jthread joins outside reconciliation_mutex_; the worker may be
  // leaving its condition wait and must be able to reacquire that mutex.
}

void TrainVMService::notify_reconciliation(const std::string& run_id) {
  if (run_id.empty() || run_id.size() > 256U) {
    throw std::invalid_argument(
        "reconciliation wake requires a bounded run identity");
  }
  {
    std::scoped_lock lock(reconciliation_mutex_);
    if (!reconciliation_started_) return;
    if (reconciliation_wake_runs_.size() < kMaximumSupervisorWakeRuns) {
      reconciliation_wake_runs_.insert(run_id);
    }
    // An explicit wake means something happened, so the run's accumulated
    // idle backoff no longer describes it.
    if (const auto found = reconciliation_schedules_.find(run_id);
        found != reconciliation_schedules_.end()) {
      found->second.backoff_ns = 0;
      found->second.idle_passes = 0U;
      found->second.next_due_ns = 0;
    }
  }
  reconciliation_condition_.notify_one();
}

std::optional<ReconcileDisposition> TrainVMService::reconcile_until_quiescent(
    const std::string& run_id) {
  {
    std::scoped_lock lock(reconciliation_mutex_);
    ++reconciliation_metrics_.reconcile_passes;
  }
  for (std::size_t step = 0U; step < kMaximumImmediateReconcileSteps;
       ++step) {
    {
      std::scoped_lock lock(reconciliation_mutex_);
      ++reconciliation_metrics_.reconcile_steps;
    }
    const ReconcileResult result = reconcile_once(run_id);
    switch (result.disposition) {
      case ReconcileDisposition::lease_acquired:
      case ReconcileDisposition::host_grant_acquired:
      case ReconcileDisposition::host_process_exited:
      case ReconcileDisposition::external_profiler_artifact_published:
      case ReconcileDisposition::host_grant_released:
      case ReconcileDisposition::builtin_completed:
      // A committed qualification verdict advances the node either way, so the
      // run keeps draining within this wake instead of sleeping a full cadence
      // between the gate and whatever the plan routes it to.
      case ReconcileDisposition::qualification_completed:
      case ReconcileDisposition::qualification_rejected:
        continue;
      case ReconcileDisposition::no_action:
      case ReconcileDisposition::lease_busy:
      case ReconcileDisposition::host_grant_busy:
      case ReconcileDisposition::launch_prepared:
      case ReconcileDisposition::launch_replayed:
      case ReconcileDisposition::awaiting_worker:
      // Evidence has not been published yet. This is a wait, not a failure:
      // the next supervisor wake retries the gate.
      case ReconcileDisposition::qualification_evidence_required:
      case ReconcileDisposition::input_required:
        return result.disposition;
    }
  }
  // A legal cyclic workflow can consume the per-wake work budget. The caller
  // requeues it at the supervisor cadence instead of monopolizing the authority
  // thread: the old unconditional notify_reconciliation() here made the next
  // condition wait return immediately, so a run that kept consuming its budget
  // reconciled in a zero-delay loop and pinned a core.
  std::scoped_lock lock(reconciliation_mutex_);
  ++reconciliation_metrics_.budget_requeues;
  return std::nullopt;
}

void TrainVMService::synchronize_lease_renewal(
    const std::string& run_id) {
  std::scoped_lock lock(command_mutex_);
  const auto projection = journal_.projection(run_id);
  if (!projection) return;
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot synchronize lease renewal without a persisted plan");
  }
  const std::string& concurrency_key =
      plan->experiment.spec.workspace.concurrency_key;
  const AuthorityTimeSample now = authority_now();
  const auto active = journal_.active_lease(concurrency_key, now);
  if (active && active->owner_run_id == run_id) {
    const std::int64_t timeout_seconds =
        plan->experiment.spec.resources.lease_timeout_seconds.value_or(30);
    if (timeout_seconds <= 0 ||
        timeout_seconds >
            LeaseRenewalCoordinator::kMaximumTimeoutNs / 1'000'000'000LL) {
      throw std::runtime_error(
          "compiled lease timeout exceeds renewal policy");
    }
    const std::int64_t timeout_ns = timeout_seconds * 1'000'000'000LL;
    const std::int64_t renewal_margin_ns = std::min<std::int64_t>(
        LeaseRenewalCoordinator::kMaximumMarginNs,
        std::max<std::int64_t>(1'000'000'000LL, timeout_ns / 3LL));
    (void)lease_renewals_.untrack(concurrency_key);
    lease_renewals_.track({.lease = *active,
                           .timeout_ns = timeout_ns,
                           .renewal_margin_ns = renewal_margin_ns});
    return;
  }
  const bool terminal = projection->observed_state == "completed" ||
                        projection->observed_state == "failed" ||
                        projection->observed_state == "cancelled";
  if (!active && terminal) {
    (void)lease_renewals_.untrack(concurrency_key);
  }
}

std::optional<std::string> TrainVMService::reconciliation_failure(
    const std::string& run_id) const {
  std::scoped_lock lock(reconciliation_mutex_);
  const auto found = reconciliation_failures_.find(run_id);
  return found == reconciliation_failures_.end()
             ? std::nullopt
             : std::optional<std::string>{found->second};
}

std::optional<ReconcileDisposition>
TrainVMService::reconcile_resource_releasing_pause(
    const std::string& run_id) {
  std::scoped_lock lock(command_mutex_);
  const auto projection = journal_.projection(run_id);
  if (!projection || projection->desired_state != "paused" ||
      projection->observed_state != "pausing") {
    return std::nullopt;
  }
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot reconcile resource pause without a persisted plan");
  }
  std::string command_id;
  for (const Event& event : journal_.events_for_run(run_id)) {
    if (event.event_type == "run.observed_state_changed" &&
        event.payload.value("state", std::string{}) == "pausing") {
      const std::string candidate =
          event.payload.value("cause_command_id", std::string{});
      const auto command = journal_.lifecycle_command(candidate);
      if (command && command->kind == LifecycleCommandKind::pause &&
          command->release_resources) {
        command_id = candidate;
      }
    }
  }
  const auto command = journal_.lifecycle_command(command_id);
  if (!command || command->kind != LifecycleCommandKind::pause ||
      !command->release_resources || !command->checkpoint_first ||
      command->status != LifecycleCommandStatus::applied ||
      !command->acknowledgement || !command->acknowledged_at_ns ||
      !command->optimizer_step || command->artifact_id.empty() ||
      command->node_id != projection->current_node_id ||
      command->attempt_id != projection->current_attempt_id) {
    throw OperationPreconditionError(
        "pausing projection has no exact resource-release command");
  }
  const AuthorityTimeSample now = authority_now();
  const std::int64_t grace_ns =
      plan->experiment.spec.recovery.graceful_stop_seconds *
      1'000'000'000LL;
  const std::int64_t deadline =
      grace_ns > std::numeric_limits<std::int64_t>::max() -
                     *command->acknowledged_at_ns
          ? std::numeric_limits<std::int64_t>::max()
          : *command->acknowledged_at_ns + grace_ns;
  if (now.wall.nanoseconds < deadline) return std::nullopt;

  if (!host_process_saga_ || !host_grant_saga_) {
    Controller controller(*plan, journal_, run_id);
    controller.recover();
    (void)controller.complete_resource_releasing_pause(command_id, now);
    return ReconcileDisposition::builtin_completed;
  }

  const std::string launch_id = run_id + ":worker-launch:" +
                                command->node_id + ":" +
                                command->attempt_id;
  auto process = journal_.host_process_saga(launch_id);
  if (!process || !process->committed) {
    throw OperationPreconditionError(
        "resource pause has no durable committed worker process");
  }
  if (!process->exited) {
    process = host_process_saga_->reconcile_exit(launch_id, true, now);
    if (!process->exited) {
      throw std::runtime_error(
          "resource pause process exit returned no terminal receipt");
    }
    return ReconcileDisposition::host_process_exited;
  }

  const ResourceBundleGrant& grant = process->prepare.grant;
  auto grant_saga = journal_.host_grant_saga(grant.request_id);
  if (!grant_saga || !grant_saga->grant ||
      grant_saga->grant->receipt_digest != grant.receipt_digest ||
      grant_saga->busy_outcome_digest) {
    throw OperationPreconditionError(
        "resource pause has no exact durable physical grant");
  }
  if (!grant_saga->release_receipt) {
    const ResourceReleaseRequest release = seal_resource_release_request({
        .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
        .release_request_id =
            "host-release-" +
            sha256_hex(grant.request_id + "\n" + grant.receipt_digest),
        .allocation_id = grant.allocation_id,
        .grant_digest = grant.receipt_digest,
        .journal_id = grant.journal_id,
        .run_id = grant.run_id,
        .logical_lease_id = grant.logical_lease_id,
        .logical_fencing_token = grant.logical_fencing_token,
        .canonical_request_digest = {},
    });
    grant_saga = host_grant_saga_->reconcile_release(
        grant.request_id, release, now);
    if (!grant_saga->release_receipt) {
      throw std::runtime_error(
          "resource pause release returned no receipt");
    }
    return ReconcileDisposition::host_grant_released;
  }

  const auto& identity = *command->acknowledgement;
  (void)journal_.release_lease(identity.concurrency_key, run_id,
                               identity.lease_id,
                               identity.fencing_token, now);
  Controller controller(*plan, journal_, run_id);
  controller.recover();
  (void)controller.complete_resource_releasing_pause(command_id, now);
  return ReconcileDisposition::builtin_completed;
}

std::optional<ReconcileDisposition> TrainVMService::reconcile_cancellation(
    const std::string& run_id) {
  std::scoped_lock lock(command_mutex_);
  const auto projection = journal_.projection(run_id);
  if (!projection || projection->desired_state != "cancelled" ||
      projection->observed_state != "cancelling") {
    return std::nullopt;
  }
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot reconcile cancellation without a persisted plan");
  }
  std::string command_id;
  for (const Event& event : journal_.events_for_run(run_id)) {
    if (event.event_type == "run.observed_state_changed" &&
        event.payload.value("state", std::string{}) == "cancelling") {
      command_id = event.payload.value("cause_command_id", std::string{});
    }
  }
  const auto command = journal_.lifecycle_command(command_id);
  if (!command || command->kind != LifecycleCommandKind::cancel ||
      command->status != LifecycleCommandStatus::applied ||
      !command->acknowledgement || !command->acknowledged_at_ns ||
      command->node_id != projection->current_node_id ||
      command->attempt_id != projection->current_attempt_id) {
    throw OperationPreconditionError(
        "cancelling projection has no exact applied command");
  }
  const AuthorityTimeSample now = authority_now();
  const std::int64_t deadline =
      command->graceful_timeout_ns >
              std::numeric_limits<std::int64_t>::max() -
                  *command->acknowledged_at_ns
          ? std::numeric_limits<std::int64_t>::max()
          : *command->acknowledged_at_ns + command->graceful_timeout_ns;
  if (now.wall.nanoseconds < deadline) return std::nullopt;

  if (!host_process_saga_ || !host_grant_saga_) {
    Controller controller(*plan, journal_, run_id);
    controller.recover();
    (void)controller.complete_cancellation(command_id, now);
    return ReconcileDisposition::builtin_completed;
  }

  const std::string launch_id = run_id + ":worker-launch:" +
                                command->node_id + ":" +
                                command->attempt_id;
  auto process = journal_.host_process_saga(launch_id);
  if (!process || !process->committed) {
    throw OperationPreconditionError(
        "cancellation has no durable committed worker process");
  }
  if (!process->exited) {
    process = host_process_saga_->reconcile_exit(launch_id, true, now);
    if (!process->exited) {
      throw std::runtime_error(
          "cancellation process exit returned no terminal receipt");
    }
    return ReconcileDisposition::host_process_exited;
  }

  const ResourceBundleGrant& grant = process->prepare.grant;
  auto grant_saga = journal_.host_grant_saga(grant.request_id);
  if (!grant_saga || !grant_saga->grant ||
      grant_saga->grant->receipt_digest != grant.receipt_digest ||
      grant_saga->busy_outcome_digest) {
    throw OperationPreconditionError(
        "cancellation has no exact durable physical grant");
  }
  if (!grant_saga->release_receipt) {
    const ResourceReleaseRequest release = seal_resource_release_request({
        .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
        .release_request_id =
            "host-release-" +
            sha256_hex(grant.request_id + "\n" + grant.receipt_digest),
        .allocation_id = grant.allocation_id,
        .grant_digest = grant.receipt_digest,
        .journal_id = grant.journal_id,
        .run_id = grant.run_id,
        .logical_lease_id = grant.logical_lease_id,
        .logical_fencing_token = grant.logical_fencing_token,
        .canonical_request_digest = {},
    });
    grant_saga = host_grant_saga_->reconcile_release(
        grant.request_id, release, now);
    if (!grant_saga->release_receipt) {
      throw std::runtime_error(
          "cancellation resource release returned no receipt");
    }
    return ReconcileDisposition::host_grant_released;
  }

  const auto& identity = *command->acknowledgement;
  (void)journal_.release_lease(identity.concurrency_key, run_id,
                               identity.lease_id,
                               identity.fencing_token, now);
  Controller controller(*plan, journal_, run_id);
  controller.recover();
  (void)controller.complete_cancellation(command_id, now);
  return ReconcileDisposition::builtin_completed;
}

void TrainVMService::record_reconciliation_failure(std::string run_id,
                                                   std::string message) {
  if (message.size() > kMaximumSupervisorFailureBytes) {
    message.resize(kMaximumSupervisorFailureBytes);
  }
  std::scoped_lock lock(reconciliation_mutex_);
  const auto existing = reconciliation_failures_.find(run_id);
  if (existing != reconciliation_failures_.end()) {
    existing->second = std::move(message);
    return;
  }
  if (reconciliation_failures_.size() < kMaximumSupervisorFailures) {
    reconciliation_failures_.emplace(std::move(run_id), std::move(message));
    return;
  }
  if (!reconciliation_failures_.contains("__overflow__")) {
    reconciliation_failures_.erase(reconciliation_failures_.begin());
  }
  reconciliation_failures_.insert_or_assign(
      "__overflow__",
      "reconciliation failure retention limit is exhausted");
}

namespace {

// The wait a disposition puts a run into, and whether the clock alone can end
// it. A run parked on worker evidence, operator input, published qualification
// evidence, or nothing at all cannot change disposition until something is
// written to the journal, so re-reconciling it on a timer is pure waste; a run
// parked on lease or host-grant contention can be released by another run's
// lease expiring, with no journal write of its own to wake it.
struct DispositionWait final {
  std::string_view reason;
  bool clock_sensitive{};
};

DispositionWait disposition_wait(ReconcileDisposition disposition) {
  switch (disposition) {
    case ReconcileDisposition::lease_busy:
      return {"waiting for a conflicting logical lease to expire", true};
    case ReconcileDisposition::host_grant_busy:
      return {"waiting for host resource admission", true};
    case ReconcileDisposition::awaiting_worker:
      return {"waiting for worker evidence", false};
    case ReconcileDisposition::launch_prepared:
    case ReconcileDisposition::launch_replayed:
      return {"waiting for the launched worker to report", false};
    case ReconcileDisposition::qualification_evidence_required:
      return {"waiting for published cache qualification evidence", false};
    case ReconcileDisposition::input_required:
      return {"waiting for operator artifact validation", false};
    case ReconcileDisposition::no_action:
      return {"no reconciliation action is available", false};
    // Progress dispositions never park a run: reconcile_until_quiescent keeps
    // draining, so they are only reachable here defensively.
    case ReconcileDisposition::lease_acquired:
    case ReconcileDisposition::host_grant_acquired:
    case ReconcileDisposition::host_process_exited:
    case ReconcileDisposition::external_profiler_artifact_published:
    case ReconcileDisposition::host_grant_released:
    case ReconcileDisposition::builtin_completed:
    case ReconcileDisposition::qualification_completed:
    case ReconcileDisposition::qualification_rejected:
      return {"reconciliation is making progress", true};
  }
  return {"reconciliation disposition is unclassified", true};
}

std::int64_t next_backoff_ns(std::int64_t current, std::int64_t ceiling,
                             std::int64_t base) {
  if (current < base) return base;
  if (current >= ceiling) return ceiling;
  return current > ceiling / 2 ? ceiling : current * 2;
}

}  // namespace

TrainVMService::SupervisorRunSchedule&
TrainVMService::reconciliation_schedule(const std::string& run_id) {
  if (const auto found = reconciliation_schedules_.find(run_id);
      found != reconciliation_schedules_.end()) {
    return found->second;
  }
  // Bounded retention. Evicting the entry due soonest costs at most one extra
  // reconcile for a run that is about to be reconciled anyway, and never
  // suppresses one: an absent schedule always means "due now".
  while (reconciliation_schedules_.size() >= kMaximumSupervisorSchedules) {
    auto victim = reconciliation_schedules_.begin();
    for (auto candidate = reconciliation_schedules_.begin();
         candidate != reconciliation_schedules_.end(); ++candidate) {
      if (candidate->second.next_due_ns < victim->second.next_due_ns) {
        victim = candidate;
      }
    }
    reconciliation_schedules_.erase(victim);
  }
  return reconciliation_schedules_[run_id];
}

bool TrainVMService::reconciliation_due(const RunProjection& projection,
                                        std::int64_t now_ns) const {
  std::scoped_lock lock(reconciliation_mutex_);
  const auto found = reconciliation_schedules_.find(projection.run_id);
  if (found == reconciliation_schedules_.end()) return true;
  // A journal write is the event-driven path: any new event for this run makes
  // it due immediately, whatever its backoff was.
  if (found->second.last_event_sequence != projection.last_event_sequence) {
    return true;
  }
  return now_ns >= found->second.next_due_ns;
}

void TrainVMService::record_reconciliation_outcome(
    const std::string& run_id,
    std::optional<ReconcileDisposition> disposition,
    std::uint64_t observed_event_sequence, std::int64_t now_ns) {
  std::scoped_lock lock(reconciliation_mutex_);
  SupervisorRunSchedule& schedule = reconciliation_schedule(run_id);
  // Anything that moved the journal is fresh work, so the run starts its wait
  // over rather than inheriting the backoff it had accumulated while idle.
  if (schedule.last_event_sequence != observed_event_sequence) {
    schedule.backoff_ns = 0;
    schedule.idle_passes = 0U;
  }
  schedule.last_event_sequence = observed_event_sequence;
  schedule.retries = 0U;
  if (!disposition) {
    // The pass exhausted its step budget, which means it was making progress.
    // Requeue at the cadence rather than at zero delay.
    schedule.wait_reason = "reconciliation exceeded its per-wake step budget";
    schedule.backoff_ns = 0;
    schedule.next_due_ns = now_ns;
    schedule.idle_passes = 0U;
    return;
  }
  const DispositionWait wait = disposition_wait(*disposition);
  schedule.wait_reason = std::string(wait.reason);
  ++schedule.idle_passes;
  schedule.backoff_ns = next_backoff_ns(
      schedule.backoff_ns,
      wait.clock_sensitive ? kContendedBackoffCeilingNs : kIdleBackoffCeilingNs,
      kSupervisorCadenceNs);
  schedule.next_due_ns =
      now_ns > std::numeric_limits<std::int64_t>::max() - schedule.backoff_ns
          ? std::numeric_limits<std::int64_t>::max()
          : now_ns + schedule.backoff_ns;
  reconciliation_metrics_.tracked_runs = reconciliation_schedules_.size();
}

void TrainVMService::record_reconciliation_retry(
    const std::string& run_id, std::uint64_t observed_event_sequence,
    std::int64_t now_ns) {
  std::scoped_lock lock(reconciliation_mutex_);
  ++reconciliation_metrics_.failures;
  SupervisorRunSchedule& schedule = reconciliation_schedule(run_id);
  if (schedule.last_event_sequence != observed_event_sequence) {
    schedule.backoff_ns = 0;
  }
  schedule.last_event_sequence = observed_event_sequence;
  ++schedule.retries;
  schedule.wait_reason = "retrying after a reconciliation failure";
  schedule.backoff_ns = next_backoff_ns(schedule.backoff_ns,
                                        kIdleBackoffCeilingNs,
                                        kSupervisorCadenceNs);
  schedule.next_due_ns =
      now_ns > std::numeric_limits<std::int64_t>::max() - schedule.backoff_ns
          ? std::numeric_limits<std::int64_t>::max()
          : now_ns + schedule.backoff_ns;
  reconciliation_metrics_.tracked_runs = reconciliation_schedules_.size();
}

ReconciliationSupervisorMetrics TrainVMService::reconciliation_metrics() const {
  std::scoped_lock lock(reconciliation_mutex_);
  ReconciliationSupervisorMetrics metrics = reconciliation_metrics_;
  metrics.tracked_runs = reconciliation_schedules_.size();
  return metrics;
}

std::vector<ReconciliationRunWait> TrainVMService::reconciliation_waits(
    std::size_t limit) const {
  std::vector<ReconciliationRunWait> waits;
  std::scoped_lock lock(reconciliation_mutex_);
  waits.reserve(std::min(limit, reconciliation_schedules_.size()));
  for (const auto& [run_id, schedule] : reconciliation_schedules_) {
    if (waits.size() >= limit) break;
    waits.push_back({.run_id = run_id,
                     .wait_reason = schedule.wait_reason,
                     .idle_passes = schedule.idle_passes,
                     .retries = schedule.retries,
                     .backoff_ns = schedule.backoff_ns,
                     .next_due_ns = schedule.next_due_ns,
                     .last_event_sequence = schedule.last_event_sequence});
  }
  return waits;
}

void TrainVMService::reconciliation_loop(std::stop_token stop) {
  constexpr auto kSupervisorCadence =
      std::chrono::nanoseconds(kSupervisorCadenceNs);
  while (!stop.stop_requested()) {
    std::set<std::string, std::less<>> run_ids;
    {
      std::unique_lock lock(reconciliation_mutex_);
      const bool woken = reconciliation_condition_.wait_for(
          lock, kSupervisorCadence,
          [&] { return stop.stop_requested() ||
                       !reconciliation_wake_runs_.empty(); });
      if (stop.stop_requested()) break;
      ++reconciliation_metrics_.wakes;
      if (woken && !reconciliation_wake_runs_.empty()) {
        ++reconciliation_metrics_.explicit_wakes;
      } else {
        ++reconciliation_metrics_.cadence_wakes;
      }
      run_ids.swap(reconciliation_wake_runs_);
    }
    // An explicitly named run is reconciled unconditionally: a wake is the
    // event-driven path, and its whole point is to bypass any backoff.
    const std::set<std::string, std::less<>> woken_runs = run_ids;

    try {
      std::vector<RunProjection> page;
      {
        std::scoped_lock lock(command_mutex_);
        page = journal_.reconcilable_projections(
            reconciliation_scan_cursor_, kReconciliationPageSize);
      }
      const std::int64_t now_ns = authority_now().boot.nanoseconds;
      {
        std::scoped_lock lock(reconciliation_mutex_);
        ++reconciliation_metrics_.scans;
        reconciliation_scan_cursor_ =
            page.size() == kReconciliationPageSize
                ? page.back().run_id
                : std::string{};
      }
      for (const RunProjection& projection : page) {
        if (woken_runs.contains(projection.run_id)) continue;
        if (reconciliation_due(projection, now_ns)) {
          run_ids.insert(projection.run_id);
        } else {
          std::scoped_lock lock(reconciliation_mutex_);
          ++reconciliation_metrics_.skipped_idle_runs;
        }
      }
    } catch (const std::exception& exception) {
      record_reconciliation_failure("__scan__", exception.what());
      return;
    }

    for (const std::string& run_id : run_ids) {
      if (stop.stop_requested()) break;
      // Sampled before the pass: an event written while the pass runs must
      // leave the run due again rather than be swallowed by the schedule. A
      // failing pass records it too, or a run whose every pass throws would
      // read as "the journal moved" forever and never back off.
      std::uint64_t observed_sequence = 0U;
      try {
        std::scoped_lock lock(command_mutex_);
        if (const auto projection = journal_.projection(run_id)) {
          observed_sequence = projection->last_event_sequence;
        }
      } catch (...) {
        // Leave the sequence at zero; the pass below reports the real failure.
      }
      try {
        const std::optional<ReconcileDisposition> disposition =
            reconcile_until_quiescent(run_id);
        synchronize_lease_renewal(run_id);
        record_reconciliation_outcome(run_id, disposition, observed_sequence,
                                      authority_now().boot.nanoseconds);
        std::scoped_lock lock(reconciliation_mutex_);
        reconciliation_failures_.erase(run_id);
      } catch (const std::exception& exception) {
        record_reconciliation_failure(run_id, exception.what());
        record_reconciliation_retry(run_id, observed_sequence,
                                    authority_now().boot.nanoseconds);
      } catch (...) {
        record_reconciliation_failure(
            run_id,
            "reconciliation failed with a non-standard exception");
        record_reconciliation_retry(run_id, observed_sequence,
                                    authority_now().boot.nanoseconds);
      }
    }
    if (stop.stop_requested()) break;
    try {
      std::scoped_lock lock(command_mutex_);
      (void)lease_renewals_.tick();
    } catch (const std::exception& exception) {
      record_reconciliation_failure("__lease_renewal__", exception.what());
      return;
    } catch (...) {
      record_reconciliation_failure(
          "__lease_renewal__",
          "lease renewal failed with a non-standard exception");
      return;
    }
  }
}

std::optional<ReconcileDisposition> TrainVMService::reconcile_host_release(
    const std::string& run_id) {
  std::scoped_lock lock(command_mutex_);
  const AuthorityTimeSample now = authority_now();
  const auto projection = journal_.projection(run_id);
  if (!projection) {
    throw std::invalid_argument("cannot reconcile host release for unknown run");
  }
  if (projection->desired_state != "running" ||
      projection->observed_state != "running") {
    return std::nullopt;
  }
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot reconcile host release without persisted plan");
  }
  Controller controller(*plan, journal_, run_id);
  controller.recover();
  if (controller.state().status != ExecutionStatus::running ||
      controller.state().current_node_id.empty()) {
    return std::nullopt;
  }
  const Node& node = plan->experiment.spec.workflow.nodes.at(
      controller.state().current_node_id);
  const Component& component =
      plan->experiment.spec.components.at(node.invoke.component);
  if (component.runtime != ComponentRuntime::builtin ||
      component.adapter != "trainvm.core" ||
      node.invoke.operation != "release_resources") {
    return std::nullopt;
  }

  // Every committed process must have terminal host evidence before its
  // physical bundle can be released. Event order, not retained in-memory
  // descriptors, discovers all attempts after controller restart.
  for (const Event& event : journal_.events_for_run(run_id)) {
    if (event.event_type != "host.process_prepared" ||
        !event.payload.contains("prepare")) {
      continue;
    }
    const HostdProcessPrepareRequest prepare =
        hostd_process_prepare_from_canonical_json(
            event.payload.at("prepare").dump());
    const std::string& launch_id =
        prepare.launch.identity.launch_event_id;
    const auto process = journal_.host_process_saga(launch_id);
    if (!process || !process->committed) {
      throw OperationPreconditionError(
          "release node has a prepared process without durable exec authority");
    }
    if (!process->exited) {
      const auto exited =
          host_process_saga_->reconcile_exit(launch_id, true, now);
      if (!exited.exited) {
        throw std::runtime_error(
            "host process exit reconciliation returned no receipt");
      }
      return ReconcileDisposition::host_process_exited;
    }
    const auto binding = journal_.launch_binding(launch_id);
    if (!binding) {
      throw OperationPreconditionError(
          "release node has a process without its durable launch binding");
    }
    if (binding->identity.profiler) {
      const auto window = std::filesystem::path(
          binding->identity.profiler->raw_output_path + ".window.json");
      std::error_code filesystem_error;
      const bool has_window = std::filesystem::exists(window, filesystem_error);
      if (filesystem_error) {
        throw std::runtime_error(
            "external profiler window identity could not be inspected");
      }
      if (has_window) {
        const auto& exit = process->exited->receipt.request;
        if (exit.wait_code != CLD_EXITED || exit.wait_status != 0) {
          throw OperationPreconditionError(
              "external profiler exited unsuccessfully after sealing a capture window");
        }
        if (reconcile_external_profiler_artifact(*projection, *plan, *process,
                                                 *binding)) {
          return ReconcileDisposition::external_profiler_artifact_published;
        }
      }
    }
  }

  const std::string& concurrency_key =
      plan->experiment.spec.workspace.concurrency_key;
  const auto lease = journal_.active_lease(concurrency_key, now);
  if (!lease || lease->owner_run_id != run_id) {
    throw OperationPreconditionError(
        "host release reconciliation lost its logical lease");
  }
  const ResourceBundleRequest request = build_resource_bundle_request({
      .journal_id = journal_.journal_id(),
      .plan_hash = plan->plan_hash,
      .run_id = run_id,
      .resources = plan->experiment.spec.resources,
      .lease = *lease,
  });
  const auto saga = journal_.host_grant_saga(request.request_id);
  if (!saga) return std::nullopt;
  if (!saga->grant || saga->busy_outcome_digest) {
    throw OperationPreconditionError(
        "release node has no exact durable physical grant");
  }
  if (saga->release_receipt) return std::nullopt;
  const ResourceReleaseRequest release = seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id =
          "host-release-" + sha256_hex(request.request_id + "\n" +
                                       saga->grant->receipt_digest),
      .allocation_id = saga->grant->allocation_id,
      .grant_digest = saga->grant->receipt_digest,
      .journal_id = saga->grant->journal_id,
      .run_id = saga->grant->run_id,
      .logical_lease_id = saga->grant->logical_lease_id,
      .logical_fencing_token = saga->grant->logical_fencing_token,
      .canonical_request_digest = {},
  });
  const HostGrantSagaSnapshot released = host_grant_saga_->reconcile_release(
      request.request_id, release, now);
  if (!released.release_receipt) {
    throw std::runtime_error(
        "host release reconciliation returned no receipt");
  }
  return ReconcileDisposition::host_grant_released;
}

bool TrainVMService::reconcile_external_profiler_artifact(
    const RunProjection& projection, const CompiledPlan& plan,
    const HostProcessSagaSnapshot& process,
    const ResolvedLaunchSpec& binding) {
  if (!binding.identity.profiler || !process.exited) {
    throw std::invalid_argument(
        "external profiler publication requires a terminal profiler launch");
  }
  const auto& profiler = *binding.identity.profiler;
  const std::string dispatch_id = projection.run_id + ":dispatch:" +
                                  binding.identity.node_id + ":" +
                                  binding.identity.attempt_id;
  const auto dispatch = journal_.dispatch(dispatch_id);
  const auto invocation = journal_.worker_invocation(dispatch_id);
  if (!dispatch || dispatch->status != DispatchStatus::completed ||
      !dispatch->result_event_id || !invocation ||
      invocation->run_id != projection.run_id ||
      invocation->node_id != binding.identity.node_id ||
      invocation->attempt_id != binding.identity.attempt_id) {
    throw OperationPreconditionError(
        "external profiler publication has no completed immutable invocation");
  }
  if (!profiler.capture.output_artifact) {
    throw OperationPreconditionError(
        "external profiler publication has no declared logical artifact");
  }
  const nlohmann::json* publication = nullptr;
  for (const auto& [output_name, candidate] : invocation->publishes.items()) {
    (void)output_name;
    if (candidate.is_object() &&
        candidate.value("logical_name", std::string{}) ==
            *profiler.capture.output_artifact) {
      if (publication != nullptr) {
        throw OperationPreconditionError(
            "external profiler publication authority is ambiguous");
      }
      publication = &candidate;
    }
  }
  if (!publication || !publication->contains("declaration") ||
      !publication->at("declaration").is_object() ||
      publication->at("declaration").value("type", std::string{}) !=
          "opaque" ||
      publication->at("declaration").value("schema", std::string{}) !=
          "trainvm.gpu-trace.v1" ||
      publication->at("declaration").value("fingerprint", std::string{}) !=
          "adapter") {
    throw OperationPreconditionError(
        "external profiler output declaration is incompatible");
  }
  const ExternalProfilerPublishedArtifact artifact =
      publish_external_profiler_artifact({
          .backend = profiler.backend,
          .run_id = projection.run_id,
          .node_id = binding.identity.node_id,
          .attempt_id = binding.identity.attempt_id,
          .authority_digest = profiler.authority.authority_digest,
          .invocation_digest = invocation->invocation_digest,
          .capture = profiler.capture,
          .raw_output_prefix = profiler.raw_output_path,
          .run_directory = plan.experiment.spec.workspace.run_directory,
      });
  const std::int64_t published_at_ns =
      process.exited->receipt.observed_wall_time_ns;
  const Event event{
      .event_id = dispatch_id + ":artifact:" +
                  sha256_hex(artifact.artifact_id),
      .run_id = projection.run_id,
      .run_revision = projection.run_revision,
      .plan_revision = invocation->plan_revision,
      .node_id = binding.identity.node_id,
      .attempt_id = binding.identity.attempt_id,
      .worker_sequence = 0U,
      .event_type = "artifact.published",
      .event_version = 1U,
      .wall_time_ns = published_at_ns,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {
          {"artifact_id", artifact.artifact_id},
          {"logical_name", *profiler.capture.output_artifact},
          {"kind", "opaque"},
          {"schema", "trainvm.gpu-trace.v1"},
          {"uri", "file://" + artifact.manifest_path.string()},
          {"size_bytes", artifact.size_bytes},
          {"fingerprint_algorithm", "adapter"},
          {"fingerprint", artifact.manifest_fingerprint},
          {"complete", true},
          {"producer_node_id", binding.identity.node_id},
          {"producer_attempt_id", binding.identity.attempt_id},
          {"parent_artifact_ids", nlohmann::json::array()},
          {"published_at_ns", published_at_ns},
      },
  };
  return journal_.publish_external_profiler_artifact(event);
}

std::optional<ReconcileDisposition> TrainVMService::reconcile_host_grant(
    const std::string& run_id) {
  std::scoped_lock lock(command_mutex_);
  const AuthorityTimeSample now = authority_now();
  const auto projection = journal_.projection(run_id);
  if (!projection) {
    throw std::invalid_argument("cannot reconcile host grant for unknown run");
  }
  if (projection->desired_state != "running" ||
      projection->observed_state != "acquiring") {
    return std::nullopt;
  }
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot reconcile host grant without persisted plan");
  }
  Controller controller(*plan, journal_, run_id);
  controller.recover();
  if (controller.state().status != ExecutionStatus::running ||
      controller.state().current_node_id.empty()) {
    throw std::runtime_error(
        "acquiring run has no deterministic current node");
  }
  const Node& node = plan->experiment.spec.workflow.nodes.at(
      controller.state().current_node_id);
  const Component& component =
      plan->experiment.spec.components.at(node.invoke.component);
  if (component.runtime == ComponentRuntime::builtin) return std::nullopt;
  const std::string& concurrency_key =
      plan->experiment.spec.workspace.concurrency_key;
  const auto lease = journal_.active_lease(concurrency_key, now);
  if (!lease || lease->owner_run_id != run_id) {
    throw OperationPreconditionError(
        "host grant reconciliation lost its logical lease");
  }
  const ResourceBundleRequest request = build_resource_bundle_request({
      .journal_id = journal_.journal_id(),
      .plan_hash = plan->plan_hash,
      .run_id = run_id,
      .resources = plan->experiment.spec.resources,
      .lease = *lease,
  });
  if (const auto existing = journal_.host_grant_saga(request.request_id)) {
    if (existing->grant) return std::nullopt;
    if (existing->busy_outcome_digest) {
      return ReconcileDisposition::host_grant_busy;
    }
  }
  const HostGrantSagaSnapshot saga =
      host_grant_saga_->reconcile_request(request, now);
  return saga.grant ? ReconcileDisposition::host_grant_acquired
                    : ReconcileDisposition::host_grant_busy;
}

void TrainVMService::prune_retained_launches(
    const AuthorityTimeSample& now) {
  for (auto retained = resolved_launches_.begin();
       retained != resolved_launches_.end();) {
    const ResolvedLaunchIdentity& identity = retained->second.spec().identity;
    const auto projection = journal_.projection(identity.run_id);
    const auto active = journal_.active_lease(identity.concurrency_key, now);
    const bool owns_active_lease =
        active && active->owner_run_id == identity.run_id &&
        active->lease_id == identity.lease_id &&
        active->fencing_token == identity.fencing_token;
    const bool terminal =
        !projection || projection->observed_state == "completed" ||
        projection->observed_state == "failed" ||
        projection->observed_state == "cancelled";
    const std::string dispatch_id = identity.run_id + ":dispatch:" +
                                    identity.node_id + ":" +
                                    identity.attempt_id;
    const auto dispatch = journal_.dispatch(dispatch_id);
    const bool completed_attempt =
        dispatch && dispatch->status == DispatchStatus::completed;
    const bool different_attempt =
        projection && !projection->current_node_id.empty() &&
        !projection->current_attempt_id.empty() &&
        (projection->current_node_id != identity.node_id ||
         projection->current_attempt_id != identity.attempt_id);
    if (terminal || completed_attempt || different_attempt ||
        !owns_active_lease) {
      retained = resolved_launches_.erase(retained);
    } else {
      ++retained;
    }
  }
}

void TrainVMService::require_retained_launch_capacity(
    const ResolvedLaunchSpec& candidate) const {
  const auto bundle_bytes = [](const ResolvedLaunchSpec& launch) {
    std::uint64_t bytes = launch.identity.executable.source_size;
    if (launch.identity.code) {
      if (launch.identity.code->source_size >
          std::numeric_limits<std::uint64_t>::max() - bytes) {
        throw HostLaunchResolutionError(
            "retained host launch byte accounting overflowed");
      }
      bytes += launch.identity.code->source_size;
    }
    if (launch.identity.profiler) {
      if (launch.identity.profiler->executable.source_size >
          std::numeric_limits<std::uint64_t>::max() - bytes) {
        throw HostLaunchResolutionError(
            "retained host launch byte accounting overflowed");
      }
      bytes += launch.identity.profiler->executable.source_size;
    }
    return bytes;
  };
  if (resolved_launches_.size() >= kMaximumRetainedLaunches) {
    throw HostLaunchResolutionError(
        "retained host launch count quota is exhausted");
  }
  std::uint64_t retained_bytes = 0U;
  for (const auto& [launch_id, retained] : resolved_launches_) {
    (void)launch_id;
    const std::uint64_t bytes = bundle_bytes(retained.spec());
    if (bytes > kMaximumRetainedLaunchBytes - retained_bytes) {
      throw HostLaunchResolutionError(
          "retained host launch byte quota is exhausted");
    }
    retained_bytes += bytes;
  }
  const std::uint64_t candidate_bytes = bundle_bytes(candidate);
  if (candidate_bytes > kMaximumRetainedLaunchBytes - retained_bytes) {
    throw HostLaunchResolutionError(
        "retained host launch byte quota is exhausted");
  }
}

ResolvedLaunchSpec TrainVMService::bind_worker_launch(
    const WorkerLaunchTicket& launch) {
  if (launch.run_id.empty()) {
    throw std::invalid_argument(
        "host launch binding requires a run identity");
  }
  std::scoped_lock lock(command_mutex_);
  const AuthorityTimeSample now = authority_now();
  prune_retained_launches(now);
  const auto projection = journal_.projection(launch.run_id);
  if (!projection) {
    throw std::invalid_argument(
        "cannot bind a host launch for an unknown run");
  }
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot bind a host launch without its persisted compiled plan");
  }

  // Portable adapter authority is revalidated before any host path is opened.
  // Host launch profiles may narrow it, but can never repair or override skew.
  adapter_registry_.validate_plan(*plan);
  const auto created = journal_.event(launch.run_id + ":created");
  if (!created || created->event_type != "run.created" ||
      !created->payload.contains("submission")) {
    throw std::runtime_error(
        "run has no durable adapter lock identity");
  }
  adapter_registry_.validate_submission_lock(
      *plan, created->payload.at("submission"));

  Controller controller(*plan, journal_, launch.run_id);
  controller.recover();
  const ExecutionState& state = controller.state();
  if (state.status != ExecutionStatus::running ||
      state.current_node_id.empty() || state.current_attempt_id.empty()) {
    throw std::logic_error(
        "host launch binding requires an active external attempt");
  }
  const Node& node =
      plan->experiment.spec.workflow.nodes.at(state.current_node_id);
  const Component& component =
      plan->experiment.spec.components.at(node.invoke.component);
  if (component.runtime == ComponentRuntime::builtin) {
    throw std::logic_error(
        "builtin operations cannot receive a host launch binding");
  }
  const Operation& operation = component.operations.at(node.invoke.operation);
  const AdapterKey key{
      .adapter = component.adapter,
      .version = component.version,
      .runtime = component.runtime,
      .operation = node.invoke.operation,
      .contract = operation.contract,
  };
  const WorkerLaunchRequest request =
      training_components_.augment_worker_launch_request(
          adapter_registry_.worker_launch_request(component,
                                                  node.invoke.operation),
          node.invoke.training);
  const std::string launch_id =
      launch.run_id + ":worker-launch:" + state.current_node_id + ":" +
      state.current_attempt_id;
  const auto durable_launch = journal_.event(launch_id);
  const auto active_lease =
      journal_.active_lease(launch.concurrency_key, now);
  nlohmann::json expected_payload{
      {"launch_nonce", launch.launch_nonce},
      {"adapter", launch.adapter},
      {"adapter_version", launch.adapter_version},
      {"code_fingerprint", launch.code_fingerprint},
      {"required_capabilities", launch.required_capabilities},
      {"concurrency_key", launch.concurrency_key},
      {"lease_id", launch.lease_id},
      {"fencing_token", launch.fencing_token},
  };
  if (launch.host_grant) {
    expected_payload["host_grant"] = encode_json(*launch.host_grant);
  }
  if (launch.node_id != state.current_node_id ||
      launch.attempt_id != state.current_attempt_id ||
      launch.adapter != key.adapter ||
      launch.adapter_version != key.version ||
      launch.code_fingerprint != request.code_fingerprint ||
      launch.required_capabilities != request.required_capabilities ||
      !active_lease || active_lease->owner_run_id != launch.run_id ||
      active_lease->lease_id != launch.lease_id ||
      active_lease->fencing_token != launch.fencing_token ||
      !durable_launch || durable_launch->event_id != launch_id ||
      durable_launch->event_type != "worker.launch_requested" ||
      durable_launch->run_id != launch.run_id ||
      durable_launch->node_id != launch.node_id ||
      durable_launch->attempt_id != launch.attempt_id ||
      durable_launch->payload != expected_payload) {
    throw AdapterResolutionError(
        "host launch ticket disagrees with portable adapter authority or durable request");
  }

  if (const auto retained = resolved_launches_.find(launch_id);
      retained != resolved_launches_.end()) {
    const ResolvedLaunchSpec durable = controller.bind_worker_launch(
        retained->second, host_launch_registry_, authority_host_,
        now);
    if (durable != retained->second.spec()) {
      throw std::runtime_error(
          "durable host launch binding disagrees with retained authority bundle");
    }
    return durable;
  }

  std::optional<GpuTraceCapture> profiler_capture;
  std::string profiler_output_path;
  if (plan->experiment.spec.execution &&
      plan->experiment.spec.execution->component == node.invoke.component &&
      plan->experiment.spec.execution->operation == node.invoke.operation &&
      plan->experiment.spec.execution->gpu_trace &&
      plan->experiment.spec.execution->gpu_trace->enabled) {
    profiler_capture = plan->experiment.spec.execution->gpu_trace;
    if (profiler_capture->backend &&
        *profiler_capture->backend != ProfilerBackend::torch) {
      profiler_output_path =
          (std::filesystem::path(plan->experiment.spec.workspace.run_directory) /
           "trainvm_artifacts" / "gpu_traces" / ".external" /
           sha256_hex(launch_id))
              .string();
    }
  }
  ResolvedLaunch resolved = host_launch_resolver_.resolve(
      launch, key, std::move(profiler_capture),
      std::move(profiler_output_path));
  require_retained_launch_capacity(resolved.spec());
  const ResolvedLaunchSpec durable = controller.bind_worker_launch(
      resolved, host_launch_registry_, authority_host_, authority_now());
  if (durable != resolved.spec()) {
    throw std::runtime_error(
        "durable host launch binding disagrees with resolved authority bundle");
  }
  const auto [retained, inserted] =
      resolved_launches_.emplace(launch_id, std::move(resolved));
  if (!inserted || retained->second.spec() != durable) {
    throw std::runtime_error(
        "host launch bundle cache rejected an exact committed binding");
  }
  return durable;
}

HostProcessSagaSnapshot TrainVMService::launch_worker_process(
    const std::string& launch_id) {
  if (!host_process_saga_) {
    throw OperationPreconditionError(
        "host process authority is disabled for this service");
  }
  std::scoped_lock lock(command_mutex_);
  const auto retained = resolved_launches_.find(launch_id);
  if (retained == resolved_launches_.end()) {
    throw OperationPreconditionError(
        "host process launch has no retained sealed descriptor authority");
  }
  const auto& identity = retained->second.spec().identity;
  if (!identity.host_grant) {
    throw OperationPreconditionError(
        "host process launch has no durable physical grant claim");
  }
  const auto grant = journal_.host_grant_saga(
      identity.host_grant->request_id);
  if (!grant || !grant->grant ||
      grant->grant->receipt_digest != identity.host_grant->grant_digest ||
      grant->grant->fences != identity.host_grant->fences) {
    throw OperationPreconditionError(
        "host process launch cannot recover its exact physical grant");
  }
  const auto projection = journal_.projection(identity.run_id);
  const auto plan =
      projection ? journal_.compiled_plan(projection->plan_hash) : std::nullopt;
  if (!projection)
    throw OperationPreconditionError(
        "host process launch cannot recover its exact resource policy: no "
        "projection for run " + identity.run_id);
  if (!plan)
    throw OperationPreconditionError(
        "host process launch cannot recover its exact resource policy: no "
        "compiled plan " + projection->plan_hash);
  // The node and attempt are re-derived from the recovered execution state,
  // not from the projection. A run is still `acquiring` when its first worker
  // is launched — entering the node is what the launch accomplishes — and the
  // projection deliberately clears current_node_id and current_attempt_id on
  // entering `acquiring`, setting them only once `node.entered` is journaled,
  // which happens after the worker reports ready. Comparing the sealed launch
  // against the projection therefore compared "train" against "" on every
  // first launch and could never hold. reconcile_host_grant already recovers
  // the controller for exactly this question, one step earlier in the same
  // reconciliation.
  Controller recovered(*plan, journal_, identity.run_id);
  recovered.recover();
  if (recovered.state().current_node_id != identity.node_id)
    throw OperationPreconditionError(
        "host process launch cannot recover its exact resource policy: the "
        "sealed launch names node " + identity.node_id +
        "; the recovered run is at " + recovered.state().current_node_id);
  if (recovered.state().current_attempt_id != identity.attempt_id)
    throw OperationPreconditionError(
        "host process launch cannot recover its exact resource policy: the "
        "sealed launch names attempt " + identity.attempt_id +
        "; the recovered run is at " + recovered.state().current_attempt_id);
  const LinuxProcessPolicy process_policy = compile_linux_process_policy(
      plan->experiment.spec.resources.cpu_io_policy);
  return host_process_saga_->reconcile(
      retained->second, *grant->grant, process_policy, controller_target_,
      authority_now());
}

bool TrainVMService::claim_worker_attempt(const std::string& key) {
  std::scoped_lock lock(worker_sessions_mutex_);
  return active_worker_attempts_.insert(key).second;
}

void TrainVMService::release_worker_attempt(const std::string& key) {
  std::scoped_lock lock(worker_sessions_mutex_);
  active_worker_attempts_.erase(key);
}

grpc::Status TrainVMService::open_worker_connection(
    const v1::WorkerHello& wire_hello, WorkerConnection& connection) {
  if (wire_hello.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker hello exceeds 64 KiB"};
  }
  try {
    WorkerHelloEvidence hello = worker_hello_evidence(wire_hello);
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(hello.run_id);
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "worker run does not exist"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no persisted compiled plan"};
    }
    Controller controller(*plan, journal_, hello.run_id);
    controller.recover();

    const auto created = journal_.event(hello.run_id + ":created");
    if (!created || created->event_type != "run.created" ||
        !created->payload.contains("submission")) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no durable authority lock identity"};
    }
    adapter_registry_.validate_submission_lock(
        *plan, created->payload.at("submission"));
    training_components_.validate_submission_lock(
        *plan, created->payload.at("submission"));

    const std::string dispatch_id =
        hello.run_id + ":dispatch:" + hello.node_id + ":" + hello.attempt_id;
    const auto historical_dispatch = journal_.dispatch(dispatch_id);
    if (historical_dispatch &&
        historical_dispatch->status == DispatchStatus::completed) {
      const std::string launch_id = hello.run_id + ":worker-launch:" +
                                    hello.node_id + ":" + hello.attempt_id;
      const auto launch = journal_.event(launch_id);
      const auto ready = journal_.event(launch_id + ":ready");
      const auto result = historical_dispatch->result_event_id
                              ? journal_.event(*historical_dispatch->result_event_id)
                              : std::nullopt;
      std::vector<std::string> ready_capabilities;
      if (ready && ready->payload.contains("capabilities")) {
        ready_capabilities = ready->payload.at("capabilities")
                                 .get<std::vector<std::string>>();
      }
      if (!launch || launch->event_type != "worker.launch_requested" ||
          !ready || ready->event_type != "worker.ready" || !result ||
          launch->node_id != hello.node_id ||
          launch->attempt_id != hello.attempt_id ||
          ready->payload.value("cause_event_id", std::string{}) != launch_id ||
          ready->payload.value("launch_nonce", std::string{}) !=
              hello.launch_nonce ||
          ready->payload.value("adapter", std::string{}) != hello.adapter ||
          ready->payload.value("adapter_version", std::string{}) !=
              hello.adapter_version ||
          ready->payload.value("code_fingerprint", std::string{}) !=
              hello.code_fingerprint ||
          ready_capabilities != hello.capabilities ||
          ready->payload.value("last_acked_controller_sequence",
                               std::uint64_t{}) !=
              hello.last_acked_controller_sequence ||
          ready->payload.value("concurrency_key", std::string{}) !=
              hello.concurrency_key ||
          ready->payload.value("lease_id", std::string{}) != hello.lease_id ||
          ready->payload.value("fencing_token", std::uint64_t{}) !=
              hello.fencing_token ||
          result->run_id != hello.run_id || result->node_id != hello.node_id ||
          result->attempt_id != hello.attempt_id ||
          result->worker_sequence == 0U) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "completed worker attempt disagrees with its durable session"};
      }
      connection.identity = worker_session(hello);
      connection.dispatch = *historical_dispatch;
      const auto invocation =
          journal_.worker_invocation(historical_dispatch->dispatch_id);
      if (!invocation) {
        return {grpc::StatusCode::DATA_LOSS,
                "completed worker attempt has no immutable invocation"};
      }
      connection.publishes = invocation->publishes;
      connection.attempt_baseline_optimizer_step =
          invocation_attempt_baseline_optimizer_step(*invocation);
      auto& welcome = connection.welcome;
      welcome.set_disposition(
          v1::WorkerWelcome::DISPOSITION_ALREADY_COMPLETED);
      welcome.set_journal_id(journal_.journal_id());
      welcome.set_plan_hash(projection->plan_hash);
      welcome.set_plan_revision(historical_dispatch->plan_revision);
      welcome.set_run_id(hello.run_id);
      welcome.set_run_revision(historical_dispatch->run_revision);
      welcome.set_node_id(hello.node_id);
      welcome.set_attempt_id(hello.attempt_id);
      welcome.set_launch_nonce(hello.launch_nonce);
      welcome.set_concurrency_key(hello.concurrency_key);
      welcome.set_lease_id(hello.lease_id);
      welcome.set_fencing_token(hello.fencing_token);
      welcome.set_dispatch_id(historical_dispatch->dispatch_id);
      welcome.set_component(historical_dispatch->component);
      welcome.set_operation(historical_dispatch->operation);
      welcome.set_acknowledged_worker_sequence(result->worker_sequence);
      welcome.set_step_zero_eval_gate_required(
          invocation_requires_step_zero_eval_gate(invocation->publishes));
      welcome.set_attempt_baseline_optimizer_step(
          connection.attempt_baseline_optimizer_step);
      welcome.set_step_zero_eval_gate_satisfied(
          durable_attempt_baseline_eval_gate_satisfied(
              journal_.events_for_run(hello.run_id), hello.run_id,
              hello.node_id, hello.attempt_id,
              connection.attempt_baseline_optimizer_step));
      populate_invocation(welcome, *invocation);
      v1::WorkerReceipt receipt;
      receipt.set_event_id(result->event_id);
      receipt.set_acknowledged_worker_sequence(result->worker_sequence);
      receipt.set_run_id(hello.run_id);
      receipt.set_committed_run_revision(projection->run_revision);
      receipt.set_observed_state(observed_state(projection->observed_state));
      receipt.set_next_node_id(projection->current_node_id);
      receipt.set_next_attempt_id(projection->current_attempt_id);
      connection.completed_receipt = std::move(receipt);
      return grpc::Status::OK;
    }

    const std::string launch_id =
        hello.run_id + ":worker-launch:" + hello.node_id + ":" +
        hello.attempt_id;
    const auto binding = journal_.launch_binding(launch_id);
    const auto retained = resolved_launches_.find(launch_id);
    if (!binding || retained == resolved_launches_.end() ||
        retained->second.spec() != *binding ||
        binding->identity.host != authority_host_ ||
        binding->identity.host_registry_digest !=
            host_launch_registry_.registry_digest()) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker readiness requires the current authority's exact retained host launch bundle"};
    }
    try {
      if (binding->identity.host_profile_digest !=
          host_launch_registry_.profile_digest(
              binding->identity.adapter_key,
              binding->identity.code_fingerprint)) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "worker host launch profile has drifted from current authority"};
      }
    } catch (const HostLaunchResolutionError& exception) {
      return {grpc::StatusCode::FAILED_PRECONDITION, exception.what()};
    }

    const WorkerReadinessResult readiness =
        controller.accept_worker_hello(hello, authority_now());
    const Dispatch dispatch = controller.prepare_dispatch(authority_now());
    const auto ready_projection = journal_.projection(hello.run_id);
    if (!ready_projection || ready_projection->observed_state != "running" ||
        ready_projection->current_node_id != hello.node_id ||
        ready_projection->current_attempt_id != hello.attempt_id ||
        ready_projection->run_revision != dispatch.run_revision) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker readiness projection disagrees with durable dispatch"};
    }
    connection.identity = worker_session(hello);
    connection.dispatch = dispatch;
    auto invocation = journal_.worker_invocation(dispatch.dispatch_id);
    if (!invocation) {
      const EffectiveControlSnapshot controls =
          journal_.effective_controls(hello.run_id);
      const Node& invocation_node =
          plan->experiment.spec.workflow.nodes.at(hello.node_id);
      nlohmann::json resolved_training = nullptr;
      if (invocation_node.invoke.training) {
        resolved_training = resolved_training_composition_json(
            training_components_.resolve_composition(
                *invocation_node.invoke.training));
      }
      WorkerInvocationContext context{
          .run_id = hello.run_id,
          .node_id = hello.node_id,
          .attempt_id = hello.attempt_id,
          .dispatch_id = dispatch.dispatch_id,
          .plan_revision = dispatch.plan_revision,
          .host_id = authority_host_.host_id,
          .artifacts = invocation_artifacts(
              journal_, *plan, hello.run_id, hello.node_id),
          .effective_controls = controls.values,
          .effective_control_revision = controls.revision,
          .resolved_training = std::move(resolved_training),
          .resume = invocation_resume_checkpoint(
              journal_, hello.run_id, hello.node_id, hello.attempt_id),
      };
      const Component& invocation_component =
          plan->experiment.spec.components.at(
              invocation_node.invoke.component);
      const Operation& invocation_operation =
          invocation_component.operations.at(invocation_node.invoke.operation);
      const AdapterKey invocation_key{
          .adapter = invocation_component.adapter,
          .version = invocation_component.version,
          .runtime = invocation_component.runtime,
          .operation = invocation_node.invoke.operation,
          .contract = invocation_operation.contract,
      };
      const FinalizationPolicyRegistry finalization_registry(
          {adapter_registry_.resolve(invocation_key)});
      const WorkerInvocationSpec candidate =
          build_worker_invocation(
              *plan, context,
              finalization_policy_digest(
                  finalization_registry.resolve(invocation_key)));
      invocation = controller.bind_worker_invocation(
          candidate, connection.identity, authority_now());
    }
    auto& welcome = connection.welcome;
    connection.publishes = invocation->publishes;
    connection.attempt_baseline_optimizer_step =
        invocation_attempt_baseline_optimizer_step(*invocation);
    welcome.set_disposition(
        readiness.disposition == WorkerReadinessDisposition::accepted
            ? v1::WorkerWelcome::DISPOSITION_ACCEPTED
            : v1::WorkerWelcome::DISPOSITION_REPLAYED);
    welcome.set_journal_id(journal_.journal_id());
    welcome.set_plan_hash(ready_projection->plan_hash);
    welcome.set_plan_revision(dispatch.plan_revision);
    welcome.set_run_id(hello.run_id);
    welcome.set_run_revision(dispatch.run_revision);
    welcome.set_node_id(hello.node_id);
    welcome.set_attempt_id(hello.attempt_id);
    welcome.set_launch_nonce(hello.launch_nonce);
    welcome.set_concurrency_key(hello.concurrency_key);
    welcome.set_lease_id(hello.lease_id);
    welcome.set_fencing_token(hello.fencing_token);
    welcome.set_dispatch_id(dispatch.dispatch_id);
    welcome.set_component(dispatch.component);
    welcome.set_operation(dispatch.operation);
    welcome.set_acknowledged_worker_sequence(journal_.latest_worker_sequence(
        hello.run_id, hello.node_id, hello.attempt_id));
    welcome.set_step_zero_eval_gate_required(
        invocation_requires_step_zero_eval_gate(invocation->publishes));
    welcome.set_attempt_baseline_optimizer_step(
        connection.attempt_baseline_optimizer_step);
    welcome.set_step_zero_eval_gate_satisfied(
        durable_attempt_baseline_eval_gate_satisfied(
            journal_.events_for_run(hello.run_id), hello.run_id,
            hello.node_id, hello.attempt_id,
            connection.attempt_baseline_optimizer_step));
    populate_invocation(welcome, *invocation);
    return grpc::Status::OK;
  } catch (const nlohmann::json::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::complete_worker_connection(
    const v1::EventEnvelope& envelope, const WorkerConnection& connection,
    v1::WorkerReceipt& receipt) {
  if (envelope.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker result exceeds 64 KiB"};
  }
  if (envelope.journal_sequence() != 0U ||
      envelope.event_id() != connection.dispatch.dispatch_id + ":result" ||
      envelope.run_id() != connection.identity.run_id ||
      envelope.run_revision() != connection.dispatch.run_revision ||
      envelope.plan_revision() != connection.dispatch.plan_revision ||
      envelope.node_id() != connection.identity.node_id ||
      envelope.attempt_id() != connection.identity.attempt_id ||
      envelope.worker_sequence() == 0U || envelope.event_type().empty() ||
      envelope.event_type().size() > 256U || envelope.event_version() != 1U ||
      !envelope.has_wall_time() || envelope.has_payload() ||
      envelope.canonical_json_payload().empty() ||
      envelope.canonical_json_payload().size() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "worker result envelope is not the canonical dispatched result"};
  }
  nlohmann::json payload;
  try {
    payload = nlohmann::json::parse(envelope.canonical_json_payload());
    if (!payload.is_object() || payload.dump() != envelope.canonical_json_payload()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "worker result payload must be a canonical JSON object"};
    }
  } catch (const nlohmann::json::exception& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  }
  try {
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(connection.identity.run_id);
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "worker run does not exist"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no persisted compiled plan"};
    }
    if (envelope.has_optimizer_step() &&
        envelope.optimizer_step() <
            connection.attempt_baseline_optimizer_step) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker result optimizer step precedes its immutable attempt "
              "baseline"};
    }
    if (invocation_requires_step_zero_eval_gate(connection.publishes) &&
        envelope.event_type() == "worker.completed" &&
        !durable_attempt_baseline_eval_gate_satisfied(
            journal_.events_for_run(connection.identity.run_id),
            connection.identity.run_id,
            connection.identity.node_id, connection.identity.attempt_id,
            connection.attempt_baseline_optimizer_step)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker completion is blocked until durable attempt-baseline "
              "scalar and eval-examples evidence"};
    }
    Controller controller(*plan, journal_, connection.identity.run_id);
    controller.recover();
    if (envelope.event_type() == "worker.completed") {
      const auto invocation =
          journal_.worker_invocation(connection.dispatch.dispatch_id);
      if (!invocation) {
        return {grpc::StatusCode::DATA_LOSS,
                "worker completion has no immutable invocation"};
      }
      const std::vector<Event> durable =
          journal_.events_for_run(connection.identity.run_id);
      const auto matches_declaration =
          [](const nlohmann::json& artifact,
             const nlohmann::json& publication) {
            const auto& declaration = publication.at("declaration");
            const auto schema = declaration.find("schema");
            return artifact.is_object() &&
                   artifact.value("complete", false) &&
                   artifact.value("logical_name", std::string{}) ==
                       publication.value("logical_name", std::string{}) &&
                   artifact.value("kind", std::string{}) ==
                       declaration.value("type", std::string{}) &&
                   artifact.value("fingerprint_algorithm", std::string{}) ==
                       declaration.value("fingerprint", std::string{}) &&
                   (schema == declaration.end() || schema->is_null() ||
                    (schema->is_string() &&
                     artifact.value("schema", std::string{}) ==
                         schema->get<std::string>()));
          };
      const auto current_artifact =
          [&](const nlohmann::json& publication,
              const std::optional<std::string>& required_parent =
                  std::nullopt) -> const Event* {
        const auto match = std::ranges::find_if(
            durable, [&](const Event& candidate) {
              if (candidate.event_type != "artifact.published" ||
                  candidate.node_id != connection.identity.node_id ||
                  candidate.attempt_id != connection.identity.attempt_id ||
                  !matches_declaration(candidate.payload, publication)) {
                return false;
              }
              if (!required_parent) return true;
              const auto parents =
                  candidate.payload.find("parent_artifact_ids");
              return parents != candidate.payload.end() &&
                     parents->is_array() &&
                     std::ranges::any_of(
                         *parents, [&](const nlohmann::json& parent) {
                           return parent.is_string() &&
                                  parent.get_ref<const std::string&>() ==
                                      *required_parent;
                         });
            });
        return match == durable.end() ? nullptr : &*match;
      };

      const nlohmann::json* selected_resume_checkpoint = nullptr;
      std::string selected_resume_artifact_id;
      std::optional<std::uint64_t> declared_terminal_step;
      if (invocation->training.is_object()) {
        const auto components = invocation->training.find("components");
        if (components != invocation->training.end() &&
            components->is_object()) {
          const auto learning_rate = components->find("learning_rate");
          if (learning_rate != components->end() &&
              learning_rate->is_object()) {
            const auto configuration = learning_rate->find("configuration");
            if (configuration != learning_rate->end() &&
                configuration->is_object()) {
              const auto maximum = configuration->find("max_steps");
              if (maximum != configuration->end() &&
                  maximum->is_number_unsigned())
                declared_terminal_step = maximum->get<std::uint64_t>();
            }
          }
        }
      }
      bool terminal_resume_checkpoint_reuse = false;
      if (invocation->resume.is_object() &&
          invocation->resume.value("api_version", std::string{}) ==
              "trainvm.resume-checkpoint/v1" &&
          invocation->resume.contains("checkpoint") &&
          invocation->resume.at("checkpoint").is_object() &&
          invocation->resume.contains("optimizer_step") &&
          invocation->resume.at("optimizer_step").is_number_unsigned() &&
          envelope.has_optimizer_step() &&
          envelope.optimizer_step() ==
              invocation->resume.at("optimizer_step").get<std::uint64_t>() &&
          declared_terminal_step &&
          envelope.optimizer_step() == *declared_terminal_step) {
        selected_resume_checkpoint = &invocation->resume.at("checkpoint");
        selected_resume_artifact_id =
            selected_resume_checkpoint->value("artifact_id", std::string{});
        terminal_resume_checkpoint_reuse =
            !selected_resume_artifact_id.empty();
      }

      std::size_t required_noncheckpoint_outputs = 0U;
      bool resume_children_are_current_and_bound =
          terminal_resume_checkpoint_reuse;
      for (const auto& [output_name, publication] :
           invocation->publishes.items()) {
        (void)output_name;
        const auto declaration = publication.find("declaration");
        if (declaration == publication.end() || !declaration->is_object() ||
            !declaration->value("required", false) ||
            declaration->value("type", std::string{}) == "checkpoint") {
          continue;
        }
        ++required_noncheckpoint_outputs;
        if (!current_artifact(publication, selected_resume_artifact_id)) {
          resume_children_are_current_and_bound = false;
        }
      }
      terminal_resume_checkpoint_reuse =
          terminal_resume_checkpoint_reuse &&
          required_noncheckpoint_outputs > 0U &&
          resume_children_are_current_and_bound;

      for (const auto& [output_name, publication] :
           invocation->publishes.items()) {
        (void)output_name;
        const auto& declaration =
            publication.contains("declaration")
                ? publication.at("declaration")
                : nlohmann::json{};
        if (!declaration.is_object() ||
            !declaration.value("required", false)) {
          continue;
        }
        const std::string logical_name =
            publication.value("logical_name", std::string{});
        const std::string required_type =
            declaration.value("type", std::string{});
        const std::string required_fingerprint =
            declaration.value("fingerprint", std::string{});
        const bool published = current_artifact(publication) != nullptr;
        const bool selected_resume_satisfies_checkpoint =
            !published && terminal_resume_checkpoint_reuse &&
            required_type == "checkpoint" &&
            selected_resume_checkpoint != nullptr &&
            matches_declaration(*selected_resume_checkpoint, publication);
        if (logical_name.empty() || required_type.empty() ||
            required_fingerprint.empty() ||
            (!published && !selected_resume_satisfies_checkpoint)) {
          throw OperationPreconditionError(
              "worker completion is missing required operation output " +
              logical_name);
        }
      }

      const AdapterProfile& completion_profile =
          adapter_registry_.resolve(invocation->adapter);
      const FinalizationPolicyRegistry completion_registry(
          {completion_profile});
      const OperationFinalizationPolicy& finalization_policy =
          completion_registry.resolve(invocation->adapter);
      if (finalization_policy.closure_required) {
        if (finalization_policy.migration_pending ||
            !envelope.has_optimizer_step() ||
            (declared_terminal_step &&
             envelope.optimizer_step() != *declared_terminal_step)) {
          const FinalizationVerdict verdict{
              .disposition = FinalizationDisposition::pending,
              .cause = "terminal optimizer-step authority is incomplete",
              .unresolved_members = {},
              .selected_artifact_id = std::nullopt,
              .selected_artifact_fingerprint = std::nullopt};
          controller.record_finalization_verdict(
              verdict,
              envelope.has_optimizer_step()
                  ? std::optional<std::uint64_t>{envelope.optimizer_step()}
                  : std::nullopt,
              authority_now());
          throw OperationPreconditionError(
              "finalization_pending: " + verdict.cause);
        }
        FinalizationVerdict verdict;
        try {
          FinalEvaluationExpectation expectation;
          const auto frozen = journal_.event(
              connection.dispatch.dispatch_id + ":finalization-expectation");
          if (frozen) {
            std::vector<Diagnostic> diagnostics;
            if (frozen->event_type != "finalization.expectation_frozen" ||
                !decode_json(frozen->payload, expectation, "", diagnostics) ||
                !diagnostics.empty() ||
                encode_json(expectation) != frozen->payload) {
              throw std::runtime_error(
                  "durable finalization expectation freeze is malformed");
            }
          } else {
            if (!declared_terminal_step) {
              const FinalizationVerdict pending{
                  .disposition = FinalizationDisposition::pending,
                  .cause = "immutable terminal step is undeclared",
                  .unresolved_members = {},
                  .selected_artifact_id = std::nullopt,
                  .selected_artifact_fingerprint = std::nullopt};
              controller.record_finalization_verdict(
                  pending, envelope.optimizer_step(), authority_now());
              throw OperationPreconditionError(
                  "finalization_pending: " + pending.cause);
            }
            expectation = controller.freeze_final_evaluation_expectation(
                derive_hf_final_evaluation_expectation(
                    finalization_policy, *invocation,
                    envelope.optimizer_step(), durable),
                authority_now());
          }
          const auto finalization_history =
              resolve_final_evaluation_receipts(*invocation, expectation,
                                                durable);
          verdict = reduce_final_evaluation(
              finalization_policy, expectation, finalization_history);
        } catch (const std::invalid_argument& exception) {
          verdict = {.disposition = FinalizationDisposition::failed,
                     .cause = exception.what(),
                     .unresolved_members = {},
                     .selected_artifact_id = std::nullopt,
                     .selected_artifact_fingerprint = std::nullopt};
        }
        controller.record_finalization_verdict(
            verdict, envelope.optimizer_step(), authority_now());
        if (verdict.disposition != FinalizationDisposition::complete) {
          const std::string state =
              verdict.disposition == FinalizationDisposition::pending
                  ? "finalization_pending: "
                  : "finalization_failed: ";
          throw OperationPreconditionError(state + verdict.cause);
        }
      }
    }
    Event event{
        .event_id = envelope.event_id(),
        .run_id = envelope.run_id(),
        .run_revision = envelope.run_revision(),
        .plan_revision = envelope.plan_revision(),
        .node_id = envelope.node_id(),
        .attempt_id = envelope.attempt_id(),
        .worker_sequence = envelope.worker_sequence(),
        .event_type = envelope.event_type(),
        .event_version = envelope.event_version(),
        .wall_time_ns = timestamp_ns(envelope.wall_time()),
        .monotonic_time_ns = envelope.monotonic_time_ns(),
        .optimizer_step = envelope.has_optimizer_step()
                              ? std::optional<std::uint64_t>{
                                    envelope.optimizer_step()}
                              : std::nullopt,
        .payload = payload,
    };
    const std::uint64_t latest = journal_.latest_worker_sequence(
        event.run_id, event.node_id, event.attempt_id);
    const auto stored = journal_.event(event.event_id);
    event.run_revision = stored ? stored->run_revision
                                : projection->run_revision;
    if (latest == std::numeric_limits<std::uint64_t>::max() ||
        event.worker_sequence > latest + 1U ||
        (event.worker_sequence <= latest && !stored)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker result sequence is not the next durable message or an exact replay"};
    }
    const ExecutionState& committed =
        controller.handle_event(event, connection.identity, authority_now());
    const auto committed_projection =
        journal_.projection(connection.identity.run_id);
    if (!committed_projection ||
        committed_projection->run_revision != committed.revision) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker result projection disagrees with committed FSM state"};
    }
    receipt.set_event_id(event.event_id);
    receipt.set_acknowledged_worker_sequence(event.worker_sequence);
    receipt.set_run_id(event.run_id);
    receipt.set_committed_run_revision(committed.revision);
    receipt.set_observed_state(
        observed_state(committed_projection->observed_state));
    receipt.set_next_node_id(committed_projection->current_node_id);
    receipt.set_next_attempt_id(committed_projection->current_attempt_id);
    const std::string launch_id = connection.identity.run_id +
                                  ":worker-launch:" +
                                  connection.identity.node_id + ":" +
                                  connection.identity.attempt_id;
    resolved_launches_.erase(launch_id);
    notify_reconciliation(connection.identity.run_id);
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::commit_worker_observation(
    Event event, const WorkerConnection& connection,
    std::uint64_t& acknowledged) {
  try {
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(connection.identity.run_id);
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "worker run does not exist"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no persisted compiled plan"};
    }
    if (event.optimizer_step &&
        *event.optimizer_step < connection.attempt_baseline_optimizer_step) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker observation optimizer step precedes its immutable "
              "attempt baseline"};
    }
    if (invocation_requires_step_zero_eval_gate(connection.publishes) &&
        event.optimizer_step &&
        *event.optimizer_step > connection.attempt_baseline_optimizer_step &&
        !durable_attempt_baseline_eval_gate_satisfied(
            journal_.events_for_run(event.run_id), event.run_id,
            event.node_id, event.attempt_id,
            connection.attempt_baseline_optimizer_step)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "optimizer step is blocked until durable attempt-baseline "
              "scalar and eval-examples evidence"};
    }
    if (event.event_type == "metric.sampled") {
      const std::string name =
          event.payload.value("name", std::string{});
      const auto& declared = plan->canonical_plan.at("spec")
                                 .at("observability")
                                 .at("metrics");
      const auto match = std::ranges::find_if(
          declared, [&](const nlohmann::json& metric) {
            return metric.value("name", std::string{}) == name;
          });
      if (match == declared.end()) {
        throw std::invalid_argument(
            "worker metric is not present in the sealed observability declaration");
      }
      if (match->value("unit", std::string{}) !=
              event.payload.value("unit", std::string{}) ||
          match->value("step_domain", std::string{}) !=
              event.payload.value("step_domain", std::string{})) {
        throw std::invalid_argument(
            "worker metric unit or step domain disagrees with the sealed observability declaration");
      }
    }
    if (event.event_type == "artifact.published") {
      const auto& parents = event.payload.at("parent_artifact_ids");
      const std::vector<Event> durable = journal_.events_for_run(event.run_id);
      for (const nlohmann::json& parent : parents) {
        const std::string parent_id = parent.get<std::string>();
        const auto matches = std::ranges::count_if(
            durable, [&](const Event& candidate) {
              return candidate.event_type == "artifact.published" &&
                     candidate.payload.is_object() &&
                     candidate.payload.value("complete", false) &&
                     candidate.payload.value("artifact_id", std::string{}) ==
                         parent_id;
            });
        if (matches != 1) {
          throw std::invalid_argument(
              "artifact parent identity is not one prior durable artifact in the current run");
        }
      }
    }
    const std::uint64_t latest = journal_.latest_worker_sequence(
        connection.identity.run_id, connection.identity.node_id,
        connection.identity.attempt_id);
    const auto stored = journal_.event(event.event_id);
    event.run_revision = stored ? stored->run_revision
                                : projection->run_revision;
    if (latest == std::numeric_limits<std::uint64_t>::max() ||
        event.worker_sequence > latest + 1U ||
        (event.worker_sequence <= latest && !stored)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker observation sequence is not the next durable message or an exact replay"};
    }
    const AuthorityTimeSample now = authority_now();
    if (event.event_type == "worker.heartbeat") {
      event.wall_time_ns = stored ? stored->wall_time_ns : now.wall.nanoseconds;
    }
    Controller controller(*plan, journal_, connection.identity.run_id);
    controller.recover();
    (void)controller.record_worker_observation(
        event, connection.identity, now);
    acknowledged = event.worker_sequence;
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::record_worker_heartbeat(
    const v1::WorkerHeartbeat& heartbeat,
    const WorkerConnection& connection, std::uint64_t& acknowledged) {
  if (heartbeat.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker heartbeat exceeds 64 KiB"};
  }
  try {
    if (heartbeat.worker_sequence() == 0U || heartbeat.phase().empty() ||
        !heartbeat.has_observed_at()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "worker heartbeat requires sequence, phase, and observed_at"};
    }

    // `phase` is an operator label and stays free text. `execution_phase` is
    // the routed one, so it is checked against this attempt's immutable
    // requests rather than believed. Two separate refusals below: an unknown
    // or unrequested phase, and a label that claims a phase the typed field
    // does not. Without the second, a worker could report `phase: "compile"`
    // while the authority never requested compilation, and every downstream
    // reader of the journal would show a compile it never authorized.
    const bool in_phase = heartbeat.execution_phase() !=
                          v1::WorkerExecutionPhaseRequest::PHASE_UNSPECIFIED;
    std::string execution_phase_label;
    if (in_phase) {
      const auto name = execution_phase_name(heartbeat.execution_phase());
      if (!name) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "worker heartbeat names an unknown execution phase"};
      }
      const v1::WorkerExecutionPhaseRequest* const request =
          find_phase_request(connection.welcome, heartbeat.execution_phase());
      if (request == nullptr || !request->enabled()) {
        return {grpc::StatusCode::PERMISSION_DENIED,
                "worker heartbeat claims an execution phase this attempt did "
                "not request"};
      }
      execution_phase_label.assign(*name);
    }
    if ((heartbeat.phase() == "compile" || heartbeat.phase() == "warmup") &&
        heartbeat.phase() != execution_phase_label) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "worker heartbeat label impersonates an execution phase its "
              "typed field does not name"};
    }
    const std::int64_t observed_at_ns = timestamp_ns(heartbeat.observed_at());
    const Event event{
        .event_id = connection.dispatch.dispatch_id + ":heartbeat:" +
                    std::to_string(heartbeat.worker_sequence()),
        .run_id = connection.identity.run_id,
        .run_revision = connection.dispatch.run_revision,
        .plan_revision = connection.dispatch.plan_revision,
        .node_id = connection.identity.node_id,
        .attempt_id = connection.identity.attempt_id,
        .worker_sequence = heartbeat.worker_sequence(),
        .event_type = "worker.heartbeat",
        .event_version = 1,
        .wall_time_ns = 0,
        .monotonic_time_ns = 0,
        .optimizer_step = heartbeat.optimizer_step(),
        .payload = {{"phase", heartbeat.phase()},
                    {"execution_phase", execution_phase_label},
                    {"observed_at_ns", observed_at_ns}},
    };
    return commit_worker_observation(event, connection, acknowledged);
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::record_worker_metric(
    const v1::MetricSample& metric, const WorkerConnection& connection,
    std::uint64_t& acknowledged) {
  if (metric.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker metric exceeds 64 KiB"};
  }
  try {
    if (metric.worker_sequence() == 0U || metric.name().empty() ||
        metric.step_domain().empty() || !metric.has_observed_at() ||
        metric.value().value_case() == v1::ScalarValue::VALUE_NOT_SET ||
        !std::isfinite(metric.sample_weight()) || metric.sample_weight() <= 0.0 ||
        (metric.value().value_case() == v1::ScalarValue::kNumberValue &&
         !std::isfinite(metric.value().number_value()))) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "worker metric requires a finite value, positive weight, identity, sequence, and timestamp"};
    }
    nlohmann::json labels = nlohmann::json::object();
    for (const auto& [key, value] : metric.labels()) {
      if (key.empty()) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "worker metric label keys must not be empty"};
      }
      labels[key] = value;
    }
    const Event event{
        .event_id = connection.dispatch.dispatch_id + ":metric:" +
                    std::to_string(metric.worker_sequence()),
        .run_id = connection.identity.run_id,
        .run_revision = connection.dispatch.run_revision,
        .plan_revision = connection.dispatch.plan_revision,
        .node_id = connection.identity.node_id,
        .attempt_id = connection.identity.attempt_id,
        .worker_sequence = metric.worker_sequence(),
        .event_type = "metric.sampled",
        .event_version = 1,
        .wall_time_ns = timestamp_ns(metric.observed_at()),
        .monotonic_time_ns = 0,
        .optimizer_step = metric.step_domain() == "optimizer_step"
                              ? std::optional<std::uint64_t>{metric.step()}
                              : std::nullopt,
        .payload = {{"name", metric.name()},
                    {"value", assignment_value(metric.value())},
                    {"unit", metric.unit()},
                    {"step_domain", metric.step_domain()},
                    {"step", metric.step()},
                    {"sample_weight", metric.sample_weight()},
                    {"labels", std::move(labels)}},
    };
    return commit_worker_observation(event, connection, acknowledged);
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::record_worker_artifact(
    const v1::ArtifactManifest& artifact,
    const WorkerConnection& connection, std::uint64_t& acknowledged) {
  if (artifact.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker artifact manifest exceeds 64 KiB"};
  }
  try {
    if (artifact.worker_sequence() == 0U || artifact.artifact_id().empty() ||
        artifact.logical_name().empty() || artifact.uri().empty() ||
        artifact.fingerprint_algorithm().empty() || artifact.fingerprint().empty() ||
        !artifact.complete() || !artifact.has_published_at() ||
        artifact.producer_node_id() != connection.identity.node_id ||
        artifact.producer_attempt_id() != connection.identity.attempt_id) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "published artifact requires complete content identity and the active producer attempt"};
    }
    const bool eval_examples =
        artifact.kind() == v1::ARTIFACT_KIND_EVAL_EXAMPLES;
    const bool checkpoint = artifact.kind() == v1::ARTIFACT_KIND_CHECKPOINT;
    const bool image_gallery =
        artifact.kind() == v1::ARTIFACT_KIND_IMAGE_GALLERY;
    const bool strict_stepped_report =
        artifact.kind() == v1::ARTIFACT_KIND_REPORT &&
        (artifact.schema() == "rwkv-lab.final-evaluation.v1" ||
         artifact.schema() ==
             "rwkv-lab.hf-test-caption-evidence-bundle.v1");
    nlohmann::json eval_examples_document = nullptr;
    std::optional<EvalExamplesManifest> eval_examples_manifest;
    std::optional<WorkerInvocationSpec> eval_examples_invocation;
    if (eval_examples) {
      if (artifact.schema() != kEvalExamplesSchema ||
          !artifact.has_optimizer_step() ||
          artifact.fingerprint_algorithm() != "manifest_sha256" ||
          artifact.canonical_manifest_json().empty() ||
          artifact.canonical_manifest_json().size() > 60U * 1024U ||
          artifact.size_bytes() != artifact.canonical_manifest_json().size()) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "eval-examples artifact requires stepped canonical manifest content"};
      }
      try {
        eval_examples_document =
            nlohmann::json::parse(artifact.canonical_manifest_json());
      } catch (const nlohmann::json::exception& exception) {
        return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
      }
      if (!eval_examples_document.is_object() ||
          eval_examples_document.dump() != artifact.canonical_manifest_json() ||
          artifact.fingerprint() !=
              "sha256:" + sha256_hex(artifact.canonical_manifest_json())) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "eval-examples manifest bytes or fingerprint are noncanonical"};
      }
      eval_examples_invocation =
          journal_.worker_invocation(connection.dispatch.dispatch_id);
      if (!eval_examples_invocation) {
        return {grpc::StatusCode::DATA_LOSS,
                "eval-examples publication has no immutable invocation"};
      }
      const std::string run_directory =
          eval_examples_invocation->workspace.value("run_directory",
                                                     std::string{});
      try {
        eval_examples_manifest =
            validate_eval_examples_manifest(eval_examples_document);
        validate_eval_examples_payload(
            *eval_examples_manifest, artifact.uri(),
            artifact.canonical_manifest_json(), artifact.fingerprint(),
            run_directory, artifact.artifact_id());
      } catch (const std::invalid_argument& exception) {
        return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
      }
      if (eval_examples_manifest->run_id != connection.identity.run_id ||
          eval_examples_manifest->node_id != connection.identity.node_id ||
          eval_examples_manifest->attempt_id !=
              connection.identity.attempt_id ||
          eval_examples_manifest->optimizer_step !=
              artifact.optimizer_step()) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "eval-examples manifest disagrees with producer or step identity"};
      }
      try {
        validate_eval_examples_gate_provenance(
            *eval_examples_manifest, eval_examples_invocation->training,
            journal_.events_for_run(connection.identity.run_id));
      } catch (const std::invalid_argument& exception) {
        return {grpc::StatusCode::FAILED_PRECONDITION, exception.what()};
      }
    } else if (((checkpoint || image_gallery || strict_stepped_report) !=
                artifact.has_optimizer_step()) ||
               !artifact.canonical_manifest_json().empty()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "checkpoint, gallery, and strict final reports require an optimizer step; "
              "other artifacts must omit eval contract fields"};
    }
    const nlohmann::json* publication = nullptr;
    if (!connection.publishes.is_object()) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker invocation publication authority is malformed"};
    }
    for (const auto& [output_name, candidate] : connection.publishes.items()) {
      (void)output_name;
      if (!candidate.is_object() || !candidate.contains("logical_name") ||
          !candidate.contains("declaration")) {
        return {grpc::StatusCode::DATA_LOSS,
                "worker invocation publication declaration is malformed"};
      }
      if (candidate.value("logical_name", std::string{}) ==
          artifact.logical_name()) {
        if (publication != nullptr) {
          return {grpc::StatusCode::DATA_LOSS,
                  "worker invocation has ambiguous publication authority"};
        }
        publication = &candidate;
      }
    }
    if (publication == nullptr) {
      return {grpc::StatusCode::PERMISSION_DENIED,
              "worker artifact is not declared by its immutable invocation"};
    }
    const auto& declaration = publication->at("declaration");
    if (!declaration.is_object() ||
        declaration.value("type", std::string{}) !=
            artifact_kind_name(artifact.kind()) ||
        declaration.value("schema", std::string{}) != artifact.schema() ||
        declaration.value("fingerprint", std::string{}) !=
            artifact.fingerprint_algorithm()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "worker artifact disagrees with its immutable output declaration"};
    }
    std::set<std::string> parents;
    nlohmann::json parent_ids = nlohmann::json::array();
    for (const auto& parent : artifact.parent_artifact_ids()) {
      if (parent.empty() || parent == artifact.artifact_id() ||
          !parents.insert(parent).second) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "artifact parent identities must be unique, nonempty, and non-self"};
      }
      parent_ids.push_back(parent);
    }
    if (eval_examples_manifest &&
        !parents.contains(eval_examples_manifest->checkpoint.artifact_id)) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "eval-examples checkpoint must be a declared artifact parent"};
    }
    const std::int64_t published_at_ns = timestamp_ns(artifact.published_at());
    nlohmann::json payload =
        {{"artifact_id", artifact.artifact_id()},
         {"logical_name", artifact.logical_name()},
         {"kind", artifact_kind_name(artifact.kind())},
         {"schema", artifact.schema()},
         {"uri", artifact.uri()},
         {"size_bytes", artifact.size_bytes()},
         {"fingerprint_algorithm", artifact.fingerprint_algorithm()},
         {"fingerprint", artifact.fingerprint()},
         {"complete", artifact.complete()},
         {"producer_node_id", artifact.producer_node_id()},
         {"producer_attempt_id", artifact.producer_attempt_id()},
         {"parent_artifact_ids", std::move(parent_ids)},
         {"published_at_ns", published_at_ns}};
    if (eval_examples) {
      payload["eval_examples_manifest"] = std::move(eval_examples_document);
    }
    const Event event{
        .event_id = connection.dispatch.dispatch_id + ":artifact:" +
                    sha256_hex(artifact.artifact_id()),
        .run_id = connection.identity.run_id,
        .run_revision = connection.dispatch.run_revision,
        .plan_revision = connection.dispatch.plan_revision,
        .node_id = connection.identity.node_id,
        .attempt_id = connection.identity.attempt_id,
        .worker_sequence = artifact.worker_sequence(),
        .event_type = "artifact.published",
        .event_version = 1,
        .wall_time_ns = published_at_ns,
        .monotonic_time_ns = 0,
        .optimizer_step =
            (eval_examples || checkpoint || image_gallery || strict_stepped_report)
                              ? std::optional<std::uint64_t>{
                                    artifact.optimizer_step()}
                              : std::nullopt,
        .payload = std::move(payload),
    };
    return commit_worker_observation(event, connection, acknowledged);
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::record_worker_execution_phase_receipt(
    const v1::WorkerExecutionPhaseReceipt& receipt,
    const WorkerConnection& connection, std::uint64_t& acknowledged) {
  if (receipt.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker execution-phase receipt exceeds 64 KiB"};
  }
  try {
    const auto valid_digest = [](std::string_view value) {
      return value.size() == 71U && value.starts_with("sha256:") &&
             std::ranges::all_of(value.substr(7U), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
             });
    };
    if (receipt.worker_sequence() == 0U ||
        !valid_digest(receipt.request_digest()) ||
        !valid_digest(receipt.state_fingerprint_before()) ||
        !valid_digest(receipt.state_fingerprint_after()) ||
        receipt.concurrency_key() != connection.identity.concurrency_key ||
        receipt.lease_id() != connection.identity.lease_id ||
        receipt.fencing_token() != connection.identity.fencing_token ||
        !receipt.has_started_at() || !receipt.has_completed_at()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "execution-phase receipt has invalid content or fenced identity"};
    }

    const auto receipt_phase_name = execution_phase_name(receipt.phase());
    if (!receipt_phase_name) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "execution-phase receipt phase is invalid"};
    }
    const std::string phase_name{*receipt_phase_name};
    const v1::WorkerExecutionPhaseRequest* request = nullptr;
    for (const auto& candidate :
         connection.welcome.execution_phase_requests()) {
      if (candidate.phase() != receipt.phase()) continue;
      if (request != nullptr) {
        return {grpc::StatusCode::DATA_LOSS,
                "worker invocation has duplicate execution-phase requests"};
      }
      request = &candidate;
    }
    if (request == nullptr ||
        request->request_digest() != receipt.request_digest()) {
      return {grpc::StatusCode::PERMISSION_DENIED,
              "execution-phase receipt is not authorized by its immutable request"};
    }

    std::string disposition;
    switch (receipt.disposition()) {
      case v1::WorkerExecutionPhaseReceipt::DISPOSITION_COMPLETED:
        disposition = "completed";
        break;
      case v1::WorkerExecutionPhaseReceipt::DISPOSITION_SKIPPED:
        disposition = "skipped";
        break;
      case v1::WorkerExecutionPhaseReceipt::DISPOSITION_FAILED:
        disposition = "failed";
        break;
      case v1::WorkerExecutionPhaseReceipt::DISPOSITION_CANCELLED:
        disposition = "cancelled";
        break;
      case v1::WorkerExecutionPhaseReceipt::DISPOSITION_UNSPECIFIED:
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "execution-phase receipt disposition is invalid"};
    }
    const nlohmann::json diagnostics = acknowledgement_diagnostics(receipt);
    const bool completed = disposition == "completed";
    const bool skipped = disposition == "skipped";
    const bool failed = disposition == "failed";
    // A cancellation stops an enabled phase at a step boundary. Nothing went
    // wrong, so unlike a failure the trajectory must still be restored — the
    // fingerprint equality below covers it. It carries diagnostics for the
    // same reason a failure does: the receipt has to say what stopped it, or
    // a partial step count in the journal is unattributable.
    const bool cancelled = disposition == "cancelled";
    if ((request->enabled() && skipped) ||
        (!request->enabled() && !skipped) ||
        ((failed || cancelled) && diagnostics.empty()) ||
        ((completed || skipped || cancelled) &&
         receipt.state_fingerprint_before() !=
             receipt.state_fingerprint_after())) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "execution-phase disposition or restored-state proof is inconsistent"};
    }
    const std::uint64_t requested_steps =
        request->has_steps() ? request->steps() : 0U;
    if ((skipped && receipt.steps_executed() != 0U) ||
        (completed && receipt.steps_executed() != requested_steps) ||
        ((failed || cancelled) &&
         receipt.steps_executed() > requested_steps)) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "execution-phase receipt step count disagrees with its request"};
    }
    const std::int64_t started_at_ns = timestamp_ns(receipt.started_at());
    const std::int64_t completed_at_ns = timestamp_ns(receipt.completed_at());
    if (completed_at_ns < started_at_ns) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "execution-phase receipt timestamps are reversed"};
    }

    const Event event{
        .event_id = connection.dispatch.dispatch_id + ":phase:" + phase_name +
                    ":" + std::to_string(receipt.worker_sequence()),
        .run_id = connection.identity.run_id,
        .run_revision = connection.dispatch.run_revision,
        .plan_revision = connection.dispatch.plan_revision,
        .node_id = connection.identity.node_id,
        .attempt_id = connection.identity.attempt_id,
        .worker_sequence = receipt.worker_sequence(),
        .event_type = "worker.execution_phase_receipted",
        .event_version = 1,
        .wall_time_ns = completed_at_ns,
        .monotonic_time_ns = 0,
        .optimizer_step = std::nullopt,
        .payload = {{"phase", phase_name},
                    {"enabled", request->enabled()},
                    {"requested_steps", requested_steps},
                    {"request_digest", receipt.request_digest()},
                    {"disposition", disposition},
                    {"steps_executed", receipt.steps_executed()},
                    {"state_fingerprint_before",
                     receipt.state_fingerprint_before()},
                    {"state_fingerprint_after",
                     receipt.state_fingerprint_after()},
                    {"started_at_ns", started_at_ns},
                    {"completed_at_ns", completed_at_ns},
                    {"diagnostics", diagnostics}},
    };
    return commit_worker_observation(event, connection, acknowledged);
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::record_worker_runtime_evidence(
    const v1::WorkerRuntimeEvidence& evidence,
    const WorkerConnection& connection) {
  if (evidence.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker runtime evidence exceeds 64 KiB"};
  }
  // Decoding is the only place the proto arm and the C++ struct meet, so it
  // is a plain field-for-field copy with nothing derived, defaulted, or
  // repaired. `worker_runtime_evidence_wire_hop` asserts the two
  // literals still describe the same document.
  WorkerRuntimeEvidenceReport report{
      .api_version = evidence.api_version(),
      .run_id = evidence.run_id(),
      .node_id = evidence.node_id(),
      .attempt_id = evidence.attempt_id(),
      .launch_nonce = evidence.launch_nonce(),
      .concurrency_key = evidence.concurrency_key(),
      .lease_id = evidence.lease_id(),
      .fencing_token = evidence.fencing_token(),
      .compute_device_vendor = evidence.compute_device_vendor(),
      .compute_architecture = evidence.compute_architecture(),
      .compute_device_uuid =
          evidence.has_compute_device_uuid()
              ? std::optional<std::string>{evidence.compute_device_uuid()}
              : std::nullopt,
      .compute_device_pci_address =
          evidence.has_compute_device_pci_address()
              ? std::optional<std::string>{
                    evidence.compute_device_pci_address()}
              : std::nullopt,
      .driver_version = evidence.driver_version(),
      .runtime_versions = {},
      .runtime_closure_fingerprint = evidence.runtime_closure_fingerprint(),
      .host_abi_digest = evidence.host_abi_digest(),
      .compute_compatibility_digest = evidence.compute_compatibility_digest(),
  };
  report.runtime_versions.reserve(
      static_cast<std::size_t>(evidence.runtime_versions_size()));
  for (const auto& version : evidence.runtime_versions()) {
    report.runtime_versions.push_back(
        {.name = version.name(), .version = version.version()});
  }
  try {
    // Shape only, and the canonical ordering the namespace digest closes over.
    (void)worker_runtime_evidence_json(report);
  } catch (const WorkerRuntimeEvidenceError& error) {
    return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
  }

  // Every refusal below happens on the wire, before a publisher exists to
  // refuse at. The three checks are deliberately separate because they answer
  // different questions.
  //
  // First: is this report even about the connection it arrived on? A worker
  // holds one attempt's stream and may only speak for that attempt.
  if (report.run_id != connection.identity.run_id ||
      report.node_id != connection.identity.node_id ||
      report.attempt_id != connection.identity.attempt_id ||
      report.launch_nonce != connection.identity.launch_nonce ||
      report.concurrency_key != connection.identity.concurrency_key ||
      report.lease_id != connection.identity.lease_id ||
      report.fencing_token != connection.identity.fencing_token) {
    return {grpc::StatusCode::PERMISSION_DENIED,
            "worker runtime evidence claims an attempt fence this connection "
            "does not hold"};
  }
  try {
    std::scoped_lock lock(command_mutex_);
    // Second: is that attempt fence still the live one? A connection's
    // identity was true when the stream opened and says nothing about now. A
    // worker whose lease moved -- released, taken by a relaunch, superseded by
    // a higher fencing token -- is still holding an open stream, and its
    // evidence would otherwise be admitted against a launch binding the
    // authority has already replaced. Re-reading the durable binding is what
    // makes late arrival a refusal here rather than at the publisher.
    const std::string launch_id =
        connection.identity.run_id + ":worker-launch:" +
        connection.identity.node_id + ":" + connection.identity.attempt_id;
    const auto binding = journal_.launch_binding(launch_id);
    if (!binding) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "worker runtime evidence has no durable launch binding to be "
              "bound to"};
    }
    const ResolvedLaunchIdentity& identity = binding->identity;
    if (identity.launch_nonce != connection.identity.launch_nonce ||
        identity.concurrency_key != connection.identity.concurrency_key ||
        identity.lease_id != connection.identity.lease_id ||
        identity.fencing_token != connection.identity.fencing_token ||
        identity.host != authority_host_ ||
        identity.host_registry_digest !=
            host_launch_registry_.registry_digest()) {
      return {grpc::StatusCode::PERMISSION_DENIED,
              "worker runtime evidence arrived after its attempt's lease or "
              "launch authority moved"};
    }
    // Third: can this deployment publish at all? A deployment with no
    // configured immutable receipt root holds no evidence authority, and
    // silently accepting a report it cannot publish would look to a worker
    // exactly like a published one.
    if (worker_runtime_evidence_ == nullptr) {
      // Two different missing things, and a deployment operator needs to be
      // able to tell them apart. Without a receipt root there is nowhere to
      // write. With one, the root is attested and writable and what is still
      // missing is the host inventory receipt the launch was granted against
      // -- the controller receives only `inventory_digest` and
      // `inventory_receipt_digest` over the hostd transport, and
      // `cache_resource_binding` selects rows out of `inventory.resources`, so
      // a digest cannot stand in for it. Reporting "no receipt root" to a
      // deployment that configured one would send its operator to fix a
      // correctly provisioned directory.
      return {grpc::StatusCode::FAILED_PRECONDITION,
              cache_evidence_
                  ? "authority has no grant-time host inventory receipt to "
                    "publish worker runtime evidence against"
                  : "authority has no configured cache evidence receipt root "
                    "to publish worker runtime evidence to"};
    }
    (void)worker_runtime_evidence_->publish(report, authority_host_, *binding);
    return grpc::Status::OK;
  } catch (const WorkerRuntimeEvidenceError& error) {
    return {grpc::StatusCode::PERMISSION_DENIED, error.what()};
  } catch (const CacheNamespaceAuthorityError& error) {
    return {grpc::StatusCode::PERMISSION_DENIED, error.what()};
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::acknowledge_worker_control(
    const v1::ControlPatchAcknowledgement& acknowledgement,
    const WorkerConnection& connection, std::uint64_t& acknowledged) {
  if (acknowledgement.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker control acknowledgement exceeds 64 KiB"};
  }
  try {
    if (acknowledgement.command_id().empty() ||
        acknowledgement.control_revision() == 0U ||
        acknowledgement.worker_sequence() == 0U ||
        acknowledgement.concurrency_key() != connection.identity.concurrency_key ||
        acknowledgement.lease_id() != connection.identity.lease_id ||
        acknowledgement.fencing_token() != connection.identity.fencing_token ||
        !acknowledgement.has_acknowledged_at()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "control acknowledgement has an invalid command or fenced identity"};
    }
    (void)timestamp_ns(acknowledgement.acknowledged_at());
    ControlCommandStatus status;
    switch (acknowledgement.disposition()) {
      case v1::ControlPatchAcknowledgement::DISPOSITION_APPLIED:
        status = ControlCommandStatus::applied;
        break;
      case v1::ControlPatchAcknowledgement::DISPOSITION_REJECTED:
        status = ControlCommandStatus::rejected;
        break;
      case v1::ControlPatchAcknowledgement::DISPOSITION_RESTART_REQUIRED:
        status = ControlCommandStatus::restart_required;
        break;
      case v1::ControlPatchAcknowledgement::DISPOSITION_UNSPECIFIED:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "control acknowledgement disposition is unspecified"};
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "control acknowledgement disposition is invalid"};
    }
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(connection.identity.run_id);
    const auto command = journal_.control_command(acknowledgement.command_id());
    if (!projection || !command || command->run_id != connection.identity.run_id ||
        command->control_revision != acknowledgement.control_revision() ||
        wire_apply_point(command->apply_point) != acknowledgement.apply_point()) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "control acknowledgement does not match a durable command"};
    }
    const std::uint64_t latest = journal_.latest_worker_sequence(
        connection.identity.run_id, connection.identity.node_id,
        connection.identity.attempt_id);
    const auto stored = journal_.event(acknowledgement.command_id() + ":ack");
    if (latest == std::numeric_limits<std::uint64_t>::max() ||
        acknowledgement.worker_sequence() > latest + 1U ||
        (acknowledgement.worker_sequence() <= latest && !stored)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "control acknowledgement sequence is not the next durable message or an exact replay"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no persisted compiled plan"};
    }
    std::optional<std::uint64_t> effective_step;
    if (status == ControlCommandStatus::applied &&
        command->apply_point != ApplyPoint::immediate) {
      if (acknowledgement.effective_step() == 0U) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "safe-point control acknowledgement requires an effective step"};
      }
      effective_step = acknowledgement.effective_step();
    } else if (acknowledgement.effective_step() != 0U) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "control acknowledgement has an inapplicable effective step"};
    }
    Controller controller(*plan, journal_, connection.identity.run_id);
    controller.recover();
    (void)controller.acknowledge_controls(
        acknowledgement.command_id(),
        ControlAcknowledgementIdentity{
            .concurrency_key = connection.identity.concurrency_key,
            .lease_id = connection.identity.lease_id,
            .fencing_token = connection.identity.fencing_token,
            .node_id = connection.identity.node_id,
            .attempt_id = connection.identity.attempt_id,
            .worker_sequence = acknowledgement.worker_sequence()},
        status, effective_step,
        acknowledgement_assignments(acknowledgement),
        acknowledgement_diagnostics(acknowledgement), authority_now());
    acknowledged = acknowledgement.worker_sequence();
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::acknowledge_worker_checkpoint(
    const v1::CheckpointAcknowledgement& acknowledgement,
    const WorkerConnection& connection, std::uint64_t& acknowledged) {
  if (acknowledgement.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker checkpoint acknowledgement exceeds 64 KiB"};
  }
  try {
    if (acknowledgement.command_id().empty() ||
        acknowledgement.worker_sequence() == 0U ||
        acknowledgement.concurrency_key() != connection.identity.concurrency_key ||
        acknowledgement.lease_id() != connection.identity.lease_id ||
        acknowledgement.fencing_token() != connection.identity.fencing_token ||
        !acknowledgement.has_acknowledged_at()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "checkpoint acknowledgement has an invalid command or fenced identity"};
    }
    (void)timestamp_ns(acknowledgement.acknowledged_at());
    CheckpointCommandStatus status;
    switch (acknowledgement.disposition()) {
      case v1::CheckpointAcknowledgement::DISPOSITION_APPLIED:
        status = CheckpointCommandStatus::applied;
        break;
      case v1::CheckpointAcknowledgement::DISPOSITION_REJECTED:
        status = CheckpointCommandStatus::rejected;
        break;
      case v1::CheckpointAcknowledgement::DISPOSITION_UNSPECIFIED:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "checkpoint acknowledgement disposition is unspecified"};
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "checkpoint acknowledgement disposition is invalid"};
    }
    const bool applied = status == CheckpointCommandStatus::applied;
    if ((applied && (acknowledgement.optimizer_step() == 0U ||
                     acknowledgement.artifact_id().empty())) ||
        (!applied && (acknowledgement.optimizer_step() != 0U ||
                      !acknowledgement.artifact_id().empty()))) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "checkpoint acknowledgement result is inconsistent"};
    }
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(connection.identity.run_id);
    const auto command =
        journal_.checkpoint_command(acknowledgement.command_id());
    if (!projection || !command ||
        command->run_id != connection.identity.run_id ||
        command->node_id != connection.identity.node_id ||
        command->attempt_id != connection.identity.attempt_id) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "checkpoint acknowledgement does not match a durable command"};
    }
    const std::uint64_t latest = journal_.latest_worker_sequence(
        connection.identity.run_id, connection.identity.node_id,
        connection.identity.attempt_id);
    const auto stored =
        journal_.event(acknowledgement.command_id() + ":ack");
    if (latest == std::numeric_limits<std::uint64_t>::max() ||
        acknowledgement.worker_sequence() > latest + 1U ||
        (acknowledgement.worker_sequence() <= latest && !stored)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "checkpoint acknowledgement sequence is not the next durable message or an exact replay"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no persisted compiled plan"};
    }
    Controller controller(*plan, journal_, connection.identity.run_id);
    controller.recover();
    (void)controller.acknowledge_checkpoint(
        acknowledgement.command_id(),
        ControlAcknowledgementIdentity{
            .concurrency_key = connection.identity.concurrency_key,
            .lease_id = connection.identity.lease_id,
            .fencing_token = connection.identity.fencing_token,
            .node_id = connection.identity.node_id,
            .attempt_id = connection.identity.attempt_id,
            .worker_sequence = acknowledgement.worker_sequence()},
        status,
        applied ? std::optional<std::uint64_t>{
                      acknowledgement.optimizer_step()}
                : std::nullopt,
        acknowledgement.artifact_id(),
        acknowledgement_diagnostics(acknowledgement), authority_now());
    acknowledged = acknowledgement.worker_sequence();
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::acknowledge_worker_lifecycle(
    const v1::LifecycleAcknowledgement& acknowledgement,
    const WorkerConnection& connection, std::uint64_t& acknowledged) {
  if (acknowledgement.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "worker lifecycle acknowledgement exceeds 64 KiB"};
  }
  try {
    if (acknowledgement.command_id().empty() ||
        acknowledgement.worker_sequence() == 0U ||
        acknowledgement.concurrency_key() != connection.identity.concurrency_key ||
        acknowledgement.lease_id() != connection.identity.lease_id ||
        acknowledgement.fencing_token() != connection.identity.fencing_token ||
        !acknowledgement.has_acknowledged_at()) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "lifecycle acknowledgement has an invalid command or fenced identity"};
    }
    (void)timestamp_ns(acknowledgement.acknowledged_at());
    LifecycleCommandKind kind;
    switch (acknowledgement.kind()) {
      case v1::LifecycleAcknowledgement::KIND_PAUSE:
        kind = LifecycleCommandKind::pause;
        break;
      case v1::LifecycleAcknowledgement::KIND_RESUME:
        kind = LifecycleCommandKind::resume;
        break;
      case v1::LifecycleAcknowledgement::KIND_CANCEL:
        kind = LifecycleCommandKind::cancel;
        break;
      case v1::LifecycleAcknowledgement::KIND_UNSPECIFIED:
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "lifecycle acknowledgement kind is invalid"};
    }
    LifecycleCommandStatus status;
    switch (acknowledgement.disposition()) {
      case v1::LifecycleAcknowledgement::DISPOSITION_APPLIED:
        status = LifecycleCommandStatus::applied;
        break;
      case v1::LifecycleAcknowledgement::DISPOSITION_REJECTED:
        status = LifecycleCommandStatus::rejected;
        break;
      case v1::LifecycleAcknowledgement::DISPOSITION_UNSPECIFIED:
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "lifecycle acknowledgement disposition is invalid"};
    }
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(connection.identity.run_id);
    const auto command = journal_.lifecycle_command(acknowledgement.command_id());
    if (!projection || !command ||
        command->run_id != connection.identity.run_id ||
        command->node_id != connection.identity.node_id ||
        command->attempt_id != connection.identity.attempt_id ||
        command->kind != kind) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "lifecycle acknowledgement does not match a durable command"};
    }
    const bool applied = status == LifecycleCommandStatus::applied;
    const bool needs_checkpoint =
        kind == LifecycleCommandKind::pause && command->checkpoint_first;
    if ((applied && needs_checkpoint &&
         (acknowledgement.optimizer_step() == 0U ||
          acknowledgement.artifact_id().empty())) ||
        (applied && !needs_checkpoint &&
         (acknowledgement.optimizer_step() != 0U ||
          !acknowledgement.artifact_id().empty())) ||
        (!applied && (acknowledgement.optimizer_step() != 0U ||
                      !acknowledgement.artifact_id().empty()))) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "lifecycle acknowledgement result is inconsistent"};
    }
    const std::uint64_t latest = journal_.latest_worker_sequence(
        connection.identity.run_id, connection.identity.node_id,
        connection.identity.attempt_id);
    const auto stored = journal_.event(acknowledgement.command_id() + ":ack");
    if (latest == std::numeric_limits<std::uint64_t>::max() ||
        acknowledgement.worker_sequence() > latest + 1U ||
        (acknowledgement.worker_sequence() <= latest && !stored)) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "lifecycle acknowledgement sequence is not the next durable message or an exact replay"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no persisted compiled plan"};
    }
    Controller controller(*plan, journal_, connection.identity.run_id);
    controller.recover();
    const AuthorityTimeSample now = authority_now();
    (void)controller.acknowledge_lifecycle(
        acknowledgement.command_id(),
        ControlAcknowledgementIdentity{
            .concurrency_key = connection.identity.concurrency_key,
            .lease_id = connection.identity.lease_id,
            .fencing_token = connection.identity.fencing_token,
            .node_id = connection.identity.node_id,
            .attempt_id = connection.identity.attempt_id,
            .worker_sequence = acknowledgement.worker_sequence()},
        status,
        applied && needs_checkpoint
            ? std::optional<std::uint64_t>{acknowledgement.optimizer_step()}
            : std::nullopt,
        acknowledgement.artifact_id(),
        acknowledgement_diagnostics(acknowledgement), now);
    if (applied && kind == LifecycleCommandKind::resume) {
      controller.recover();
      (void)controller.prepare_dispatch(now);
    }
    if (applied && kind == LifecycleCommandKind::cancel) {
      notify_reconciliation(connection.identity.run_id);
    }
    acknowledged = acknowledgement.worker_sequence();
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return worker_failure(exception);
  }
}

grpc::Status TrainVMService::Connect(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<v1::ControllerToWorker,
                             v1::WorkerToController>* stream) {
  if (stream == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "worker stream is required"};
  }
  v1::WorkerToController first;
  if (!stream->Read(&first)) {
    return cancelled(context)
               ? grpc::Status(grpc::StatusCode::CANCELLED,
                              "worker stream cancelled before hello")
               : grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                              "worker hello must be the first stream message");
  }
  if (first.ByteSizeLong() > kMaximumWorkerMessageBytes || !first.has_hello()) {
    return {first.ByteSizeLong() > kMaximumWorkerMessageBytes
                ? grpc::StatusCode::RESOURCE_EXHAUSTED
                : grpc::StatusCode::INVALID_ARGUMENT,
            "worker hello must be the first bounded stream message"};
  }
  const auto& hello = first.hello();
  if (hello.run_id().empty() || hello.node_id().empty() ||
      hello.attempt_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "worker hello attempt identity is required"};
  }
  const std::string attempt_key = hello.run_id() + "\n" + hello.node_id() +
                                  "\n" + hello.attempt_id();
  if (!claim_worker_attempt(attempt_key)) {
    return {grpc::StatusCode::ALREADY_EXISTS,
            "worker attempt already has an active stream"};
  }
  const auto finish = [&](grpc::Status status) {
    release_worker_attempt(attempt_key);
    return status;
  };
  WorkerConnection connection;
  grpc::Status status = open_worker_connection(hello, connection);
  if (!status.ok()) return finish(std::move(status));
  v1::ControllerToWorker welcome;
  *welcome.mutable_welcome() = connection.welcome;
  if (!stream->Write(welcome)) {
    return finish({grpc::StatusCode::CANCELLED,
                   "worker disconnected before durable welcome"});
  }
  if (connection.completed_receipt) {
    v1::ControllerToWorker receipt;
    *receipt.mutable_receipt() = *connection.completed_receipt;
    if (!stream->Write(receipt)) {
      return finish({grpc::StatusCode::CANCELLED,
                     "worker disconnected before replayed receipt"});
    }
    return finish(grpc::Status::OK);
  }
  std::uint64_t last_sent_controller_sequence =
      hello.last_acked_controller_sequence();
  const auto send_pending_commands = [&]() -> grpc::Status {
    std::vector<v1::WorkerCommand> commands;
    try {
      std::scoped_lock lock(command_mutex_);
      for (const auto& control : journal_.pending_control_commands(
               connection.identity.run_id, 0U)) {
        const std::uint64_t sequence =
            journal_.control_command_sequence(control.command_id);
        if (sequence <= last_sent_controller_sequence) continue;
        auto command = worker_control_command(control);
        command.set_controller_sequence(sequence);
        commands.push_back(std::move(command));
      }
      for (const auto& checkpoint : journal_.pending_checkpoint_commands(
               connection.identity.run_id, last_sent_controller_sequence)) {
        commands.push_back(worker_checkpoint_command(checkpoint));
      }
      for (const auto& lifecycle : journal_.pending_lifecycle_commands(
               connection.identity.run_id, last_sent_controller_sequence)) {
        commands.push_back(worker_lifecycle_command(lifecycle));
      }
      std::ranges::sort(commands, {}, &v1::WorkerCommand::controller_sequence);
    } catch (const std::exception& exception) {
      return worker_failure(exception);
    }
    for (const auto& command : commands) {
      v1::ControllerToWorker response;
      *response.mutable_command() = command;
      if (!stream->Write(response)) {
        return {grpc::StatusCode::CANCELLED,
                "worker disconnected before a durable controller command"};
      }
      last_sent_controller_sequence = command.controller_sequence();
    }
    return grpc::Status::OK;
  };
  status = send_pending_commands();
  if (!status.ok()) return finish(std::move(status));
  for (;;) {
    v1::WorkerToController message;
    if (!stream->Read(&message)) {
      return finish(cancelled(context)
                        ? grpc::Status(grpc::StatusCode::CANCELLED,
                                       "worker stream cancelled before result")
                        : grpc::Status(
                              grpc::StatusCode::FAILED_PRECONDITION,
                              "worker stream closed before its required result"));
    }
    if (message.ByteSizeLong() > kMaximumWorkerMessageBytes) {
      return finish({grpc::StatusCode::RESOURCE_EXHAUSTED,
                     "worker stream message exceeds 64 KiB"});
    }
    if (message.has_event()) {
      v1::WorkerReceipt committed;
      status = complete_worker_connection(message.event(), connection, committed);
      if (!status.ok()) return finish(std::move(status));
      v1::ControllerToWorker response;
      *response.mutable_receipt() = std::move(committed);
      if (!stream->Write(response)) {
        return finish({grpc::StatusCode::CANCELLED,
                       "worker disconnected after durable result commit"});
      }
      return finish(grpc::Status::OK);
    }

    if (message.has_runtime_evidence()) {
      // Not acknowledged, because it carries no worker_sequence to
      // acknowledge: the transport is the report struct exactly. Acceptance
      // is the immutable receipt the authority published; refusal is this
      // terminal status.
      status = record_worker_runtime_evidence(message.runtime_evidence(),
                                              connection);
      if (!status.ok()) return finish(std::move(status));
      status = send_pending_commands();
      if (!status.ok()) return finish(std::move(status));
      continue;
    }

    std::uint64_t acknowledged = 0U;
    if (message.has_heartbeat()) {
      status = record_worker_heartbeat(message.heartbeat(), connection,
                                       acknowledged);
    } else if (message.has_metric()) {
      status = record_worker_metric(message.metric(), connection, acknowledged);
    } else if (message.has_artifact()) {
      status = record_worker_artifact(message.artifact(), connection,
                                      acknowledged);
    } else if (message.has_control_ack()) {
      status = acknowledge_worker_control(message.control_ack(), connection,
                                          acknowledged);
    } else if (message.has_checkpoint_ack()) {
      status = acknowledge_worker_checkpoint(message.checkpoint_ack(), connection,
                                             acknowledged);
    } else if (message.has_lifecycle_ack()) {
      status = acknowledge_worker_lifecycle(message.lifecycle_ack(), connection,
                                            acknowledged);
    } else if (message.has_phase_receipt()) {
      status = record_worker_execution_phase_receipt(
          message.phase_receipt(), connection, acknowledged);
    } else {
      return finish({grpc::StatusCode::INVALID_ARGUMENT,
                     message.has_hello()
                         ? "worker hello may only be the first stream message"
                         : "worker stream message variant is required"});
    }
    if (!status.ok()) return finish(std::move(status));
    v1::ControllerToWorker response;
    response.set_acknowledge_worker_sequence(acknowledged);
    if (!stream->Write(response)) {
      return finish({grpc::StatusCode::CANCELLED,
                     "worker disconnected after durable message commit"});
    }
    status = send_pending_commands();
    if (!status.ok()) return finish(std::move(status));
  }
}

grpc::Status TrainVMService::SubmitExperiment(grpc::ServerContext* context,
                                              const v1::SubmitExperimentRequest* request,
                                              v1::SubmitExperimentResponse* response) {
  return submit_experiment(context, request, response, std::nullopt);
}

grpc::Status TrainVMService::submit_experiment(
    grpc::ServerContext* context,
    const v1::SubmitExperimentRequest* request,
    v1::SubmitExperimentResponse* response,
    const std::optional<AuthorRunAuthority>& authoring) {
  if (request == nullptr || response == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "request and response are required"};
  }
  if (request->ByteSizeLong() > kMaximumSubmissionBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, "submission exceeds 2 MiB"};
  }
  try {
    if (request->expected_journal_id().empty() ||
        request->expected_journal_id() != journal_.journal_id()) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "dashboard journal identity differs from the authority"};
    }
    if (request->source_document().empty() || request->source_format().empty()) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "source document and format are required"};
    }
    if (request->create_run() &&
        (request->idempotency_key().empty() || request->author().empty() ||
         request->reason().empty() || request->expected_plan_hash().empty() ||
         request->expected_adapter_lock_digest().empty())) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "run creation requires an idempotency key, author, reason, and "
              "expected plan and adapter-lock hashes"};
    }
    const bool has_fork_identity =
        !request->forked_from_run_id().empty() ||
        request->expected_parent_run_revision() != 0U ||
        !request->expected_parent_plan_hash().empty();
    if (has_fork_identity &&
        (!request->create_run() || request->forked_from_run_id().empty() ||
         request->forked_from_run_id().size() > 256U ||
         request->expected_parent_run_revision() == 0U ||
         request->expected_parent_plan_hash().empty())) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "run fork requires a bounded parent run ID, revision, and plan hash"};
    }
    if (cancelled(context))
      return cancellation_status();

    const CompileResult compiled =
        compile_document_source(request->source_document(), request->source_format());
    for (const auto& diagnostic : compiled.diagnostics) {
      add_diagnostic(*response, diagnostic);
    }
    if (!compiled.valid() || !compiled.plan) {
      return grpc::Status::OK;
    }
    const std::string canonical = compiled.plan->canonical_plan.dump();
    response->set_canonical_document(canonical);
    // v1 has no separate lowered execution IR yet, so both canonical fields
    // intentionally carry the same normalized, hashed experiment document.
    response->set_canonical_plan(canonical);
    response->set_plan_hash(compiled.plan->plan_hash);
    if (authoring) {
      const AuthorityTimeSample now = authority_now();
      const auto& receipt = authoring->receipt;
      const bool training_plan = std::ranges::any_of(
          compiled.plan->experiment.spec.workflow.nodes, [](const auto &entry) {
            return entry.second.invoke.training.has_value();
          });
      const auto &content_receipt = authoring->content_measurement_receipt;
      bool content_receipt_sealed = true;
      bool content_receipt_roots_match = true;
      if (content_receipt) {
        nlohmann::json principal = encode_json(*content_receipt);
        principal.erase("receipt_digest");
        content_receipt_sealed = content_receipt->receipt_digest ==
                                 "sha256:" + sha256_hex(principal.dump());
        const auto &locked_roots =
            compiled.plan->experiment.spec.workspace.input_content_roots;
        content_receipt_roots_match =
            locked_roots &&
            locked_roots->size() == content_receipt->roots.size();
        if (content_receipt_roots_match) {
          for (std::size_t index = 0U; index < locked_roots->size(); ++index) {
            const auto &locked = locked_roots->at(index);
            const auto &measured = content_receipt->roots[index];
            if (locked.path != measured.path ||
                locked.tree_sha256 != measured.tree_sha256 ||
                locked.file_count != measured.file_count ||
                locked.total_bytes != measured.total_bytes) {
              content_receipt_roots_match = false;
              break;
            }
          }
        }
      }
      if (!request->create_run() ||
          authoring->request_digest.size() != 71U ||
          !authoring->request_digest.starts_with("sha256:") ||
          receipt.api_version != kTrainingPreflightReceiptApiVersion ||
          !receipt.passed || !receipt.cacheable ||
          receipt.plan_hash != compiled.plan->plan_hash ||
          receipt.receipt_digest.size() != 71U ||
          !receipt.receipt_digest.starts_with("sha256:") ||
          (training_plan && !content_receipt) ||
          (content_receipt &&
           (content_receipt->api_version !=
                kInputContentMeasurementReceiptApiVersion ||
            content_receipt->cache_api_version !=
                kInputContentMeasurementCacheApiVersion ||
            content_receipt->cache_policy_digest !=
                input_content_measurement_cache_.policy_digest() ||
            content_receipt->request_digest != authoring->request_digest ||
            content_receipt->plan_hash != compiled.plan->plan_hash ||
            content_receipt->receipt_digest.size() != 71U ||
            !content_receipt->receipt_digest.starts_with("sha256:") ||
            !content_receipt_sealed || !content_receipt_roots_match)) ||
          now.boot.nanoseconds < 0 ||
          static_cast<std::uint64_t>(now.boot.nanoseconds) >=
              receipt.valid_until_monotonic_ns) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "canonical author-run requires a verified unexpired passive "
                "preflight receipt bound to the exact plan"};
      }
    }
    if (request->create_run() &&
        request->expected_plan_hash() != compiled.plan->plan_hash) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "authority compiler plan hash differs from the validated preview"};
    }
    std::string adapter_lock_manifest;
    try {
      adapter_lock_manifest =
          adapter_registry_.plan_lock_manifest(*compiled.plan);
    } catch (const AdapterResolutionError& exception) {
      add_diagnostic(*response,
                     Diagnostic{.severity = Diagnostic::Severity::error,
                                .code = "adapter.registry",
                                .path = "/spec/components",
                                .message = exception.what()});
      return grpc::Status::OK;
    }
    const std::string adapter_lock_digest =
        "sha256:" + sha256_hex(adapter_lock_manifest);
    response->set_adapter_lock_digest(adapter_lock_digest);
    response->set_canonical_adapter_lock(adapter_lock_manifest);

    const bool uses_training_components =
        training_components_.plan_uses_components(*compiled.plan);
    std::string training_lock_manifest;
    std::string training_lock_digest;
    if (uses_training_components) {
      try {
        training_lock_manifest =
            training_components_.plan_lock_manifest(*compiled.plan);
        training_lock_digest =
            "sha256:" + sha256_hex(training_lock_manifest);
        response->set_training_component_lock_digest(
            training_lock_digest);
        response->set_canonical_training_component_lock(
            training_lock_manifest);
      } catch (const TrainingComponentResolutionError& exception) {
        add_diagnostic(*response,
                       Diagnostic{.severity = Diagnostic::Severity::error,
                                  .code = "training_component.registry",
                                  .path = "/spec/workflow/nodes",
                                  .message = exception.what()});
        return grpc::Status::OK;
      }
    }
    if (!request->create_run()) return grpc::Status::OK;
    if (request->expected_adapter_lock_digest() != adapter_lock_digest) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "authority adapter lock differs from the validated preview"};
    }
    if (uses_training_components &&
        request->expected_training_component_lock_digest() !=
            training_lock_digest) {
      return {
          grpc::StatusCode::FAILED_PRECONDITION,
          "authority training-component lock differs from the validated preview"};
    }
    if (!uses_training_components &&
        !request->expected_training_component_lock_digest().empty()) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "submission supplies a training-component lock for a plan without one"};
    }
    if (cancelled(context))
      return cancellation_status();

    std::scoped_lock lock(command_mutex_);
    if (cancelled(context))
      return cancellation_status();
    const std::string run_id =
        "run-" + sha256_hex(nlohmann::json({{"journal_id", journal_.journal_id()},
                                            {"idempotency_key", request->idempotency_key()}})
                                .dump());
    if (const auto created = journal_.event(run_id + ":created")) {
      if (created->event_type != "run.created" || created->run_id != run_id ||
          created->payload.value("plan_hash", std::string{}) !=
              compiled.plan->plan_hash ||
          !created->payload.contains("submission") ||
          !created->payload.at("submission").is_object()) {
        throw RunCreationConflict(
            "run already exists with a different run.created event");
      }
      const auto stored_plan = journal_.compiled_plan(compiled.plan->plan_hash);
      if (!stored_plan ||
          stored_plan->canonical_plan != compiled.plan->canonical_plan ||
          stored_plan->experiment.metadata.name !=
              compiled.plan->experiment.metadata.name) {
        throw RunCreationConflict(
            "run already exists with a different compiled plan");
      }
      const nlohmann::json& stored_submission =
          created->payload.at("submission");
      nlohmann::json expected_submission{
          {"idempotency_key", request->idempotency_key()},
          {"source_format", request->source_format()},
          {"create_run", true},
          {"author", request->author()},
          {"reason", request->reason()},
          {"plan_hash", compiled.plan->plan_hash},
          {"adapter_lock_digest", adapter_lock_digest},
          {"adapter_lock", nlohmann::json::parse(adapter_lock_manifest)},
      };
      if (authoring) {
        expected_submission["author_run"] = {
            {"request_digest", authoring->request_digest},
            {"preflight_receipt", encode_json(authoring->receipt)},
        };
        if (authoring->content_measurement_receipt)
          expected_submission["author_run"]["content_measurement_receipt"] =
              encode_json(*authoring->content_measurement_receipt);
      }
      if (uses_training_components) {
        expected_submission["training_component_lock_digest"] =
            training_lock_digest;
        expected_submission["training_component_lock"] =
            nlohmann::json::parse(training_lock_manifest);
      }
      if (has_fork_identity) {
        expected_submission["forked_from"] = {
            {"run_id", request->forked_from_run_id()},
            {"run_revision", request->expected_parent_run_revision()},
            {"plan_hash", request->expected_parent_plan_hash()},
        };
      }
      nlohmann::json comparable_stored = stored_submission;
      nlohmann::json comparable_expected = expected_submission;
      if (authoring && comparable_stored.contains("author_run") &&
          comparable_stored.at("author_run").is_object() &&
          comparable_stored.at("author_run").value(
              "request_digest", std::string{}) == authoring->request_digest) {
        // A retry re-collects fresh passive evidence by design. The first
        // durable receipt remains the launch record; idempotency is fenced by
        // the exact request/plan/registry locks, not the observation time.
        comparable_stored["author_run"].erase("preflight_receipt");
        comparable_expected["author_run"].erase("preflight_receipt");
        comparable_stored["author_run"].erase("content_measurement_receipt");
        comparable_expected["author_run"].erase("content_measurement_receipt");
      }
      if (comparable_stored != comparable_expected) {
        throw RunCreationConflict(
            "run already exists with a different submission identity");
      }
      Controller recovered(*stored_plan, journal_, run_id);
      const ExecutionState& recovered_state = recovered.recover();
      const auto projection = journal_.projection(run_id);
      if (!projection || projection->plan_hash != compiled.plan->plan_hash ||
          projection->run_revision != recovered_state.revision) {
        throw std::runtime_error(
            "durable run creation has no matching projection");
      }
      auto* identity = response->mutable_run();
      identity->set_run_id(projection->run_id);
      identity->set_revision(projection->run_revision);
      identity->set_plan_hash(projection->plan_hash);
      notify_reconciliation(projection->run_id);
      return grpc::Status::OK;
    }

    if (has_fork_identity) {
      if (request->forked_from_run_id() == run_id) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "a run cannot be forked from itself"};
      }
      const auto parent = journal_.projection(request->forked_from_run_id());
      if (!parent) {
        return {grpc::StatusCode::NOT_FOUND, "fork parent run does not exist"};
      }
      if (parent->run_revision != request->expected_parent_run_revision() ||
          parent->plan_hash != request->expected_parent_plan_hash()) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "fork parent identity is stale"};
      }
    }

    nlohmann::json submission_identity{
        {"idempotency_key", request->idempotency_key()},
        {"source_format", request->source_format()},
        {"create_run", request->create_run()},
        {"author", request->author()},
        {"reason", request->reason()},
        {"plan_hash", compiled.plan->plan_hash},
        {"adapter_lock_digest", adapter_lock_digest},
        {"adapter_lock", nlohmann::json::parse(adapter_lock_manifest)},
    };
    if (authoring) {
      submission_identity["author_run"] = {
          {"request_digest", authoring->request_digest},
          {"preflight_receipt", encode_json(authoring->receipt)},
      };
      if (authoring->content_measurement_receipt)
        submission_identity["author_run"]["content_measurement_receipt"] =
            encode_json(*authoring->content_measurement_receipt);
    }
    if (uses_training_components) {
      submission_identity["training_component_lock_digest"] =
          training_lock_digest;
      submission_identity["training_component_lock"] =
          nlohmann::json::parse(training_lock_manifest);
    }
    if (has_fork_identity) {
      submission_identity["forked_from"] = {
          {"run_id", request->forked_from_run_id()},
          {"run_revision", request->expected_parent_run_revision()},
          {"plan_hash", request->expected_parent_plan_hash()},
      };
    }
    Controller controller(*compiled.plan, journal_, run_id);
    controller.create_queued(submission_identity);
    const auto projection = journal_.projection(run_id);
    if (!projection) {
      return {grpc::StatusCode::INTERNAL, "created run has no durable projection"};
    }
    auto* identity = response->mutable_run();
    identity->set_run_id(projection->run_id);
    identity->set_revision(projection->run_revision);
    identity->set_plan_hash(projection->plan_hash);
    notify_reconciliation(projection->run_id);
    return grpc::Status::OK;
  } catch (const RunCreationConflict& exception) {
    return {grpc::StatusCode::ALREADY_EXISTS, exception.what()};
  } catch (const std::invalid_argument& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::AuthorRun(
    grpc::ServerContext* context, const v1::AuthorRunRequest* request,
    grpc::ServerWriter<v1::AuthorRunUpdate>* writer) {
  if (request == nullptr || writer == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "author-run request and update stream are required"};
  }
  if (request->ByteSizeLong() > kMaximumSubmissionBytes ||
      request->request_document().empty() ||
      (request->source_format() != "json" &&
       request->source_format() != "yaml")) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "author-run requires a bounded closed JSON or YAML document"};
  }
  const auto canonical_plan_hash = [](const std::string& value) {
    return value.size() == 64U &&
           std::ranges::all_of(value, [](const unsigned char character) {
             return (character >= static_cast<unsigned char>('0') &&
                     character <= static_cast<unsigned char>('9')) ||
                    (character >= static_cast<unsigned char>('a') &&
                     character <= static_cast<unsigned char>('f'));
           });
  };
  if ((request->dry_run() && !request->expected_plan_hash().empty()) ||
      (!request->dry_run() &&
       !canonical_plan_hash(request->expected_plan_hash()))) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            request->dry_run()
                ? "dry-run authoring cannot carry an expected plan hash"
                : "launch authoring requires a 64-character lowercase "
                  "expected plan hash from a completed dry-run"};
  }
  const auto write = [&](v1::AuthorRunUpdate& update) {
    update.set_dry_run(request->dry_run());
    return writer->Write(update);
  };
  const auto stage = [&](AuthorRunStage value, std::string detail) {
    v1::AuthorRunUpdate update;
    update.set_stage(wire_author_run_stage(value));
    update.set_detail(std::move(detail));
    return write(update);
  };
  const auto fail = [&](std::string code, std::string path,
                        std::string message, std::string help) {
    v1::AuthorRunUpdate update;
    update.set_stage(v1::AUTHOR_RUN_STAGE_FAILED);
    update.set_detail("author-run failed before submission");
    update.set_terminal(true);
    add_diagnostic(update,
                   TrainingPreflightDiagnostic{
                       .severity = Diagnostic::Severity::error,
                       .code = std::move(code),
                       .path = std::move(path),
                       .message = std::move(message),
                       .help = std::move(help),
                   });
    (void)write(update);
  };

  if (!stage(AuthorRunStage::validating,
             "validating closed author-run document"))
    return cancellation_status();
  try {
    const AuthorRunDocument document = decode_author_run_document(
        request->request_document(), request->source_format());
    if (document.source.recipe &&
        std::filesystem::path(document.source.recipe->registry_path) !=
            recipe_registry_path_) {
      fail("author_run.recipe_registry_authority", "/source/recipe/registry_path",
           "recipe source does not name the deployed authority registry",
           "Select a recipe returned by trainvm.recipe-profiles@1.0.0; author "
           "documents cannot redirect recipe authority.");
      return grpc::Status::OK;
    }
    if (!stage(AuthorRunStage::resolving,
               "resolving recipe and component graph"))
      return cancellation_status();
    if (!stage(AuthorRunStage::locking_inputs,
               "locking immutable static input content"))
      return cancellation_status();
    ResolvedAuthorRun resolved = resolve_and_lock_author_run(document, &input_content_measurement_cache_);
    if (!request->dry_run() &&
        request->expected_plan_hash() != resolved.plan.plan_hash) {
      fail("author_run.plan_changed", "/expected_plan_hash",
           "resolved plan no longer matches the frozen preview",
           "Run a new dry-run preview and review the changed recipe or input "
           "content identities before submission.");
      return grpc::Status::OK;
    }

    // Resolve every adapter/component before family probes. These are pure
    // authority reads and make dry-run an honest preview of submission.
    (void)adapter_registry_.plan_lock_manifest(resolved.plan);
    if (training_components_.plan_uses_components(resolved.plan))
      (void)training_components_.plan_lock_manifest(resolved.plan);

    v1::AuthorRunUpdate resolved_update;
    resolved_update.set_stage(v1::AUTHOR_RUN_STAGE_LOCKING_INPUTS);
    std::uint64_t cache_hits = 0U;
    std::uint64_t cache_misses = 0U;
    std::uint64_t cache_bypasses = 0U;
    std::uint64_t staging_saturations = 0U;
    std::uint64_t bytes_hashed = 0U;
    std::uint64_t elapsed_nanoseconds = 0U;
    for (const auto &measurement : resolved.content_measurements) {
      cache_hits += measurement.cache_hits;
      cache_misses += measurement.cache_misses;
      cache_bypasses += measurement.cache_bypasses;
      staging_saturations += measurement.staging_saturations;
      bytes_hashed += measurement.bytes_hashed;
      elapsed_nanoseconds += measurement.elapsed_nanoseconds;
    }
    std::string measurement_detail =
        "resolved canonical plan and provenance; " +
        std::string(kInputContentMeasurementCacheApiVersion) + " "
        "hits=" +
        std::to_string(cache_hits) + " misses=" +
        std::to_string(cache_misses) + " bypasses=" +
        std::to_string(cache_bypasses) + " staging_saturations=" +
        std::to_string(staging_saturations) + " bytes_hashed=" +
        std::to_string(bytes_hashed) + " elapsed_nanoseconds=" +
        std::to_string(elapsed_nanoseconds);
    if (resolved.content_measurement_receipt) {
      const auto &commit = resolved.content_measurement_receipt->cache_commit;
      measurement_detail +=
          " capacity=" + std::to_string(commit.capacity) +
          " entries_before=" + std::to_string(commit.entries_before) +
          " entries_after=" + std::to_string(commit.entries_after) +
          " staged=" + std::to_string(commit.staged_entries) +
          " staging_saturations=" +
          std::to_string(commit.staging_saturations) +
          " evictions=" + std::to_string(commit.evictions) +
          " saturations=" + std::to_string(commit.saturations) +
          " corruptions=" + std::to_string(commit.corruptions);
    }
    resolved_update.set_detail(std::move(measurement_detail));
    resolved_update.set_plan_hash(resolved.plan.plan_hash);
    resolved_update.set_canonical_plan_json(
        resolved.plan.canonical_plan.dump());
    resolved_update.set_content_lock_reused(resolved.content_lock_reused);
    if (resolved.recipe_expansion)
      resolved_update.set_recipe_expansion_json(
          resolved.recipe_expansion->dump());
    if (resolved.content_measurement_receipt)
      resolved_update.set_content_measurement_receipt_json(
          encode_json(*resolved.content_measurement_receipt).dump());
    if (!write(resolved_update))
      return cancellation_status();

    if (!stage(AuthorRunStage::preflight,
               "collecting passive host and exact family-probe evidence"))
      return cancellation_status();
    TrainingPreflightEvidenceResult evidence =
        preflight_evidence_->collect(resolved.plan,
                                     resolved.recipe_provenance);
    if (!evidence.environment || !evidence.diagnostics.empty()) {
      v1::AuthorRunUpdate update;
      update.set_stage(v1::AUTHOR_RUN_STAGE_FAILED);
      update.set_detail("passive evidence collection failed");
      update.set_plan_hash(resolved.plan.plan_hash);
      update.set_terminal(true);
      for (const auto& diagnostic : evidence.diagnostics)
        add_diagnostic(update, diagnostic);
      (void)write(update);
      return grpc::Status::OK;
    }
    TrainingPreflightReceipt receipt =
        run_training_preflight(resolved.plan, *evidence.environment);
    v1::AuthorRunUpdate preflight_update;
    preflight_update.set_stage(receipt.passed
                                   ? v1::AUTHOR_RUN_STAGE_PREFLIGHT
                                   : v1::AUTHOR_RUN_STAGE_FAILED);
    preflight_update.set_detail(receipt.passed
                                    ? "passive preflight passed"
                                    : "passive preflight rejected submission");
    preflight_update.set_plan_hash(resolved.plan.plan_hash);
    preflight_update.set_canonical_plan_json(
        resolved.plan.canonical_plan.dump());
    preflight_update.set_preflight_receipt_json(encode_json(receipt).dump());
    preflight_update.set_content_lock_reused(resolved.content_lock_reused);
    if (resolved.recipe_expansion)
      preflight_update.set_recipe_expansion_json(
          resolved.recipe_expansion->dump());
    if (resolved.content_measurement_receipt)
      preflight_update.set_content_measurement_receipt_json(
          encode_json(*resolved.content_measurement_receipt).dump());
    for (const auto& diagnostic : receipt.diagnostics)
      add_diagnostic(preflight_update, diagnostic);
    preflight_update.set_terminal(!receipt.passed);
    if (!write(preflight_update))
      return cancellation_status();
    if (!receipt.passed)
      return grpc::Status::OK;

    if (request->dry_run()) {
      v1::AuthorRunUpdate update;
      update.set_stage(v1::AUTHOR_RUN_STAGE_COMPLETE);
      update.set_detail("dry-run complete; no directory, journal row, lease, "
                        "or process was created");
      update.set_plan_hash(resolved.plan.plan_hash);
      update.set_canonical_plan_json(resolved.plan.canonical_plan.dump());
      update.set_preflight_receipt_json(encode_json(receipt).dump());
      update.set_content_lock_reused(resolved.content_lock_reused);
      if (resolved.recipe_expansion)
        update.set_recipe_expansion_json(resolved.recipe_expansion->dump());
      if (resolved.content_measurement_receipt)
        update.set_content_measurement_receipt_json(
            encode_json(*resolved.content_measurement_receipt).dump());
      update.set_terminal(true);
      (void)write(update);
      return grpc::Status::OK;
    }
    if (cancelled(context))
      return cancellation_status();

    // Serialize provisioning for one deterministic request identity. The
    // normal SubmitExperiment command lock remains separate and is acquired
    // only after this scoped lock is released by function return/retry.
    std::scoped_lock authoring_lock(author_run_mutex_);
    if (!stage(AuthorRunStage::provisioning,
               "provisioning authority-owned run directory"))
      return cancellation_status();
    auto provision = provision_authorized_run_directory(
        resolved.plan, *evidence.environment, resolved.request_digest);
    // Recollect the complete passive host/family evidence after workspace
    // creation. This is not merely a second path check over a stale snapshot.
    TrainingPreflightEvidenceResult provisioned_evidence =
        preflight_evidence_->collect(resolved.plan,
                                     resolved.recipe_provenance);
    if (!provisioned_evidence.environment ||
        !provisioned_evidence.diagnostics.empty()) {
      fail("author_run.provisioned_evidence", "/host",
           "passive evidence could not be recollected after provisioning",
           "Inspect hostd/family probe diagnostics and retry; no run or lease "
           "was created.");
      return grpc::Status::OK;
    }
    const TrainingPreflightReceipt provisioned_receipt =
        run_training_preflight(resolved.plan,
                               *provisioned_evidence.environment);
    if (!provisioned_receipt.passed) {
      fail("author_run.provisioned_preflight", "/spec/workspace/run_directory",
           "run-directory provisioning changed or invalidated passive evidence",
           "Inspect the authority-owned directory and retry; no run or lease "
           "was created.");
      return grpc::Status::OK;
    }
    if (cancelled(context))
      return cancellation_status();

    if (!stage(AuthorRunStage::submitting,
               "submitting idempotent queued run"))
      return cancellation_status();
    v1::SubmitExperimentRequest preview_request;
    preview_request.set_source_document(resolved.plan.canonical_plan.dump());
    preview_request.set_source_format("json");
    preview_request.set_create_run(false);
    preview_request.set_expected_journal_id(journal_.journal_id());
    v1::SubmitExperimentResponse preview;
    grpc::Status status =
        submit_experiment(context, &preview_request, &preview, std::nullopt);
    if (!status.ok())
      return status;
    if (preview.diagnostics_size() != 0 ||
        preview.plan_hash() != resolved.plan.plan_hash ||
        preview.adapter_lock_digest().empty()) {
      v1::AuthorRunUpdate update;
      update.set_stage(v1::AUTHOR_RUN_STAGE_FAILED);
      update.set_detail("authority submission preview failed");
      update.set_terminal(true);
      for (const auto& diagnostic : preview.diagnostics())
        *update.add_diagnostics() = diagnostic;
      (void)write(update);
      return grpc::Status::OK;
    }

    v1::SubmitExperimentRequest create = preview_request;
    create.set_create_run(true);
    create.set_idempotency_key(resolved.request_digest);
    create.set_expected_journal_id(journal_.journal_id());
    create.set_author(document.author);
    create.set_reason(document.reason);
    create.set_expected_plan_hash(preview.plan_hash());
    create.set_expected_adapter_lock_digest(preview.adapter_lock_digest());
    create.set_expected_training_component_lock_digest(
        preview.training_component_lock_digest());
    v1::SubmitExperimentResponse submitted;
    status = submit_experiment(
        context, &create, &submitted,
        AuthorRunAuthority{.request_digest = resolved.request_digest,
                           .receipt = provisioned_receipt,
                           .content_measurement_receipt =
                               resolved.content_measurement_receipt});
    if (!status.ok())
      return status;
    if (!submitted.has_run() || submitted.run().run_id().empty() ||
        submitted.run().plan_hash() != resolved.plan.plan_hash ||
        submitted.run().plan_hash() != provisioned_receipt.plan_hash) {
      v1::AuthorRunUpdate update;
      update.set_stage(v1::AUTHOR_RUN_STAGE_FAILED);
      update.set_detail("authority rejected queued run creation");
      update.set_terminal(true);
      for (const auto& diagnostic : submitted.diagnostics())
        *update.add_diagnostics() = diagnostic;
      (void)write(update);
      return grpc::Status::OK;
    }
    provision.mark_durable();
    v1::AuthorRunUpdate complete;
    complete.set_stage(v1::AUTHOR_RUN_STAGE_COMPLETE);
    complete.set_detail("run is durably visible to the dashboard");
    complete.set_plan_hash(resolved.plan.plan_hash);
    complete.set_canonical_plan_json(resolved.plan.canonical_plan.dump());
    complete.set_preflight_receipt_json(
        encode_json(provisioned_receipt).dump());
    *complete.mutable_run() = submitted.run();
    complete.set_dashboard_url("/api/trainvm/runs/" +
                               submitted.run().run_id());
    complete.set_content_lock_reused(resolved.content_lock_reused);
    if (resolved.recipe_expansion)
      complete.set_recipe_expansion_json(resolved.recipe_expansion->dump());
    if (resolved.content_measurement_receipt)
      complete.set_content_measurement_receipt_json(
          encode_json(*resolved.content_measurement_receipt).dump());
    complete.set_terminal(true);
    (void)write(complete);
    return grpc::Status::OK;
  } catch (const RunAuthoringError& error) {
    fail("author_run.invalid", "", error.what(),
         "Correct the closed author-run document; no run or lease was created.");
    return grpc::Status::OK;
  } catch (const RecipeProfileError& error) {
    fail("author_run.recipe", "/source/recipe", error.what(),
         "Select an exact deployed recipe/version and valid bounded overrides.");
    return grpc::Status::OK;
  } catch (const AdapterResolutionError& error) {
    fail("author_run.adapter", "/source", error.what(),
         "Install or select the exact registered adapter profile.");
    return grpc::Status::OK;
  } catch (const TrainingComponentResolutionError& error) {
    fail("author_run.training_component", "/source", error.what(),
         "Install or select the exact registered training components.");
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    fail("author_run.authority_failure", "", error.what(),
         "Inspect the authority diagnostic; no unverified run was created.");
    return grpc::Status::OK;
  }
}

grpc::Status TrainVMService::DiffPlan(grpc::ServerContext* context,
                                      const v1::PlanDiffRequest* request,
                                      v1::PlanDiffResponse* response) {
  if (request == nullptr || response == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "plan diff request and response are required"};
  }
  if (request->ByteSizeLong() > kMaximumSubmissionBytes ||
      request->run_id().empty() || request->run_id().size() > 256U ||
      request->proposed_source_document().empty() ||
      request->source_format().empty() ||
      request->expected_journal_id().empty() ||
      request->expected_current_plan_hash().empty() ||
      request->expected_proposed_plan_hash().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "plan diff requires a bounded run ID, source, and format"};
  }
  if (cancelled(context)) return cancellation_status();
  try {
    if (request->expected_journal_id() != journal_.journal_id()) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "plan diff journal identity is stale"};
    }
    const CompileResult proposed = compile_document_source(
        request->proposed_source_document(), request->source_format());
    for (const Diagnostic& diagnostic : proposed.diagnostics) {
      add_diagnostic(*response, diagnostic);
    }
    if (!proposed.valid() || !proposed.plan) return grpc::Status::OK;
    response->set_proposed_plan_hash(proposed.plan->plan_hash);
    if (proposed.plan->plan_hash != request->expected_proposed_plan_hash()) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "authority compiler plan hash differs from the validated preview"};
    }
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(request->run_id());
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "run does not exist"};
    }
    if (request->expected_revision() != projection->run_revision) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "plan diff run revision is stale"};
    }
    if (request->expected_current_plan_hash() != projection->plan_hash) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "plan diff current plan identity is stale"};
    }
    const auto current = journal_.compiled_plan(projection->plan_hash);
    if (!current) {
      return {grpc::StatusCode::DATA_LOSS,
              "run has no persisted compiled plan"};
    }
    response->set_semantic_diff(
        nlohmann::json::diff(current->canonical_plan,
                             proposed.plan->canonical_plan)
            .dump());
    const bool unchanged = current->plan_hash == proposed.plan->plan_hash;
    response->set_adoptable_in_place(unchanged);
    if (!unchanged) {
      add_diagnostic(
          *response,
          {.severity = Diagnostic::Severity::warning,
           .code = "plan.adoption_requires_new_run",
           .path = "/",
           .message =
               "this protocol revision can preview the semantic diff but "
               "does not mutate an active plan in place"});
    }
    return grpc::Status::OK;
  } catch (const nlohmann::json::exception& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::invalid_argument& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::CommandRun(grpc::ServerContext* context,
                                        const v1::RunCommandRequest* request,
                                        v1::RunCommandResponse* response) {
  if (request == nullptr || response == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "request and response are required"};
  }
  if (request->ByteSizeLong() > kMaximumCommandBytes) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, "command exceeds 64 KiB"};
  }
  if (!request->has_controls() && !request->has_checkpoint() &&
      !request->has_pause() && !request->has_resume() &&
      !request->has_cancel()) {
    return {grpc::StatusCode::UNIMPLEMENTED,
            "this lifecycle command is not implemented yet"};
  }
  if (request->run_id().empty() || request->idempotency_key().empty() ||
      request->author().empty() || request->reason().empty() ||
      request->expected_journal_id().empty() || request->expected_plan_hash().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "run ID, journal ID, plan hash, idempotency key, author, and reason are required"};
  }
  if (cancelled(context)) return cancellation_status();

  std::scoped_lock lock(command_mutex_);
  if (cancelled(context)) return cancellation_status();
  try {
    const auto projection = journal_.projection(request->run_id());
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "run does not exist"};
    }
    if (request->expected_journal_id() != journal_.journal_id() ||
        request->expected_plan_hash() != projection->plan_hash) {
      response->set_disposition(v1::RunCommandResponse::DISPOSITION_CONFLICT);
      auto* diagnostic = response->add_diagnostics();
      diagnostic->set_severity(v1::Diagnostic::SEVERITY_ERROR);
      diagnostic->set_code("control.authority_identity_conflict");
      diagnostic->set_message("dashboard journal or plan identity differs from the authority");
      fill_run_summary(*projection, journal_, *response);
      return grpc::Status::OK;
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS, "run has no persisted compiled plan"};
    }
    if (cancelled(context)) return cancellation_status();
    Controller controller(*plan, journal_, request->run_id());
    controller.recover();
    if (cancelled(context)) return cancellation_status();
    if (request->has_cancel()) {
      const auto& state = controller.state();
      if (state.current_node_id.empty() || request->cancel().reason().empty()) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "cancel requires an active worker node and command reason"};
      }
      const Node& node =
          plan->experiment.spec.workflow.nodes.at(state.current_node_id);
      const Component& component =
          plan->experiment.spec.components.at(node.invoke.component);
      const AdapterProfile& profile =
          adapter_registry_.resolve(component, node.invoke.operation);
      if (const auto refused = admit_lifecycle_control(
              profile.lifecycle, LifecycleControlVerb::cancel, false)) {
        response->set_disposition(
            v1::RunCommandResponse::DISPOSITION_REJECTED);
        auto* diagnostic = response->add_diagnostics();
        diagnostic->set_severity(v1::Diagnostic::SEVERITY_ERROR);
        diagnostic->set_code(refused->code);
        diagnostic->set_message(refused->message);
        fill_run_summary(*projection, journal_, *response);
        return grpc::Status::OK;
      }
      const std::int64_t timeout = duration_ns(
          request->cancel().graceful_timeout(),
          plan->experiment.spec.recovery.graceful_stop_seconds);
      const auto submission = controller.request_cancel(
          request->idempotency_key(), request->expected_run_revision(),
          request->cancel().reason(), timeout, request->author(),
          request->reason());
      response->set_disposition(
          submission.inserted
              ? v1::RunCommandResponse::DISPOSITION_ACCEPTED
              : replay_disposition(submission.command));
      fill_lifecycle_result(submission.command, *response);
      fill_run_summary(*journal_.projection(request->run_id()), journal_,
                       *response);
      return grpc::Status::OK;
    }
    if (request->has_pause() || request->has_resume()) {
      const bool pause = request->has_pause();
      const auto& state = controller.state();
      if (state.current_node_id.empty()) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "pause and resume require an active worker node"};
      }
      const Node& node =
          plan->experiment.spec.workflow.nodes.at(state.current_node_id);
      const Component& component =
          plan->experiment.spec.components.at(node.invoke.component);
      const AdapterProfile& profile =
          adapter_registry_.resolve(component, node.invoke.operation);
      const bool release = pause && request->pause().release_resources();
      if (release && !request->pause().checkpoint_first()) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "resource-releasing pause requires checkpoint_first"};
      }
      const LifecycleControlVerb verb =
          !pause ? LifecycleControlVerb::resume
                 : (release ? LifecycleControlVerb::pause_release_resources
                            : LifecycleControlVerb::pause_keep_resources);
      if (const auto refused = admit_lifecycle_control(
              profile.lifecycle, verb,
              pause && request->pause().checkpoint_first())) {
        response->set_disposition(
            v1::RunCommandResponse::DISPOSITION_REJECTED);
        auto* diagnostic = response->add_diagnostics();
        diagnostic->set_severity(v1::Diagnostic::SEVERITY_ERROR);
        diagnostic->set_code(refused->code);
        diagnostic->set_message(refused->message);
        fill_run_summary(*projection, journal_, *response);
        return grpc::Status::OK;
      }
      const auto submission = controller.request_lifecycle(
          pause ? LifecycleCommandKind::pause
                : LifecycleCommandKind::resume,
          request->idempotency_key(), request->expected_run_revision(),
          pause && request->pause().checkpoint_first(), release,
          request->author(), request->reason());
      response->set_disposition(
          submission.inserted
              ? v1::RunCommandResponse::DISPOSITION_ACCEPTED
              : replay_disposition(submission.command));
      fill_lifecycle_result(submission.command, *response);
      fill_run_summary(*journal_.projection(request->run_id()), journal_,
                       *response);
      return grpc::Status::OK;
    }
    if (request->has_checkpoint()) {
      if (request->checkpoint().reason().empty()) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "checkpoint-now requires a command reason"};
      }
      const auto& state = controller.state();
      if (state.current_node_id.empty()) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "checkpoint-now requires an active worker node"};
      }
      const Node& node =
          plan->experiment.spec.workflow.nodes.at(state.current_node_id);
      const Component& component =
          plan->experiment.spec.components.at(node.invoke.component);
      const AdapterProfile& profile =
          adapter_registry_.resolve(component, node.invoke.operation);
      if (const auto refused = admit_lifecycle_control(
              profile.lifecycle, LifecycleControlVerb::checkpoint_now,
              false)) {
        response->set_disposition(
            v1::RunCommandResponse::DISPOSITION_REJECTED);
        auto* diagnostic = response->add_diagnostics();
        diagnostic->set_severity(v1::Diagnostic::SEVERITY_ERROR);
        diagnostic->set_code(refused->code);
        diagnostic->set_message(refused->message);
        fill_run_summary(*projection, journal_, *response);
        return grpc::Status::OK;
      }
      const auto submission = controller.request_checkpoint(
          request->idempotency_key(), request->expected_run_revision(),
          request->checkpoint().reason(), request->author(), request->reason());
      response->set_disposition(
          submission.inserted
              ? v1::RunCommandResponse::DISPOSITION_ACCEPTED
              : replay_disposition(submission.command));
      fill_checkpoint_result(submission.command, *response);
      fill_run_summary(*journal_.projection(request->run_id()), journal_,
                       *response);
      return grpc::Status::OK;
    }
    const auto assignments = assignments_json(request->controls());
    if (cancelled(context)) return cancellation_status();
    const auto validation = controller.request_controls(
        request->idempotency_key(), request->expected_run_revision(),
        request->controls().expected_control_revision(), assignments, request->author(),
        request->reason());
    if (!validation.valid()) {
      response->set_disposition(v1::RunCommandResponse::DISPOSITION_REJECTED);
      for (const auto& diagnostic : validation.diagnostics) {
        add_diagnostic(*response, diagnostic);
      }
      fill_run_summary(*journal_.projection(request->run_id()), journal_, *response);
      return grpc::Status::OK;
    }
    if (!validation.command) {
      return {grpc::StatusCode::INTERNAL, "validated command was not persisted"};
    }
    response->set_disposition(validation.replayed
                                  ? replay_disposition(*validation.command)
                                  : v1::RunCommandResponse::DISPOSITION_ACCEPTED);
    fill_control_result(*validation.command, *response);
    response->set_command_sequence(
        journal_.control_command_sequence(validation.command->command_id));
    fill_run_summary(*journal_.projection(request->run_id()), journal_, *response);
    return grpc::Status::OK;
  } catch (const std::invalid_argument& exception) {
    response->set_disposition(v1::RunCommandResponse::DISPOSITION_CONFLICT);
    auto* diagnostic = response->add_diagnostics();
    diagnostic->set_severity(v1::Diagnostic::SEVERITY_ERROR);
    diagnostic->set_code("command.conflict");
    diagnostic->set_message(exception.what());
    if (const auto projection = journal_.projection(request->run_id())) {
      fill_run_summary(*projection, journal_, *response);
    }
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::GetRun(grpc::ServerContext* context,
                                    const v1::GetRunRequest* request,
                                    v1::RunSummary* response) {
  if (request == nullptr || response == nullptr ||
      request->ByteSizeLong() > 4'096U || request->run_id().empty() ||
      request->run_id().size() > 256U) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "get-run requires a bounded run identity"};
  }
  if (cancelled(context)) return cancellation_status();
  try {
    {
      std::scoped_lock lock(command_mutex_);
      const auto projection = journal_.projection(request->run_id());
      if (!projection) {
        return {grpc::StatusCode::NOT_FOUND, "run does not exist"};
      }
      fill_run_summary(*projection, journal_, *response);
    }
    if (const auto failure = reconciliation_failure(request->run_id())) {
      response->set_wait_reason("reconciliation failure: " + *failure);
    }
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::GetCompiledPlan(
    grpc::ServerContext* context,
    const v1::GetCompiledPlanRequest* request,
    v1::GetCompiledPlanResponse* response) {
  if (request == nullptr || response == nullptr ||
      request->ByteSizeLong() > 4'096U || request->run_id().empty() ||
      request->run_id().size() > 256U) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "compiled-plan read requires a bounded run identity"};
  }
  if (cancelled(context)) return cancellation_status();
  try {
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(request->run_id());
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "run does not exist"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "run has no persisted compiled plan"};
    }
    const std::string canonical = plan->canonical_plan.dump();
    if (canonical.empty() || canonical.size() > kMaximumSubmissionBytes) {
      return {grpc::StatusCode::DATA_LOSS,
              "persisted compiled plan exceeds the authority read bound"};
    }
    response->set_journal_id(journal_.journal_id());
    auto* identity = response->mutable_run();
    identity->set_run_id(projection->run_id);
    identity->set_revision(projection->run_revision);
    identity->set_plan_hash(projection->plan_hash);
    response->set_canonical_plan_json(canonical);
    return grpc::Status::OK;
  } catch (const std::invalid_argument& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::ListRuns(grpc::ServerContext* context,
                                      const v1::ListRunsRequest* request,
                                      v1::ListRunsResponse* response) {
  constexpr std::size_t kDefaultLimit = 100U;
  constexpr std::size_t kMaximumLimit = 250U;
  if (request == nullptr || response == nullptr ||
      request->ByteSizeLong() > kMaximumCommandBytes ||
      request->observed_states_size() > 64 || request->labels_size() > 64 ||
      request->page_token().size() > 4'096U) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "list-runs query exceeds its bounds"};
  }
  if (cancelled(context)) return cancellation_status();
  try {
    std::set<std::string, std::less<>> states;
    for (const int state : request->observed_states()) {
      if (!v1::ObservedState_IsValid(state)) {
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "list-runs observed-state filter is unknown"};
      }
      states.insert(
          observed_state_name(static_cast<v1::ObservedState>(state)));
    }
    if (states.size() !=
        static_cast<std::size_t>(request->observed_states_size())) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "list-runs observed-state filters must be unique"};
    }
    std::map<std::string, std::string, std::less<>> labels;
    for (const auto& [key, value] : request->labels()) {
      labels.emplace(key, value);
    }
    const std::size_t limit =
        request->limit() == 0U ? kDefaultLimit
                              : static_cast<std::size_t>(request->limit());
    if (limit > kMaximumLimit) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "list-runs limit exceeds 250"};
    }
    std::vector<RunProjection> projections;
    std::string next_page_token;
    {
      std::scoped_lock lock(command_mutex_);
      const std::string filter_digest =
          run_list_filter_digest(journal_.journal_id(), states, labels);
      std::optional<RunProjectionCursor> after;
      if (!request->page_token().empty()) {
        after = decode_run_page_token(request->page_token(), filter_digest);
      }
      projections = journal_.run_projections({
          .observed_states = states,
          .labels = labels,
          .after = std::move(after),
          .limit = limit + 1U,
      });
      if (projections.size() > limit) {
        next_page_token = encode_run_page_token(
            projections.at(limit - 1U), filter_digest);
        projections.resize(limit);
      }
      for (const RunProjection& projection : projections) {
        fill_run_summary(projection, journal_, *response->add_runs());
      }
    }
    response->set_journal_id(journal_.journal_id());
    response->set_next_page_token(std::move(next_page_token));
    for (int index = 0; index < response->runs_size(); ++index) {
      if (const auto failure = reconciliation_failure(
              response->runs(index).identity().run_id())) {
        response->mutable_runs(index)->set_wait_reason(
            "reconciliation failure: " + *failure);
      }
    }
    return grpc::Status::OK;
  } catch (const nlohmann::json::exception& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::invalid_argument& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::WatchEvents(
    grpc::ServerContext* context, const v1::WatchEventsRequest* request,
    grpc::ServerWriter<v1::EventEnvelope>* writer) {
  constexpr std::size_t kEventPageSize = 256U;
  constexpr std::uint32_t kMaximumReplayEvents = 1'000U;
  if (context == nullptr || request == nullptr || writer == nullptr ||
      request->ByteSizeLong() > kMaximumCommandBytes ||
      request->run_ids_size() > 64 || request->event_types_size() > 64 ||
      request->replay_limit() > kMaximumReplayEvents ||
      ((request->through_journal_sequence() != 0U || request->newest_first()) &&
       request->replay_limit() == 0U) ||
      (request->through_journal_sequence() != 0U &&
       request->after_journal_sequence() >=
           request->through_journal_sequence()) ||
      (request->newest_first() &&
       request->through_journal_sequence() == 0U) ||
      (request->newest_per_metric_series() &&
       (request->newest_first() || request->replay_limit() == 0U ||
        request->through_journal_sequence() == 0U ||
        request->run_ids_size() != 1 || request->event_types_size() != 1 ||
        request->event_types(0) != "metric.sampled"))) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "watch-events request exceeds its bounds"};
  }
  try {
    std::set<std::string, std::less<>> run_ids(request->run_ids().begin(),
                                               request->run_ids().end());
    std::set<std::string, std::less<>> event_types(
        request->event_types().begin(), request->event_types().end());
    if (run_ids.size() != static_cast<std::size_t>(request->run_ids_size()) ||
        event_types.size() !=
            static_cast<std::size_t>(request->event_types_size())) {
      return {grpc::StatusCode::INVALID_ARGUMENT,
              "watch-events filters must be unique"};
    }
    std::uint64_t cursor = request->after_journal_sequence();
    std::uint64_t through = request->through_journal_sequence();
    std::size_t replayed = 0U;
    while (!cancelled(context)) {
      const std::size_t query_limit =
          request->replay_limit() == 0U
              ? kEventPageSize
              : std::min<std::size_t>(
                    kEventPageSize,
                    static_cast<std::size_t>(request->replay_limit()) -
                        replayed);
      std::vector<SequencedEvent> events;
      {
        std::scoped_lock lock(command_mutex_);
        events = journal_.sequenced_events({
            .after_journal_sequence = cursor,
            .through_journal_sequence = through,
            .run_ids = run_ids,
            .event_types = event_types,
            .limit = query_limit,
            .newest_first = request->newest_first(),
            .newest_per_metric_series =
                request->newest_per_metric_series(),
        });
      }
      for (const SequencedEvent& event : events) {
        v1::EventEnvelope output = wire_event(event);
        if (!writer->Write(output)) {
          return {grpc::StatusCode::CANCELLED,
                  "event stream closed by its reader"};
        }
        if (request->newest_first()) {
          through = event.journal_sequence - 1U;
        } else {
          cursor = event.journal_sequence;
        }
        ++replayed;
      }
      if (request->newest_first() && through <= cursor) {
        return grpc::Status::OK;
      }
      if (request->replay_limit() != 0U &&
          (replayed == static_cast<std::size_t>(request->replay_limit()) ||
           events.size() < query_limit)) {
        return grpc::Status::OK;
      }
      if (events.size() == query_limit) continue;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return {grpc::StatusCode::CANCELLED, "event stream was cancelled"};
  } catch (const std::invalid_argument& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::GetControlView(
    grpc::ServerContext* context, const v1::GetControlViewRequest* request,
    v1::GetControlViewResponse* response) {
  constexpr std::size_t kControlHistoryLimit = 50U;
  if (request == nullptr || response == nullptr ||
      request->ByteSizeLong() > 4'096U || request->run_id().empty() ||
      request->run_id().size() > 256U) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "control view requires a bounded run identity"};
  }
  if (cancelled(context)) return cancellation_status();
  try {
    std::scoped_lock lock(command_mutex_);
    const auto projection = journal_.projection(request->run_id());
    if (!projection) {
      return {grpc::StatusCode::NOT_FOUND, "run does not exist"};
    }
    const auto plan = journal_.compiled_plan(projection->plan_hash);
    if (!plan) {
      return {grpc::StatusCode::DATA_LOSS,
              "run has no persisted compiled plan"};
    }
    nlohmann::json effective_values = nlohmann::json::object();
    for (const auto& [name, control] :
         plan->experiment.spec.controls.catalog) {
      v1::ControlDescriptor& descriptor =
          (*response->mutable_catalog())[name];
      descriptor.set_type(wire_control_type(control.type));
      set_wire_scalar(control.default_value,
                      *descriptor.mutable_default_value());
      if (control.minimum) descriptor.set_minimum(*control.minimum);
      if (control.maximum) descriptor.set_maximum(*control.maximum);
      if (control.values) {
        for (const nlohmann::json& value : *control.values) {
          set_wire_scalar(value, *descriptor.add_values());
        }
      }
      descriptor.set_apply_point(wire_apply_point(control.apply));
      descriptor.set_mutable_after_start(control.mutable_after_start);
      descriptor.set_requires_pause(control.requires_pause.value_or(false));
      descriptor.set_description(control.description.value_or(""));
      descriptor.set_unit(control.unit.value_or(""));
      effective_values[name] = control.default_value;
    }
    const EffectiveControlSnapshot effective =
        journal_.effective_controls(request->run_id());
    for (const auto& [name, value] : effective.values.items()) {
      if (!effective_values.contains(name)) {
        throw std::runtime_error(
            "effective controls contain an unknown catalog key");
      }
      effective_values[name] = value;
    }
    fill_wire_assignments(effective_values,
                          [&] { return response->add_effective_values(); });
    response->set_latest_requested_revision(
        journal_.latest_control_revision(request->run_id()));
    response->set_latest_effective_revision(effective.revision);
    for (const ControlCommand& command : journal_.control_commands(
             request->run_id(), kControlHistoryLimit)) {
      auto* output = response->add_commands();
      output->set_command_id(command.command_id);
      output->set_control_revision(command.control_revision);
      output->set_apply_point(wire_apply_point(command.apply_point));
      fill_wire_assignments(command.assignments,
                            [&] { return output->add_assignments(); });
      output->set_author(command.author);
      output->set_reason(command.reason);
      output->set_status(wire_command_status(command.status));
      if (command.effective_step) {
        output->set_effective_step(*command.effective_step);
      }
      fill_wire_assignments(command.effective_values,
                            [&] { return output->add_effective_values(); });
      if (!command.diagnostics.is_array()) {
        throw std::runtime_error(
            "stored control diagnostics are not an array");
      }
      for (const nlohmann::json& diagnostic : command.diagnostics) {
        auto* wire_diagnostic = output->add_diagnostics();
        fill_stored_diagnostic(diagnostic, *wire_diagnostic);
        if (wire_diagnostic->severity() ==
            v1::Diagnostic::SEVERITY_UNSPECIFIED) {
          throw std::runtime_error(
              "stored control diagnostic has an invalid severity");
        }
      }
    }
    return grpc::Status::OK;
  } catch (const std::invalid_argument& exception) {
    return {grpc::StatusCode::INVALID_ARGUMENT, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::GetDescriptor(
    grpc::ServerContext* context, const v1::DescriptorRequest* request,
    v1::DescriptorResponse* response) {
  if (request == nullptr || response == nullptr) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "descriptor request and response are required"};
  }
  if (request->ByteSizeLong() > 4096U || request->adapter().size() > 256U ||
      request->version().size() > 256U) {
    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
            "descriptor selector exceeds its bound"};
  }
  if (cancelled(context)) return cancellation_status();
  if (request->version() != "1.0.0" ||
      (request->adapter() != "trainvm.training-components" &&
       request->adapter() != "trainvm.operations" &&
       request->adapter() != "trainvm.recipe-profiles")) {
    return {grpc::StatusCode::NOT_FOUND,
            "no descriptor matches the exact requested provider and version"};
  }
  try {
    if (request->adapter() == "trainvm.operations") {
      response->set_schema_json(
          adapter_registry_.operation_descriptors_json().dump());
      response->set_schema_hash(
          adapter_registry_.operation_descriptors_digest());
    } else if (request->adapter() == "trainvm.training-components") {
      response->set_schema_json(training_components_.document_json().dump());
      response->set_schema_hash(training_components_.registry_digest());
    } else {
      const RecipeProfileRegistry recipes =
          RecipeProfileRegistry::load_file(recipe_registry_path_);
      nlohmann::json document = recipes.document_json();
      document["registry_path"] = recipe_registry_path_.string();
      document["default_registry_path"] = recipe_registry_path_.string();
      document["registry_digest"] = recipes.registry_digest();
      const std::string canonical = document.dump();
      response->set_schema_json(canonical);
      response->set_schema_hash("sha256:" + sha256_hex(canonical));
    }
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

grpc::Status TrainVMService::GetReconciliationStatus(
    grpc::ServerContext* context,
    const v1::GetReconciliationStatusRequest* request,
    v1::GetReconciliationStatusResponse* response) {
  if (request == nullptr || response == nullptr ||
      request->ByteSizeLong() > 64U) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "reconciliation status requires an empty bounded request"};
  }
  if (cancelled(context)) return cancellation_status();
  const ReconciliationSupervisorMetrics metrics = reconciliation_metrics();
  const std::vector<ReconciliationRunWait> waits =
      reconciliation_waits(kMaximumReportedWaits);
  {
    std::scoped_lock lock(reconciliation_mutex_);
    response->set_supervisor_running(reconciliation_started_);
  }
  response->set_wakes(metrics.wakes);
  response->set_explicit_wakes(metrics.explicit_wakes);
  response->set_cadence_wakes(metrics.cadence_wakes);
  response->set_scans(metrics.scans);
  response->set_reconcile_passes(metrics.reconcile_passes);
  response->set_reconcile_steps(metrics.reconcile_steps);
  response->set_skipped_idle_runs(metrics.skipped_idle_runs);
  response->set_budget_requeues(metrics.budget_requeues);
  response->set_failures(metrics.failures);
  response->set_tracked_runs(metrics.tracked_runs);
  response->set_waits_truncated(metrics.tracked_runs > waits.size());
  for (const ReconciliationRunWait& wait : waits) {
    auto* output = response->add_waits();
    output->set_run_id(wait.run_id);
    output->set_wait_reason(wait.wait_reason);
    output->set_idle_passes(wait.idle_passes);
    output->set_retries(wait.retries);
    output->set_backoff_ns(wait.backoff_ns);
    output->set_next_due_boottime_ns(wait.next_due_ns);
    output->set_last_event_sequence(wait.last_event_sequence);
  }
  return grpc::Status::OK;
}

grpc::Status TrainVMService::GetHostAuthorityStatus(
    grpc::ServerContext* context,
    const v1::GetHostAuthorityStatusRequest* request,
    v1::GetHostAuthorityStatusResponse* response) {
  if (request == nullptr || response == nullptr ||
      request->ByteSizeLong() > 64U) {
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "host authority status requires an empty bounded request"};
  }
  if (cancelled(context)) return cancellation_status();
  if (!hostd_status_client_ || hostd_status_timeout_ns_ <= 0) {
    return {grpc::StatusCode::FAILED_PRECONDITION,
            "TrainVM has no configured hostd authority status source"};
  }
  try {
    const std::int64_t now = hostd_monotonic_now_ns();
    if (hostd_status_timeout_ns_ >
        std::numeric_limits<std::int64_t>::max() - now) {
      return {grpc::StatusCode::INTERNAL,
              "hostd status deadline exceeds the authority clock range"};
    }
    std::uint64_t correlation =
        hostd_status_correlation_.fetch_add(1U, std::memory_order_relaxed);
    if (correlation == 0U) {
      correlation =
          hostd_status_correlation_.fetch_add(1U, std::memory_order_relaxed);
    }
    const HostdStatusReply reply = hostd_request_status(
        *hostd_status_client_, correlation, now + hostd_status_timeout_ns_);
    if (cancelled(context)) return cancellation_status();
    if (reply.kind == HostdStatusReplyKind::error) {
      const std::string detail = reply.error
                                     ? reply.error->code + ": " +
                                           reply.error->message
                                     : "hostd returned an incomplete typed error";
      return {grpc::StatusCode::UNAVAILABLE, detail};
    }
    if (!reply.status || !reply.authority_status) {
      return {grpc::StatusCode::DATA_LOSS,
              "hostd omitted its typed authority snapshot"};
    }
    const HostdCoordinatorStatus& coordinator = *reply.status;
    const HostdAuthorityStatus& authority = *reply.authority_status;
    response->set_api_version(authority.api_version);
    auto* wire_coordinator = response->mutable_coordinator();
    wire_coordinator->set_api_version(coordinator.api_version);
    wire_coordinator->set_lifecycle(
        wire_hostd_lifecycle(coordinator.lifecycle));
    wire_coordinator->set_host_id(coordinator.host_id);
    wire_coordinator->set_boot_id(coordinator.boot_id);
    wire_coordinator->set_broker_epoch(coordinator.broker_epoch);
    wire_coordinator->set_inventory_digest(coordinator.inventory_digest);
    wire_coordinator->set_live_sessions(coordinator.live_sessions);
    wire_coordinator->set_admission_sessions(coordinator.admission_sessions);
    wire_coordinator->set_stale_admission_sessions(
        coordinator.stale_admission_sessions);
    wire_coordinator->set_release_only_sessions(
        coordinator.release_only_sessions);
    wire_coordinator->set_admission_counts_are_cached_evidence(
        coordinator.admission_counts_are_cached_evidence);
    if (coordinator.startup_audit) {
      wire_coordinator->set_startup_audit_receipt_digest(
          coordinator.startup_audit->receipt_digest);
      wire_coordinator->set_startup_audit_passed(
          coordinator.startup_audit->disposition ==
          HostStartupAuditDisposition::passed);
    }
    wire_coordinator->set_poison_reason(coordinator.poison_reason);

    response->set_startup_phase(
        wire_hostd_startup_phase(authority.startup_phase));
    response->set_startup_recovery_steps(authority.startup_recovery_steps);
    response->set_remaining_unclosed_process_records(
        authority.remaining_unclosed_process_records);
    response->set_remaining_terminal_release_records(
        authority.remaining_terminal_release_records);
    response->set_ledger_verified(authority.ledger_verified);
    response->set_ledger_verification_reason(
        authority.ledger_verification_reason);
    response->set_ledger_sequence(
        authority.ledger_chain_head.ledger_sequence);
    response->set_ledger_chain_hash(authority.ledger_chain_head.chain_hash);
    response->set_ledger_record_count(authority.ledger_record_count);
    response->set_occupancy_ledger_sequence(
        authority.occupancy_ledger_sequence);
    response->set_occupancy_digest(authority.occupancy_digest);
    response->set_active_fence_count(authority.active_fence_count);
    response->set_active_fences_truncated(
        authority.active_fences_truncated);
    for (const ResourceFence& fence : authority.active_fences) {
      auto* output = response->add_active_fences();
      output->set_kind(wire_host_resource_kind(fence.resource.kind));
      output->set_vendor(
          wire_host_accelerator_vendor(fence.resource.vendor));
      output->set_stable_id(fence.resource.stable_id);
      output->set_parent_id(fence.resource.parent_id.value_or(""));
      output->set_generation(fence.generation);
      output->set_inventory_digest(fence.inventory_digest);
      output->set_topology_digest(fence.topology_digest);
    }
    response->set_active_process_count(authority.active_process_count);
    response->set_active_processes_truncated(
        authority.active_processes_truncated);
    for (const HostdProcessAuthorityStatus& process :
         authority.active_processes) {
      auto* output = response->add_active_processes();
      output->set_allocation_id(process.allocation_id);
      output->set_journal_id(process.journal_id);
      output->set_run_id(process.run_id);
      output->set_logical_lease_id(process.logical_lease_id);
      output->set_logical_fencing_token(process.logical_fencing_token);
      output->set_launch_id(process.launch_id);
      output->set_phase(wire_hostd_process_phase(process.phase));
      output->set_cgroup_path(process.cgroup_path);
      if (process.host_pid) output->set_host_pid(*process.host_pid);
      if (process.process_starttime_ticks) {
        output->set_process_starttime_ticks(
            *process.process_starttime_ticks);
      }
      output->set_device_policy_intended(process.device_policy_intended);
      output->set_device_policy_installed(process.device_policy_installed);
      output->set_device_policy_digest(process.device_policy_digest);
      output->set_device_policy_installation_digest(
          process.device_policy_installation_digest);
      output->set_process_policy_intended(process.process_policy_intended);
      output->set_process_policy_installed(process.process_policy_installed);
      output->set_process_policy_digest(process.process_policy_digest);
      output->set_process_policy_installation_digest(
          process.process_policy_installation_digest);
      if (process.cgroup_empty)
        output->set_cgroup_empty(*process.cgroup_empty);
      if (process.accelerator_contexts_empty) {
        output->set_accelerator_contexts_empty(
            *process.accelerator_contexts_empty);
      }
      output->set_context_audit_digest(process.context_audit_digest);
      output->set_terminal_receipt_digest(process.terminal_receipt_digest);
    }
    response->set_process_launch_enabled(authority.process_launch_enabled);
    response->set_mutation_enabled(authority.mutation_enabled);
    response->set_mutation_disabled_reason(
        authority.mutation_disabled_reason);
    const LeaseRenewalCoordinatorSnapshot renewal =
        lease_renewals_.snapshot();
    response->set_lease_renewal_tracked_count(renewal.tracked_count);
    response->set_lease_renewal_poisoned(renewal.poisoned);
    const std::optional<std::string> renewal_failure =
        reconciliation_failure("__lease_renewal__");
    response->set_lease_renewal_failure(
        renewal.poison_reason.empty()
            ? renewal_failure.value_or("")
            : renewal.poison_reason);
    response->set_resource_inventory_observed(
        authority.resource_inventory_observed);
    response->set_resource_inventory_observation_age_ns(
        authority.resource_inventory_observation_age_ns);
    response->set_current_inventory_digest(
        authority.current_inventory_digest);
    response->set_current_inventory_receipt_digest(
        authority.current_inventory_receipt_digest);
    response->set_degraded_resource_count(
        authority.degraded_resource_count);
    response->set_resource_degradation_reason(
        authority.resource_degradation_reason);
    return grpc::Status::OK;
  } catch (const HostdTransportError& exception) {
    return {grpc::StatusCode::UNAVAILABLE, exception.what()};
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path,
          AdapterRegistry adapter_registry,
          HostLaunchRegistry host_launch_registry,
          TrainingComponentRegistry training_components,
          std::optional<HostdClientConfiguration> hostd_configuration,
          std::optional<std::uint32_t> worker_socket_gid,
          std::filesystem::path recipe_registry_path,
          std::optional<std::filesystem::path> cache_evidence_root) {
  if (journal_path.empty() || socket_path.empty()) {
    throw std::invalid_argument("serve requires journal and socket paths");
  }
  const auto absolute_socket = std::filesystem::absolute(socket_path);
  const auto parent = absolute_socket.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  if (worker_socket_gid &&
      parent == std::filesystem::absolute(journal_path)
                    .lexically_normal()
                    .parent_path()) {
    throw std::runtime_error(
        "group-accessible worker socket requires a directory separate from "
        "the owner-only journal authority");
  }

  SignalMaskGuard signal_mask;
  // Acquire journal authority before touching the socket. Otherwise a second
  // daemon pointed at the same paths could unlink the live authority's socket
  // and only then discover that it cannot acquire the journal lock.
  TrainVMService service(journal_path, std::move(adapter_registry),
                         std::move(host_launch_registry), {},
                         std::move(training_components),
                         std::move(hostd_configuration),
                         "unix:" + absolute_socket.string(), nullptr,
                         SqliteAuthorityEnforcementGrade::strict_filesystem, {},
                         std::move(recipe_registry_path), nullptr,
                         std::move(cache_evidence_root));
  if (worker_socket_gid) {
    struct stat parent_status {};
    if (*worker_socket_gid != static_cast<std::uint32_t>(::getegid()) ||
        ::lstat(parent.c_str(), &parent_status) != 0 ||
        !S_ISDIR(parent_status.st_mode) ||
        parent_status.st_uid != ::geteuid() ||
        parent_status.st_gid != static_cast<gid_t>(*worker_socket_gid) ||
        (parent_status.st_mode & (S_IWGRP | S_IRWXO)) != 0 ||
        ::chmod(parent.c_str(), S_IRWXU | S_IXGRP) != 0) {
      throw std::runtime_error(
          "worker socket group requires an owned, same-egid, nonwritable "
          "authority directory");
    }
  }
  SocketAuthorityLock socket_authority(absolute_socket);
  remove_stale_socket(absolute_socket);
  SocketCleanupGuard socket_cleanup(absolute_socket);
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(static_cast<int>(kMaximumSubmissionBytes));
  builder.AddListeningPort("unix:" + absolute_socket.string(), grpc::InsecureServerCredentials());
  builder.RegisterService(static_cast<v1::TrainVM::Service*>(&service));
  builder.RegisterService(static_cast<v1::WorkerControl::Service*>(&service));
  std::unique_ptr<grpc::Server> server;
  const mode_t socket_mode = worker_socket_gid
                                 ? S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP
                                 : S_IRUSR | S_IWUSR;
  {
    // The socket is born at its final access grade. chmod below is retained as
    // defense in depth, but is too late to close the bind-to-chmod window.
    UmaskGuard restricted(static_cast<mode_t>(0777U & ~socket_mode));
    server = builder.BuildAndStart();
  }
  if (!server) {
    throw std::runtime_error("could not start TrainVM authority on " + absolute_socket.string());
  }
  socket_cleanup.claim();
  const auto shutdown_server = [&] {
    socket_cleanup.preserve_replacement();
    server->Shutdown();
    service.stop_reconciliation_supervisor();
    server->Wait();
    socket_cleanup.restore_replacement();
  };
  struct stat socket_status {};
  if (::chmod(absolute_socket.c_str(), socket_mode) != 0 ||
      ::lstat(absolute_socket.c_str(), &socket_status) != 0 ||
      !S_ISSOCK(socket_status.st_mode) ||
      socket_status.st_uid != ::geteuid() ||
      socket_status.st_gid != ::getegid() ||
      (socket_status.st_mode & 0777U) != socket_mode ||
      socket_status.st_nlink != 1) {
    shutdown_server();
    throw std::runtime_error("could not attest TrainVM authority socket permissions: " +
                             std::string(std::strerror(errno)));
  }
  try {
    service.start_reconciliation_supervisor();
  } catch (...) {
    shutdown_server();
    throw;
  }
  int received_signal = 0;
  if (::sigwait(signal_mask.signals(), &received_signal) != 0) {
    shutdown_server();
    throw std::runtime_error("TrainVM authority signal wait failed");
  }
  shutdown_server();
  return 0;
}

}  // namespace trainvm
