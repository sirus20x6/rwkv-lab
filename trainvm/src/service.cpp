#include "trainvm/service.hpp"

#include "trainvm/controller.hpp"
#include "trainvm/reflection_json.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/security/server_credentials.h>

namespace trainvm {

AuthorityLock::AuthorityLock(const std::filesystem::path& journal_path) {
  const auto absolute_journal = std::filesystem::absolute(journal_path);
  if (!absolute_journal.parent_path().empty()) {
    std::filesystem::create_directories(absolute_journal.parent_path());
  }
  // Lock the journal inode so symlink and hardlink aliases cannot create
  // independent writer authorities. flock and SQLite's POSIX record locks are
  // separate lock domains on Linux, so this does not interfere with SQLite.
  journal_descriptor_ =
      ::open(absolute_journal.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, S_IRUSR | S_IWUSR);
  if (journal_descriptor_ < 0) {
    throw std::runtime_error("could not open authority journal " + absolute_journal.string() +
                             ": " + std::strerror(errno));
  }
  if (::flock(journal_descriptor_, LOCK_EX | LOCK_NB) != 0) {
    const std::string message = std::strerror(errno);
    ::close(journal_descriptor_);
    journal_descriptor_ = -1;
    throw std::runtime_error("another TrainVM authority owns " + absolute_journal.string() +
                             ": " + message);
  }

  const std::filesystem::path path =
      std::filesystem::weakly_canonical(absolute_journal).string() + ".authority.lock";
  descriptor_ = ::open(path.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, S_IRUSR | S_IWUSR);
  if (descriptor_ < 0) {
    const std::string message = std::strerror(errno);
    ::close(journal_descriptor_);
    journal_descriptor_ = -1;
    throw std::runtime_error("could not open authority lock " + path.string() + ": " + message);
  }
  if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
    const std::string message = std::strerror(errno);
    ::close(descriptor_);
    descriptor_ = -1;
    ::close(journal_descriptor_);
    journal_descriptor_ = -1;
    throw std::runtime_error("another TrainVM authority owns " + path.string() + ": " +
                               message);
  }
  if (::fchmod(descriptor_, S_IRUSR | S_IWUSR) != 0) {
    const std::string message = std::strerror(errno);
    ::close(descriptor_);
    descriptor_ = -1;
    ::close(journal_descriptor_);
    journal_descriptor_ = -1;
    throw std::runtime_error("could not restrict authority lock " + path.string() + ": " +
                             message);
  }
}

AuthorityLock::~AuthorityLock() {
  if (descriptor_ >= 0) {
    ::close(descriptor_);
  }
  if (journal_descriptor_ >= 0) {
    ::close(journal_descriptor_);
  }
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

void add_stored_diagnostics(v1::RunCommandResponse& response,
                            const nlohmann::json& diagnostics) {
  if (!diagnostics.is_array()) return;
  for (const auto& diagnostic : diagnostics) {
    if (!diagnostic.is_object()) continue;
    auto* output = response.add_diagnostics();
    const auto severity = diagnostic.find("severity");
    if (severity != diagnostic.end() && severity->is_string()) {
      const auto value = severity->get<std::string>();
      if (value == "info") output->set_severity(v1::Diagnostic::SEVERITY_INFO);
      if (value == "warning") output->set_severity(v1::Diagnostic::SEVERITY_WARNING);
      if (value == "error") output->set_severity(v1::Diagnostic::SEVERITY_ERROR);
    }
    const auto string_value = [&](std::string_view key) -> std::string {
      const auto value = diagnostic.find(std::string(key));
      if (value != diagnostic.end() && value->is_string()) {
        return value->get<std::string>();
      }
      return {};
    };
    output->set_code(string_value("code"));
    output->set_document_path(string_value("document_path"));
    if (output->document_path().empty()) {
      output->set_document_path(string_value("path"));
    }
    output->set_message(string_value("message"));
    output->set_help(string_value("help"));
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

void fill_control_result(const ControlCommand& command, v1::RunCommandResponse& response) {
  response.set_command_sequence(command.control_revision);
  auto* output = response.mutable_control();
  output->set_command_id(command.command_id);
  output->set_control_revision(command.control_revision);
  output->set_apply_point(wire_apply_point(command.apply_point));
  output->set_requires_pause(command.requires_pause);
  output->set_status(wire_command_status(command.status));
  for (auto iterator = command.assignments.begin(); iterator != command.assignments.end(); ++iterator) {
    auto* assignment = output->add_assignments();
    assignment->set_key(iterator.key());
    set_wire_scalar(iterator.value(), *assignment->mutable_value());
  }
  add_stored_diagnostics(response, command.diagnostics);
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

void fill_run_summary(const RunProjection& projection, const Journal& journal,
                      v1::RunCommandResponse& response) {
  auto* output = response.mutable_run();
  output->mutable_identity()->set_run_id(projection.run_id);
  output->mutable_identity()->set_revision(projection.run_revision);
  output->mutable_identity()->set_plan_hash(projection.plan_hash);
  output->set_experiment_name(projection.experiment_name);
  output->set_desired_state(desired_state(projection.desired_state));
  output->set_observed_state(observed_state(projection.observed_state));
  output->set_current_node_id(projection.current_node_id);
  output->set_current_attempt_id(projection.current_attempt_id);
  output->set_optimizer_step(projection.optimizer_step);
  output->set_failure_summary(projection.failure_summary);
  const auto requested = journal.latest_control_revision(projection.run_id);
  const auto effective = journal.latest_effective_control_revision(projection.run_id);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  output->set_effective_control_revision(effective);
#pragma GCC diagnostic pop
  output->set_latest_requested_control_revision(requested);
  output->set_latest_effective_control_revision(effective);
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

}  // namespace

TrainVMService::TrainVMService(
    const std::filesystem::path& journal_path,
    AdapterRegistry adapter_registry,
    HostLaunchRegistry host_launch_registry,
    std::function<std::int64_t()> authority_clock)
    : TrainVMService(journal_path, std::move(adapter_registry),
                     std::move(host_launch_registry),
                     HostLaunchResolver::local_host_identity(),
                     std::move(authority_clock)) {}

TrainVMService::TrainVMService(
    const std::filesystem::path& journal_path,
    AdapterRegistry adapter_registry,
    HostLaunchRegistry host_launch_registry,
    HostIdentity authority_host,
    std::function<std::int64_t()> authority_clock)
    : authority_lock_(std::make_unique<AuthorityLock>(journal_path)),
      journal_(journal_path),
      authority_clock_(authority_clock ? std::move(authority_clock)
                                       : std::function<std::int64_t()>{[] {
                                           return std::chrono::duration_cast<
                                                      std::chrono::nanoseconds>(
                                                      std::chrono::system_clock::now()
                                                          .time_since_epoch())
                                               .count();
                                         }}),
      adapter_registry_(std::move(adapter_registry)),
      host_launch_registry_(std::move(host_launch_registry)),
      authority_host_(std::move(authority_host)),
      host_launch_resolver_(host_launch_registry_, authority_host_),
      reconciler_(journal_, adapter_registry_, command_mutex_,
                  [this] { return authority_now_ns(); }) {}

TrainVMService::~TrainVMService() = default;

std::int64_t TrainVMService::authority_now_ns() const {
  const std::int64_t now_ns = authority_clock_();
  if (now_ns < 0) {
    throw std::runtime_error("authority clock returned a negative timestamp");
  }
  return now_ns;
}

ReconcileResult TrainVMService::reconcile_once(const std::string& run_id) {
  return reconciler_.step(run_id);
}

void TrainVMService::prune_retained_launches(std::int64_t now_ns) {
  if (now_ns < 0) {
    throw std::invalid_argument(
        "retained host launch pruning clock must be nonnegative");
  }
  for (auto retained = resolved_launches_.begin();
       retained != resolved_launches_.end();) {
    const ResolvedLaunchIdentity& identity = retained->second.spec().identity;
    const auto projection = journal_.projection(identity.run_id);
    const auto active = journal_.active_lease(identity.concurrency_key, now_ns);
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
  const std::int64_t now_ns = authority_now_ns();
  prune_retained_launches(now_ns);
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
      adapter_registry_.worker_launch_request(component,
                                              node.invoke.operation);
  const std::string launch_id =
      launch.run_id + ":worker-launch:" + state.current_node_id + ":" +
      state.current_attempt_id;
  const auto durable_launch = journal_.event(launch_id);
  const auto active_lease =
      journal_.active_lease(launch.concurrency_key, now_ns);
  const nlohmann::json expected_payload{
      {"launch_nonce", launch.launch_nonce},
      {"adapter", launch.adapter},
      {"adapter_version", launch.adapter_version},
      {"code_fingerprint", launch.code_fingerprint},
      {"required_capabilities", launch.required_capabilities},
      {"concurrency_key", launch.concurrency_key},
      {"lease_id", launch.lease_id},
      {"fencing_token", launch.fencing_token},
  };
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
        now_ns);
    if (durable != retained->second.spec()) {
      throw std::runtime_error(
          "durable host launch binding disagrees with retained authority bundle");
    }
    return durable;
  }

  ResolvedLaunch resolved = host_launch_resolver_.resolve(launch, key);
  require_retained_launch_capacity(resolved.spec());
  const ResolvedLaunchSpec durable = controller.bind_worker_launch(
      resolved, host_launch_registry_, authority_host_, authority_now_ns());
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

    const auto created = journal_.event(hello.run_id + ":created");
    if (!created || created->event_type != "run.created" ||
        !created->payload.contains("submission")) {
      return {grpc::StatusCode::DATA_LOSS,
              "worker run has no durable adapter lock identity"};
    }
    adapter_registry_.validate_submission_lock(
        *plan, created->payload.at("submission"));

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
        controller.accept_worker_hello(hello, authority_now_ns());
    const Dispatch dispatch = controller.prepare_dispatch(authority_now_ns());
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
    auto& welcome = connection.welcome;
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
    welcome.set_acknowledged_worker_sequence(0);
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
      envelope.worker_sequence() != 1U || envelope.event_type().empty() ||
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
    Controller controller(*plan, journal_, connection.identity.run_id);
    controller.recover();
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
    const ExecutionState& committed =
        controller.handle_event(event, connection.identity, authority_now_ns());
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
  v1::WorkerToController result;
  if (!stream->Read(&result)) {
    return finish(cancelled(context)
                      ? grpc::Status(grpc::StatusCode::CANCELLED,
                                     "worker stream cancelled before result")
                      : grpc::Status(
                            grpc::StatusCode::FAILED_PRECONDITION,
                            "worker stream closed before its required result"));
  }
  if (result.ByteSizeLong() > kMaximumWorkerMessageBytes) {
    return finish({grpc::StatusCode::RESOURCE_EXHAUSTED,
                   "worker result exceeds 64 KiB"});
  }
  if (!result.has_event()) {
    return finish({grpc::StatusCode::UNIMPLEMENTED,
                   "this WorkerControl revision accepts one result event only"});
  }
  v1::WorkerReceipt committed;
  status = complete_worker_connection(result.event(), connection, committed);
  if (!status.ok()) return finish(std::move(status));
  v1::ControllerToWorker response;
  *response.mutable_receipt() = std::move(committed);
  if (!stream->Write(response)) {
    return finish({grpc::StatusCode::CANCELLED,
                   "worker disconnected after durable result commit"});
  }
  return finish(grpc::Status::OK);
}

grpc::Status TrainVMService::SubmitExperiment(grpc::ServerContext* context,
                                              const v1::SubmitExperimentRequest* request,
                                              v1::SubmitExperimentResponse* response) {
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
    if (request->create_run() &&
        request->expected_plan_hash() != compiled.plan->plan_hash) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "authority compiler plan hash differs from the validated preview"};
    }
    if (!request->create_run()) {
      try {
        const std::string adapter_lock_manifest =
            adapter_registry_.plan_lock_manifest(*compiled.plan);
        response->set_adapter_lock_digest(
            "sha256:" + sha256_hex(adapter_lock_manifest));
        response->set_canonical_adapter_lock(adapter_lock_manifest);
      } catch (const AdapterResolutionError& exception) {
        add_diagnostic(*response,
                       Diagnostic{.severity = Diagnostic::Severity::error,
                                  .code = "adapter.registry",
                                  .path = "/spec/components",
                                  .message = exception.what()});
      }
      return grpc::Status::OK;
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
      const auto stored_lock = stored_submission.find("adapter_lock");
      if (stored_lock == stored_submission.end() || !stored_lock->is_object()) {
        throw std::runtime_error(
            "durable run submission has no canonical adapter lock");
      }
      const std::string stored_lock_manifest = stored_lock->dump();
      const std::string stored_lock_digest =
          "sha256:" + sha256_hex(stored_lock_manifest);
      const nlohmann::json expected_submission{
          {"idempotency_key", request->idempotency_key()},
          {"source_format", request->source_format()},
          {"create_run", true},
          {"author", request->author()},
          {"reason", request->reason()},
          {"plan_hash", compiled.plan->plan_hash},
          {"adapter_lock_digest", request->expected_adapter_lock_digest()},
          {"adapter_lock", *stored_lock},
      };
      if (stored_lock_digest != request->expected_adapter_lock_digest() ||
          stored_submission != expected_submission) {
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
      response->set_adapter_lock_digest(stored_lock_digest);
      response->set_canonical_adapter_lock(stored_lock_manifest);
      auto* identity = response->mutable_run();
      identity->set_run_id(projection->run_id);
      identity->set_revision(projection->run_revision);
      identity->set_plan_hash(projection->plan_hash);
      return grpc::Status::OK;
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
    if (request->expected_adapter_lock_digest() != adapter_lock_digest) {
      return {grpc::StatusCode::FAILED_PRECONDITION,
              "authority adapter lock differs from the validated preview"};
    }
    const nlohmann::json submission_identity{
        {"idempotency_key", request->idempotency_key()},
        {"source_format", request->source_format()},
        {"create_run", request->create_run()},
        {"author", request->author()},
        {"reason", request->reason()},
        {"plan_hash", compiled.plan->plan_hash},
        {"adapter_lock_digest", adapter_lock_digest},
        {"adapter_lock", nlohmann::json::parse(adapter_lock_manifest)},
    };
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
    return grpc::Status::OK;
  } catch (const RunCreationConflict& exception) {
    return {grpc::StatusCode::ALREADY_EXISTS, exception.what()};
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
  if (!request->has_controls()) {
    return {grpc::StatusCode::UNIMPLEMENTED, "only live-control commands are implemented"};
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
    fill_run_summary(*journal_.projection(request->run_id()), journal_, *response);
    return grpc::Status::OK;
  } catch (const std::invalid_argument& exception) {
    response->set_disposition(v1::RunCommandResponse::DISPOSITION_CONFLICT);
    auto* diagnostic = response->add_diagnostics();
    diagnostic->set_severity(v1::Diagnostic::SEVERITY_ERROR);
    diagnostic->set_code("control.conflict");
    diagnostic->set_message(exception.what());
    if (const auto projection = journal_.projection(request->run_id())) {
      fill_run_summary(*projection, journal_, *response);
    }
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path,
          AdapterRegistry adapter_registry,
          HostLaunchRegistry host_launch_registry) {
  if (journal_path.empty() || socket_path.empty()) {
    throw std::invalid_argument("serve requires journal and socket paths");
  }
  const auto absolute_socket = std::filesystem::absolute(socket_path);
  const auto parent = absolute_socket.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  SignalMaskGuard signal_mask;
  // Acquire journal authority before touching the socket. Otherwise a second
  // daemon pointed at the same paths could unlink the live authority's socket
  // and only then discover that it cannot acquire the journal lock.
  TrainVMService service(journal_path, std::move(adapter_registry),
                         std::move(host_launch_registry));
  SocketAuthorityLock socket_authority(absolute_socket);
  remove_stale_socket(absolute_socket);
  SocketCleanupGuard socket_cleanup(absolute_socket);
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(static_cast<int>(kMaximumSubmissionBytes));
  builder.AddListeningPort("unix:" + absolute_socket.string(), grpc::InsecureServerCredentials());
  builder.RegisterService(static_cast<v1::TrainVM::Service*>(&service));
  builder.RegisterService(static_cast<v1::WorkerControl::Service*>(&service));
  std::unique_ptr<grpc::Server> server;
  {
    // The Unix socket must be born owner-only. chmod after binding is retained as
    // defense in depth, but is too late to close the bind-to-chmod access window.
    UmaskGuard owner_only(S_IRWXG | S_IRWXO);
    server = builder.BuildAndStart();
  }
  if (!server) {
    throw std::runtime_error("could not start TrainVM authority on " + absolute_socket.string());
  }
  socket_cleanup.claim();
  const auto shutdown_server = [&] {
    socket_cleanup.preserve_replacement();
    server->Shutdown();
    server->Wait();
    socket_cleanup.restore_replacement();
  };
  if (::chmod(absolute_socket.c_str(), S_IRUSR | S_IWUSR) != 0) {
    shutdown_server();
    throw std::runtime_error("could not restrict TrainVM authority socket permissions: " +
                             std::string(std::strerror(errno)));
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
