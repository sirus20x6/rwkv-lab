#include "trainvm/service.hpp"

#include "trainvm/lifecycle_admission.hpp"

#include "trainvm/controller.hpp"
#include "trainvm/document.hpp"
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
namespace {

int open_directory_by_components(const std::filesystem::path& absolute_path,
                                 bool create_missing) {
  if (!absolute_path.is_absolute()) {
    throw std::runtime_error("authority directory path must be absolute");
  }
  int current = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (current < 0) {
    throw std::runtime_error("could not open filesystem root: " +
                             std::string(std::strerror(errno)));
  }
  for (const auto& part : absolute_path.relative_path()) {
    const std::string component = part.string();
    if (component.empty() || component == "." || component == ".." ||
        component.find('/') != std::string::npos) {
      (void)::close(current);
      throw std::runtime_error("authority directory has a noncanonical component");
    }
    int next = ::openat(current, component.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (next < 0 && errno == ENOENT && create_missing) {
      if (::mkdirat(current, component.c_str(), S_IRWXU) != 0 &&
          errno != EEXIST) {
        const std::string message = std::strerror(errno);
        (void)::close(current);
        throw std::runtime_error("could not create authority directory component " +
                                 component + ": " + message);
      }
      next = ::openat(current, component.c_str(),
                      O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    }
    if (next < 0) {
      const std::string message = std::strerror(errno);
      (void)::close(current);
      throw std::runtime_error("could not securely resolve authority directory component " +
                               component + ": " + message);
    }
    (void)::close(current);
    current = next;
  }
  return current;
}

bool safe_owned_regular(const struct stat& status, uid_t owner) {
  return S_ISREG(status.st_mode) && status.st_uid == owner &&
         status.st_nlink == 1 &&
         (status.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}

void require_safe_sqlite_auxiliary(int directory_descriptor,
                                   std::string_view name, uid_t owner) {
  struct stat status {};
  const std::string owned_name(name);
  if (::fstatat(directory_descriptor, owned_name.c_str(), &status,
                AT_SYMLINK_NOFOLLOW) != 0) {
    if (errno == ENOENT) return;
    throw std::runtime_error("could not inspect SQLite auxiliary " + owned_name +
                             ": " + std::strerror(errno));
  }
  if (!safe_owned_regular(status, owner)) {
    throw std::runtime_error(
        "SQLite auxiliary is not a safe unique owned regular file " +
        owned_name);
  }
}

}  // namespace

AuthorityLock::AuthorityLock(const std::filesystem::path& journal_path) {
  const auto absolute_journal =
      std::filesystem::absolute(journal_path).lexically_normal();
  const std::filesystem::path parent = absolute_journal.parent_path();
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
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
      descriptor_ = -1;
    }
    if (journal_descriptor_ >= 0) {
      (void)::close(journal_descriptor_);
      journal_descriptor_ = -1;
    }
    if (directory_descriptor_ >= 0) {
      (void)::close(directory_descriptor_);
      directory_descriptor_ = -1;
    }
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
    directory_descriptor_ = open_directory_by_components(parent, true);
  } catch (const std::exception& exception) {
    fail(exception.what());
  }
  struct stat directory_status {};
  if (directory_descriptor_ < 0 ||
      ::fstat(directory_descriptor_, &directory_status) != 0 ||
      !S_ISDIR(directory_status.st_mode) ||
      directory_status.st_uid != ::geteuid() ||
      (directory_status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    const std::string message = std::strerror(errno);
    fail("authority journal directory is not a safe owned directory " +
         parent.string() + ": " + message);
  }

  const std::string lock_name = filename + ".authority.lock";
  descriptor_ = ::openat(directory_descriptor_, lock_name.c_str(),
                         O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_RDWR,
                         S_IRUSR | S_IWUSR);
  struct stat lock_status {};
  if (descriptor_ < 0 || ::fstat(descriptor_, &lock_status) != 0 ||
      !S_ISREG(lock_status.st_mode) || lock_status.st_uid != ::geteuid() ||
      lock_status.st_nlink != 1 ||
      (lock_status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    const std::string message = std::strerror(errno);
    fail("authority sidecar is not a safe unique regular file " + lock_name +
         ": " + message);
  }
  if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
    const std::string message = std::strerror(errno);
    fail("another TrainVM authority owns " + lock_name + ": " + message);
  }
  if (::fchmod(descriptor_, S_IRUSR | S_IWUSR) != 0) {
    const std::string message = std::strerror(errno);
    fail("could not restrict authority lock " + lock_name + ": " + message);
  }

  journal_descriptor_ = ::openat(
      directory_descriptor_, filename.c_str(),
      O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_RDWR, S_IRUSR | S_IWUSR);
  struct stat journal_status {};
  if (journal_descriptor_ < 0 ||
      ::fstat(journal_descriptor_, &journal_status) != 0 ||
      !S_ISREG(journal_status.st_mode) ||
      journal_status.st_uid != ::geteuid() || journal_status.st_nlink != 1 ||
      (journal_status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    const std::string message = std::strerror(errno);
    fail("authority journal is not a safe unique regular file " + filename +
         ": " + message);
  }
  if (::flock(journal_descriptor_, LOCK_EX | LOCK_NB) != 0) {
    const std::string message = std::strerror(errno);
    fail("another TrainVM authority owns " + filename + ": " + message);
  }
  if (::fchmod(journal_descriptor_, S_IRUSR | S_IWUSR) != 0) {
    const std::string message = std::strerror(errno);
    fail("could not restrict authority journal " + filename + ": " + message);
  }

  try {
    for (const std::string_view suffix :
         {std::string_view{"-journal"}, std::string_view{"-wal"},
          std::string_view{"-shm"}}) {
      require_safe_sqlite_auxiliary(directory_descriptor_, filename +
                                        std::string(suffix),
                                    ::geteuid());
    }
  } catch (const std::exception& exception) {
    fail(exception.what());
  }

  journal_identity_ = {
      .directory_path = parent.string(),
      .journal_name = filename,
      .authority_name = lock_name,
      .directory_device = static_cast<std::uint64_t>(directory_status.st_dev),
      .directory_inode = static_cast<std::uint64_t>(directory_status.st_ino),
      .device = static_cast<std::uint64_t>(journal_status.st_dev),
      .inode = static_cast<std::uint64_t>(journal_status.st_ino),
      .authority_device = static_cast<std::uint64_t>(lock_status.st_dev),
      .authority_inode = static_cast<std::uint64_t>(lock_status.st_ino),
      .owner_uid = static_cast<std::uint64_t>(::geteuid()),
  };
  stable_journal_path_ = std::filesystem::path("/proc/self/fd") /
                         std::to_string(directory_descriptor_) / filename;
}

AuthorityLock::~AuthorityLock() {
  if (kernel_namespace_descriptor_ >= 0) {
    ::close(kernel_namespace_descriptor_);
  }
  if (descriptor_ >= 0) {
    ::close(descriptor_);
  }
  if (journal_descriptor_ >= 0) {
    ::close(journal_descriptor_);
  }
  if (directory_descriptor_ >= 0) {
    ::close(directory_descriptor_);
  }
}

const std::filesystem::path& AuthorityLock::journal_path() const noexcept {
  return stable_journal_path_;
}

const JournalFileIdentity& AuthorityLock::journal_identity() const noexcept {
  return journal_identity_;
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

void populate_invocation(v1::WorkerWelcome& welcome,
                         const WorkerInvocationSpec& invocation) {
  const std::string canonical =
      worker_invocation_canonical_json(invocation);
  welcome.set_canonical_invocation_json(canonical);
  welcome.set_invocation_digest(invocation.invocation_digest);
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
    ICacheQualificationEvidenceResolver* cache_qualification)
    : TrainVMService(journal_path, std::move(adapter_registry),
                     std::move(host_launch_registry),
                     HostLaunchResolver::local_host_identity(),
                     std::move(authority_clock),
                     HostGrantEnforcement::required,
                     std::move(training_components), {}, {}, {},
                     cache_qualification) {
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
    ICacheQualificationEvidenceResolver* cache_qualification)
    : authority_lock_(std::make_unique<AuthorityLock>(journal_path)),
      journal_(authority_lock_->journal_path(),
               authority_lock_->journal_identity(),
               host_grant_enforcement, authority_host),
      authority_clock_(
          authority_clock
              ? std::make_shared<AuthorityClock>(std::move(authority_clock))
              : std::make_shared<AuthorityClock>()),
      lease_renewals_(journal_, authority_clock_),
      adapter_registry_(std::move(adapter_registry)),
      host_launch_registry_(std::move(host_launch_registry)),
      training_components_(std::move(training_components)),
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
                  [this] { return authority_now(); }, cache_qualification) {
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
  }
  reconciliation_condition_.notify_one();
}

void TrainVMService::reconcile_until_quiescent(const std::string& run_id) {
  for (std::size_t step = 0U; step < kMaximumImmediateReconcileSteps;
       ++step) {
    const ReconcileResult result = reconcile_once(run_id);
    switch (result.disposition) {
      case ReconcileDisposition::lease_acquired:
      case ReconcileDisposition::host_grant_acquired:
      case ReconcileDisposition::host_process_exited:
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
        return;
    }
  }
  // A legal cyclic workflow can consume the per-wake work budget. Requeue it
  // instead of monopolizing the authority thread or treating bounded work as
  // a workflow failure.
  notify_reconciliation(run_id);
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

void TrainVMService::reconciliation_loop(std::stop_token stop) {
  constexpr auto kSupervisorCadence = std::chrono::milliseconds(250);
  while (!stop.stop_requested()) {
    std::set<std::string, std::less<>> run_ids;
    {
      std::unique_lock lock(reconciliation_mutex_);
      (void)reconciliation_condition_.wait_for(
          lock, kSupervisorCadence,
          [&] { return stop.stop_requested() ||
                       !reconciliation_wake_runs_.empty(); });
      if (stop.stop_requested()) break;
      run_ids.swap(reconciliation_wake_runs_);
    }

    try {
      std::vector<RunProjection> page;
      {
        std::scoped_lock lock(command_mutex_);
        page = journal_.reconcilable_projections(
            reconciliation_scan_cursor_, kReconciliationPageSize);
      }
      {
        std::scoped_lock lock(reconciliation_mutex_);
        reconciliation_scan_cursor_ =
            page.size() == kReconciliationPageSize
                ? page.back().run_id
                : std::string{};
      }
      for (const RunProjection& projection : page) {
        run_ids.insert(projection.run_id);
      }
    } catch (const std::exception& exception) {
      record_reconciliation_failure("__scan__", exception.what());
      return;
    }

    for (const std::string& run_id : run_ids) {
      if (stop.stop_requested()) break;
      try {
        reconcile_until_quiescent(run_id);
        synchronize_lease_renewal(run_id);
        std::scoped_lock lock(reconciliation_mutex_);
        reconciliation_failures_.erase(run_id);
      } catch (const std::exception& exception) {
        record_reconciliation_failure(run_id, exception.what());
      } catch (...) {
        record_reconciliation_failure(
            run_id,
            "reconciliation failed with a non-standard exception");
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

  ResolvedLaunch resolved = host_launch_resolver_.resolve(launch, key);
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
  if (!projection || !plan || projection->current_node_id != identity.node_id ||
      projection->current_attempt_id != identity.attempt_id) {
    throw OperationPreconditionError(
        "host process launch cannot recover its exact resource policy");
  }
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
      const WorkerInvocationSpec candidate =
          build_worker_invocation(*plan, context);
      invocation = controller.bind_worker_invocation(
          candidate, connection.identity, authority_now());
    }
    auto& welcome = connection.welcome;
    connection.publishes = invocation->publishes;
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
        .optimizer_step = metric.step(),
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
    const std::int64_t published_at_ns = timestamp_ns(artifact.published_at());
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
        .optimizer_step = std::nullopt,
        .payload = {{"artifact_id", artifact.artifact_id()},
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
                    {"published_at_ns", published_at_ns}},
    };
    return commit_worker_observation(event, connection, acknowledged);
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
      if (stored_submission != expected_submission) {
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
      request->replay_limit() > kMaximumReplayEvents) {
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
            .run_ids = run_ids,
            .event_types = event_types,
            .limit = query_limit,
        });
      }
      for (const SequencedEvent& event : events) {
        v1::EventEnvelope output = wire_event(event);
        if (!writer->Write(output)) {
          return {grpc::StatusCode::CANCELLED,
                  "event stream closed by its reader"};
        }
        cursor = event.journal_sequence;
        ++replayed;
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
  if (request->adapter() != "trainvm.training-components" ||
      request->version() != "1.0.0") {
    return {grpc::StatusCode::NOT_FOUND,
            "no descriptor matches the exact requested provider and version"};
  }
  try {
    const std::string canonical =
        training_components_.document_json().dump();
    response->set_schema_json(canonical);
    response->set_schema_hash(training_components_.registry_digest());
    return grpc::Status::OK;
  } catch (const std::exception& exception) {
    return {grpc::StatusCode::DATA_LOSS, exception.what()};
  }
}

int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path,
          AdapterRegistry adapter_registry,
          HostLaunchRegistry host_launch_registry,
          TrainingComponentRegistry training_components,
          std::optional<HostdClientConfiguration> hostd_configuration) {
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
                         std::move(host_launch_registry), {},
                         std::move(training_components),
                         std::move(hostd_configuration),
                         "unix:" + absolute_socket.string());
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
    service.stop_reconciliation_supervisor();
    server->Wait();
    socket_cleanup.restore_replacement();
  };
  if (::chmod(absolute_socket.c_str(), S_IRUSR | S_IWUSR) != 0) {
    shutdown_server();
    throw std::runtime_error("could not restrict TrainVM authority socket permissions: " +
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
