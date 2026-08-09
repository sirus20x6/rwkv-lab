#include "trainvm/gpu_fault_observer.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumStateBytes = 16U << 10U;

[[noreturn]] void reject(std::string message) {
  throw HostdStateError(std::move(message));
}

std::string system_error(std::string_view operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool boot_uuid(std::string_view value) {
  if (value.size() != 36U || value[8U] != '-' || value[13U] != '-' ||
      value[18U] != '-' || value[23U] != '-')
    return false;
  for (std::size_t index = 0U; index < value.size(); ++index) {
    if (index == 8U || index == 13U || index == 18U || index == 23U)
      continue;
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f')))
      return false;
  }
  return true;
}

bool digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

nlohmann::json state_body(const GpuFaultObserverState& state) {
  return {{"api_version", state.api_version},
          {"blocked", state.blocked},
          {"boot_id", state.boot_id},
          {"event_count", state.event_count},
          {"last_event_digest", state.last_event_digest},
          {"last_xid", state.last_xid},
          {"observed_boottime_ns", state.observed_boottime_ns}};
}

GpuFaultObserverState seal(GpuFaultObserverState state) {
  const bool no_events = state.event_count == 0U;
  if (state.api_version != kGpuFaultObserverStateApiVersion ||
      !boot_uuid(state.boot_id) || state.observed_boottime_ns == 0U ||
      (no_events && (state.blocked || state.last_xid != 0U ||
                     !state.last_event_digest.empty())) ||
      (!no_events && (!state.blocked || state.last_xid == 0U ||
                      !digest(state.last_event_digest)))) {
    reject("GPU fault observer state is not canonical");
  }
  state.state_digest = "sha256:" + sha256_hex(state_body(state).dump());
  return state;
}

std::string read_all(int descriptor, std::size_t expected_size) {
  std::string result(expected_size, '\0');
  std::size_t offset = 0U;
  while (offset < result.size()) {
    const ssize_t count =
        ::read(descriptor, result.data() + offset, result.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) reject(system_error("could not read GPU fault state"));
    if (count == 0) reject("GPU fault state was truncated while reading");
    offset += static_cast<std::size_t>(count);
  }
  std::array<char, 1> extra{};
  ssize_t count = 0;
  do {
    count = ::read(descriptor, extra.data(), extra.size());
  } while (count < 0 && errno == EINTR);
  if (count != 0) reject("GPU fault state changed size while reading");
  return result;
}

void write_all(int descriptor, std::string_view bytes) {
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    const ssize_t count =
        ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) reject(system_error("could not write GPU fault state"));
    offset += static_cast<std::size_t>(count);
  }
}

}  // namespace

std::optional<NvidiaXidEvent> parse_nvidia_xid_line(
    std::string_view line) {
  constexpr std::string_view marker = "NVRM: Xid (";
  const std::size_t begin = line.find(marker);
  if (begin == std::string_view::npos) return std::nullopt;
  const std::size_t close = line.find("):", begin + marker.size());
  if (close == std::string_view::npos) return std::nullopt;
  std::size_t cursor = close + 2U;
  while (cursor < line.size() &&
         (line[cursor] == ' ' || line[cursor] == '\t'))
    ++cursor;
  const std::size_t digits = cursor;
  while (cursor < line.size() && line[cursor] >= '0' &&
         line[cursor] <= '9')
    ++cursor;
  if (cursor == digits ||
      (cursor < line.size() && line[cursor] != ',' && line[cursor] != ' ' &&
       line[cursor] != '\t'))
    return std::nullopt;
  std::uint32_t code = 0U;
  const auto parsed =
      std::from_chars(line.data() + digits, line.data() + cursor, code);
  if (parsed.ec != std::errc{} || parsed.ptr != line.data() + cursor ||
      code == 0U)
    return std::nullopt;
  return NvidiaXidEvent{
      .code = code,
      .line_digest = "sha256:" + sha256_hex(std::string(line)),
  };
}

GpuFaultObserverState make_gpu_fault_observer_state(
    std::string boot_id, std::uint64_t observed_boottime_ns) {
  return seal({.api_version = std::string(kGpuFaultObserverStateApiVersion),
               .boot_id = std::move(boot_id),
               .blocked = false,
               .event_count = 0U,
               .last_xid = 0U,
               .last_event_digest = {},
               .observed_boottime_ns = observed_boottime_ns,
               .state_digest = {}});
}

GpuFaultObserverState update_gpu_fault_observer_state(
    GpuFaultObserverState state, std::uint64_t observed_boottime_ns,
    std::optional<NvidiaXidEvent> event) {
  state = seal(std::move(state));
  if (observed_boottime_ns < state.observed_boottime_ns)
    reject("GPU fault observer boottime moved backwards");
  state.observed_boottime_ns = observed_boottime_ns;
  if (event) {
    if (event->code == 0U || !digest(event->line_digest) ||
        state.event_count == std::numeric_limits<std::uint64_t>::max())
      reject("GPU fault event is malformed or exhausted");
    state.blocked = true;
    ++state.event_count;
    state.last_xid = event->code;
    state.last_event_digest = std::move(event->line_digest);
  }
  state.state_digest.clear();
  return seal(std::move(state));
}

std::string gpu_fault_observer_state_json(
    const GpuFaultObserverState& state) {
  const GpuFaultObserverState canonical = seal(state);
  nlohmann::json value = state_body(canonical);
  value["state_digest"] = canonical.state_digest;
  return value.dump() + "\n";
}

GpuFaultObserverState gpu_fault_observer_state_from_json(
    std::string_view source) {
  nlohmann::json value;
  try {
    bool duplicate = false;
    std::vector<std::set<std::string>> keys;
    const nlohmann::json::parser_callback_t callback =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (keys.size() <= index + 1U) keys.resize(index + 2U);
            keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (keys.size() <= index) keys.resize(index + 1U);
            if (!keys[index].insert(parsed.get<std::string>()).second)
              duplicate = true;
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     keys.size() > index + 1U) {
            keys[index + 1U].clear();
          }
          return true;
        };
    value = nlohmann::json::parse(source, callback);
    if (duplicate) reject("GPU fault observer state has a duplicate field");
    const std::array<std::string_view, 8> fields = {
        "api_version",          "blocked",    "boot_id",
        "event_count",          "last_event_digest",
        "last_xid",             "observed_boottime_ns",
        "state_digest"};
    if (!value.is_object() || value.size() != fields.size() ||
        !std::ranges::all_of(fields, [&value](std::string_view field) {
          return value.contains(std::string(field));
        }))
      reject("GPU fault observer state schema is not closed");
    GpuFaultObserverState state{
        .api_version = value.at("api_version").get<std::string>(),
        .boot_id = value.at("boot_id").get<std::string>(),
        .blocked = value.at("blocked").get<bool>(),
        .event_count = value.at("event_count").get<std::uint64_t>(),
        .last_xid = value.at("last_xid").get<std::uint32_t>(),
        .last_event_digest =
            value.at("last_event_digest").get<std::string>(),
        .observed_boottime_ns =
            value.at("observed_boottime_ns").get<std::uint64_t>(),
        .state_digest = value.at("state_digest").get<std::string>(),
    };
    const std::string supplied = state.state_digest;
    state.state_digest.clear();
    state = seal(std::move(state));
    if (state.state_digest != supplied)
      reject("GPU fault observer state digest does not match its fields");
    return state;
  } catch (const HostdStateError&) {
    throw;
  } catch (const nlohmann::json::exception&) {
    reject("GPU fault observer state is not valid canonical JSON");
  }
}

GpuFaultObserverState read_gpu_fault_observer_state(
    const std::filesystem::path& path, uid_t expected_owner) {
  const int descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) reject(system_error("could not open GPU fault state"));
  struct Close final {
    int descriptor;
    ~Close() { (void)::close(descriptor); }
  } close{descriptor};
  struct stat before {};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_uid != expected_owner || (before.st_mode & 0022U) != 0U ||
      before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) > kMaximumStateBytes)
    reject("GPU fault state file identity, owner, mode, or size is unsafe");
  const std::string source =
      read_all(descriptor, static_cast<std::size_t>(before.st_size));
  struct stat after {};
  if (::fstat(descriptor, &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
    reject("GPU fault state changed while it was being read");
  return gpu_fault_observer_state_from_json(source);
}

void write_gpu_fault_observer_state(
    const std::filesystem::path& path,
    const GpuFaultObserverState& state, uid_t expected_directory_owner) {
  const std::filesystem::path parent = path.parent_path();
  const std::string filename = path.filename().string();
  if (!path.is_absolute() || parent.empty() || filename.empty() ||
      filename == "." || filename == ".." || filename.find('/') != filename.npos)
    reject("GPU fault state path is not canonical");
  const int directory =
      ::open(parent.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  if (directory < 0)
    reject(system_error("could not open GPU fault state directory"));
  struct CloseDirectory final {
    int descriptor;
    ~CloseDirectory() { (void)::close(descriptor); }
  } close_directory{directory};
  struct stat directory_status {};
  if (::fstat(directory, &directory_status) != 0 ||
      !S_ISDIR(directory_status.st_mode) ||
      directory_status.st_uid != expected_directory_owner ||
      (directory_status.st_mode & 0022U) != 0U)
    reject("GPU fault state directory authority is unsafe");
  const std::string temporary =
      "." + filename + ".tmp." + std::to_string(::getpid()) + "." +
      std::to_string(linux_boottime_now_ns());
  const int output = ::openat(directory, temporary.c_str(),
                              O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                  O_NOFOLLOW,
                              0600);
  if (output < 0)
    reject(system_error("could not create GPU fault state temporary"));
  bool output_open = true;
  bool renamed = false;
  try {
    const std::string contents = gpu_fault_observer_state_json(state);
    write_all(output, contents);
    if (::fsync(output) != 0)
      reject(system_error("could not fsync GPU fault state"));
    const int close_result = ::close(output);
    output_open = false;
    if (close_result != 0)
      reject(system_error("could not close GPU fault state"));
    if (::renameat(directory, temporary.c_str(), directory,
                   filename.c_str()) != 0)
      reject(system_error("could not publish GPU fault state"));
    renamed = true;
    if (::fsync(directory) != 0)
      reject(system_error("could not fsync GPU fault state directory"));
  } catch (...) {
    if (!renamed) {
      if (output_open) (void)::close(output);
      (void)::unlinkat(directory, temporary.c_str(), 0);
    }
    throw;
  }
}

std::uint64_t linux_boottime_now_ns() {
  struct timespec now {};
  if (::clock_gettime(CLOCK_BOOTTIME, &now) != 0 || now.tv_sec < 0 ||
      now.tv_nsec < 0)
    reject(system_error("could not sample Linux boottime"));
  const auto seconds = static_cast<std::uint64_t>(now.tv_sec);
  const auto nanoseconds = static_cast<std::uint64_t>(now.tv_nsec);
  if (seconds >
      (std::numeric_limits<std::uint64_t>::max() - nanoseconds) /
          1'000'000'000ULL)
    reject("Linux boottime overflowed GPU fault evidence");
  return seconds * 1'000'000'000ULL + nanoseconds;
}

LinuxGpuFaultAdmissionGuard::LinuxGpuFaultAdmissionGuard(
    std::filesystem::path state_path, std::string expected_boot_id,
    std::uint64_t maximum_state_age_ns, uid_t expected_owner)
    : state_path_(std::move(state_path)),
      expected_boot_id_(std::move(expected_boot_id)),
      maximum_state_age_ns_(maximum_state_age_ns),
      expected_owner_(expected_owner) {
  if (!state_path_.is_absolute() ||
      state_path_.lexically_normal() != state_path_ ||
      !boot_uuid(expected_boot_id_) || maximum_state_age_ns_ == 0U ||
      maximum_state_age_ns_ > 60'000'000'000ULL)
    reject("GPU fault admission guard configuration is invalid");
}

void LinuxGpuFaultAdmissionGuard::require_new_grant_allowed() {
  const GpuFaultObserverState state =
      read_gpu_fault_observer_state(state_path_, expected_owner_);
  const std::uint64_t now = linux_boottime_now_ns();
  if (state.boot_id != expected_boot_id_)
    reject("GPU fault observer state belongs to another boot");
  if (state.observed_boottime_ns > now ||
      now - state.observed_boottime_ns > maximum_state_age_ns_)
    reject("GPU fault observer heartbeat is stale");
  if (state.blocked)
    reject("GPU fault observer blocked new grants after NVIDIA Xid " +
           std::to_string(state.last_xid));
}

}  // namespace trainvm
