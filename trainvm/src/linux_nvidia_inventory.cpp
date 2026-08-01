#include "trainvm/linux_nvidia_inventory.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <openssl/sha.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#include <unistd.h>

#if __has_include(<nvml.h>)
#include <nvml.h>
#define TRAINVM_HAS_OFFICIAL_NVML_ABI 1
#else
#define TRAINVM_HAS_OFFICIAL_NVML_ABI 0
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumFileBytes = 16U * 1024U;
constexpr std::size_t kNvmlStringBytes = 96U;
constexpr unsigned int kNvmlSuccess = 0U;
constexpr unsigned int kNvmlErrorNotSupported = 3U;
constexpr unsigned int kNvmlErrorInsufficientSize = 7U;
constexpr unsigned int kNvmlErrorNotFound = 6U;
constexpr unsigned int kNvmlFeatureEnabled = 1U;
constexpr unsigned int kNvmlInstanceNotApplicable =
    std::numeric_limits<unsigned int>::max();
constexpr std::string_view kNvidiaVendor = "0x10de";

class Descriptor final {
public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }
  Descriptor(Descriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor &operator=(Descriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }

private:
  int value_;
};

bool bounded_identifier(std::string_view value) {
  if (value.empty() ||
      value.size() > HostResourceBounds::maximum_identifier_bytes)
    return false;
  return std::ranges::all_of(value, [](char character) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    return alphanumeric || character == '.' || character == '_' ||
           character == ':' || character == '/' || character == '-';
  });
}

bool printable_bounded(std::string_view value, std::size_t maximum) {
  return value.size() <= maximum &&
         std::ranges::all_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return byte >= 0x20U && byte <= 0x7eU;
         });
}

bool lower_hex(char character) {
  return (character >= '0' && character <= '9') ||
         (character >= 'a' && character <= 'f');
}

bool canonical_uuid(std::string_view value, std::string_view prefix) {
  if (!value.starts_with(prefix) || value.size() != prefix.size() + 36U)
    return false;
  const auto suffix = value.substr(prefix.size());
  for (std::size_t index = 0U; index < suffix.size(); ++index) {
    const bool hyphen =
        index == 8U || index == 13U || index == 18U || index == 23U;
    if (hyphen ? suffix[index] != '-' : !lower_hex(suffix[index]))
      return false;
  }
  return true;
}

bool canonical_bdf(std::string_view value) {
  if (value.size() != 12U || value[4] != ':' || value[7] != ':' ||
      value[10] != '.')
    return false;
  for (const std::size_t index : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U, 11U})
    if (!lower_hex(value[index]))
      return false;
  return true;
}

bool known_context(ResourceContextDisposition disposition) {
  return disposition == ResourceContextDisposition::absent ||
         disposition == ResourceContextDisposition::present;
}

std::string trim_ascii(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                            value.back() == ' ' || value.back() == '\t'))
    value.pop_back();
  std::size_t begin = 0U;
  while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t' ||
                                  value[begin] == '\n' || value[begin] == '\r'))
    ++begin;
  value.erase(0U, begin);
  return value;
}

std::optional<std::string>
unique_line_value(std::string_view text, std::string_view key, char separator) {
  std::optional<std::string> result;
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const auto end = text.find('\n', begin);
    const std::string_view line = text.substr(
        begin, (end == std::string_view::npos ? text.size() : end) - begin);
    const auto split = line.find(separator);
    if (split != std::string_view::npos &&
        trim_ascii(std::string(line.substr(0U, split))) == key) {
      if (result)
        return std::nullopt;
      result = trim_ascii(std::string(line.substr(split + 1U)));
    }
    if (end == std::string_view::npos)
      break;
    begin = end + 1U;
  }
  return result;
}

std::optional<std::uint32_t>
nvidia_frontend_major_impl(std::string_view devices) {
  if (devices.empty() || devices.size() > kMaximumFileBytes)
    return std::nullopt;
  std::optional<std::uint32_t> result;
  bool character_section = false;
  std::size_t begin = 0U;
  while (begin < devices.size()) {
    const auto end = devices.find('\n', begin);
    const std::string line = trim_ascii(std::string(devices.substr(
        begin,
        (end == std::string_view::npos ? devices.size() : end) - begin)));
    if (line == "Character devices:")
      character_section = true;
    else if (line == "Block devices:")
      character_section = false;
    const auto separator = line.find_first_of(" \t");
    if (character_section && separator != std::string::npos) {
      const std::string name = trim_ascii(line.substr(separator + 1U));
      if (name == "nvidia" || name == "nvidia-frontend") {
        std::uint32_t parsed = 0U;
        const auto converted =
            std::from_chars(line.data(), line.data() + separator, parsed);
        if (converted.ec != std::errc{} ||
            converted.ptr != line.data() + separator ||
            (result && *result != parsed))
          return std::nullopt;
        result = parsed;
      }
    }
    if (end == std::string_view::npos)
      break;
    begin = end + 1U;
  }
  return result;
}

bool pinned_pci_mapping_impl(std::string_view pci_uevent,
                             std::string_view gpu_information,
                             std::string_view uuid, std::string_view bdf,
                             std::uint32_t minor_number) {
  if (pci_uevent.empty() || pci_uevent.size() > kMaximumFileBytes ||
      gpu_information.empty() || gpu_information.size() > kMaximumFileBytes ||
      !canonical_uuid(uuid, "GPU-") || !canonical_bdf(bdf))
    return false;
  const auto driver = unique_line_value(pci_uevent, "DRIVER", '=');
  const auto slot = unique_line_value(pci_uevent, "PCI_SLOT_NAME", '=');
  const auto observed_uuid =
      unique_line_value(gpu_information, "GPU UUID", ':');
  const auto observed_slot =
      unique_line_value(gpu_information, "Bus Location", ':');
  const auto observed_minor =
      unique_line_value(gpu_information, "Device Minor", ':');
  if (!driver || *driver != "nvidia" || !slot || *slot != bdf ||
      !observed_uuid || *observed_uuid != uuid || !observed_slot ||
      *observed_slot != bdf || !observed_minor)
    return false;
  std::uint32_t parsed_minor = 0U;
  const auto converted = std::from_chars(
      observed_minor->data(), observed_minor->data() + observed_minor->size(),
      parsed_minor);
  return converted.ec == std::errc{} &&
         converted.ptr == observed_minor->data() + observed_minor->size() &&
         parsed_minor == minor_number;
}

std::string normalize_bdf(std::string value) {
  std::ranges::transform(value, value.begin(), [](char character) {
    if (character >= 'A' && character <= 'F')
      return static_cast<char>(character - 'A' + 'a');
    return character;
  });
  if (value.size() == 16U && value.starts_with("0000"))
    value.erase(0U, 4U);
  return canonical_bdf(value) ? value : std::string{};
}

template <std::size_t Size>
std::optional<std::string>
bounded_c_string(const std::array<char, Size> &value) {
  const auto end = std::ranges::find(value, '\0');
  if (end == value.end())
    return std::nullopt;
  return std::string(value.begin(), end);
}

std::string pci_bdf_from_v2_impl(std::string_view legacy, std::uint32_t domain,
                                 std::uint32_t bus, std::uint32_t device) {
  const auto terminator = legacy.find('\0');
  if (terminator != std::string_view::npos) {
    const std::string legacy_value(legacy.substr(0U, terminator));
    if (const auto normalized = normalize_bdf(legacy_value);
        !normalized.empty())
      return normalized;
  }
  if (domain > 0xffffU || bus > 0xffU || device > 0x1fU)
    return {};
  std::array<char, 16U> buffer{};
  const int length = std::snprintf(buffer.data(), buffer.size(),
                                   "%04x:%02x:%02x.0", domain, bus, device);
  if (length != 12)
    return {};
  return normalize_bdf(
      std::string(buffer.data(), static_cast<std::size_t>(length)));
}

std::uint64_t monotonic_now_ns() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

bool unsafe_loader_environment() {
  static constexpr std::array names{"LD_LIBRARY_PATH", "LD_PRELOAD",
                                    "LD_AUDIT"};
  return std::ranges::any_of(names, [](const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0';
  });
}

bool root_owned_nonwritable(const char *path, bool regular) {
  struct stat status{};
  if (::stat(path, &status) != 0 || status.st_uid != 0U ||
      (status.st_mode & (S_IWGRP | S_IWOTH)) != 0)
    return false;
  return regular ? S_ISREG(status.st_mode) : S_ISDIR(status.st_mode);
}

bool attested_library_symbol(const void *symbol, std::string &detail) {
  Dl_info info{};
  if (symbol == nullptr || ::dladdr(symbol, &info) == 0 ||
      info.dli_fname == nullptr || info.dli_fname[0] != '/') {
    detail = "nvml-loader-unresolved";
    return false;
  }
  std::array<char, PATH_MAX> resolved{};
  if (::realpath(info.dli_fname, resolved.data()) == nullptr ||
      !root_owned_nonwritable(resolved.data(), true)) {
    detail = "nvml-loader-object-untrusted";
    return false;
  }
  std::string parent(resolved.data());
  for (;;) {
    const auto slash = parent.find_last_of('/');
    parent = slash == 0U ? "/" : parent.substr(0U, slash);
    if (!root_owned_nonwritable(parent.c_str(), false)) {
      detail = "nvml-loader-directory-untrusted";
      return false;
    }
    if (parent == "/")
      break;
  }
  detail = unsafe_loader_environment() ? "nvml-loader-env-unsafe"
                                       : "nvml-loader-attested";
  return !unsafe_loader_environment();
}

std::string digest_revision(std::string_view evidence) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  (void)::SHA256(reinterpret_cast<const unsigned char *>(evidence.data()),
                 evidence.size(), digest.data());
  static constexpr char digits[] = "0123456789abcdef";
  std::string result = "nvidia-";
  result.reserve(7U + digest.size() * 2U);
  for (const unsigned char byte : digest) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

class SecureRoot final {
public:
  explicit SecureRoot(const char *path,
                      std::optional<long> expected_filesystem = std::nullopt)
      : descriptor_(
            ::open(path, O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)) {
    if (descriptor_.get() < 0 || !expected_filesystem)
      return;
    struct statfs filesystem{};
    if (::fstatfs(descriptor_.get(), &filesystem) != 0 ||
        filesystem.f_type != *expected_filesystem)
      descriptor_ = Descriptor{};
  }

  [[nodiscard]] bool available() const noexcept {
    return descriptor_.get() >= 0;
  }

  [[nodiscard]] std::optional<std::string>
  read(std::string_view relative,
       std::size_t maximum = kMaximumFileBytes) const {
    const int opened = open(relative, O_RDONLY | O_CLOEXEC);
    if (opened < 0)
      return std::nullopt;
    Descriptor file(opened);
    struct stat before{};
    if (::fstat(file.get(), &before) != 0 || !S_ISREG(before.st_mode))
      return std::nullopt;
    std::string result;
    std::array<char, 1024U> buffer{};
    for (;;) {
      const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
      if (count < 0) {
        if (errno == EINTR)
          continue;
        return std::nullopt;
      }
      if (count == 0)
        break;
      if (result.size() + static_cast<std::size_t>(count) > maximum)
        return std::nullopt;
      result.append(buffer.data(), static_cast<std::size_t>(count));
    }
    struct stat after{};
    if (::fstat(file.get(), &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino)
      return std::nullopt;
    return result;
  }

  [[nodiscard]] std::optional<std::vector<std::string>>
  list(std::string_view relative, std::size_t maximum) const {
    const int opened = open(relative, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (opened < 0)
      return std::nullopt;
    DIR *directory = ::fdopendir(opened);
    if (directory == nullptr) {
      (void)::close(opened);
      return std::nullopt;
    }
    std::vector<std::string> result;
    errno = 0;
    while (dirent *entry = ::readdir(directory)) {
      const std::string_view name(entry->d_name);
      if (name == "." || name == "..")
        continue;
      if (name.size() > HostResourceBounds::maximum_identifier_bytes ||
          result.size() >= maximum) {
        (void)::closedir(directory);
        return std::nullopt;
      }
      result.emplace_back(name);
      errno = 0;
    }
    const int read_error = errno;
    (void)::closedir(directory);
    if (read_error != 0)
      return std::nullopt;
    std::ranges::sort(result);
    return result;
  }

  [[nodiscard]] std::optional<std::pair<std::uint32_t, std::uint32_t>>
  character_device(std::string_view relative) const {
    const int opened = open(relative, O_PATH | O_CLOEXEC | O_NOFOLLOW);
    if (opened < 0)
      return std::nullopt;
    Descriptor device(opened);
    struct stat status{};
    if (::fstat(device.get(), &status) != 0 || !S_ISCHR(status.st_mode))
      return std::nullopt;
    return std::pair{static_cast<std::uint32_t>(::major(status.st_rdev)),
                     static_cast<std::uint32_t>(::minor(status.st_rdev))};
  }

private:
  [[nodiscard]] int open(std::string_view relative, std::uint64_t flags) const {
    if (!available() || relative.empty() || relative.starts_with('/') ||
        relative.find('\0') != std::string_view::npos)
      return -1;
    struct open_how how{};
    how.flags = flags;
    how.resolve = RESOLVE_IN_ROOT | RESOLVE_NO_MAGICLINKS;
    return static_cast<int>(::syscall(SYS_openat2, descriptor_.get(),
                                      std::string(relative).c_str(), &how,
                                      sizeof(how)));
  }

  Descriptor descriptor_;
};

using NvmlReturn = unsigned int;
using NvmlDevice = void *;

struct NvmlPciInfo final {
  char bus_id_legacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pci_device_id;
  unsigned int pci_subsystem_id;
  char bus_id[32];
};

struct NvmlMemory final {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct NvmlProcessV2 final {
  unsigned int pid;
  unsigned long long used_gpu_memory;
  // Kept at the modern ABI size even though this collector intentionally does
  // not trust v2 instance attribution for MIG scheduling decisions.
  unsigned int gpu_instance_id;
  unsigned int compute_instance_id;
};

struct NvmlProcessV3 final {
  unsigned int pid;
  unsigned long long used_gpu_memory;
  unsigned int gpu_instance_id;
  unsigned int compute_instance_id;
};

#if TRAINVM_HAS_OFFICIAL_NVML_ABI
static_assert(sizeof(NvmlPciInfo) == sizeof(nvmlPciInfo_t));
static_assert(offsetof(NvmlPciInfo, bus_id_legacy) ==
              offsetof(nvmlPciInfo_t, busIdLegacy));
static_assert(offsetof(NvmlPciInfo, domain) == offsetof(nvmlPciInfo_t, domain));
static_assert(offsetof(NvmlPciInfo, bus_id) == offsetof(nvmlPciInfo_t, busId));
static_assert(sizeof(NvmlProcessV2) == sizeof(nvmlProcessInfo_v2_t));
static_assert(offsetof(NvmlProcessV2, used_gpu_memory) ==
              offsetof(nvmlProcessInfo_v2_t, usedGpuMemory));
static_assert(offsetof(NvmlProcessV2, gpu_instance_id) ==
              offsetof(nvmlProcessInfo_v2_t, gpuInstanceId));
static_assert(sizeof(NvmlProcessV3) == sizeof(nvmlProcessInfo_t));
static_assert(offsetof(NvmlProcessV3, compute_instance_id) ==
              offsetof(nvmlProcessInfo_t, computeInstanceId));
#endif

struct NvmlSample final {
  bool loaded{};
  bool complete{};
  bool contexts_complete{};
  bool loader_attested{};
  std::string loader_detail;
  std::string driver;
  std::vector<LinuxNvidiaRawDevice> devices;

  bool operator==(const NvmlSample &) const = default;
};

class NvmlLibrary final {
public:
  explicit NvmlLibrary(std::size_t maximum_processes,
                       std::size_t maximum_partitions)
      : maximum_processes_(maximum_processes),
        maximum_partitions_(maximum_partitions),
        library_(::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL)) {
    if (library_ == nullptr)
      return;
    init_ = symbol<Init>("nvmlInit_v2");
    shutdown_ = symbol<Shutdown>("nvmlShutdown");
    get_driver_ = symbol<GetDriver>("nvmlSystemGetDriverVersion");
    get_count_ = symbol<GetCount>("nvmlDeviceGetCount_v2");
    get_by_index_ = symbol<GetByIndex>("nvmlDeviceGetHandleByIndex_v2");
    get_uuid_ = symbol<GetString>("nvmlDeviceGetUUID");
    get_pci_v3_ = symbol<GetPci>("nvmlDeviceGetPciInfo_v3");
    get_pci_v2_ = symbol<GetPci>("nvmlDeviceGetPciInfo_v2");
    get_memory_ = symbol<GetMemory>("nvmlDeviceGetMemoryInfo");
    get_minor_ = symbol<GetMinor>("nvmlDeviceGetMinorNumber");
    get_display_active_ = symbol<GetDisplay>("nvmlDeviceGetDisplayActive");
    get_display_mode_ = symbol<GetDisplay>("nvmlDeviceGetDisplayMode");
    get_compute_v3_ =
        symbol<GetProcessesV3>("nvmlDeviceGetComputeRunningProcesses_v3");
    get_compute_v2_ =
        symbol<GetProcessesV2>("nvmlDeviceGetComputeRunningProcesses_v2");
    get_graphics_v3_ =
        symbol<GetProcessesV3>("nvmlDeviceGetGraphicsRunningProcesses_v3");
    get_graphics_v2_ =
        symbol<GetProcessesV2>("nvmlDeviceGetGraphicsRunningProcesses_v2");
    get_mig_mode_ = symbol<GetMigMode>("nvmlDeviceGetMigMode");
    get_max_mig_ = symbol<GetMaxMig>("nvmlDeviceGetMaxMigDeviceCount");
    get_mig_by_index_ =
        symbol<GetMigByIndex>("nvmlDeviceGetMigDeviceHandleByIndex");
    get_gpu_instance_id_ = symbol<GetInstanceId>("nvmlDeviceGetGpuInstanceId");
    get_compute_instance_id_ =
        symbol<GetInstanceId>("nvmlDeviceGetComputeInstanceId");
    ready_ = init_ != nullptr && shutdown_ != nullptr &&
             get_driver_ != nullptr && get_count_ != nullptr &&
             get_by_index_ != nullptr && get_uuid_ != nullptr &&
             (get_pci_v3_ != nullptr || get_pci_v2_ != nullptr) &&
             get_memory_ != nullptr && get_minor_ != nullptr &&
             get_display_active_ != nullptr && get_display_mode_ != nullptr &&
             (get_compute_v3_ != nullptr || get_compute_v2_ != nullptr) &&
             (get_graphics_v3_ != nullptr || get_graphics_v2_ != nullptr) &&
             get_mig_mode_ != nullptr && get_max_mig_ != nullptr &&
             get_mig_by_index_ != nullptr;
    loader_attested_ = attested_library_symbol(
        reinterpret_cast<const void *>(get_count_), loader_detail_);
    if (ready_ && init_() == kNvmlSuccess)
      initialized_ = true;
  }

  ~NvmlLibrary() {
    if (initialized_)
      (void)shutdown_();
    if (library_ != nullptr)
      (void)::dlclose(library_);
  }

  NvmlLibrary(const NvmlLibrary &) = delete;
  NvmlLibrary &operator=(const NvmlLibrary &) = delete;

  [[nodiscard]] NvmlSample sample(std::size_t maximum_devices) const {
    NvmlSample result;
    result.loaded = library_ != nullptr;
    result.loader_attested = loader_attested_;
    result.loader_detail = loader_detail_;
    if (!initialized_)
      return result;
    std::array<char, kNvmlStringBytes> driver{};
    unsigned int count = 0U;
    if (get_driver_(driver.data(), static_cast<unsigned int>(driver.size())) !=
            kNvmlSuccess ||
        get_count_(&count) != kNvmlSuccess || count > maximum_devices)
      return result;
    const auto driver_value = bounded_c_string(driver);
    if (!driver_value)
      return result;
    result.driver = *driver_value;
    result.complete = true;
    result.contexts_complete = true;
    for (unsigned int index = 0U; index < count; ++index) {
      NvmlDevice handle = nullptr;
      if (get_by_index_(index, &handle) != kNvmlSuccess || handle == nullptr) {
        result.complete = false;
        continue;
      }
      LinuxNvidiaRawDevice device;
      bool device_complete = true;
      std::array<char, kNvmlStringBytes> uuid{};
      NvmlPciInfo pci{};
      NvmlMemory memory{};
      unsigned int minor_number = 0U;
      const bool pci_v3 = get_pci_v3_ != nullptr;
      const NvmlReturn pci_status =
          pci_v3 ? get_pci_v3_(handle, &pci) : get_pci_v2_(handle, &pci);
      if (get_uuid_(handle, uuid.data(),
                    static_cast<unsigned int>(uuid.size())) != kNvmlSuccess ||
          pci_status != kNvmlSuccess ||
          get_memory_(handle, &memory) != kNvmlSuccess ||
          get_minor_(handle, &minor_number) != kNvmlSuccess) {
        result.complete = false;
        continue;
      }
      const auto uuid_value = bounded_c_string(uuid);
      if (!uuid_value) {
        result.complete = false;
        continue;
      }
      device.uuid = *uuid_value;
      const std::string_view v3_bus_id(pci.bus_id, sizeof(pci.bus_id));
      device.pci_bdf = pci_v3 ? normalize_bdf(std::string(
                                    v3_bus_id.substr(0U, v3_bus_id.find('\0'))))
                              : pci_bdf_from_v2_impl(
                                    std::string_view(pci.bus_id_legacy,
                                                     sizeof(pci.bus_id_legacy)),
                                    pci.domain, pci.bus, pci.device);
      device.total_memory_bytes = memory.total;
      device.device_minor = minor_number;
      unsigned int display_active = 0U;
      unsigned int display_mode = 0U;
      const bool display_complete =
          get_display_active_(handle, &display_active) == kNvmlSuccess &&
          get_display_mode_(handle, &display_mode) == kNvmlSuccess &&
          display_active <= kNvmlFeatureEnabled &&
          display_mode <= kNvmlFeatureEnabled;
      if (display_complete) {
        device.display_active = display_active == kNvmlFeatureEnabled;
        device.display_mode_enabled = display_mode == kNvmlFeatureEnabled;
      }
      device.display_evidence_complete = display_complete;
      device_complete = device_complete && display_complete;
      result.contexts_complete = result.contexts_complete && display_complete;

      unsigned int current_mig = 0U;
      unsigned int pending_mig = 0U;
      const NvmlReturn mig_status =
          get_mig_mode_(handle, &current_mig, &pending_mig);
      if (mig_status == kNvmlSuccess) {
        device.current_mig_mode = current_mig;
        device.pending_mig_mode = pending_mig;
        device_complete = device_complete && current_mig == pending_mig;
      }
      if (mig_status == kNvmlSuccess && current_mig != 0U) {
        const auto compute =
            contexts(handle, get_compute_v3_, get_compute_v2_, ParentOnly{});
        const auto graphics =
            contexts(handle, get_graphics_v3_, get_graphics_v2_, ParentOnly{});
        assign_contexts(device, compute, graphics);
        device_complete =
            device_complete && compute.complete && graphics.complete;
        result.contexts_complete =
            result.contexts_complete && compute.complete && graphics.complete;
        unsigned int maximum_mig = 0U;
        if (get_max_mig_(handle, &maximum_mig) != kNvmlSuccess ||
            maximum_mig > maximum_partitions_) {
          device_complete = false;
        } else {
          for (unsigned int mig_index = 0U; mig_index < maximum_mig;
               ++mig_index) {
            NvmlDevice mig = nullptr;
            const NvmlReturn found = get_mig_by_index_(handle, mig_index, &mig);
            if (found == kNvmlErrorNotFound)
              continue;
            if (found != kNvmlSuccess || mig == nullptr) {
              device_complete = false;
              continue;
            }
            LinuxNvidiaRawPartition partition;
            std::array<char, kNvmlStringBytes> mig_uuid{};
            NvmlMemory mig_memory{};
            unsigned int gpu_instance_id = 0U;
            unsigned int compute_instance_id = 0U;
            if (get_uuid_(mig, mig_uuid.data(),
                          static_cast<unsigned int>(mig_uuid.size())) !=
                    kNvmlSuccess ||
                get_memory_(mig, &mig_memory) != kNvmlSuccess ||
                get_gpu_instance_id_ == nullptr ||
                get_compute_instance_id_ == nullptr ||
                get_gpu_instance_id_(mig, &gpu_instance_id) != kNvmlSuccess ||
                get_compute_instance_id_(mig, &compute_instance_id) !=
                    kNvmlSuccess) {
              device_complete = false;
              continue;
            }
            const auto mig_uuid_value = bounded_c_string(mig_uuid);
            if (!mig_uuid_value) {
              device_complete = false;
              continue;
            }
            partition.uuid = *mig_uuid_value;
            partition.total_memory_bytes = mig_memory.total;
            partition.gpu_instance_id = gpu_instance_id;
            partition.compute_instance_id = compute_instance_id;
            const InstanceOnly instance{gpu_instance_id, compute_instance_id};
            const auto mig_compute =
                contexts(mig, get_compute_v3_, get_compute_v2_, instance);
            const auto mig_graphics =
                contexts(mig, get_graphics_v3_, get_graphics_v2_, instance);
            assign_contexts(partition, mig_compute, mig_graphics);
            partition.evidence_complete =
                mig_compute.complete && mig_graphics.complete;
            device_complete = device_complete && partition.evidence_complete;
            result.contexts_complete =
                result.contexts_complete && partition.evidence_complete;
            device.partitions.push_back(std::move(partition));
          }
        }
      } else if (mig_status != kNvmlSuccess &&
                 mig_status != kNvmlErrorNotSupported) {
        device_complete = false;
      } else {
        if (mig_status == kNvmlErrorNotSupported) {
          device.current_mig_mode = 0U;
          device.pending_mig_mode = 0U;
        }
        // Non-MIG devices can safely use v2 process records because no instance
        // attribution is required.
        const auto compute =
            contexts(handle, get_compute_v3_, get_compute_v2_, AllProcesses{});
        const auto graphics = contexts(handle, get_graphics_v3_,
                                       get_graphics_v2_, AllProcesses{});
        assign_contexts(device, compute, graphics);
        device_complete =
            device_complete && compute.complete && graphics.complete;
        result.contexts_complete =
            result.contexts_complete && compute.complete && graphics.complete;
      }
      device.evidence_complete = device_complete;
      result.complete = result.complete && device_complete;
      std::ranges::sort(device.partitions, {}, &LinuxNvidiaRawPartition::uuid);
      result.devices.push_back(std::move(device));
    }
    std::ranges::sort(result.devices, {}, &LinuxNvidiaRawDevice::uuid);
    return result;
  }

private:
  using Init = NvmlReturn (*)();
  using Shutdown = NvmlReturn (*)();
  using GetDriver = NvmlReturn (*)(char *, unsigned int);
  using GetCount = NvmlReturn (*)(unsigned int *);
  using GetByIndex = NvmlReturn (*)(unsigned int, NvmlDevice *);
  using GetString = NvmlReturn (*)(NvmlDevice, char *, unsigned int);
  using GetPci = NvmlReturn (*)(NvmlDevice, NvmlPciInfo *);
  using GetMemory = NvmlReturn (*)(NvmlDevice, NvmlMemory *);
  using GetMinor = NvmlReturn (*)(NvmlDevice, unsigned int *);
  using GetDisplay = NvmlReturn (*)(NvmlDevice, unsigned int *);
  using GetProcessesV2 = NvmlReturn (*)(NvmlDevice, unsigned int *,
                                        NvmlProcessV2 *);
  using GetProcessesV3 = NvmlReturn (*)(NvmlDevice, unsigned int *,
                                        NvmlProcessV3 *);
  using GetMigMode = NvmlReturn (*)(NvmlDevice, unsigned int *, unsigned int *);
  using GetMaxMig = NvmlReturn (*)(NvmlDevice, unsigned int *);
  using GetMigByIndex = NvmlReturn (*)(NvmlDevice, unsigned int, NvmlDevice *);
  using GetInstanceId = NvmlReturn (*)(NvmlDevice, unsigned int *);

  struct AllProcesses final {};
  struct ParentOnly final {};
  struct InstanceOnly final {
    unsigned int gpu;
    unsigned int compute;
  };
  struct ContextSample final {
    ResourceContextDisposition disposition{ResourceContextDisposition::unknown};
    bool complete{};
    std::vector<LinuxNvidiaRawProcess> processes;
  };

  template <typename Function>
  [[nodiscard]] Function symbol(const char *name) const {
    return reinterpret_cast<Function>(::dlsym(library_, name));
  }

  template <typename Target>
  [[nodiscard]] ContextSample
  contexts(NvmlDevice device, GetProcessesV3 query_v3, GetProcessesV2 query_v2,
           Target target) const {
    if (query_v3 != nullptr)
      return contexts_v3(device, query_v3, target);
    if constexpr (!std::is_same_v<Target, AllProcesses>)
      return {};
    return contexts_v2(device, query_v2);
  }

  [[nodiscard]] ContextSample contexts_v2(NvmlDevice device,
                                          GetProcessesV2 query) const {
    if (query == nullptr)
      return {};
    unsigned int count = 0U;
    NvmlReturn status = query(device, &count, nullptr);
    if (status == kNvmlSuccess && count == 0U)
      return {.disposition = ResourceContextDisposition::absent,
              .complete = true,
              .processes = {}};
    if (status != kNvmlErrorInsufficientSize || count == 0U ||
        count > maximum_processes_)
      return {};
    std::vector<NvmlProcessV2> processes(count);
    status = query(device, &count, processes.data());
    if (status != kNvmlSuccess || count > processes.size())
      return {};
    ContextSample result{.disposition =
                             count == 0U ? ResourceContextDisposition::absent
                                         : ResourceContextDisposition::present,
                         .complete = true,
                         .processes = {}};
    result.processes.reserve(count);
    for (unsigned int index = 0U; index < count; ++index)
      result.processes.push_back({.pid = processes[index].pid,
                                  .gpu_instance_id = std::nullopt,
                                  .compute_instance_id = std::nullopt});
    return result;
  }

  template <typename Target>
  [[nodiscard]] ContextSample
  contexts_v3(NvmlDevice device, GetProcessesV3 query, Target target) const {
    unsigned int count = 0U;
    NvmlReturn status = query(device, &count, nullptr);
    if (status == kNvmlSuccess && count == 0U)
      return {.disposition = ResourceContextDisposition::absent,
              .complete = true,
              .processes = {}};
    if (status != kNvmlErrorInsufficientSize || count == 0U ||
        count > maximum_processes_)
      return {};
    std::vector<NvmlProcessV3> processes(count);
    status = query(device, &count, processes.data());
    if (status != kNvmlSuccess || count > processes.size())
      return {};
    ContextSample result{.disposition = ResourceContextDisposition::absent,
                         .complete = true,
                         .processes = {}};
    for (unsigned int index = 0U; index < count; ++index) {
      const auto &process = processes[index];
      const bool parent =
          process.gpu_instance_id == kNvmlInstanceNotApplicable &&
          process.compute_instance_id == kNvmlInstanceNotApplicable;
      bool selected = true;
      if constexpr (std::is_same_v<Target, ParentOnly>)
        selected = parent;
      else if constexpr (std::is_same_v<Target, InstanceOnly>)
        selected = process.gpu_instance_id == target.gpu &&
                   process.compute_instance_id == target.compute;
      if (!selected)
        continue;
      result.disposition = ResourceContextDisposition::present;
      result.processes.push_back(
          {.pid = process.pid,
           .gpu_instance_id =
               parent ? std::nullopt : std::optional(process.gpu_instance_id),
           .compute_instance_id =
               parent ? std::nullopt
                      : std::optional(process.compute_instance_id)});
    }
    return result;
  }

  template <typename Raw>
  static void assign_contexts(Raw &raw, const ContextSample &compute,
                              const ContextSample &graphics) {
    raw.contexts = {.compute = compute.disposition,
                    .graphics = graphics.disposition};
    raw.compute_processes = compute.processes;
    raw.graphics_processes = graphics.processes;
    const auto less = [](const LinuxNvidiaRawProcess &left,
                         const LinuxNvidiaRawProcess &right) {
      return std::tie(left.pid, left.gpu_instance_id,
                      left.compute_instance_id) <
             std::tie(right.pid, right.gpu_instance_id,
                      right.compute_instance_id);
    };
    std::ranges::sort(raw.compute_processes, less);
    std::ranges::sort(raw.graphics_processes, less);
  }

  std::size_t maximum_processes_;
  std::size_t maximum_partitions_;
  void *library_{};
  Init init_{};
  Shutdown shutdown_{};
  GetDriver get_driver_{};
  GetCount get_count_{};
  GetByIndex get_by_index_{};
  GetString get_uuid_{};
  GetPci get_pci_v3_{};
  GetPci get_pci_v2_{};
  GetMemory get_memory_{};
  GetMinor get_minor_{};
  GetDisplay get_display_active_{};
  GetDisplay get_display_mode_{};
  GetProcessesV3 get_compute_v3_{};
  GetProcessesV2 get_compute_v2_{};
  GetProcessesV3 get_graphics_v3_{};
  GetProcessesV2 get_graphics_v2_{};
  GetMigMode get_mig_mode_{};
  GetMaxMig get_max_mig_{};
  GetMigByIndex get_mig_by_index_{};
  GetInstanceId get_gpu_instance_id_{};
  GetInstanceId get_compute_instance_id_{};
  bool loader_attested_{};
  std::string loader_detail_;
  bool ready_{};
  bool initialized_{};
};

std::string sample_revision(const std::string &boot,
                            const std::vector<std::string> &structural,
                            const NvmlSample &sample) {
  std::string evidence =
      boot + "\n" + sample.driver +
      "\nloader:" + (sample.loader_attested ? "trusted" : "untrusted") + ":" +
      sample.loader_detail + "\n";
  const auto optional_number = [](const auto &value) {
    return value ? std::to_string(*value) : std::string("-");
  };
  const auto append_processes = [&](std::string &output, std::string_view kind,
                                    const auto &processes) {
    for (const auto &process : processes)
      output += "p:" + std::string(kind) + ":" + std::to_string(process.pid) +
                ":" + optional_number(process.gpu_instance_id) + ":" +
                optional_number(process.compute_instance_id) + "\n";
  };
  for (const auto &bdf : structural)
    evidence += "s:" + bdf + "\n";
  for (const auto &device : sample.devices) {
    evidence +=
        "d:" + device.uuid + ":" + device.pci_bdf + ":" +
        std::to_string(device.total_memory_bytes) + ":" +
        (device.device_major ? std::to_string(*device.device_major) : "-") +
        ":" +
        (device.device_minor ? std::to_string(*device.device_minor) : "-") +
        ":" + (device.numa_node ? std::to_string(*device.numa_node) : "-") +
        ":" +
        std::to_string(static_cast<unsigned int>(device.contexts.compute)) +
        ":" +
        std::to_string(static_cast<unsigned int>(device.contexts.graphics)) +
        ":" + optional_number(device.current_mig_mode) + ":" +
        optional_number(device.pending_mig_mode) + ":" +
        (device.display_active ? (*device.display_active ? "1" : "0") : "-") +
        ":" +
        (device.display_mode_enabled
             ? (*device.display_mode_enabled ? "1" : "0")
             : "-") +
        "\n";
    append_processes(evidence, "dc", device.compute_processes);
    append_processes(evidence, "dg", device.graphics_processes);
    for (const auto &partition : device.partitions) {
      evidence += "m:" + partition.uuid + ":" +
                  std::to_string(partition.total_memory_bytes) + ":" +
                  optional_number(partition.gpu_instance_id) + ":" +
                  optional_number(partition.compute_instance_id) + ":" +
                  std::to_string(
                      static_cast<unsigned int>(partition.contexts.compute)) +
                  ":" +
                  std::to_string(
                      static_cast<unsigned int>(partition.contexts.graphics)) +
                  "\n";
      append_processes(evidence, "mc", partition.compute_processes);
      append_processes(evidence, "mg", partition.graphics_processes);
    }
  }
  return digest_revision(evidence);
}

class RealLinuxNvidiaKernel final : public ILinuxNvidiaReadOnlyKernel {
public:
  explicit RealLinuxNvidiaKernel(LinuxNvidiaInventoryConfig config)
      : config_(std::move(config)) {}

  LinuxNvidiaRawSnapshot capture() override {
    const std::uint64_t capture_started = monotonic_now_ns();
    SecureRoot root("/");
    SecureRoot proc("/proc", PROC_SUPER_MAGIC);
    SecureRoot sys("/sys", SYSFS_MAGIC);
    SecureRoot dev("/dev", TMPFS_MAGIC);
    LinuxNvidiaRawSnapshot result;
    result.api_version = std::string(kLinuxNvidiaInventoryApiVersion);
    result.broker_epoch = config_.broker_epoch;
    const auto host = root.read("etc/machine-id", 256U);
    const auto boot_begin = proc.read("sys/kernel/random/boot_id", 256U);
    const auto driver_begin =
        proc.read("driver/nvidia/version", kMaximumFileBytes);
    const auto structural_begin = structural_bdfs(sys);
    NvmlLibrary nvml(config_.maximum_processes_per_device,
                     config_.maximum_partitions_per_device);
    NvmlSample first = nvml.sample(config_.maximum_devices);
    const bool first_structure = apply_structural(proc, sys, dev, first);
    NvmlSample second = nvml.sample(config_.maximum_devices);
    const bool second_structure = apply_structural(proc, sys, dev, second);
    const auto structural_end = structural_bdfs(sys);
    const auto driver_end =
        proc.read("driver/nvidia/version", kMaximumFileBytes);
    const auto boot_end = proc.read("sys/kernel/random/boot_id", 256U);
    const auto host_end = root.read("etc/machine-id", 256U);
    result.host_id = host ? trim_ascii(*host) : std::string{};
    result.boot_id = boot_begin ? trim_ascii(*boot_begin) : std::string{};
    result.nvml_loaded = first.loaded;
    result.nvml_complete = first.complete && second.complete;
    result.context_details_complete =
        first.contexts_complete && second.contexts_complete;
    result.trusted_host_namespace = config_.trusted_host_namespace;
    result.trusted_nvml_loader = config_.trusted_nvml_loader &&
                                 first.loader_attested &&
                                 second.loader_attested;
    result.capture_started_monotonic_ns = capture_started;
    result.capture_finished_monotonic_ns = monotonic_now_ns();
    result.devices = first.devices;
    result.structural_pci_bdfs =
        structural_begin.value_or(std::vector<std::string>{});
    result.structural_complete = host && host_end && boot_begin && boot_end &&
                                 driver_begin && driver_end &&
                                 structural_begin && structural_end &&
                                 first_structure && second_structure;
    const std::string first_boot =
        boot_begin ? trim_ascii(*boot_begin) : "missing-boot";
    const std::string second_boot =
        boot_end ? trim_ascii(*boot_end) : "missing-boot-end";
    result.begin_revision = sample_revision(
        (host ? trim_ascii(*host) : "missing-host") + "\n" + first_boot + "\n" +
            driver_begin.value_or("missing-driver"),
        result.structural_pci_bdfs, first);
    result.end_revision = sample_revision(
        (host_end ? trim_ascii(*host_end) : "missing-host-end") + "\n" +
            second_boot + "\n" + driver_end.value_or("missing-driver-end"),
        structural_end.value_or(std::vector<std::string>{}), second);
    if (first == second && structural_begin == structural_end &&
        host == host_end && boot_begin == boot_end &&
        driver_begin == driver_end)
      result.detail =
          !result.trusted_host_namespace
              ? "host-namespace-unattested"
              : (!result.trusted_nvml_loader
                     ? first.loader_detail
                     : (first.complete && second.complete
                            ? "point-in-time-nvml-snapshot"
                            : (first.loaded ? "nvml-incomplete-point-in-time"
                                            : "nvml-unavailable")));
    else
      result.detail = "torn-nvidia-snapshot";
    return result;
  }

  [[nodiscard]] std::uint64_t monotonic_now_ns() const override {
    return trainvm::monotonic_now_ns();
  }

private:
  static std::optional<std::vector<std::string>>
  structural_bdfs(const SecureRoot &sys) {
    const auto entries = sys.list("bus/pci/drivers/nvidia",
                                  HostResourceBounds::maximum_resources + 8U);
    if (!entries)
      return std::nullopt;
    std::vector<std::string> bdfs;
    for (const auto &entry : *entries) {
      const std::string bdf = normalize_bdf(entry);
      if (!bdf.empty())
        bdfs.push_back(bdf);
    }
    std::ranges::sort(bdfs);
    bdfs.erase(std::unique(bdfs.begin(), bdfs.end()), bdfs.end());
    return bdfs;
  }

  static bool apply_structural(const SecureRoot &proc, const SecureRoot &sys,
                               const SecureRoot &dev, NvmlSample &sample) {
    const auto registered_major = nvidia_frontend_major(proc);
    bool all_complete = true;
    for (auto &device : sample.devices) {
      bool complete = !device.pci_bdf.empty() && device.device_minor;
      const std::string prefix = "bus/pci/devices/" + device.pci_bdf + "/";
      const auto vendor = sys.read(prefix + "vendor", 32U);
      complete = complete && vendor && trim_ascii(*vendor) == kNvidiaVendor;
      const auto pci_uevent = sys.read(prefix + "uevent", kMaximumFileBytes);
      const auto numa = sys.read(prefix + "numa_node", 32U);
      if (numa) {
        const std::string value = trim_ascii(*numa);
        std::int32_t parsed = -1;
        const auto conversion =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (conversion.ec == std::errc{} &&
            conversion.ptr == value.data() + value.size() && parsed >= 0)
          device.numa_node = parsed;
        else if (value != "-1")
          complete = false;
      } else {
        complete = false;
      }
      if (device.device_minor) {
        const auto node = dev.character_device(
            "nvidia" + std::to_string(*device.device_minor));
        const std::string sys_char =
            node ? "dev/char/" + std::to_string(node->first) + ":" +
                       std::to_string(node->second) + "/"
                 : std::string{};
        const auto char_vendor =
            node ? sys.read(sys_char + "device/vendor", 32U) : std::nullopt;
        const auto char_uevent =
            node ? sys.read(sys_char + "device/uevent", kMaximumFileBytes)
                 : std::nullopt;
        const bool sys_char_bdf_mapped =
            char_uevent && unique_line_value(*char_uevent, "PCI_SLOT_NAME",
                                             '=') == device.pci_bdf;
        const auto gpu_information =
            proc.read("driver/nvidia/gpus/" + device.pci_bdf + "/information",
                      kMaximumFileBytes);
        const bool pinned_pci_bdf_mapped =
            pci_uevent && gpu_information &&
            pinned_pci_mapping_impl(*pci_uevent, *gpu_information, device.uuid,
                                    device.pci_bdf, *device.device_minor);
        if (node && registered_major && node->first == *registered_major &&
            node->second == *device.device_minor &&
            ((char_vendor && trim_ascii(*char_vendor) == kNvidiaVendor &&
              sys_char_bdf_mapped) ||
             pinned_pci_bdf_mapped)) {
          device.device_major = node->first;
          device.device_node_mapping_complete = true;
        } else {
          device.device_major.reset();
          complete = false;
        }
      }
      device.evidence_complete = device.evidence_complete && complete;
      all_complete = all_complete && complete;
    }
    sample.complete = sample.complete && all_complete;
    return all_complete;
  }

  static std::optional<std::uint32_t>
  nvidia_frontend_major(const SecureRoot &proc) {
    const auto devices = proc.read("devices", kMaximumFileBytes);
    return devices ? nvidia_frontend_major_impl(*devices) : std::nullopt;
  }

  LinuxNvidiaInventoryConfig config_;
};

bool structural_matches(const LinuxNvidiaRawSnapshot &raw) {
  std::vector<std::string> observed;
  observed.reserve(raw.devices.size());
  for (const auto &device : raw.devices)
    observed.push_back(device.pci_bdf);
  std::ranges::sort(observed);
  auto structural = raw.structural_pci_bdfs;
  std::ranges::sort(structural);
  return observed == structural &&
         std::adjacent_find(observed.begin(), observed.end()) == observed.end();
}

bool process_evidence_valid(
    const std::vector<LinuxNvidiaRawProcess> &processes,
    ResourceContextDisposition disposition, std::size_t maximum,
    std::optional<std::pair<std::uint32_t, std::uint32_t>> instance) {
  if (processes.size() > maximum ||
      (disposition == ResourceContextDisposition::absent &&
       !processes.empty()) ||
      (disposition == ResourceContextDisposition::present && processes.empty()))
    return false;
  std::set<std::tuple<std::uint32_t, std::optional<std::uint32_t>,
                      std::optional<std::uint32_t>>>
      observed;
  for (const auto &process : processes) {
    if (process.pid == 0U || process.gpu_instance_id.has_value() !=
                                 process.compute_instance_id.has_value())
      return false;
    if (instance) {
      if (!process.gpu_instance_id ||
          *process.gpu_instance_id != instance->first ||
          *process.compute_instance_id != instance->second)
        return false;
    } else if (process.gpu_instance_id) {
      return false;
    }
    if (!observed
             .emplace(process.pid, process.gpu_instance_id,
                      process.compute_instance_id)
             .second)
      return false;
  }
  return true;
}

} // namespace

namespace linux_nvidia_test_seam {

std::string pci_bdf_from_v2(std::string_view legacy, std::uint32_t domain,
                            std::uint32_t bus, std::uint32_t device) {
  return pci_bdf_from_v2_impl(legacy, domain, bus, device);
}

std::optional<std::uint32_t>
nvidia_frontend_major(std::string_view proc_devices) {
  return nvidia_frontend_major_impl(proc_devices);
}

bool pinned_pci_device_mapping(std::string_view pci_uevent,
                               std::string_view proc_gpu_information,
                               std::string_view uuid, std::string_view pci_bdf,
                               std::uint32_t minor) {
  return pinned_pci_mapping_impl(pci_uevent, proc_gpu_information, uuid,
                                 pci_bdf, minor);
}

} // namespace linux_nvidia_test_seam

struct LinuxNvidiaInventoryCollector::Implementation final {
  LinuxNvidiaInventoryConfig config;
  std::shared_ptr<ILinuxNvidiaReadOnlyKernel> kernel;
};

LinuxNvidiaInventoryCollector::LinuxNvidiaInventoryCollector(
    LinuxNvidiaInventoryConfig config,
    std::shared_ptr<ILinuxNvidiaReadOnlyKernel> kernel)
    : implementation_(std::make_unique<Implementation>()) {
  if (config.api_version != kLinuxNvidiaInventoryApiVersion ||
      !bounded_identifier(config.broker_epoch) ||
      config.maximum_devices == 0U ||
      config.maximum_devices > HostResourceBounds::maximum_resources ||
      config.maximum_partitions_per_device == 0U ||
      config.maximum_partitions_per_device > 64U ||
      config.maximum_processes_per_device == 0U ||
      config.maximum_processes_per_device > 65536U ||
      config.maximum_capture_duration_ns == 0U ||
      config.maximum_capture_duration_ns > 60'000'000'000ULL ||
      config.maximum_snapshot_age_ns == 0U ||
      config.maximum_snapshot_age_ns > 60'000'000'000ULL)
    throw HostResourceError("Linux NVIDIA inventory config is invalid");
  implementation_->config = std::move(config);
  implementation_->kernel =
      kernel ? std::move(kernel)
             : std::make_shared<RealLinuxNvidiaKernel>(implementation_->config);
}

LinuxNvidiaInventoryCollector::~LinuxNvidiaInventoryCollector() = default;
LinuxNvidiaInventoryCollector::LinuxNvidiaInventoryCollector(
    LinuxNvidiaInventoryCollector &&) noexcept = default;
LinuxNvidiaInventoryCollector &LinuxNvidiaInventoryCollector::operator=(
    LinuxNvidiaInventoryCollector &&) noexcept = default;

HostKernelSnapshot LinuxNvidiaInventoryCollector::capture_inventory() {
  LinuxNvidiaRawSnapshot raw;
  std::uint64_t request_started = 0U;
  std::uint64_t observed_now = 0U;
  try {
    request_started = implementation_->kernel->monotonic_now_ns();
    raw = implementation_->kernel->capture();
    observed_now = implementation_->kernel->monotonic_now_ns();
  } catch (const std::exception &) {
    throw HostResourceError("Linux NVIDIA evidence capture failed");
  } catch (...) {
    throw HostResourceError("Linux NVIDIA evidence capture failed");
  }
  if (raw.api_version != kLinuxNvidiaInventoryApiVersion ||
      !bounded_identifier(raw.host_id) || !bounded_identifier(raw.boot_id) ||
      raw.broker_epoch != implementation_->config.broker_epoch ||
      !bounded_identifier(raw.begin_revision) ||
      !bounded_identifier(raw.end_revision) ||
      !printable_bounded(
          raw.detail, HostResourceBounds::maximum_probe_detail_bytes - 96U) ||
      raw.devices.size() > implementation_->config.maximum_devices ||
      raw.structural_pci_bdfs.size() >
          implementation_->config.maximum_devices ||
      raw.capture_started_monotonic_ns == 0U ||
      raw.capture_finished_monotonic_ns == 0U)
    throw HostResourceError("Linux NVIDIA evidence envelope is invalid");

  bool complete =
      raw.structural_complete && raw.nvml_loaded && raw.nvml_complete &&
      raw.context_details_complete &&
      implementation_->config.trusted_host_namespace &&
      implementation_->config.trusted_nvml_loader &&
      raw.trusted_host_namespace && raw.trusted_nvml_loader &&
      raw.begin_revision == raw.end_revision && structural_matches(raw) &&
      raw.capture_started_monotonic_ns >= request_started &&
      raw.capture_finished_monotonic_ns >= raw.capture_started_monotonic_ns &&
      observed_now >= raw.capture_finished_monotonic_ns &&
      raw.capture_finished_monotonic_ns - raw.capture_started_monotonic_ns <=
          implementation_->config.maximum_capture_duration_ns &&
      observed_now - raw.capture_finished_monotonic_ns <=
          implementation_->config.maximum_snapshot_age_ns;
  std::set<std::string> ids;
  std::set<std::string> bdfs;
  std::size_t raw_resource_count = raw.devices.size();
  for (const auto &device : raw.devices) {
    if (!canonical_uuid(device.uuid, "GPU-") ||
        !canonical_bdf(device.pci_bdf) || device.total_memory_bytes == 0U ||
        !device.device_major || !device.device_minor ||
        !device.device_node_mapping_complete ||
        (device.numa_node && *device.numa_node < 0) ||
        !device.current_mig_mode || !device.pending_mig_mode ||
        *device.current_mig_mode > 1U || *device.pending_mig_mode > 1U ||
        *device.current_mig_mode != *device.pending_mig_mode ||
        !device.display_active || !device.display_mode_enabled ||
        !device.display_evidence_complete ||
        !known_context(device.contexts.compute) ||
        !known_context(device.contexts.graphics) ||
        !process_evidence_valid(
            device.compute_processes, device.contexts.compute,
            implementation_->config.maximum_processes_per_device,
            std::nullopt) ||
        !process_evidence_valid(
            device.graphics_processes, device.contexts.graphics,
            implementation_->config.maximum_processes_per_device,
            std::nullopt) ||
        !ids.insert(device.uuid).second ||
        !bdfs.insert(device.pci_bdf).second ||
        device.partitions.size() >
            implementation_->config.maximum_partitions_per_device)
      complete = false;
    std::uint64_t partition_memory = 0U;
    if (device.partitions.size() >
        HostResourceBounds::maximum_resources -
            std::min(raw_resource_count, HostResourceBounds::maximum_resources))
      complete = false;
    else
      raw_resource_count += device.partitions.size();
    for (const auto &partition : device.partitions) {
      if (!canonical_uuid(partition.uuid, "MIG-") ||
          partition.total_memory_bytes == 0U || !partition.gpu_instance_id ||
          !partition.compute_instance_id || !device.current_mig_mode ||
          *device.current_mig_mode == 0U ||
          !known_context(partition.contexts.compute) ||
          !known_context(partition.contexts.graphics) ||
          !process_evidence_valid(
              partition.compute_processes, partition.contexts.compute,
              implementation_->config.maximum_processes_per_device,
              std::pair(*partition.gpu_instance_id,
                        *partition.compute_instance_id)) ||
          !process_evidence_valid(
              partition.graphics_processes, partition.contexts.graphics,
              implementation_->config.maximum_processes_per_device,
              std::pair(*partition.gpu_instance_id,
                        *partition.compute_instance_id)) ||
          partition.total_memory_bytes >
              device.total_memory_bytes -
                  std::min(partition_memory, device.total_memory_bytes) ||
          !ids.insert(partition.uuid).second)
        complete = false;
      else
        partition_memory += partition.total_memory_bytes;
      complete = complete && partition.evidence_complete;
    }
    complete = complete && device.evidence_complete;
  }
  if (raw_resource_count > HostResourceBounds::maximum_resources)
    complete = false;

  HostKernelSnapshot snapshot;
  snapshot.api_version = std::string(kHostInventoryApiVersion);
  snapshot.host_id = raw.host_id;
  snapshot.boot_id = raw.boot_id;
  snapshot.broker_epoch = raw.broker_epoch;
  const std::string revision_evidence =
      raw.begin_revision + "\n" + raw.end_revision +
      "\ntime:" + std::to_string(raw.capture_started_monotonic_ns) + ":" +
      std::to_string(raw.capture_finished_monotonic_ns) +
      "\ntrust:" + (raw.trusted_host_namespace ? "host" : "no-host") + ":" +
      (raw.trusted_nvml_loader ? "loader" : "no-loader") +
      (complete ? "\ncomplete" : "\nfail-closed");
  const std::string revision = digest_revision(revision_evidence);
  snapshot.begin_revision = revision;
  snapshot.end_revision = revision;
  const std::string probe_detail =
      (complete ? "enforcement-grade:point-in-time-scheduling:"
                : ((!implementation_->config.trusted_host_namespace ||
                    !implementation_->config.trusted_nvml_loader ||
                    !raw.trusted_host_namespace || !raw.trusted_nvml_loader)
                       ? "observation-only:untrusted-host-or-loader:"
                       : "probe-unknown:stale-torn-or-incomplete:")) +
      (raw.detail.empty() ? "nvidia" : raw.detail);
  snapshot.probes.push_back(
      {.vendor = HostAcceleratorVendor::nvidia,
       .disposition = complete
                          ? ProbeDisposition::complete
                          : (raw.nvml_loaded ? ProbeDisposition::partial
                                             : ProbeDisposition::unavailable),
       .context_details_complete = complete,
       .detail = probe_detail});

  if (raw.devices.size() <= implementation_->config.maximum_devices) {
    std::set<std::string> emitted_ids;
    std::set<std::string> emitted_bdfs;
    for (const auto &device : raw.devices) {
      if (!canonical_uuid(device.uuid, "GPU-") ||
          !canonical_bdf(device.pci_bdf) || device.total_memory_bytes == 0U ||
          !emitted_ids.insert(device.uuid).second ||
          !emitted_bdfs.insert(device.pci_bdf).second)
        continue;
      ObservedHostResource resource;
      resource.id = {.kind = HostResourceKind::accelerator,
                     .vendor = HostAcceleratorVendor::nvidia,
                     .stable_id = device.uuid,
                     .parent_id = std::nullopt};
      resource.disposition = ResourceObservationDisposition::probe_unknown;
      resource.compute_contexts = ResourceContextDisposition::unknown;
      resource.graphics_contexts = ResourceContextDisposition::unknown;
      if (complete) {
        resource.compute_contexts = device.contexts.compute;
        resource.graphics_contexts =
            *device.display_active || *device.display_mode_enabled
                ? ResourceContextDisposition::present
                : device.contexts.graphics;
        if (*device.current_mig_mode != 0U)
          resource.disposition = ResourceObservationDisposition::probe_unknown;
        else
          resource.disposition =
              resource.compute_contexts ==
                          ResourceContextDisposition::present ||
                      resource.graphics_contexts ==
                          ResourceContextDisposition::present
                  ? ResourceObservationDisposition::occupied
                  : ResourceObservationDisposition::audited_eligible;
      }
      resource.pci_bdf = device.pci_bdf;
      if (device.device_node_mapping_complete && device.device_major &&
          device.device_minor) {
        resource.device_major = device.device_major;
        resource.device_minor = device.device_minor;
      }
      if (device.numa_node && *device.numa_node >= 0)
        resource.numa_node = device.numa_node;
      resource.total_memory_bytes = device.total_memory_bytes;
      resource.labels = {{"backend", "linux-nvml"}};
      resource.labels.emplace("evidence-scope", "point-in-time-scheduling");
      resource.labels.emplace(
          "host-namespace",
          raw.trusted_host_namespace &&
                  implementation_->config.trusted_host_namespace
              ? "attested"
              : "unattested");
      resource.labels.emplace(
          "nvml-loader",
          raw.trusted_nvml_loader && implementation_->config.trusted_nvml_loader
              ? "attested"
              : "unattested");
      if (device.display_active && *device.display_active)
        resource.labels.emplace("display", "active");
      else if (device.display_mode_enabled && *device.display_mode_enabled)
        resource.labels.emplace("display", "mode-enabled");
      if (device.current_mig_mode && *device.current_mig_mode != 0U)
        resource.labels.emplace("partition-parent", "nonselectable");
      if (resource.disposition == ResourceObservationDisposition::occupied)
        resource.labels.emplace("occupancy", "foreign-observed");
      else if (resource.disposition ==
               ResourceObservationDisposition::probe_unknown)
        resource.labels.emplace("occupancy", "unknown");
      snapshot.resources.push_back(resource);
      std::uint64_t included_memory = 0U;
      for (const auto &partition : device.partitions) {
        if (!canonical_uuid(partition.uuid, "MIG-") ||
            partition.total_memory_bytes == 0U ||
            partition.total_memory_bytes >
                device.total_memory_bytes -
                    std::min(included_memory, device.total_memory_bytes) ||
            !emitted_ids.insert(partition.uuid).second)
          continue;
        included_memory += partition.total_memory_bytes;
        ObservedHostResource child;
        child.id = {.kind = HostResourceKind::accelerator_partition,
                    .vendor = HostAcceleratorVendor::nvidia,
                    .stable_id = partition.uuid,
                    .parent_id = device.uuid};
        child.disposition = ResourceObservationDisposition::probe_unknown;
        child.compute_contexts = ResourceContextDisposition::unknown;
        child.graphics_contexts = ResourceContextDisposition::unknown;
        if (complete) {
          child.compute_contexts = partition.contexts.compute;
          child.graphics_contexts = partition.contexts.graphics;
          child.disposition =
              child.compute_contexts == ResourceContextDisposition::present ||
                      child.graphics_contexts ==
                          ResourceContextDisposition::present
                  ? ResourceObservationDisposition::occupied
                  : ResourceObservationDisposition::audited_eligible;
        }
        child.pci_bdf = device.pci_bdf;
        if (device.numa_node && *device.numa_node >= 0)
          child.numa_node = device.numa_node;
        child.total_memory_bytes = partition.total_memory_bytes;
        child.labels = {
            {"backend", "linux-nvml"},
            {"partition", "mig"},
            {"evidence-scope", "point-in-time-scheduling"},
            {"host-namespace",
             raw.trusted_host_namespace &&
                     implementation_->config.trusted_host_namespace
                 ? "attested"
                 : "unattested"},
            {"nvml-loader", raw.trusted_nvml_loader &&
                                    implementation_->config.trusted_nvml_loader
                                ? "attested"
                                : "unattested"}};
        if ((device.display_active && *device.display_active) ||
            (device.display_mode_enabled && *device.display_mode_enabled))
          child.labels.emplace("parent-display-policy",
                               "partition-requires-explicit-policy");
        if (child.disposition == ResourceObservationDisposition::occupied)
          child.labels.emplace("occupancy", "foreign-observed");
        else if (child.disposition ==
                 ResourceObservationDisposition::probe_unknown)
          child.labels.emplace("occupancy", "unknown");
        snapshot.resources.push_back(std::move(child));
      }
    }
  }
  std::ranges::sort(snapshot.resources, [](const auto &left,
                                           const auto &right) {
    return canonical_resource_key(left.id) < canonical_resource_key(right.id);
  });
  if (snapshot.resources.size() > HostResourceBounds::maximum_resources)
    snapshot.resources.clear();
  return snapshot;
}

} // namespace trainvm
