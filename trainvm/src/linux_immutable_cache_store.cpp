#include "trainvm/linux_immutable_cache_store.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <linux/openat2.h>
#include <memory>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumPathBytes = 4096U;
constexpr std::size_t kMaximumDepth = 64U;

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() { reset(); }
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept { return std::exchange(value_, -1); }

 private:
  void reset() noexcept {
    if (value_ >= 0)
      (void)::close(value_);
    value_ = -1;
  }
  int value_;
};

[[noreturn]] void fail(std::string_view message) {
  throw CacheArtifactAuthorityError(std::string(message));
}

[[noreturn]] void fail_errno(std::string_view message) {
  throw CacheArtifactAuthorityError(std::string(message) + ": " +
                                    std::strerror(errno));
}

bool valid_sha256(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool safe_name(std::string_view value) {
  return !value.empty() && value != "." && value != ".." &&
         value.size() <= 255U && value.find('/') == std::string_view::npos &&
         value.find('\0') == std::string_view::npos;
}

bool path_within(const std::filesystem::path& child,
                 const std::filesystem::path& root) {
  const auto normalized_child = child.lexically_normal();
  const auto normalized_root = root.lexically_normal();
  auto child_iterator = normalized_child.begin();
  for (auto root_iterator = normalized_root.begin();
       root_iterator != normalized_root.end();
       ++root_iterator, ++child_iterator) {
    if (child_iterator == normalized_child.end() ||
        *child_iterator != *root_iterator) {
      return false;
    }
  }
  return true;
}

Descriptor openat2_beneath(int root, std::string_view relative, int flags) {
  struct open_how how{};
  how.flags = static_cast<std::uint64_t>(flags | O_CLOEXEC | O_NOFOLLOW);
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS |
                RESOLVE_NO_XDEV;
  const std::string owned(relative);
  const long result =
      ::syscall(SYS_openat2, root, owned.c_str(), &how, sizeof(how));
  if (result < 0)
    fail_errno("secure cache path resolution failed");
  return Descriptor(static_cast<int>(result));
}

struct PinnedRoot {
  std::filesystem::path path;
  Descriptor descriptor;
  dev_t device{};
};

PinnedRoot pin_root(const std::filesystem::path& source, uid_t expected_uid,
                    bool publication) {
  if (source.empty() || !source.is_absolute() ||
      source.lexically_normal() != source ||
      source.native().size() > kMaximumPathBytes) {
    fail("cache store root must be a canonical bounded absolute path");
  }
  Descriptor descriptor(
      ::open(source.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0)
    fail_errno("cache store root could not be opened");
  struct stat metadata{};
  if (::fstat(descriptor.get(), &metadata) != 0 || !S_ISDIR(metadata.st_mode) ||
      metadata.st_uid != expected_uid ||
      (metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0 ||
      (publication && (metadata.st_mode & S_IWUSR) == 0)) {
    fail("cache store root has unsafe ownership, type, or mode");
  }
  return {.path = source,
          .descriptor = std::move(descriptor),
          .device = metadata.st_dev};
}

Descriptor duplicate(int descriptor) {
  const int result = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 3);
  if (result < 0)
    fail_errno("cache descriptor duplication failed");
  return Descriptor(result);
}

std::vector<std::string> entries(int directory) {
  struct DirectoryCloser {
    void operator()(DIR* value) const noexcept { (void)::closedir(value); }
  };
  Descriptor copied = duplicate(directory);
  DIR* stream = ::fdopendir(copied.release());
  if (stream == nullptr)
    fail_errno("cache directory stream could not be opened");
  std::unique_ptr<DIR, DirectoryCloser> owned(stream);
  std::vector<std::string> result;
  errno = 0;
  while (dirent* entry = ::readdir(stream)) {
    std::string name(entry->d_name);
    if (name == "." || name == "..")
      continue;
    if (!safe_name(name))
      fail("cache tree contains an invalid entry name");
    result.push_back(std::move(name));
    errno = 0;
  }
  if (errno != 0)
    fail_errno("cache directory enumeration failed");
  std::ranges::sort(result);
  return result;
}

Descriptor ensure_directory(int parent, std::string_view name, mode_t mode,
                            uid_t expected_uid) {
  if (!safe_name(name))
    fail("cache destination directory name is invalid");
  const std::string owned(name);
  if (::mkdirat(parent, owned.c_str(), mode) != 0 && errno != EEXIST) {
    fail_errno("cache destination directory could not be created");
  }
  Descriptor result = openat2_beneath(parent, owned, O_RDONLY | O_DIRECTORY);
  struct stat metadata{};
  if (::fstat(result.get(), &metadata) != 0 || !S_ISDIR(metadata.st_mode) ||
      metadata.st_uid != expected_uid ||
      (metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    fail("cache destination directory has unsafe metadata");
  }
  return result;
}

std::string hex_digest(EVP_MD_CTX* context) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int size = 0;
  if (EVP_DigestFinal_ex(context, bytes.data(), &size) != 1 || size != 32U) {
    fail("cache SHA-256 finalization failed");
  }
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result(64U, '0');
  for (std::size_t index = 0; index < 32U; ++index) {
    result[index * 2U] = alphabet[bytes[index] >> 4U];
    result[index * 2U + 1U] = alphabet[bytes[index] & 0x0fU];
  }
  return "sha256:" + result;
}

std::string sha256_bytes(std::string_view bytes) {
  return "sha256:" + sha256_hex(bytes);
}

struct Object {
  std::string relative_path;
  std::string sha256;
  std::uint64_t size_bytes{};
};

struct ScanState {
  std::uint64_t maximum_files{};
  std::uint64_t maximum_bytes{};
  std::uint64_t maximum_single_file{};
  std::uint64_t files{};
  std::uint64_t bytes{};
  std::vector<Object> objects;
};

std::string join_relative(std::string_view prefix, std::string_view name) {
  std::string result;
  if (!prefix.empty()) {
    result.append(prefix);
    result.push_back('/');
  }
  result.append(name);
  if (result.size() > kMaximumPathBytes)
    fail("cache relative path is too long");
  return result;
}

std::string copy_regular(int source_directory, std::string_view name,
                         const struct stat& expected, int destination_directory,
                         ScanState& state) {
  const std::string owned(name);
  Descriptor source = openat2_beneath(source_directory, owned, O_RDONLY);
  struct stat before{};
  if (::fstat(source.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_dev != expected.st_dev || before.st_ino != expected.st_ino ||
      before.st_size != expected.st_size ||
      before.st_mtim.tv_sec != expected.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != expected.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != expected.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != expected.st_ctim.tv_nsec ||
      before.st_size <= 0 ||
      static_cast<std::uint64_t>(before.st_size) > state.maximum_single_file) {
    fail("cache source file changed or exceeds its bound");
  }
  Descriptor destination(
      ::openat(destination_directory, owned.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0440));
  if (destination.get() < 0)
    fail_errno("cache destination file creation failed");
  EVP_MD_CTX* raw = EVP_MD_CTX_new();
  if (raw == nullptr)
    fail("cache SHA-256 context allocation failed");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      raw, &EVP_MD_CTX_free);
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    fail("cache SHA-256 initialization failed");
  }
  std::array<unsigned char, 1U << 20U> buffer{};
  std::uint64_t copied = 0U;
  for (;;) {
    const ssize_t count = ::read(source.get(), buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0)
      fail_errno("cache source read failed");
    if (count == 0)
      break;
    const auto amount = static_cast<std::size_t>(count);
    copied += static_cast<std::uint64_t>(amount);
    if (copied > static_cast<std::uint64_t>(before.st_size) ||
        EVP_DigestUpdate(context.get(), buffer.data(), amount) != 1) {
      fail("cache source grew or hashing failed");
    }
    std::size_t offset = 0U;
    while (offset < amount) {
      const ssize_t written =
          ::write(destination.get(), buffer.data() + offset, amount - offset);
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        fail_errno("cache destination write failed");
      offset += static_cast<std::size_t>(written);
    }
  }
  struct stat after{};
  if (copied != static_cast<std::uint64_t>(before.st_size) ||
      ::fstat(source.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      ::fchmod(destination.get(), 0440) != 0 ||
      ::fsync(destination.get()) != 0) {
    fail("cache source changed during copy or destination sync failed");
  }
  return hex_digest(context.get());
}

void copy_tree(int source, int destination, std::string_view prefix,
               std::size_t depth, ScanState& state) {
  if (depth > kMaximumDepth)
    fail("cache source tree is too deep");
  for (const std::string& name : entries(source)) {
    struct stat metadata{};
    if (::fstatat(source, name.c_str(), &metadata, AT_SYMLINK_NOFOLLOW) != 0) {
      fail_errno("cache source entry stat failed");
    }
    const std::string relative = join_relative(prefix, name);
    if (S_ISDIR(metadata.st_mode)) {
      Descriptor source_child =
          openat2_beneath(source, name, O_RDONLY | O_DIRECTORY);
      struct stat opened{};
      if (::fstat(source_child.get(), &opened) != 0 ||
          !S_ISDIR(opened.st_mode) || opened.st_dev != metadata.st_dev ||
          opened.st_ino != metadata.st_ino) {
        fail("cache source directory changed during traversal");
      }
      Descriptor destination_child = ensure_directory(
          destination, name, 0750, static_cast<uid_t>(::geteuid()));
      copy_tree(source_child.get(), destination_child.get(), relative,
                depth + 1U, state);
      if (::fchmod(destination_child.get(), 0550) != 0 ||
          ::fsync(destination_child.get()) != 0) {
        fail("cache destination directory sync failed");
      }
    } else if (S_ISREG(metadata.st_mode)) {
      if (state.files == state.maximum_files || metadata.st_size <= 0 ||
          static_cast<std::uint64_t>(metadata.st_size) >
              state.maximum_bytes - state.bytes) {
        fail("cache source tree exceeds its file or byte bound");
      }
      const std::string object_digest =
          copy_regular(source, name, metadata, destination, state);
      ++state.files;
      state.bytes += static_cast<std::uint64_t>(metadata.st_size);
      state.objects.push_back(
          {.relative_path = relative,
           .sha256 = object_digest,
           .size_bytes = static_cast<std::uint64_t>(metadata.st_size)});
    } else {
      fail("cache source tree contains a symlink or nonregular entry");
    }
  }
}

std::string hash_regular(int directory, std::string_view name,
                         std::uint64_t expected_size) {
  Descriptor file = openat2_beneath(directory, name, O_RDONLY);
  struct stat before{};
  if (::fstat(file.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uint64_t>(before.st_size) != expected_size ||
      (before.st_mode & 0222) != 0) {
    fail("published cache file metadata is invalid");
  }
  EVP_MD_CTX* raw = EVP_MD_CTX_new();
  if (raw == nullptr)
    fail("cache SHA-256 context allocation failed");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      raw, &EVP_MD_CTX_free);
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    fail("cache SHA-256 initialization failed");
  }
  std::array<unsigned char, 1U << 20U> buffer{};
  std::uint64_t read_bytes = 0U;
  for (;;) {
    const ssize_t count = ::read(file.get(), buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0)
      fail_errno("published cache file read failed");
    if (count == 0)
      break;
    read_bytes += static_cast<std::uint64_t>(count);
    if (read_bytes > expected_size ||
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      fail("published cache file changed while hashing");
    }
  }
  struct stat after{};
  if (read_bytes != expected_size || ::fstat(file.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      before.st_mode != after.st_mode) {
    fail("published cache file changed while hashing");
  }
  return hex_digest(context.get());
}

void write_file(int directory, std::string_view name, std::string_view bytes) {
  const std::string owned(name);
  Descriptor file(::openat(directory, owned.c_str(),
                           O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                           0440));
  if (file.get() < 0)
    fail_errno("cache manifest creation failed");
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    const ssize_t written =
        ::write(file.get(), bytes.data() + offset, bytes.size() - offset);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      fail_errno("cache manifest write failed");
    offset += static_cast<std::size_t>(written);
  }
  if (::fchmod(file.get(), 0440) != 0 || ::fsync(file.get()) != 0) {
    fail("cache manifest sync failed");
  }
}

std::string read_file(int directory, std::string_view name,
                      std::size_t maximum_bytes) {
  Descriptor file = openat2_beneath(directory, name, O_RDONLY);
  struct stat before{};
  if (::fstat(file.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uint64_t>(before.st_size) > maximum_bytes ||
      (before.st_mode & 0222) != 0) {
    fail("cache manifest metadata is invalid");
  }
  std::string result(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0U;
  while (offset < result.size()) {
    const ssize_t count =
        ::read(file.get(), result.data() + offset, result.size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      fail_errno("cache manifest read failed");
    offset += static_cast<std::size_t>(count);
  }
  struct stat after{};
  if (::fstat(file.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      before.st_mode != after.st_mode) {
    fail("cache manifest changed while it was read");
  }
  return result;
}

std::string random_name() {
  std::array<unsigned char, 16U> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    fail("cache temporary-name entropy failed");
  }
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result = ".publish-";
  for (unsigned char byte : bytes) {
    result.push_back(alphabet[byte >> 4U]);
    result.push_back(alphabet[byte & 0x0fU]);
  }
  return result;
}

nlohmann::json objects_json(const std::vector<Object>& objects) {
  nlohmann::json result = nlohmann::json::array();
  for (const Object& object : objects) {
    result.push_back({{"relative_path", object.relative_path},
                      {"sha256", object.sha256},
                      {"size_bytes", object.size_bytes}});
  }
  return result;
}

std::string store_digest(const ImmutableCacheTreeReceipt& receipt) {
  return "sha256:" + sha256_hex(nlohmann::json{
                         {"domain", "trainvm.linux-cache-store/v1"},
                         {"namespace_digest", receipt.namespace_digest},
                         {"artifact_tree_digest", receipt.artifact_tree_digest},
                         {"manifest_digest", receipt.manifest_digest},
                         {"content_address", receipt.content_address},
                         {"file_count", receipt.file_count},
                         {"total_bytes", receipt.total_bytes},
                         {"immutable", receipt.immutable},
                     }
                                    .dump());
}

struct ParsedManifest {
  std::string namespace_digest;
  std::string tree_digest;
  std::uint64_t file_count{};
  std::uint64_t total_bytes{};
  std::vector<Object> objects;
};

ParsedManifest parse_manifest(std::string_view bytes) {
  nlohmann::json value;
  try {
    value = nlohmann::json::parse(bytes);
  } catch (const nlohmann::json::exception&) {
    fail("cache manifest is malformed");
  }
  if (!value.is_object() || value.dump() != bytes || value.size() != 8U ||
      value.value("api_version", std::string{}) !=
          "trainvm.immutable-cache-tree-manifest/v1" ||
      value.value("payload_directory", std::string{}) != "payload" ||
      !value.contains("namespace_digest") ||
      !value.at("namespace_digest").is_string() ||
      !value.contains("artifact_tree_digest") ||
      !value.at("artifact_tree_digest").is_string() ||
      !value.contains("file_count") ||
      !value.at("file_count").is_number_unsigned() ||
      !value.contains("total_bytes") ||
      !value.at("total_bytes").is_number_unsigned() ||
      !value.contains("objects") || !value.at("objects").is_array() ||
      !value.contains("format") || !value.at("format").is_string() ||
      value.at("format").get_ref<const std::string&>() !=
          "descriptor-copy-sha256-v1") {
    fail("cache manifest schema is invalid");
  }
  ParsedManifest result{
      .namespace_digest = value.at("namespace_digest").get<std::string>(),
      .tree_digest = value.at("artifact_tree_digest").get<std::string>(),
      .file_count = value.at("file_count").get<std::uint64_t>(),
      .total_bytes = value.at("total_bytes").get<std::uint64_t>(),
      .objects = {},
  };
  std::string previous;
  for (const auto& item : value.at("objects")) {
    if (!item.is_object() || item.size() != 3U ||
        !item.contains("relative_path") ||
        !item.at("relative_path").is_string() || !item.contains("sha256") ||
        !item.at("sha256").is_string() || !item.contains("size_bytes") ||
        !item.at("size_bytes").is_number_unsigned()) {
      fail("cache manifest object entry is invalid");
    }
    Object object{.relative_path = item.at("relative_path").get<std::string>(),
                  .sha256 = item.at("sha256").get<std::string>(),
                  .size_bytes = item.at("size_bytes").get<std::uint64_t>()};
    if (object.relative_path.empty() ||
        object.relative_path.size() > kMaximumPathBytes ||
        object.relative_path.starts_with('/') ||
        object.relative_path.contains("//") || object.relative_path == ".." ||
        object.relative_path.starts_with("../") ||
        object.relative_path.contains("/../") || !valid_sha256(object.sha256) ||
        object.size_bytes == 0U ||
        (!previous.empty() && object.relative_path <= previous)) {
      fail("cache manifest objects are noncanonical");
    }
    previous = object.relative_path;
    result.objects.push_back(std::move(object));
  }
  if (!valid_sha256(result.namespace_digest) ||
      !valid_sha256(result.tree_digest) || result.file_count == 0U ||
      result.file_count != result.objects.size()) {
    fail("cache manifest identity is invalid");
  }
  const auto total = std::ranges::fold_left(
      result.objects, std::uint64_t{0},
      [](std::uint64_t sum, const Object& object) {
        if (object.size_bytes > std::numeric_limits<std::uint64_t>::max() - sum)
          fail("cache manifest byte count overflows");
        return sum + object.size_bytes;
      });
  if (total != result.total_bytes)
    fail("cache manifest byte total is invalid");
  const nlohmann::json tree_body{
      {"api_version", "trainvm.immutable-cache-tree/v1"},
      {"namespace_digest", result.namespace_digest},
      {"objects", objects_json(result.objects)}};
  if ("sha256:" + sha256_hex(nlohmann::json{
                      {"domain", "trainvm.immutable-cache-tree/v1"},
                      {"value", tree_body}}
                                 .dump()) !=
      result.tree_digest) {
    fail("cache manifest tree digest is invalid");
  }
  return result;
}

Descriptor open_parent_for_object(int payload, const std::string& relative,
                                  std::string& leaf) {
  const std::filesystem::path path(relative);
  leaf = path.filename().string();
  if (!safe_name(leaf))
    fail("cache object leaf is invalid");
  const auto parent = path.parent_path();
  if (parent.empty())
    return duplicate(payload);
  return openat2_beneath(payload, parent.generic_string(),
                         O_RDONLY | O_DIRECTORY);
}

void remove_tree(int parent, const std::string& name) noexcept {
  try {
    Descriptor directory =
        openat2_beneath(parent, name, O_RDONLY | O_DIRECTORY);
    for (const std::string& child : entries(directory.get())) {
      struct stat metadata{};
      if (::fstatat(directory.get(), child.c_str(), &metadata,
                    AT_SYMLINK_NOFOLLOW) != 0)
        continue;
      if (S_ISDIR(metadata.st_mode)) {
        remove_tree(directory.get(), child);
      } else {
        (void)::unlinkat(directory.get(), child.c_str(), 0);
      }
    }
    (void)::unlinkat(parent, name.c_str(), AT_REMOVEDIR);
  } catch (...) {
  }
}

} // namespace

struct LinuxImmutableCacheStore::Implementation {
  LinuxImmutableCacheStoreConfig config;
  PinnedRoot publication;
  std::vector<PinnedRoot> sources;

  explicit Implementation(LinuxImmutableCacheStoreConfig value)
      : config(std::move(value)),
        publication(
            pin_root(config.publication_root, config.authority_uid, true)) {
    if (config.allowed_source_roots.empty() ||
        config.maximum_file_count == 0U || config.maximum_total_bytes == 0U ||
        config.maximum_single_file_bytes == 0U ||
        config.maximum_file_count > 1'000'000U ||
        config.maximum_total_bytes > (1ULL << 50U) ||
        config.maximum_single_file_bytes > config.maximum_total_bytes) {
      fail("cache store configuration is empty or exceeds hard bounds");
    }
    if (config.authority_uid != ::geteuid()) {
      fail("cache store authority UID is not the effective service identity");
    }
    for (const auto& root : config.allowed_source_roots) {
      if (path_within(root, config.publication_root) ||
          path_within(config.publication_root, root)) {
        fail("cache source and publication roots must not overlap");
      }
      if (std::ranges::any_of(sources, [&](const PinnedRoot& existing) {
            return path_within(root, existing.path) ||
                   path_within(existing.path, root);
          })) {
        fail("cache source roots must be unique and nonoverlapping");
      }
      sources.push_back(pin_root(root, config.source_uid, false));
    }
  }

  std::pair<Descriptor, const PinnedRoot*>
  open_source(const std::filesystem::path& requested) const {
    if (requested.empty() || !requested.is_absolute() ||
        requested.lexically_normal() != requested) {
      fail("cache source must be a canonical absolute directory");
    }
    const PinnedRoot* selected = nullptr;
    for (const auto& root : sources) {
      if (path_within(requested, root.path) &&
          (selected == nullptr ||
           root.path.native().size() > selected->path.native().size())) {
        selected = &root;
      }
    }
    if (selected == nullptr)
      fail("cache source is outside allowed roots");
    const auto relative = requested.lexically_relative(selected->path);
    Descriptor descriptor = relative.empty() || relative == "."
                                ? duplicate(selected->descriptor.get())
                                : openat2_beneath(selected->descriptor.get(),
                                                  relative.generic_string(),
                                                  O_RDONLY | O_DIRECTORY);
    struct stat metadata{};
    if (::fstat(descriptor.get(), &metadata) != 0 ||
        !S_ISDIR(metadata.st_mode) || metadata.st_dev != selected->device) {
      fail("cache source directory identity changed");
    }
    return {std::move(descriptor), selected};
  }

  ImmutableCacheTreeReceipt verify_address(const std::string& address) const {
    const auto lower_hex = [](char character) {
      return (character >= '0' && character <= '9') ||
             (character >= 'a' && character <= 'f');
    };
    if (address.size() != 129U || address[64U] != '/' ||
        !std::ranges::all_of(address.substr(0U, 64U), lower_hex) ||
        !std::ranges::all_of(address.substr(65U), lower_hex)) {
      fail("cache content address is invalid");
    }
    const std::string namespace_hex = address.substr(0U, 64U);
    const std::string tree_hex = address.substr(65U);
    Descriptor namespaces = openat2_beneath(
        publication.descriptor.get(), "namespaces", O_RDONLY | O_DIRECTORY);
    Descriptor namespace_directory = openat2_beneath(
        namespaces.get(), namespace_hex, O_RDONLY | O_DIRECTORY);
    Descriptor artifacts = openat2_beneath(namespace_directory.get(),
                                           "artifacts", O_RDONLY | O_DIRECTORY);
    Descriptor artifact =
        openat2_beneath(artifacts.get(), tree_hex, O_RDONLY | O_DIRECTORY);
    struct stat artifact_metadata{};
    if (::fstat(artifact.get(), &artifact_metadata) != 0 ||
        !S_ISDIR(artifact_metadata.st_mode) ||
        artifact_metadata.st_uid != config.authority_uid ||
        (artifact_metadata.st_mode & 0222) != 0) {
      fail("published cache revision metadata is not immutable");
    }
    const std::string manifest_bytes =
        read_file(artifact.get(), "manifest.json", 256U << 20U);
    const ParsedManifest manifest = parse_manifest(manifest_bytes);
    if (manifest.namespace_digest != "sha256:" + namespace_hex ||
        manifest.tree_digest != "sha256:" + tree_hex ||
        manifest.file_count > config.maximum_file_count ||
        manifest.total_bytes > config.maximum_total_bytes) {
      fail("cache content address disagrees with its manifest");
    }
    Descriptor payload =
        openat2_beneath(artifact.get(), "payload", O_RDONLY | O_DIRECTORY);
    struct stat payload_metadata{};
    if (::fstat(payload.get(), &payload_metadata) != 0 ||
        !S_ISDIR(payload_metadata.st_mode) ||
        payload_metadata.st_uid != config.authority_uid ||
        (payload_metadata.st_mode & 0222) != 0) {
      fail("published cache payload metadata is not immutable");
    }
    std::uint64_t observed_files = 0U;
    std::uint64_t observed_bytes = 0U;
    for (const Object& object : manifest.objects) {
      std::string leaf;
      Descriptor parent =
          open_parent_for_object(payload.get(), object.relative_path, leaf);
      if (hash_regular(parent.get(), leaf, object.size_bytes) !=
          object.sha256) {
        fail("published cache object digest changed");
      }
      ++observed_files;
      observed_bytes += object.size_bytes;
    }
    // A separate descriptor walk detects undeclared files and directories.
    ScanState observed{.maximum_files = config.maximum_file_count,
                       .maximum_bytes = config.maximum_total_bytes,
                       .maximum_single_file = config.maximum_single_file_bytes,
                       .files = 0U,
                       .bytes = 0U,
                       .objects = {}};
    std::function<void(int, std::string_view, std::size_t)> walk;
    std::set<std::string> actual_directories;
    walk = [&](int directory, std::string_view prefix, std::size_t depth) {
      if (depth > kMaximumDepth)
        fail("published cache tree is too deep");
      for (const std::string& name : entries(directory)) {
        struct stat metadata{};
        if (::fstatat(directory, name.c_str(), &metadata,
                      AT_SYMLINK_NOFOLLOW) != 0)
          fail_errno("published cache entry stat failed");
        const std::string relative = join_relative(prefix, name);
        if (S_ISDIR(metadata.st_mode)) {
          if ((metadata.st_mode & 0222) != 0 ||
              metadata.st_uid != config.authority_uid ||
              !actual_directories.insert(relative).second) {
            fail("published cache directory metadata is invalid");
          }
          Descriptor child =
              openat2_beneath(directory, name, O_RDONLY | O_DIRECTORY);
          walk(child.get(), relative, depth + 1U);
        } else if (S_ISREG(metadata.st_mode)) {
          ++observed.files;
          if (metadata.st_size <= 0 ||
              static_cast<std::uint64_t>(metadata.st_size) >
                  config.maximum_total_bytes - observed.bytes)
            fail("published cache tree exceeds bounds");
          observed.bytes += static_cast<std::uint64_t>(metadata.st_size);
          observed.objects.push_back(
              {.relative_path = relative, .sha256 = {}, .size_bytes = 0U});
        } else {
          fail("published cache tree contains a nonregular entry");
        }
      }
    };
    walk(payload.get(), "", 0U);
    std::vector<std::string> declared;
    std::vector<std::string> actual;
    for (const auto& object : manifest.objects)
      declared.push_back(object.relative_path);
    for (const auto& object : observed.objects)
      actual.push_back(object.relative_path);
    std::set<std::string> expected_directories;
    for (const auto& object : manifest.objects) {
      std::filesystem::path parent =
          std::filesystem::path(object.relative_path).parent_path();
      while (!parent.empty()) {
        expected_directories.insert(parent.generic_string());
        parent = parent.parent_path();
      }
    }
    if (declared != actual || observed_files != manifest.file_count ||
        observed_bytes != manifest.total_bytes ||
        observed.files != manifest.file_count ||
        observed.bytes != manifest.total_bytes ||
        expected_directories != actual_directories) {
      fail("published cache payload inventory changed");
    }
    ImmutableCacheTreeReceipt receipt{
        .api_version = "trainvm.immutable-cache-tree/v1",
        .namespace_digest = manifest.namespace_digest,
        .artifact_tree_digest = manifest.tree_digest,
        .manifest_digest = sha256_bytes(manifest_bytes),
        .content_address = address,
        .file_count = manifest.file_count,
        .total_bytes = manifest.total_bytes,
        .immutable = true,
        .store_receipt_digest = {},
    };
    receipt.store_receipt_digest = store_digest(receipt);
    return receipt;
  }
};

LinuxImmutableCacheStore::LinuxImmutableCacheStore(
    LinuxImmutableCacheStoreConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}

LinuxImmutableCacheStore::~LinuxImmutableCacheStore() = default;
LinuxImmutableCacheStore::LinuxImmutableCacheStore(
    LinuxImmutableCacheStore&&) noexcept = default;
LinuxImmutableCacheStore& LinuxImmutableCacheStore::operator=(
    LinuxImmutableCacheStore&&) noexcept = default;

ImmutableCacheTreeReceipt LinuxImmutableCacheStore::publish(
    const CacheNamespaceAuthorityReceipt& authority,
    const CacheArtifactCandidate& candidate) {
  (void)cache_namespace_authority_receipt_json(authority);
  if (candidate.maximum_file_count == 0U ||
      candidate.maximum_file_count >
          implementation_->config.maximum_file_count ||
      candidate.maximum_total_bytes == 0U ||
      candidate.maximum_total_bytes >
          implementation_->config.maximum_total_bytes) {
    fail("cache candidate bounds exceed store policy");
  }
  const std::filesystem::path source_path(candidate.source_directory);
  auto [source, source_root] = implementation_->open_source(source_path);
  (void)source_root;
  Descriptor namespaces = ensure_directory(
      implementation_->publication.descriptor.get(), "namespaces", 0750,
      implementation_->config.authority_uid);
  const std::string namespace_hex =
      authority.cache_namespace.namespace_digest.substr(7U);
  Descriptor namespace_directory =
      ensure_directory(namespaces.get(), namespace_hex, 0750,
                       implementation_->config.authority_uid);
  Descriptor artifacts =
      ensure_directory(namespace_directory.get(), "artifacts", 0750,
                       implementation_->config.authority_uid);
  if (::fsync(implementation_->publication.descriptor.get()) != 0 ||
      ::fsync(namespaces.get()) != 0 ||
      ::fsync(namespace_directory.get()) != 0) {
    fail("cache publication namespace creation was not durable");
  }
  const std::string temporary_name = random_name();
  if (::mkdirat(artifacts.get(), temporary_name.c_str(), 0750) != 0)
    fail_errno("cache temporary revision could not be created");
  try {
    Descriptor temporary = openat2_beneath(artifacts.get(), temporary_name,
                                           O_RDONLY | O_DIRECTORY);
    Descriptor payload =
        ensure_directory(temporary.get(), "payload", 0750,
                         implementation_->config.authority_uid);
    ScanState state{
        .maximum_files = candidate.maximum_file_count,
        .maximum_bytes = candidate.maximum_total_bytes,
        .maximum_single_file =
            implementation_->config.maximum_single_file_bytes,
        .files = 0U,
        .bytes = 0U,
        .objects = {},
    };
    copy_tree(source.get(), payload.get(), "", 0U, state);
    if (state.files == 0U || state.bytes == 0U ||
        ::fchmod(payload.get(), 0550) != 0 || ::fsync(payload.get()) != 0) {
      fail("cache candidate is empty or payload sync failed");
    }
    const nlohmann::json objects = objects_json(state.objects);
    const nlohmann::json tree_body{
        {"api_version", "trainvm.immutable-cache-tree/v1"},
        {"namespace_digest", authority.cache_namespace.namespace_digest},
        {"objects", objects},
    };
    const std::string tree_digest =
        "sha256:" +
        sha256_hex(nlohmann::json{{"domain", "trainvm.immutable-cache-tree/v1"},
                                  {"value", tree_body}}
                       .dump());
    const nlohmann::json manifest{
        {"api_version", "trainvm.immutable-cache-tree-manifest/v1"},
        {"namespace_digest", authority.cache_namespace.namespace_digest},
        {"artifact_tree_digest", tree_digest},
        {"payload_directory", "payload"},
        {"file_count", state.files},
        {"total_bytes", state.bytes},
        {"objects", objects},
        {"format", "descriptor-copy-sha256-v1"},
    };
    const std::string manifest_bytes = manifest.dump();
    write_file(temporary.get(), "manifest.json", manifest_bytes);
    if (::fchmod(temporary.get(), 0550) != 0 || ::fsync(temporary.get()) != 0)
      fail("cache temporary revision sync failed");
    const std::string tree_hex = tree_digest.substr(7U);
    if (::renameat(artifacts.get(), temporary_name.c_str(), artifacts.get(),
                   tree_hex.c_str()) != 0) {
      if (errno != EEXIST && errno != ENOTEMPTY)
        fail_errno("cache atomic promotion failed");
      remove_tree(artifacts.get(), temporary_name);
    }
    if (::fsync(artifacts.get()) != 0)
      fail("cache artifact namespace sync failed");
    const std::string address = namespace_hex + "/" + tree_hex;
    const ImmutableCacheTreeReceipt verified =
        implementation_->verify_address(address);
    if (verified.manifest_digest != sha256_bytes(manifest_bytes) ||
        verified.file_count != state.files ||
        verified.total_bytes != state.bytes) {
      fail("cache promoted revision differs from the candidate");
    }
    return verified;
  } catch (...) {
    remove_tree(artifacts.get(), temporary_name);
    throw;
  }
}

ImmutableCacheTreeReceipt
LinuxImmutableCacheStore::verify(const std::string& content_address) {
  return implementation_->verify_address(content_address);
}

} // namespace trainvm
