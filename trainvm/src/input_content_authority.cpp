#include "trainvm/input_content_authority.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <linux/magic.h>
#include <linux/stat.h>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

constexpr std::uint64_t kMaximumFiles = 10'000'000U;
constexpr std::uint64_t kMaximumBytes =
    std::uint64_t{16U} * 1024U * 1024U * 1024U * 1024U * 1024U;
constexpr std::size_t kMaximumDepth = 128U;
constexpr std::size_t kMaximumNameBytes = 4096U;
constexpr std::size_t kDigestBytes = 32U;
constexpr std::size_t kReadBufferBytes = 1024U * 1024U;
constexpr char kFileDomain[] = "trainvm.input-content.file/v1";
constexpr char kDirectoryDomain[] = "trainvm.input-content.directory/v1";
constexpr long kZfsSuperMagic = 0x2fc12fc1L;

using Digest = std::array<unsigned char, kDigestBytes>;

[[noreturn]] void fail_system(std::string_view operation) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::strerror(errno));
}

class Descriptor final {
public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }

  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
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

  [[nodiscard]] int get() const noexcept { return value_; }

private:
  int value_;
};

class DigestContext final {
public:
  DigestContext() : context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
    if (!context_ ||
        EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
      throw std::runtime_error("could not initialize SHA-256");
    }
  }

  void update(const void *data, std::size_t size) {
    if (size != 0U && EVP_DigestUpdate(context_.get(), data, size) != 1)
      throw std::runtime_error("could not update SHA-256");
  }

  template <std::size_t Size>
  void update(const std::array<unsigned char, Size> &value) {
    update(value.data(), value.size());
  }

  [[nodiscard]] Digest finish() {
    Digest digest{};
    unsigned int length = 0U;
    if (EVP_DigestFinal_ex(context_.get(), digest.data(), &length) != 1 ||
        length != digest.size()) {
      throw std::runtime_error("could not finalize SHA-256");
    }
    return digest;
  }

private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};

struct NodeMeasurement final {
  ContentRootKind kind{};
  std::uint64_t file_count{};
  std::uint64_t total_bytes{};
  Digest digest{};

  bool operator==(const NodeMeasurement &) const = default;
};

struct FileCacheKey final {
  std::uint64_t filesystem_type{};
  std::uint64_t unique_mount_id{};
  std::uint64_t device{};
  std::uint64_t inode{};
  std::uint64_t mode{};
  std::uint64_t links{};
  std::uint64_t owner{};
  std::uint64_t group{};
  std::uint64_t size{};
  std::int64_t modified_seconds{};
  std::int64_t modified_nanoseconds{};
  std::int64_t changed_seconds{};
  std::int64_t changed_nanoseconds{};

  bool operator==(const FileCacheKey &) const = default;
  auto operator<=>(const FileCacheKey &) const = default;
};

void hash_combine(std::size_t &seed, std::uint64_t value) noexcept {
  seed ^= std::hash<std::uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL +
          (seed << 6U) + (seed >> 2U);
}

struct FileCacheKeyHash final {
  std::size_t operator()(const FileCacheKey &key) const noexcept {
    std::size_t result = 0U;
    hash_combine(result, key.filesystem_type);
    hash_combine(result, key.unique_mount_id);
    hash_combine(result, key.device);
    hash_combine(result, key.inode);
    hash_combine(result, key.mode);
    hash_combine(result, key.links);
    hash_combine(result, key.owner);
    hash_combine(result, key.group);
    hash_combine(result, key.size);
    hash_combine(result, static_cast<std::uint64_t>(key.modified_seconds));
    hash_combine(result, static_cast<std::uint64_t>(key.modified_nanoseconds));
    hash_combine(result, static_cast<std::uint64_t>(key.changed_seconds));
    hash_combine(result, static_cast<std::uint64_t>(key.changed_nanoseconds));
    return result;
  }
};

FileCacheKey file_cache_key(const struct stat &value,
                            const InputContentFilesystemIdentity &filesystem) {
  return {
      .filesystem_type = filesystem.filesystem_type,
      .unique_mount_id = filesystem.unique_mount_id,
      .device = static_cast<std::uint64_t>(value.st_dev),
      .inode = static_cast<std::uint64_t>(value.st_ino),
      .mode = static_cast<std::uint64_t>(value.st_mode),
      .links = static_cast<std::uint64_t>(value.st_nlink),
      .owner = static_cast<std::uint64_t>(value.st_uid),
      .group = static_cast<std::uint64_t>(value.st_gid),
      .size = static_cast<std::uint64_t>(value.st_size),
      .modified_seconds = static_cast<std::int64_t>(value.st_mtim.tv_sec),
      .modified_nanoseconds = static_cast<std::int64_t>(value.st_mtim.tv_nsec),
      .changed_seconds = static_cast<std::int64_t>(value.st_ctim.tv_sec),
      .changed_nanoseconds = static_cast<std::int64_t>(value.st_ctim.tv_nsec),
  };
}

std::optional<InputContentFilesystemIdentity>
local_filesystem_identity(int descriptor) noexcept {
  struct statfs filesystem{};
  if (::fstatfs(descriptor, &filesystem) != 0)
    return std::nullopt;
  switch (filesystem.f_type) {
  case EXT4_SUPER_MAGIC:
  case BTRFS_SUPER_MAGIC:
  case XFS_SUPER_MAGIC:
  case TMPFS_MAGIC:
  case OVERLAYFS_SUPER_MAGIC:
  case kZfsSuperMagic:
    break;
  default:
    return std::nullopt;
  }
#if defined(STATX_MNT_ID_UNIQUE) && defined(AT_STATX_DONT_SYNC)
  struct statx identity{};
  if (::statx(descriptor, "",
              AT_EMPTY_PATH | AT_NO_AUTOMOUNT | AT_STATX_DONT_SYNC,
              STATX_MNT_ID_UNIQUE, &identity) != 0 ||
      (identity.stx_mask & STATX_MNT_ID_UNIQUE) == 0U ||
      identity.stx_mnt_id == 0U) {
    return std::nullopt;
  }
  return InputContentFilesystemIdentity{
      .filesystem_type = static_cast<std::uint64_t>(filesystem.f_type),
      .unique_mount_id = identity.stx_mnt_id,
  };
#else
  // Older build headers cannot express the non-reusable mount identity. A
  // device/inode tuple alone is insufficient across remounts, so compile to a
  // safe cache bypass rather than weakening the key.
  return std::nullopt;
#endif
}

Digest cache_record_seal(const FileCacheKey &key,
                         const NodeMeasurement &value) {
  DigestContext digest;
  constexpr std::string_view domain = "trainvm.input-content-cache-record/v1";
  digest.update(domain.data(), domain.size());
  const auto add = [&digest](const auto &field) {
    digest.update(&field, sizeof(field));
  };
  add(key.filesystem_type);
  add(key.unique_mount_id);
  add(key.device);
  add(key.inode);
  add(key.mode);
  add(key.links);
  add(key.owner);
  add(key.group);
  add(key.size);
  add(key.modified_seconds);
  add(key.modified_nanoseconds);
  add(key.changed_seconds);
  add(key.changed_nanoseconds);
  const auto kind = static_cast<std::uint64_t>(value.kind);
  add(kind);
  add(value.file_count);
  add(value.total_bytes);
  digest.update(value.digest);
  return digest.finish();
}

bool valid_cached_record(const FileCacheKey &key, const NodeMeasurement &value,
                         const Digest &seal) {
  return value.kind == ContentRootKind::file && value.file_count == 1U &&
         value.total_bytes == key.size && cache_record_seal(key, value) == seal;
}

} // namespace

struct InputContentMeasurementCache::Impl final {
  struct Entry final {
    NodeMeasurement value;
    Digest seal{};
    std::list<FileCacheKey>::iterator recency;
  };

  explicit Impl(std::uint64_t maximum, FilesystemIdentitySource identity_source,
                IntegrityFaultForTesting integrity_fault,
                PublicationFaultForTesting publication_fault)
      : maximum_entries(maximum),
        filesystem_identity(std::move(identity_source)),
        integrity_fault_for_testing(std::move(integrity_fault)),
        publication_fault_for_testing(std::move(publication_fault)) {}

  std::mutex mutex;
  std::uint64_t maximum_entries{};
  FilesystemIdentitySource filesystem_identity;
  IntegrityFaultForTesting integrity_fault_for_testing;
  PublicationFaultForTesting publication_fault_for_testing;
  std::list<FileCacheKey> recency;
  std::unordered_map<FileCacheKey, Entry, FileCacheKeyHash> entries;
};

struct InputContentMeasurementTransaction::Impl final {
  struct StagedEntry final {
    NodeMeasurement value;
    Digest seal{};
  };

  explicit Impl(InputContentMeasurementCache::Impl *owner_value)
      : owner(owner_value) {}

  [[nodiscard]] std::optional<InputContentFilesystemIdentity>
  filesystem(int descriptor) const {
    return owner->filesystem_identity(descriptor);
  }

  [[nodiscard]] std::optional<NodeMeasurement> find(const FileCacheKey &key) {
    if (const auto staged_entry = staged.find(key);
        staged_entry != staged.end()) {
      return staged_entry->second.value;
    }
    std::scoped_lock lock(owner->mutex);
    const auto found = owner->entries.find(key);
    if (found == owner->entries.end())
      return std::nullopt;
    const bool injected = owner->integrity_fault_for_testing &&
                          owner->integrity_fault_for_testing();
    if (injected ||
        !valid_cached_record(key, found->second.value, found->second.seal)) {
      ++corruptions;
      return std::nullopt;
    }
    if (touched.insert(key).second)
      touch_order.push_back(key);
    return found->second.value;
  }

  bool stage(const FileCacheKey &key, const NodeMeasurement &value) {
    const StagedEntry replacement{
        .value = value, .seal = cache_record_seal(key, value)};
    if (const auto existing = staged.find(key); existing != staged.end()) {
      existing->second = replacement;
      return true;
    }
    if (staged.size() >= owner->maximum_entries) {
      ++staging_saturations;
      return false;
    }
    staged.emplace(key, replacement);
    return true;
  }

  InputContentMeasurementCache::Impl *owner;
  std::unordered_map<FileCacheKey, StagedEntry, FileCacheKeyHash> staged;
  std::unordered_set<FileCacheKey, FileCacheKeyHash> touched;
  std::vector<FileCacheKey> touch_order;
  std::uint64_t corruptions{};
  std::uint64_t staging_saturations{};
  bool committed{};
};

namespace {

struct Child final {
  std::string name;
  NodeMeasurement measurement;
};

struct RetainedLink final {
  Descriptor parent;
  std::string name;
  struct stat expected{};
};

struct OpenedRoot final {
  Descriptor descriptor;
  std::vector<RetainedLink> links;
};

bool valid_utf8(std::string_view value) {
  std::size_t offset = 0U;
  while (offset < value.size()) {
    const unsigned char lead = static_cast<unsigned char>(value[offset]);
    std::size_t continuation = 0U;
    std::uint32_t codepoint = 0U;
    if (lead <= 0x7fU) {
      continuation = 0U;
      codepoint = lead;
    } else if (lead >= 0xc2U && lead <= 0xdfU) {
      continuation = 1U;
      codepoint = lead & 0x1fU;
    } else if (lead >= 0xe0U && lead <= 0xefU) {
      continuation = 2U;
      codepoint = lead & 0x0fU;
    } else if (lead >= 0xf0U && lead <= 0xf4U) {
      continuation = 3U;
      codepoint = lead & 0x07U;
    } else {
      return false;
    }
    if (offset + continuation >= value.size())
      return false;
    for (std::size_t index = 1U; index <= continuation; ++index) {
      const unsigned char byte =
          static_cast<unsigned char>(value[offset + index]);
      if ((byte & 0xc0U) != 0x80U)
        return false;
      codepoint = (codepoint << 6U) | (byte & 0x3fU);
    }
    if ((continuation == 2U && codepoint < 0x800U) ||
        (continuation == 3U && codepoint < 0x10000U) ||
        (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
        codepoint > 0x10ffffU) {
      return false;
    }
    offset += continuation + 1U;
  }
  return true;
}

// Identity of a path component above the measured root is its device/inode
// pair, plus the ownership and mode that policy depends on. A directory's
// st_nlink, st_size, and timestamps all change whenever an unrelated entry is
// created or removed inside it, so comparing those on an ancestor reports
// benign concurrent activity in a shared parent such as /tmp as a substitution
// and fails the measurement. Content stability *inside* the root is a different
// question and is still checked with the full same_stat comparison below.
bool same_namespace_component(const struct stat &before,
                              const struct stat &after) {
  return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_mode == after.st_mode && before.st_uid == after.st_uid &&
         before.st_gid == after.st_gid;
}

bool same_stat(const struct stat &before, const struct stat &after) {
  return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_mode == after.st_mode && before.st_nlink == after.st_nlink &&
         before.st_uid == after.st_uid && before.st_gid == after.st_gid &&
         before.st_size == after.st_size &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

OpenedRoot open_root_by_components(const std::filesystem::path &path) {
  const int slash =
      ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (slash < 0)
    fail_system("could not open filesystem root");
  Descriptor current(slash);
  std::vector<std::string> components;
  for (const std::filesystem::path &component : path.relative_path()) {
    const std::string name = component.native();
    if (name.empty() || name == "." || name == ".." ||
        name.size() > kMaximumNameBytes || !valid_utf8(name)) {
      throw std::invalid_argument(
          "input content root has an invalid UTF-8 path component");
    }
    components.push_back(name);
  }
  if (components.empty())
    return {.descriptor = std::move(current), .links = {}};

  std::vector<RetainedLink> links;
  links.reserve(components.size());
  for (std::size_t index = 0U; index < components.size(); ++index) {
    struct stat before{};
    if (::fstatat(current.get(), components[index].c_str(), &before,
                  AT_SYMLINK_NOFOLLOW) != 0)
      fail_system("could not inspect input content root component");
    if (S_ISLNK(before.st_mode))
      throw std::runtime_error("input content root has a symlinked ancestor");
    const bool final = index + 1U == components.size();
    if (!S_ISDIR(before.st_mode) && !(final && S_ISREG(before.st_mode)))
      throw std::runtime_error(
          final ? "input content root is a special node"
                : "input content root ancestor is not a directory");
    const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK |
                      (S_ISDIR(before.st_mode) ? O_DIRECTORY : 0);
    const int next = ::openat(current.get(), components[index].c_str(), flags);
    if (next < 0) {
      if (errno == ELOOP)
        throw std::runtime_error("input content root has a symlinked ancestor");
      fail_system("could not open input content root component");
    }
    struct stat opened{};
    if (::fstat(next, &opened) != 0) {
      const int saved_errno = errno;
      (void)::close(next);
      errno = saved_errno;
      fail_system("could not inspect opened input content root component");
    }
    if (!same_namespace_component(before, opened)) {
      (void)::close(next);
      throw std::runtime_error(
          "input content root component changed while opening");
    }
    links.push_back({.parent = std::move(current),
                     .name = components[index],
                     .expected = opened});
    current = Descriptor(next);
  }
  return {.descriptor = std::move(current), .links = std::move(links)};
}

void require_unchanged_links(const std::vector<RetainedLink> &links) {
  for (auto iterator = links.rbegin(); iterator != links.rend(); ++iterator) {
    struct stat observed{};
    if (::fstatat(iterator->parent.get(), iterator->name.c_str(), &observed,
                  AT_SYMLINK_NOFOLLOW) != 0)
      fail_system("could not reinspect input content root component");
    if (!same_namespace_component(iterator->expected, observed))
      throw std::runtime_error(
          "input content root namespace changed while hashing");
  }
}

std::array<unsigned char, 8U> big_endian_64(std::uint64_t value) {
  std::array<unsigned char, 8U> encoded{};
  for (std::size_t index = 0U; index < encoded.size(); ++index) {
    encoded[encoded.size() - 1U - index] =
        static_cast<unsigned char>(value & 0xffU);
    value >>= 8U;
  }
  return encoded;
}

std::array<unsigned char, 4U> big_endian_32(std::uint32_t value) {
  std::array<unsigned char, 4U> encoded{};
  for (std::size_t index = 0U; index < encoded.size(); ++index) {
    encoded[encoded.size() - 1U - index] =
        static_cast<unsigned char>(value & 0xffU);
    value >>= 8U;
  }
  return encoded;
}

void add_totals(NodeMeasurement &parent, const NodeMeasurement &child) {
  if (child.file_count > kMaximumFiles - parent.file_count)
    throw std::runtime_error("input content root exceeds the file-count bound");
  if (child.total_bytes > kMaximumBytes - parent.total_bytes)
    throw std::runtime_error("input content root exceeds the byte bound");
  parent.file_count += child.file_count;
  parent.total_bytes += child.total_bytes;
}

NodeMeasurement
measure_node(Descriptor descriptor, std::size_t depth,
             std::span<unsigned char> read_buffer,
             InputContentMeasurementStats *stats,
             InputContentMeasurementTransaction::Impl *transaction) {
  if (depth > kMaximumDepth)
    throw std::runtime_error("input content root exceeds the depth bound");
  struct stat before{};
  if (::fstat(descriptor.get(), &before) != 0)
    fail_system("could not inspect input content node");

  if (S_ISREG(before.st_mode)) {
    if (before.st_size < 0)
      throw std::runtime_error("input content file has a negative size");
    const auto size = static_cast<std::uint64_t>(before.st_size);
    if (size > kMaximumBytes)
      throw std::runtime_error("input content root exceeds the byte bound");
    const auto filesystem = transaction != nullptr
                                ? transaction->filesystem(descriptor.get())
                                : std::nullopt;
    std::optional<FileCacheKey> cache_key;
    if (filesystem) {
      cache_key = file_cache_key(before, *filesystem);
      if (const auto cached = transaction->find(*cache_key)) {
        struct stat after{};
        if (::fstat(descriptor.get(), &after) != 0)
          fail_system("could not reinspect cached input content file");
        if (!same_stat(before, after))
          throw std::runtime_error(
              "input content file changed while reusing its measurement");
        if (stats != nullptr)
          ++stats->cache_hits;
        return *cached;
      }
      if (stats != nullptr)
        ++stats->cache_misses;
    } else if (stats != nullptr) {
      ++stats->cache_bypasses;
    }
    DigestContext content;
    std::uint64_t observed = 0U;
    for (;;) {
      const ssize_t count =
          ::read(descriptor.get(), read_buffer.data(), read_buffer.size());
      if (count < 0) {
        if (errno == EINTR)
          continue;
        fail_system("could not read input content file");
      }
      if (count == 0)
        break;
      const auto bytes = static_cast<std::size_t>(count);
      if (static_cast<std::uint64_t>(bytes) > size - observed)
        throw std::runtime_error("input content file changed while hashing");
      content.update(read_buffer.data(), bytes);
      observed += static_cast<std::uint64_t>(bytes);
      if (stats != nullptr)
        stats->bytes_hashed += bytes;
    }
    struct stat after{};
    if (::fstat(descriptor.get(), &after) != 0)
      fail_system("could not reinspect input content file");
    if (!same_stat(before, after) || observed != size)
      throw std::runtime_error("input content file changed while hashing");
    const Digest content_digest = content.finish();
    DigestContext node;
    node.update(kFileDomain, sizeof(kFileDomain));
    node.update(big_endian_64(size));
    node.update(content_digest);
    const NodeMeasurement measured{.kind = ContentRootKind::file,
                                   .file_count = 1U,
                                   .total_bytes = size,
                                   .digest = node.finish()};
    if (cache_key && !transaction->stage(*cache_key, measured) &&
        stats != nullptr)
      ++stats->staging_saturations;
    return measured;
  }

  if (!S_ISDIR(before.st_mode))
    throw std::runtime_error(
        "input content root contains a symlink or special node");

  const int duplicate = ::fcntl(descriptor.get(), F_DUPFD_CLOEXEC, 0);
  if (duplicate < 0)
    fail_system("could not duplicate input directory");
  DIR *raw_directory = ::fdopendir(duplicate);
  if (raw_directory == nullptr) {
    const int saved_errno = errno;
    (void)::close(duplicate);
    errno = saved_errno;
    fail_system("could not enumerate input directory");
  }
  struct CloseDirectory final {
    void operator()(DIR *directory) const noexcept {
      if (directory != nullptr)
        (void)::closedir(directory);
    }
  };
  std::unique_ptr<DIR, CloseDirectory> directory(raw_directory);
  std::vector<std::string> names;
  errno = 0;
  while (dirent *entry = ::readdir(directory.get())) {
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..")
      continue;
    if (name.empty() || name.size() > kMaximumNameBytes || !valid_utf8(name))
      throw std::runtime_error(
          "input directory contains an invalid UTF-8 name");
    names.emplace_back(name);
    errno = 0;
  }
  if (errno != 0)
    fail_system("could not enumerate input directory");
  std::ranges::sort(names,
                    [](const std::string &left, const std::string &right) {
                      return std::lexicographical_compare(
                          left.begin(), left.end(), right.begin(), right.end(),
                          [](char left_byte, char right_byte) {
                            return static_cast<unsigned char>(left_byte) <
                                   static_cast<unsigned char>(right_byte);
                          });
                    });

  std::vector<Child> children;
  children.reserve(names.size());
  NodeMeasurement result{.kind = ContentRootKind::directory};
  for (const std::string &name : names) {
    struct stat child_before{};
    if (::fstatat(descriptor.get(), name.c_str(), &child_before,
                  AT_SYMLINK_NOFOLLOW) != 0)
      fail_system("could not inspect input content child");
    if (S_ISLNK(child_before.st_mode))
      throw std::runtime_error("input content root contains a symlink");
    if (!S_ISREG(child_before.st_mode) && !S_ISDIR(child_before.st_mode))
      throw std::runtime_error("input content root contains a special node");
    const int child_flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK |
                            (S_ISDIR(child_before.st_mode) ? O_DIRECTORY : 0);
    const int child_fd = ::openat(descriptor.get(), name.c_str(), child_flags);
    if (child_fd < 0) {
      if (errno == ELOOP)
        throw std::runtime_error("input content root contains a symlink");
      fail_system("could not open input content child");
    }
    struct stat child_opened{};
    if (::fstat(child_fd, &child_opened) != 0) {
      const int saved_errno = errno;
      (void)::close(child_fd);
      errno = saved_errno;
      fail_system("could not inspect opened input content child");
    }
    if (!same_stat(child_before, child_opened)) {
      (void)::close(child_fd);
      throw std::runtime_error("input content child changed while opening");
    }
    NodeMeasurement child = measure_node(Descriptor(child_fd), depth + 1U,
                                         read_buffer, stats, transaction);
    struct stat child_after{};
    if (::fstatat(descriptor.get(), name.c_str(), &child_after,
                  AT_SYMLINK_NOFOLLOW) != 0)
      fail_system("could not reinspect input content child");
    if (!same_stat(child_opened, child_after))
      throw std::runtime_error("input content child was substituted");
    add_totals(result, child);
    children.push_back({.name = name, .measurement = std::move(child)});
  }

  struct stat after{};
  if (::fstat(descriptor.get(), &after) != 0)
    fail_system("could not reinspect input directory");
  if (!same_stat(before, after))
    throw std::runtime_error("input directory changed while hashing");

  DigestContext node;
  node.update(kDirectoryDomain, sizeof(kDirectoryDomain));
  for (const Child &child : children) {
    if (child.name.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("input content child name exceeds its encoding");
    node.update(big_endian_32(static_cast<std::uint32_t>(child.name.size())));
    node.update(child.name.data(), child.name.size());
    const unsigned char kind = child.measurement.kind == ContentRootKind::file
                                   ? static_cast<unsigned char>('f')
                                   : static_cast<unsigned char>('d');
    node.update(&kind, 1U);
    node.update(child.measurement.digest);
  }
  result.digest = node.finish();
  return result;
}

std::string digest_hex(const Digest &digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2U);
  for (const unsigned char byte : digest) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

bool path_within(const std::filesystem::path &child,
                 const std::filesystem::path &parent) {
  auto child_component = child.begin();
  for (auto parent_component = parent.begin(); parent_component != parent.end();
       ++parent_component, ++child_component) {
    if (child_component == child.end() || *child_component != *parent_component)
      return false;
  }
  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Owner-only persistent digest store
// ---------------------------------------------------------------------------
//
// The cache above answers "have I already hashed this exact file?" for one
// process. `trainvm lock-input-content` is not one process -- it is a fresh
// process per lock -- so it never answered it at all, and re-read every byte of
// every root each time. This store carries the same answers between those
// processes.
//
// Nothing here weakens the question. A record is admitted only under the key
// the in-memory cache already uses (mount incarnation, device, inode, mode,
// link count, ownership, size, mtime and ctime to the nanosecond) and only
// after its seal is recomputed, so a record cannot be moved to another key or
// edited in place without detection. Changed bytes move ctime, which is part of
// the key, so a changed file misses. The one case a metadata key cannot see is
// a write that lands inside the same timestamp tick the record was sealed
// against; `kTimestampSettlingNanoseconds` below refuses to persist or admit a
// record that recent, which is the same defence git's index applies to its
// racily-clean entries.
//
// The store is trusted for the reason the runtime-closure evidence directory is
// trusted: it is owned by this user, on a directory no one else can write, with
// exactly one link, opened without following symlinks. It is not authenticated
// against a privileged writer and does not claim to be.

namespace {

constexpr std::string_view kDigestStoreMagic =
    "trainvm.input-content-digest-store/v1\n";
constexpr std::size_t kDigestStorePolicyBytes = 71U;
constexpr std::size_t kDigestStoreMaximumBootIdBytes = 256U;
// Thirteen key fields plus kind, file count and byte count, each a big-endian
// 64-bit word, then the measured digest and its seal. Derived rather than
// written out so a field added to the record cannot leave the bound behind.
constexpr std::size_t kDigestStoreEntryBytes = 16U * 8U + 2U * kDigestBytes;
constexpr std::uint64_t kDigestStoreMaximumBytes = 64U * 1024U * 1024U;
constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000;
// Filesystem timestamp granularity is not observable, and a coarse one lets a
// second write share the tick of the write we measured. One second is far
// beyond any granularity Linux uses, so a record older than this cannot be
// overwritten without moving ctime. Files touched within the last second are
// simply hashed again next time and cached then.
constexpr std::int64_t kTimestampSettlingNanoseconds = kNanosecondsPerSecond;
// Rejects absurd or hostile timestamps before they are multiplied out.
constexpr std::int64_t kMaximumTimestampSeconds = 100'000'000'000;

[[nodiscard]] std::optional<std::int64_t> flat_nanoseconds(std::int64_t seconds,
                                                           std::int64_t nanos) {
  if (seconds < 0 || seconds > kMaximumTimestampSeconds || nanos < 0 ||
      nanos >= kNanosecondsPerSecond)
    return std::nullopt;
  return seconds * kNanosecondsPerSecond + nanos;
}

// True when both of a key's timestamps are old enough that any later write must
// have moved one of them.
[[nodiscard]] bool timestamps_have_settled(const FileCacheKey &key,
                                           std::int64_t sealed_at) {
  const auto modified =
      flat_nanoseconds(key.modified_seconds, key.modified_nanoseconds);
  const auto changed =
      flat_nanoseconds(key.changed_seconds, key.changed_nanoseconds);
  if (!modified || !changed || sealed_at <= 0)
    return false;
  const std::int64_t newest = std::max(*modified, *changed);
  return newest <= sealed_at - kTimestampSettlingNanoseconds;
}

[[nodiscard]] std::int64_t realtime_nanoseconds() {
  struct timespec now{};
  if (::clock_gettime(CLOCK_REALTIME, &now) != 0)
    fail_system("could not read the wall clock");
  const auto flat = flat_nanoseconds(static_cast<std::int64_t>(now.tv_sec),
                                     static_cast<std::int64_t>(now.tv_nsec));
  if (!flat)
    throw std::runtime_error("the wall clock reported an unusable time");
  return *flat;
}

// The mount incarnation in a cache key is unique only for the life of one boot,
// so a store written before this boot describes mounts that no longer mean what
// its records say. Binding the store to the boot identity is what makes the
// mount identity in the key sound across processes.
[[nodiscard]] std::optional<std::string> current_boot_identity() {
  const int raw = ::open("/proc/sys/kernel/random/boot_id",
                         O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (raw < 0)
    return std::nullopt;
  Descriptor descriptor(raw);
  std::string value;
  std::array<char, 128U> buffer{};
  for (;;) {
    const ssize_t count =
        ::read(descriptor.get(), buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return std::nullopt;
    }
    if (count == 0)
      break;
    value.append(buffer.data(), static_cast<std::size_t>(count));
    if (value.size() > kDigestStoreMaximumBootIdBytes)
      return std::nullopt;
  }
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r'))
    value.pop_back();
  if (value.empty() || value.size() > kDigestStoreMaximumBootIdBytes)
    return std::nullopt;
  return value;
}

// A directory nobody but this user can write, on the device it claims to be on.
[[nodiscard]] Descriptor open_store_directory(const std::filesystem::path &directory) {
  const int raw = ::open(directory.c_str(),
                         O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (raw < 0)
    fail_system("could not open the input content digest store directory");
  Descriptor descriptor(raw);
  struct stat metadata{};
  if (::fstat(descriptor.get(), &metadata) != 0)
    fail_system("could not inspect the input content digest store directory");
  if (!S_ISDIR(metadata.st_mode) || metadata.st_uid != ::geteuid() ||
      (metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0)
    throw std::runtime_error(
        "input content digest store directory is not owner-only");
  return descriptor;
}

void require_store_path(const std::filesystem::path &store_path) {
  if (!store_path.is_absolute() || store_path.empty() ||
      store_path.lexically_normal() != store_path ||
      store_path.native().size() > kMaximumNameBytes ||
      !store_path.has_filename() || !store_path.has_parent_path())
    throw std::invalid_argument(
        "input content digest store path must be an absolute normalized file");
  const std::string name = store_path.filename().native();
  if (name == "." || name == ".." || name.size() > kMaximumNameBytes ||
      !valid_utf8(name))
    throw std::invalid_argument(
        "input content digest store name is not bounded UTF-8");
}

class ByteWriter final {
public:
  void raw(const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + size);
  }
  void text(std::string_view value) { raw(value.data(), value.size()); }
  void unsigned64(std::uint64_t value) {
    const auto encoded = big_endian_64(value);
    raw(encoded.data(), encoded.size());
  }
  void signed64(std::int64_t value) {
    unsigned64(static_cast<std::uint64_t>(value));
  }
  void unsigned32(std::uint32_t value) {
    const auto encoded = big_endian_32(value);
    raw(encoded.data(), encoded.size());
  }
  void digest(const Digest &value) { raw(value.data(), value.size()); }
  [[nodiscard]] const std::vector<unsigned char> &bytes() const noexcept {
    return buffer_;
  }

private:
  std::vector<unsigned char> buffer_;
};

// Every read is bounds-checked against the declared length; a truncated or
// oversized store therefore fails as a refusal rather than reading past its
// buffer.
class ByteReader final {
public:
  ByteReader(const unsigned char *data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  [[nodiscard]] bool take(void *destination, std::size_t size) noexcept {
    if (size > size_ - offset_)
      return false;
    std::memcpy(destination, data_ + offset_, size);
    offset_ += size;
    return true;
  }
  [[nodiscard]] bool matches(std::string_view value) noexcept {
    if (value.size() > size_ - offset_)
      return false;
    const bool equal =
        std::memcmp(data_ + offset_, value.data(), value.size()) == 0;
    offset_ += value.size();
    return equal;
  }
  [[nodiscard]] bool unsigned64(std::uint64_t &value) noexcept {
    std::array<unsigned char, 8U> encoded{};
    if (!take(encoded.data(), encoded.size()))
      return false;
    value = 0U;
    for (const unsigned char byte : encoded)
      value = (value << 8U) | byte;
    return true;
  }
  [[nodiscard]] bool signed64(std::int64_t &value) noexcept {
    std::uint64_t raw = 0U;
    if (!unsigned64(raw))
      return false;
    value = static_cast<std::int64_t>(raw);
    return true;
  }
  [[nodiscard]] bool unsigned32(std::uint32_t &value) noexcept {
    std::array<unsigned char, 4U> encoded{};
    if (!take(encoded.data(), encoded.size()))
      return false;
    value = 0U;
    for (const unsigned char byte : encoded)
      value = (value << 8U) | byte;
    return true;
  }
  [[nodiscard]] bool digest(Digest &value) noexcept {
    return take(value.data(), value.size());
  }
  [[nodiscard]] std::size_t remaining() const noexcept {
    return size_ - offset_;
  }

private:
  const unsigned char *data_;
  std::size_t size_;
  std::size_t offset_{};
};

void encode_store_entry(ByteWriter &writer, const FileCacheKey &key,
                        const NodeMeasurement &value, const Digest &seal) {
  writer.unsigned64(key.filesystem_type);
  writer.unsigned64(key.unique_mount_id);
  writer.unsigned64(key.device);
  writer.unsigned64(key.inode);
  writer.unsigned64(key.mode);
  writer.unsigned64(key.links);
  writer.unsigned64(key.owner);
  writer.unsigned64(key.group);
  writer.unsigned64(key.size);
  writer.signed64(key.modified_seconds);
  writer.signed64(key.modified_nanoseconds);
  writer.signed64(key.changed_seconds);
  writer.signed64(key.changed_nanoseconds);
  writer.unsigned64(static_cast<std::uint64_t>(value.kind));
  writer.unsigned64(value.file_count);
  writer.unsigned64(value.total_bytes);
  writer.digest(value.digest);
  writer.digest(seal);
}

[[nodiscard]] bool decode_store_entry(ByteReader &reader, FileCacheKey &key,
                                      NodeMeasurement &value, Digest &seal) {
  std::uint64_t kind = 0U;
  if (!reader.unsigned64(key.filesystem_type) ||
      !reader.unsigned64(key.unique_mount_id) || !reader.unsigned64(key.device) ||
      !reader.unsigned64(key.inode) || !reader.unsigned64(key.mode) ||
      !reader.unsigned64(key.links) || !reader.unsigned64(key.owner) ||
      !reader.unsigned64(key.group) || !reader.unsigned64(key.size) ||
      !reader.signed64(key.modified_seconds) ||
      !reader.signed64(key.modified_nanoseconds) ||
      !reader.signed64(key.changed_seconds) ||
      !reader.signed64(key.changed_nanoseconds) || !reader.unsigned64(kind) ||
      !reader.unsigned64(value.file_count) ||
      !reader.unsigned64(value.total_bytes) || !reader.digest(value.digest) ||
      !reader.digest(seal))
    return false;
  if (kind != static_cast<std::uint64_t>(ContentRootKind::file))
    return false;
  value.kind = ContentRootKind::file;
  return true;
}

Digest digest_of(std::span<const unsigned char> bytes) {
  DigestContext context;
  context.update(bytes.data(), bytes.size());
  return context.finish();
}

// Reads the whole store under the safety rules above. An absent or unsafe-shaped
// file yields nullopt; unsafe *ownership* throws, because that one is an
// operator mistake rather than a cold cache.
[[nodiscard]] std::optional<std::vector<unsigned char>>
read_store_file(int directory, const std::string &name) {
  const int raw =
      ::openat(directory, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (raw < 0) {
    if (errno == ENOENT || errno == ELOOP)
      return std::nullopt;
    fail_system("could not open the input content digest store");
  }
  Descriptor descriptor(raw);
  struct stat before{};
  if (::fstat(descriptor.get(), &before) != 0)
    fail_system("could not inspect the input content digest store");
  if (!S_ISREG(before.st_mode))
    throw std::runtime_error(
        "input content digest store is not a regular file");
  if (before.st_uid != ::geteuid() ||
      (before.st_mode & (S_IWGRP | S_IWOTH | S_IRGRP | S_IROTH)) != 0 ||
      before.st_nlink != 1)
    throw std::runtime_error(
        "input content digest store is not owner-only with a single link");
  if (before.st_size < 0 ||
      static_cast<std::uint64_t>(before.st_size) > kDigestStoreMaximumBytes)
    return std::nullopt;
  std::vector<unsigned char> bytes(static_cast<std::size_t>(before.st_size));
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    const ssize_t count =
        ::read(descriptor.get(), bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      fail_system("could not read the input content digest store");
    }
    if (count == 0)
      return std::nullopt;
    offset += static_cast<std::size_t>(count);
  }
  struct stat after{};
  if (::fstat(descriptor.get(), &after) != 0)
    fail_system("could not reinspect the input content digest store");
  if (!same_stat(before, after))
    return std::nullopt;
  return bytes;
}

void write_store_file(int directory, const std::string &name,
                      const std::vector<unsigned char> &bytes) {
  const std::string temporary = name + ".staged";
  (void)::unlinkat(directory, temporary.c_str(), 0);
  const int raw = ::openat(directory, temporary.c_str(),
                           O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                           0600);
  if (raw < 0)
    fail_system("could not stage the input content digest store");
  try {
    Descriptor descriptor(raw);
    std::size_t offset = 0U;
    while (offset < bytes.size()) {
      const ssize_t count = ::write(descriptor.get(), bytes.data() + offset,
                                    bytes.size() - offset);
      if (count < 0) {
        if (errno == EINTR)
          continue;
        fail_system("could not write the input content digest store");
      }
      offset += static_cast<std::size_t>(count);
    }
    if (::fchmod(descriptor.get(), 0600) != 0 || ::fsync(descriptor.get()) != 0)
      throw std::runtime_error(
          "could not seal the staged input content digest store");
    if (::renameat(directory, temporary.c_str(), directory, name.c_str()) != 0)
      fail_system("could not publish the input content digest store");
    if (::fsync(directory) != 0)
      throw std::runtime_error(
          "input content digest store publication was not durable");
  } catch (...) {
    (void)::unlinkat(directory, temporary.c_str(), 0);
    throw;
  }
}

} // namespace

InputContentDigestStoreStats
InputContentMeasurementCache::admit_persistent_digests(
    const std::filesystem::path &store_path) {
  require_store_path(store_path);
  InputContentDigestStoreStats result;
  const Descriptor directory = open_store_directory(store_path.parent_path());
  const auto bytes = read_store_file(directory.get(), store_path.filename().native());
  if (!bytes)
    return result;
  result.present = true;
  if (bytes->size() < kDigestBytes)
    return result;
  const std::size_t body = bytes->size() - kDigestBytes;
  Digest trailer{};
  std::memcpy(trailer.data(), bytes->data() + body, kDigestBytes);
  if (digest_of(std::span<const unsigned char>(bytes->data(), body)) != trailer)
    return result;

  ByteReader reader(bytes->data(), body);
  const std::string policy = policy_digest();
  std::array<char, kDigestStorePolicyBytes> stored_policy{};
  std::uint32_t boot_length = 0U;
  std::int64_t sealed_at = 0;
  std::uint64_t declared = 0U;
  if (!reader.matches(kDigestStoreMagic) ||
      !reader.take(stored_policy.data(), stored_policy.size()) ||
      policy.size() != stored_policy.size() ||
      std::memcmp(policy.data(), stored_policy.data(), stored_policy.size()) !=
          0 ||
      !reader.unsigned32(boot_length) ||
      boot_length > kDigestStoreMaximumBootIdBytes || boot_length == 0U)
    return result;
  std::string stored_boot(boot_length, '\0');
  const auto boot = current_boot_identity();
  if (!reader.take(stored_boot.data(), stored_boot.size()) || !boot ||
      *boot != stored_boot || !reader.signed64(sealed_at) ||
      !reader.unsigned64(declared) || declared > impl_->maximum_entries ||
      declared > reader.remaining() / kDigestStoreEntryBytes ||
      declared * kDigestStoreEntryBytes != reader.remaining())
    return result;
  result.accepted = true;
  result.offered_entries = declared;

  // Records were written most-recently-used first. Insert in reverse so the
  // published order reproduces the recency the writing process observed.
  std::vector<std::pair<FileCacheKey, NodeMeasurement>> admitted;
  admitted.reserve(static_cast<std::size_t>(declared));
  for (std::uint64_t index = 0U; index < declared; ++index) {
    FileCacheKey key{};
    NodeMeasurement value{};
    Digest seal{};
    if (!decode_store_entry(reader, key, value, seal)) {
      result.refused_entries = declared - index;
      break;
    }
    if (!valid_cached_record(key, value, seal) ||
        !timestamps_have_settled(key, sealed_at)) {
      ++result.refused_entries;
      continue;
    }
    admitted.emplace_back(key, value);
  }

  std::scoped_lock lock(impl_->mutex);
  for (auto entry = admitted.rbegin(); entry != admitted.rend(); ++entry) {
    if (impl_->entries.contains(entry->first))
      continue;
    if (impl_->entries.size() >= impl_->maximum_entries) {
      const auto evicted = impl_->entries.find(impl_->recency.back());
      if (evicted == impl_->entries.end())
        throw std::logic_error("input content cache LRU index lost an entry");
      impl_->recency.erase(evicted->second.recency);
      impl_->entries.erase(evicted);
    }
    impl_->recency.push_front(entry->first);
    impl_->entries.emplace(
        entry->first,
        Impl::Entry{.value = entry->second,
                    .seal = cache_record_seal(entry->first, entry->second),
                    .recency = impl_->recency.begin()});
    ++result.admitted_entries;
  }
  return result;
}

InputContentDigestStoreStats
InputContentMeasurementCache::publish_persistent_digests(
    const std::filesystem::path &store_path) const {
  require_store_path(store_path);
  InputContentDigestStoreStats result;
  const auto boot = current_boot_identity();
  if (!boot)
    return result;
  const Descriptor directory = open_store_directory(store_path.parent_path());
  // Sealed after every measurement in this process finished, so a record that
  // clears the settling window here cannot have been rewritten inside the tick
  // its own timestamps name.
  const std::int64_t sealed_at = realtime_nanoseconds();
  const std::string policy = policy_digest();
  if (policy.size() != kDigestStorePolicyBytes)
    throw std::logic_error("input content cache policy digest changed shape");

  ByteWriter writer;
  writer.text(kDigestStoreMagic);
  writer.text(policy);
  writer.unsigned32(static_cast<std::uint32_t>(boot->size()));
  writer.text(*boot);
  writer.signed64(sealed_at);

  std::scoped_lock lock(impl_->mutex);
  std::vector<const FileCacheKey *> publishable;
  publishable.reserve(impl_->recency.size());
  for (const FileCacheKey &key : impl_->recency) {
    const auto entry = impl_->entries.find(key);
    if (entry == impl_->entries.end() ||
        !valid_cached_record(key, entry->second.value, entry->second.seal)) {
      ++result.refused_entries;
      continue;
    }
    if (!timestamps_have_settled(key, sealed_at)) {
      ++result.withheld_entries;
      continue;
    }
    publishable.push_back(&key);
  }
  writer.unsigned64(static_cast<std::uint64_t>(publishable.size()));
  const std::size_t header_bytes = writer.bytes().size();
  for (const FileCacheKey *key : publishable) {
    const auto &entry = impl_->entries.at(*key);
    encode_store_entry(writer, *key, entry.value, entry.seal);
  }
  std::vector<unsigned char> bytes = writer.bytes();
  // The reader locates records by multiplying this width out, so a record that
  // stopped matching it has to fail here rather than at the next lock.
  if (bytes.size() - header_bytes !=
      publishable.size() * kDigestStoreEntryBytes)
    throw std::logic_error(
        "input content digest store record width no longer matches its bound");
  const Digest trailer = digest_of(bytes);
  bytes.insert(bytes.end(), trailer.begin(), trailer.end());
  if (static_cast<std::uint64_t>(bytes.size()) > kDigestStoreMaximumBytes)
    throw std::runtime_error(
        "input content digest store exceeds its publication bound");
  write_store_file(directory.get(), store_path.filename().native(), bytes);
  result.present = true;
  result.accepted = true;
  result.offered_entries = static_cast<std::uint64_t>(publishable.size());
  result.admitted_entries = result.offered_entries;
  return result;
}

InputContentMeasurementCache::InputContentMeasurementCache(
    std::uint64_t maximum_entries, FilesystemIdentitySource filesystem_identity,
    IntegrityFaultForTesting integrity_fault_for_testing,
    PublicationFaultForTesting publication_fault_for_testing) {
  if (maximum_entries == 0U || maximum_entries > 10'000'000U)
    throw std::invalid_argument(
        "input content cache capacity must be between 1 and 10000000");
  if (!filesystem_identity)
    filesystem_identity = local_filesystem_identity;
  impl_ =
      std::make_unique<Impl>(maximum_entries, std::move(filesystem_identity),
                             std::move(integrity_fault_for_testing),
                             std::move(publication_fault_for_testing));
}

InputContentMeasurementCache::~InputContentMeasurementCache() = default;

InputContentMeasurementTransaction
InputContentMeasurementCache::begin_transaction() {
  return InputContentMeasurementTransaction(
      std::make_unique<InputContentMeasurementTransaction::Impl>(impl_.get()));
}

std::string InputContentMeasurementCache::policy_digest() const {
  DigestContext digest;
  constexpr std::string_view policy =
      "trainvm.input-content-measurement-cache-policy/v2;"
      "key=filesystem-type,unique-mount-id,device,inode,mode,nlink,uid,gid,"
      "size,mtime,ctime;filesystems=ext4,btrfs,xfs,tmpfs,overlay,zfs;"
      "namespace=reenumerated;publication=post-compile-transactional;"
      "transaction-capacity=cache-capacity;touches=transactional;"
      "replacement=lru;integrity=sealed-record";
  digest.update(policy.data(), policy.size());
  digest.update(&impl_->maximum_entries, sizeof(impl_->maximum_entries));
  return "sha256:" + digest_hex(digest.finish());
}

InputContentMeasurementTransaction::InputContentMeasurementTransaction(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

InputContentMeasurementTransaction::~InputContentMeasurementTransaction() =
    default;
InputContentMeasurementTransaction::InputContentMeasurementTransaction(
    InputContentMeasurementTransaction &&) noexcept = default;
InputContentMeasurementTransaction &
InputContentMeasurementTransaction::operator=(
    InputContentMeasurementTransaction &&) noexcept = default;

InputContentMeasurementCacheCommitStats
InputContentMeasurementTransaction::commit() {
  if (!impl_ || impl_->committed)
    throw std::logic_error(
        "input content cache transaction was already committed or moved");
  auto &owner = *impl_->owner;
  std::scoped_lock lock(owner.mutex);
  InputContentMeasurementCacheCommitStats result{
      .capacity = owner.maximum_entries,
      .entries_before = static_cast<std::uint64_t>(owner.entries.size()),
      .entries_after = 0U,
      .staged_entries = static_cast<std::uint64_t>(impl_->staged.size()),
      .staging_saturations = impl_->staging_saturations,
      .evictions = 0U,
      .saturations = 0U,
      .corruptions = impl_->corruptions,
  };

  std::vector<FileCacheKey> keys;
  keys.reserve(impl_->staged.size());
  for (const auto &[key, unused] : impl_->staged) {
    (void)unused;
    keys.push_back(key);
  }
  std::ranges::sort(keys);

  // A cache hit records its LRU touch in the transaction. Publish those
  // no-allocation list moves only after the caller accepts the measured plan;
  // abandoning a transaction is therefore a complete cache no-op. This hot
  // path does not copy the potentially million-entry LRU list.
  if (keys.empty()) {
    using EntryMap = decltype(owner.entries);
    std::vector<EntryMap::iterator> touches;
    touches.reserve(impl_->touch_order.size());
    for (const auto &key : impl_->touch_order) {
      const auto entry = owner.entries.find(key);
      if (entry != owner.entries.end() &&
          valid_cached_record(key, entry->second.value, entry->second.seal))
        touches.push_back(entry);
    }
    // Fault injection and every potentially throwing record validation finish
    // before the first splice. Once publication starts, it consists only of
    // no-throw list operations while the owner mutex excludes replacement.
    if (owner.publication_fault_for_testing) {
      for (std::size_t index = 0U; index < touches.size(); ++index) {
        if (owner.publication_fault_for_testing())
          throw std::bad_alloc();
      }
    }
    for (const auto entry : touches) {
      owner.recency.splice(owner.recency.begin(), owner.recency,
                           entry->second.recency);
    }
    result.entries_after = static_cast<std::uint64_t>(owner.entries.size());
    impl_->committed = true;
    return result;
  }

  // Validate the complete locked snapshot before changing any entry. Reserve
  // buckets up front so allocation failure cannot follow an eviction. The
  // mutation phase keeps extracted map and list nodes and rolls every inserted
  // entry back if a remaining allocation unexpectedly fails.
  std::uint64_t prospective_new = 0U;
  for (const auto &key : keys) {
    const auto &staged = impl_->staged.at(key);
    if (!valid_cached_record(key, staged.value, staged.seal))
      throw std::logic_error(
          "input content cache refused an invalid staged record");
    if (const auto existing = owner.entries.find(key);
        existing != owner.entries.end()) {
      if (valid_cached_record(key, existing->second.value,
                              existing->second.seal) &&
          existing->second.value != staged.value)
        throw std::runtime_error(
            "input content cache observed conflicting exact metadata");
      if (!valid_cached_record(key, existing->second.value,
                               existing->second.seal))
        ++prospective_new;
    } else {
      ++prospective_new;
    }
  }
  const auto maximum_size = static_cast<std::size_t>(owner.maximum_entries);
  owner.entries.reserve(
      std::min(maximum_size, owner.entries.size() +
                                 static_cast<std::size_t>(prospective_new)));

  using EntryMap = decltype(owner.entries);
  struct RemovedEntry final {
    EntryMap::node_type node;
  };
  std::vector<RemovedEntry> removed;
  removed.reserve(static_cast<std::size_t>(prospective_new));
  std::vector<FileCacheKey> inserted;
  inserted.reserve(static_cast<std::size_t>(prospective_new));
  // Copy the list before the first mutation. This is intentionally paid only
  // by transactions that publish new measurements; a fully warm transaction
  // has no staged entries. The snapshot makes rollback exact even if an
  // allocation fails after push_front() but before the matching map entry is
  // installed, and preserves the pre-commit LRU order after touches/evictions.
  std::list<FileCacheKey> original_recency = owner.recency;
  const auto remove_existing = [&](EntryMap::iterator entry) {
    owner.recency.erase(entry->second.recency);
    removed.push_back({.node = owner.entries.extract(entry)});
  };
  const auto rollback = [&]() noexcept {
    for (const auto &key : inserted) {
      const auto found = owner.entries.find(key);
      if (found != owner.entries.end())
        owner.entries.erase(found);
    }
    for (auto &entry : removed) {
      const auto restored = owner.entries.insert(std::move(entry.node));
      if (!restored.inserted)
        std::terminate();
    }
    owner.recency.swap(original_recency);
    for (auto recency = owner.recency.begin(); recency != owner.recency.end();
         ++recency) {
      const auto entry = owner.entries.find(*recency);
      if (entry == owner.entries.end())
        std::terminate();
      entry->second.recency = recency;
    }
  };

  try {
    for (const auto &key : impl_->touch_order) {
      const auto entry = owner.entries.find(key);
      if (entry != owner.entries.end() &&
          valid_cached_record(key, entry->second.value, entry->second.seal))
        owner.recency.splice(owner.recency.begin(), owner.recency,
                             entry->second.recency);
    }
    for (const auto &key : keys) {
      const auto &staged = impl_->staged.at(key);
      if (const auto existing = owner.entries.find(key);
          existing != owner.entries.end()) {
        if (valid_cached_record(key, existing->second.value,
                                existing->second.seal)) {
          owner.recency.splice(owner.recency.begin(), owner.recency,
                               existing->second.recency);
          continue;
        }
        remove_existing(existing);
        ++result.corruptions;
      }
      if (owner.entries.size() >= owner.maximum_entries) {
        ++result.saturations;
        const auto evicted = owner.entries.find(owner.recency.back());
        if (evicted == owner.entries.end())
          throw std::logic_error("input content cache LRU index lost an entry");
        remove_existing(evicted);
        ++result.evictions;
      }
      owner.recency.push_front(key);
      const auto [unused, was_inserted] =
          owner.entries.emplace(key, InputContentMeasurementCache::Impl::Entry{
                                         .value = staged.value,
                                         .seal = staged.seal,
                                         .recency = owner.recency.begin(),
                                     });
      (void)unused;
      if (!was_inserted)
        throw std::logic_error(
            "input content cache insertion unexpectedly conflicted");
      inserted.push_back(key);
      if (owner.publication_fault_for_testing &&
          owner.publication_fault_for_testing())
        throw std::bad_alloc();
    }
  } catch (...) {
    rollback();
    throw;
  }
  result.entries_after = static_cast<std::uint64_t>(owner.entries.size());
  impl_->committed = true;
  return result;
}

InputContentRootIdentity
measure_input_content_root(const std::filesystem::path &path,
                           InputContentMeasurementStats *measurement_stats,
                           InputContentMeasurementTransaction *transaction) {
  const auto measurement_started = std::chrono::steady_clock::now();
  if (!path.is_absolute() || path.empty() || path.lexically_normal() != path)
    throw std::invalid_argument(
        "input content root path must be absolute and lexically normalized");
  const std::string native = path.native();
  if (native.find('\0') != std::string::npos ||
      native.size() > kMaximumNameBytes || native.starts_with("//") ||
      !valid_utf8(native) || (native.size() > 1U && native.ends_with('/')))
    throw std::invalid_argument(
        "input content root path must be bounded strict UTF-8");

  OpenedRoot opened = open_root_by_components(path);
  if (measurement_stats != nullptr)
    *measurement_stats = {};
  // Authoring runs on a gRPC worker whose stack is deliberately bounded. Keep
  // the large streaming buffer on the heap and reuse it through recursion so
  // deep trees neither overflow that stack nor allocate once per file.
  std::vector<unsigned char> read_buffer(kReadBufferBytes);
  NodeMeasurement root = measure_node(
      std::move(opened.descriptor), 0U, read_buffer, measurement_stats,
      transaction != nullptr ? transaction->impl_.get() : nullptr);
  require_unchanged_links(opened.links);
  if (root.kind == ContentRootKind::directory && root.file_count == 0U)
    throw std::runtime_error("input content directory root is empty");
  if (measurement_stats != nullptr) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - measurement_started);
    measurement_stats->elapsed_nanoseconds =
        static_cast<std::uint64_t>(std::max(elapsed.count(), std::int64_t{0}));
  }
  return {.api_version = std::string(kInputContentRootApiVersion),
          .path = path.native(),
          .kind = root.kind,
          .file_count = root.file_count,
          .total_bytes = root.total_bytes,
          .tree_sha256 = "sha256:" + digest_hex(root.digest)};
}

std::vector<InputContentRootIdentity> measure_input_content_root_set(
    const InputContentRootSet &root_set,
    std::vector<InputContentMeasurementStats> *measurement_stats,
    InputContentMeasurementTransaction *transaction) {
  if (root_set.api_version != kInputContentRootSetApiVersion)
    throw std::invalid_argument(
        "input content root set API version is unsupported");
  if (root_set.paths.empty() || root_set.paths.size() > 256U)
    throw std::invalid_argument(
        "input content root set must contain between 1 and 256 paths");
  std::vector<std::filesystem::path> paths;
  paths.reserve(root_set.paths.size());
  for (const std::string &value : root_set.paths) {
    const std::filesystem::path path(value);
    if (!path.is_absolute() || path.empty() || path.lexically_normal() != path)
      throw std::invalid_argument(
          "input content root set contains a noncanonical path");
    paths.push_back(path);
  }
  std::ranges::sort(paths, {}, [](const std::filesystem::path &path) {
    return path.native();
  });
  for (std::size_t index = 0U; index < paths.size(); ++index) {
    if (index > 0U && (paths[index] == paths[index - 1U] ||
                       path_within(paths[index], paths[index - 1U])))
      throw std::invalid_argument(
          "input content root set paths must be unique and nonoverlapping");
  }
  std::vector<InputContentRootIdentity> identities;
  identities.reserve(paths.size());
  if (measurement_stats != nullptr) {
    measurement_stats->clear();
    measurement_stats->reserve(paths.size());
  }
  for (const auto &path : paths) {
    InputContentMeasurementStats stats;
    identities.push_back(measure_input_content_root(
        path, measurement_stats != nullptr ? &stats : nullptr, transaction));
    if (measurement_stats != nullptr)
      measurement_stats->push_back(stats);
  }
  return identities;
}

} // namespace trainvm
