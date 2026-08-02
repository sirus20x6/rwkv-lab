#include "trainvm/input_content_authority.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
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
constexpr char kFileDomain[] = "trainvm.input-content.file/v1";
constexpr char kDirectoryDomain[] = "trainvm.input-content.directory/v1";

using Digest = std::array<unsigned char, kDigestBytes>;

[[noreturn]] void fail_system(std::string_view operation) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::strerror(errno));
}

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0) (void)::close(value_);
  }

  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) (void)::close(value_);
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
  DigestContext()
      : context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
    if (!context_ ||
        EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
      throw std::runtime_error("could not initialize SHA-256");
    }
  }

  void update(const void* data, std::size_t size) {
    if (size != 0U && EVP_DigestUpdate(context_.get(), data, size) != 1)
      throw std::runtime_error("could not update SHA-256");
  }

  template <std::size_t Size>
  void update(const std::array<unsigned char, Size>& value) {
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
};

struct Child final {
  std::string name;
  NodeMeasurement measurement;
};

struct RetainedLink final {
  Descriptor parent;
  std::string name;
  struct stat expected {};
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
    if (offset + continuation >= value.size()) return false;
    for (std::size_t index = 1U; index <= continuation; ++index) {
      const unsigned char byte =
          static_cast<unsigned char>(value[offset + index]);
      if ((byte & 0xc0U) != 0x80U) return false;
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
bool same_namespace_component(const struct stat& before,
                              const struct stat& after) {
  return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_mode == after.st_mode && before.st_uid == after.st_uid &&
         before.st_gid == after.st_gid;
}

bool same_stat(const struct stat& before, const struct stat& after) {
  return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_mode == after.st_mode && before.st_nlink == after.st_nlink &&
         before.st_size == after.st_size &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

OpenedRoot open_root_by_components(const std::filesystem::path& path) {
  const int slash =
      ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (slash < 0) fail_system("could not open filesystem root");
  Descriptor current(slash);
  std::vector<std::string> components;
  for (const std::filesystem::path& component : path.relative_path()) {
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
    struct stat before {};
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
    const int next =
        ::openat(current.get(), components[index].c_str(), flags);
    if (next < 0) {
      if (errno == ELOOP)
        throw std::runtime_error("input content root has a symlinked ancestor");
      fail_system("could not open input content root component");
    }
    struct stat opened {};
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
  return {.descriptor = std::move(current),
          .links = std::move(links)};
}

void require_unchanged_links(const std::vector<RetainedLink>& links) {
  for (auto iterator = links.rbegin(); iterator != links.rend(); ++iterator) {
    struct stat observed {};
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

void add_totals(NodeMeasurement& parent, const NodeMeasurement& child) {
  if (child.file_count > kMaximumFiles - parent.file_count)
    throw std::runtime_error("input content root exceeds the file-count bound");
  if (child.total_bytes > kMaximumBytes - parent.total_bytes)
    throw std::runtime_error("input content root exceeds the byte bound");
  parent.file_count += child.file_count;
  parent.total_bytes += child.total_bytes;
}

NodeMeasurement measure_node(Descriptor descriptor, std::size_t depth) {
  if (depth > kMaximumDepth)
    throw std::runtime_error("input content root exceeds the depth bound");
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0)
    fail_system("could not inspect input content node");

  if (S_ISREG(before.st_mode)) {
    if (before.st_size < 0)
      throw std::runtime_error("input content file has a negative size");
    const auto size = static_cast<std::uint64_t>(before.st_size);
    if (size > kMaximumBytes)
      throw std::runtime_error("input content root exceeds the byte bound");
    DigestContext content;
    std::array<unsigned char, 1024U * 1024U> buffer{};
    std::uint64_t observed = 0U;
    for (;;) {
      const ssize_t count = ::read(descriptor.get(), buffer.data(),
                                   buffer.size());
      if (count < 0) {
        if (errno == EINTR) continue;
        fail_system("could not read input content file");
      }
      if (count == 0) break;
      const auto bytes = static_cast<std::size_t>(count);
      if (static_cast<std::uint64_t>(bytes) > size - observed)
        throw std::runtime_error("input content file changed while hashing");
      content.update(buffer.data(), bytes);
      observed += static_cast<std::uint64_t>(bytes);
    }
    struct stat after {};
    if (::fstat(descriptor.get(), &after) != 0)
      fail_system("could not reinspect input content file");
    if (!same_stat(before, after) || observed != size)
      throw std::runtime_error("input content file changed while hashing");
    const Digest content_digest = content.finish();
    DigestContext node;
    node.update(kFileDomain, sizeof(kFileDomain));
    node.update(big_endian_64(size));
    node.update(content_digest);
    return {.kind = ContentRootKind::file,
            .file_count = 1U,
            .total_bytes = size,
            .digest = node.finish()};
  }

  if (!S_ISDIR(before.st_mode))
    throw std::runtime_error(
        "input content root contains a symlink or special node");

  const int duplicate = ::fcntl(descriptor.get(), F_DUPFD_CLOEXEC, 0);
  if (duplicate < 0) fail_system("could not duplicate input directory");
  DIR* raw_directory = ::fdopendir(duplicate);
  if (raw_directory == nullptr) {
    const int saved_errno = errno;
    (void)::close(duplicate);
    errno = saved_errno;
    fail_system("could not enumerate input directory");
  }
  struct CloseDirectory final {
    void operator()(DIR* directory) const noexcept {
      if (directory != nullptr) (void)::closedir(directory);
    }
  };
  std::unique_ptr<DIR, CloseDirectory> directory(raw_directory);
  std::vector<std::string> names;
  errno = 0;
  while (dirent* entry = ::readdir(directory.get())) {
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..") continue;
    if (name.empty() || name.size() > kMaximumNameBytes || !valid_utf8(name))
      throw std::runtime_error("input directory contains an invalid UTF-8 name");
    names.emplace_back(name);
    errno = 0;
  }
  if (errno != 0) fail_system("could not enumerate input directory");
  std::ranges::sort(names, [](const std::string& left,
                             const std::string& right) {
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
  for (const std::string& name : names) {
    struct stat child_before {};
    if (::fstatat(descriptor.get(), name.c_str(), &child_before,
                  AT_SYMLINK_NOFOLLOW) != 0)
      fail_system("could not inspect input content child");
    if (S_ISLNK(child_before.st_mode))
      throw std::runtime_error("input content root contains a symlink");
    if (!S_ISREG(child_before.st_mode) && !S_ISDIR(child_before.st_mode))
      throw std::runtime_error("input content root contains a special node");
    const int child_flags =
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK |
        (S_ISDIR(child_before.st_mode) ? O_DIRECTORY : 0);
    const int child_fd =
        ::openat(descriptor.get(), name.c_str(), child_flags);
    if (child_fd < 0) {
      if (errno == ELOOP)
        throw std::runtime_error("input content root contains a symlink");
      fail_system("could not open input content child");
    }
    struct stat child_opened {};
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
    NodeMeasurement child = measure_node(Descriptor(child_fd), depth + 1U);
    struct stat child_after {};
    if (::fstatat(descriptor.get(), name.c_str(), &child_after,
                  AT_SYMLINK_NOFOLLOW) != 0)
      fail_system("could not reinspect input content child");
    if (!same_stat(child_opened, child_after))
      throw std::runtime_error("input content child was substituted");
    add_totals(result, child);
    children.push_back({.name = name, .measurement = std::move(child)});
  }

  struct stat after {};
  if (::fstat(descriptor.get(), &after) != 0)
    fail_system("could not reinspect input directory");
  if (!same_stat(before, after))
    throw std::runtime_error("input directory changed while hashing");

  DigestContext node;
  node.update(kDirectoryDomain, sizeof(kDirectoryDomain));
  for (const Child& child : children) {
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

std::string digest_hex(const Digest& digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2U);
  for (const unsigned char byte : digest) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

}  // namespace

InputContentRootIdentity measure_input_content_root(
    const std::filesystem::path& path) {
  if (!path.is_absolute() || path.empty() || path.lexically_normal() != path)
    throw std::invalid_argument(
        "input content root path must be absolute and lexically normalized");
  const std::string native = path.native();
  if (native.find('\0') != std::string::npos ||
      native.size() > kMaximumNameBytes || native.starts_with("//") ||
      !valid_utf8(native) ||
      (native.size() > 1U && native.ends_with('/')))
    throw std::invalid_argument(
        "input content root path must be bounded strict UTF-8");

  OpenedRoot opened = open_root_by_components(path);
  NodeMeasurement root = measure_node(std::move(opened.descriptor), 0U);
  require_unchanged_links(opened.links);
  if (root.kind == ContentRootKind::directory && root.file_count == 0U)
    throw std::runtime_error("input content directory root is empty");
  return {.api_version = std::string(kInputContentRootApiVersion),
          .path = path.native(),
          .kind = root.kind,
          .file_count = root.file_count,
          .total_bytes = root.total_bytes,
          .tree_sha256 = "sha256:" + digest_hex(root.digest)};
}

}  // namespace trainvm
