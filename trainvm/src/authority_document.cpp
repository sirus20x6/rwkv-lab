#include "trainvm/authority_document.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/openat2.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

namespace trainvm {
namespace {

class FileDescriptor final {
 public:
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() {
    if (descriptor_ >= 0) (void)::close(descriptor_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      if (descriptor_ >= 0) (void)::close(descriptor_);
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return descriptor_; }

 private:
  int descriptor_;
};

[[noreturn]] void publication_error(std::string_view label,
                                    std::string_view detail) {
  throw std::invalid_argument(std::string(label) + ": " +
                              std::string(detail));
}

bool safe_basename(std::string_view value) {
  return !value.empty() && value.size() <= 128U && value != "." &&
         value != ".." && value.find('/') == std::string_view::npos &&
         value.find('\0') == std::string_view::npos;
}

FileDescriptor open_parent_from_root(const std::filesystem::path& parent,
                                    std::string_view label) {
  FileDescriptor current(
      ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (current.get() < 0) publication_error(label, "cannot open filesystem root");
  for (const auto& part_path : parent.relative_path()) {
    const std::string part = part_path.string();
    if (!safe_basename(part))
      publication_error(label, "parent path is noncanonical");
    open_how how{};
    how.flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW;
    how.resolve =
        RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS;
    const int opened = static_cast<int>(::syscall(
        SYS_openat2, current.get(), part.c_str(), &how, sizeof(how)));
    if (opened < 0)
      publication_error(label, "cannot securely resolve parent path");
    current = FileDescriptor(opened);
  }
  return current;
}

void write_all(int descriptor, std::string_view contents,
               std::string_view label) {
  std::size_t offset = 0U;
  while (offset < contents.size()) {
    const ssize_t count =
        ::write(descriptor, contents.data() + offset, contents.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) publication_error(label, "write failed");
    offset += static_cast<std::size_t>(count);
  }
}

}  // namespace

std::string read_authority_document(const std::filesystem::path& path,
                                    std::string_view document_kind,
                                    std::uintmax_t maximum_bytes) {
  const std::string label(document_kind);
  if (path.empty() || !path.is_absolute() || label.empty() ||
      maximum_bytes == 0U)
    throw std::invalid_argument(
        "authority document path, kind, and bound must be explicit");
  const int raw =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (raw < 0)
    throw std::invalid_argument("could not securely open " + label + " " +
                                path.string() + ": " +
                                std::strerror(errno));
  FileDescriptor descriptor(raw);
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 ||
      !S_ISREG(before.st_mode) || before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) > maximum_bytes ||
      (before.st_uid != 0U && before.st_uid != ::geteuid()) ||
      (before.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw std::invalid_argument(
        label + " must be an owner/root-owned regular file that is not "
                "group/world-writable and fits its declared size bound");
  }
  std::string text(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0U;
  while (offset < text.size()) {
    const ssize_t count =
        ::read(descriptor.get(), text.data() + offset, text.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0)
      throw std::invalid_argument(label +
                                  " changed or failed while being read");
    offset += static_cast<std::size_t>(count);
  }
  char extra = '\0';
  const ssize_t trailing = ::read(descriptor.get(), &extra, 1U);
  struct stat after {};
  if (trailing != 0 || ::fstat(descriptor.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    throw std::invalid_argument(label + " changed while it was being read");
  }
  return text;
}

void publish_authority_document(
    const std::filesystem::path& path, std::string_view document_kind,
    std::string_view contents, AuthorityDocumentPublicationPolicy policy,
    std::uintmax_t maximum_bytes) {
  const std::string label(document_kind);
  const std::filesystem::path parent = path.parent_path();
  const std::string filename = path.filename().string();
  if (path.empty() || !path.is_absolute() || path.lexically_normal() != path ||
      parent.empty() || !safe_basename(filename) || label.empty() ||
      label.size() > 128U || contents.empty() || maximum_bytes == 0U ||
      static_cast<std::uintmax_t>(contents.size()) > maximum_bytes ||
      (policy.file_mode != 0600 && policy.file_mode != 0640)) {
    publication_error(label.empty() ? "authority document" : label,
                      "publication policy, path, or size is invalid");
  }

  FileDescriptor directory = open_parent_from_root(parent, label);
  struct stat parent_status {};
  if (::fstat(directory.get(), &parent_status) != 0 ||
      !S_ISDIR(parent_status.st_mode) ||
      parent_status.st_uid != policy.parent_owner_uid ||
      parent_status.st_gid != policy.parent_owner_gid ||
      (parent_status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    publication_error(label, "parent directory authority is unsafe");
  }

  static std::atomic<std::uint64_t> sequence{1U};
  FileDescriptor temporary(-1);
  std::string temporary_name;
  for (std::size_t attempt = 0U; attempt < 16U; ++attempt) {
    const std::uint64_t suffix =
        sequence.fetch_add(1U, std::memory_order_relaxed);
    temporary_name = "." + filename + ".tmp." +
                     std::to_string(static_cast<long long>(::getpid())) +
                     "." + std::to_string(suffix);
    const int opened = ::openat(directory.get(), temporary_name.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                    O_NOFOLLOW,
                                0600);
    if (opened >= 0) {
      temporary = FileDescriptor(opened);
      break;
    }
    if (errno != EEXIST)
      publication_error(label, "cannot create atomic temporary file");
  }
  if (temporary.get() < 0)
    publication_error(label, "temporary filename collision bound exhausted");

  bool renamed = false;
  struct TemporaryCleanup final {
    int directory;
    const std::string& name;
    bool& renamed;
    ~TemporaryCleanup() {
      if (!renamed) (void)::unlinkat(directory, name.c_str(), 0);
    }
  } cleanup{directory.get(), temporary_name, renamed};

  struct stat temporary_status {};
  if (::fstat(temporary.get(), &temporary_status) != 0 ||
      !S_ISREG(temporary_status.st_mode) || temporary_status.st_nlink != 1)
    publication_error(label, "temporary file identity is unsafe");
  if ((temporary_status.st_uid != policy.owner_uid ||
       temporary_status.st_gid != policy.owner_gid) &&
      ::fchown(temporary.get(), policy.owner_uid, policy.owner_gid) != 0)
    publication_error(label, "cannot assign authority document owner");
  if (::fchmod(temporary.get(), policy.file_mode) != 0)
    publication_error(label, "cannot seal authority document permissions");
  write_all(temporary.get(), contents, label);
  if (::fsync(temporary.get()) != 0)
    publication_error(label, "cannot fsync authority document");
  if (::renameat(directory.get(), temporary_name.c_str(), directory.get(),
                 filename.c_str()) != 0)
    publication_error(label, "cannot atomically publish authority document");
  renamed = true;
  if (::fsync(directory.get()) != 0)
    publication_error(label, "cannot fsync authority document directory");

  const int opened = ::openat(directory.get(), filename.c_str(),
                              O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (opened < 0)
    publication_error(label, "cannot re-open published authority document");
  FileDescriptor published(opened);
  struct stat published_status {};
  if (::fstat(published.get(), &published_status) != 0 ||
      !S_ISREG(published_status.st_mode) || published_status.st_nlink != 1 ||
      published_status.st_uid != policy.owner_uid ||
      published_status.st_gid != policy.owner_gid ||
      (published_status.st_mode & 07777) != policy.file_mode ||
      published_status.st_size != static_cast<off_t>(contents.size())) {
    publication_error(label, "published authority document is inexact");
  }
}

}  // namespace trainvm
