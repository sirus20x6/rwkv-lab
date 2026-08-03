#include "trainvm/authority_document.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

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
  [[nodiscard]] int get() const noexcept { return descriptor_; }

 private:
  int descriptor_;
};

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

}  // namespace trainvm
