#include "io/mmap_source.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "io/icursor.h"

namespace reader {

namespace {

class MmapCursor : public ICursor {
 public:
  explicit MmapCursor(std::shared_ptr<MmapSource::Mapping> mapping, std::uint64_t offset)
      : mapping_(std::move(mapping)), position_(std::min<std::uint64_t>(offset, mapping_->length)) {}

  bool eof() const override { return position_ >= mapping_->length; }

  std::size_t readRaw(char* dst, std::size_t n) override {
    if (eof()) {
      return 0;
    }
    std::size_t remaining = mapping_->length - position_;
    std::size_t to_copy = std::min<std::size_t>(n, remaining);
    std::memcpy(dst, mapping_->data + position_, to_copy);
    position_ += to_copy;
    return to_copy;
  }

  bool nextLine(std::string_view& out) override {
    if (eof()) {
      return false;
    }
    auto* start = mapping_->data + position_;
    auto* ptr = start;
    auto* end = mapping_->data + mapping_->length;
    while (ptr < end && *ptr != '\n' && *ptr != '\r') {
      ++ptr;
    }
    out = std::string_view(start, static_cast<std::size_t>(ptr - start));
    if (ptr < end && *ptr == '\r') {
      ++ptr;
    }
    if (ptr < end && *ptr == '\n') {
      ++ptr;
    }
    position_ = static_cast<std::uint64_t>(ptr - mapping_->data);
    return true;
  }

  std::uint64_t position() const override { return position_; }

  void seek(std::uint64_t pos) override {
    position_ = std::min<std::uint64_t>(pos, mapping_->length);
  }

 private:
  std::shared_ptr<MmapSource::Mapping> mapping_;
  std::uint64_t position_ = 0;
};

}  // namespace

MmapSource::MmapSource(std::experimental::filesystem::path path) : mapping_(std::make_shared<Mapping>()) {
  mapping_->path = std::move(path);
  int fd = ::open(mapping_->path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(),
                            "Unable to open file: " + mapping_->path.string());
  }

  struct stat stat_buf;
  if (::fstat(fd, &stat_buf) != 0) {
    ::close(fd);
    throw std::system_error(errno, std::generic_category(),
                            "Unable to stat file: " + mapping_->path.string());
  }

  mapping_->length = static_cast<std::size_t>(stat_buf.st_size);
  if (mapping_->length == 0) {
    mapping_->data = nullptr;
    size_ = 0;
    ::close(fd);
    return;
  }

  void* mapped = ::mmap(nullptr, mapping_->length, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapped == MAP_FAILED) {
    throw std::system_error(errno, std::generic_category(),
                            "Unable to mmap file: " + mapping_->path.string());
  }
  mapping_->data = static_cast<char*>(mapped);
  size_ = mapping_->length;
}

std::unique_ptr<ICursor> MmapSource::cursor(std::uint64_t offset) const {
  if (!mapping_) {
    return nullptr;
  }
  return std::make_unique<MmapCursor>(mapping_, offset);
}

MmapSource::~MmapSource() {
  if (mapping_ && mapping_->data) {
    ::munmap(mapping_->data, mapping_->length);
    mapping_->data = nullptr;
    mapping_->length = 0;
  }
}

}  // namespace reader
