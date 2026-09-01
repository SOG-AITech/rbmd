#include "io/file_stream_source.h"

#include <fstream>
#include <stdexcept>
#include <string_view>

#include "io/icursor.h"

namespace reader {

namespace {

class FileStreamCursor : public ICursor {
 public:
  FileStreamCursor(std::experimental::filesystem::path path, std::uint64_t offset, std::uint64_t size)
      : path_(std::move(path)),
        stream_(path_, std::ios::binary),
        position_(offset),
        size_(size) {
    if (!stream_.is_open()) {
      throw std::runtime_error("Failed to open file: " + path_.string());
    }
    stream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!stream_) {
      throw std::runtime_error("Failed to seek file: " + path_.string());
    }
  }

  bool eof() const override { return stream_.eof(); }

  std::size_t readRaw(char* dst, std::size_t n) override {
    stream_.read(dst, static_cast<std::streamsize>(n));
    auto read = static_cast<std::size_t>(stream_.gcount());
    position_ += read;
    return read;
  }

  bool nextLine(std::string_view& out) override {
    if (!std::getline(stream_, buffer_)) {
      return false;
    }
    position_ = static_cast<std::uint64_t>(stream_.tellg());
    if (stream_.eof() || position_ == static_cast<std::uint64_t>(-1)) {
      position_ = size_;
    }
    out = std::string_view(buffer_);
    return true;
  }

  std::uint64_t position() const override { return position_; }

  void seek(std::uint64_t position) override {
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(position), std::ios::beg);
    if (!stream_) {
      throw std::runtime_error("Failed to seek file: " + path_.string());
    }
    position_ = position;
  }

 private:
    std::experimental::filesystem::path path_;
    mutable std::ifstream stream_;
    std::uint64_t position_;
    std::uint64_t size_;
    std::string buffer_;
};

}  // namespace

FileStreamSource::FileStreamSource(std::experimental::filesystem::path path)
    : path_(std::move(path)),
      size_(static_cast<std::uint64_t>(std::experimental::filesystem::file_size(path_))) {}

std::unique_ptr<ICursor> FileStreamSource::cursor(std::uint64_t offset) const {
  if (offset > size_) {
    offset = size_;
  }
  return std::make_unique<FileStreamCursor>(path_, offset, size_);
}

}  // namespace reader
