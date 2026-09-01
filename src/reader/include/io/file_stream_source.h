#pragma once

#include <experimental/filesystem>
#include <memory>

#include "io/isource.h"
#include "io/icursor.h"

namespace reader {

class FileStreamSource : public ISource {
public:
  explicit FileStreamSource(std::experimental::filesystem::path path);
  ~FileStreamSource() override = default;

  std::unique_ptr<ICursor> cursor(std::uint64_t offset) const override;
  std::uint64_t size() const override { return size_; }

private:
  std::experimental::filesystem::path path_;
  std::uint64_t size_;
};

}  // namespace reader

