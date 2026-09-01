#pragma once

#include <cstddef>
#include <cstdint>
#include <experimental/filesystem>
#include <memory>

#include "io/isource.h"

namespace reader {

class MmapSource : public ISource {
public:
  struct Mapping {
    std::experimental::filesystem::path path;
    std::size_t length = 0;
    char* data = nullptr;
  };

  explicit MmapSource(std::experimental::filesystem::path path);
  ~MmapSource() override;

  std::unique_ptr<ICursor> cursor(std::uint64_t offset) const override;
  std::uint64_t size() const override { return size_; }

  std::shared_ptr<Mapping> mapping_;
  std::uint64_t size_ = 0;
};

}  // namespace reader
