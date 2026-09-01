#pragma once

#include <cstdint>
#include <memory>

namespace reader {

class ICursor;

class ISource {
public:
  virtual ~ISource() = default;

  virtual std::unique_ptr<ICursor> cursor(std::uint64_t offset) const = 0;
  virtual std::uint64_t size() const = 0;
};

}  // namespace reader

