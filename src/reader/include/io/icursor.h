#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace reader {

class ICursor {
public:
  virtual ~ICursor() = default;

  virtual bool eof() const = 0;
  virtual std::size_t readRaw(char* dst, std::size_t n) = 0;
  virtual bool nextLine(std::string_view& out) = 0;
  virtual std::uint64_t position() const = 0;
  virtual void seek(std::uint64_t position) = 0;
};

}  // namespace reader

