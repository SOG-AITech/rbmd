#pragma once

#include <experimental/filesystem>
#include <string>

namespace reader {

class IResourceLocator {
public:
  virtual ~IResourceLocator() = default;

  virtual std::experimental::filesystem::path resolve(const std::string& path) const = 0;
  virtual void addSearchPath(const std::string& dir) = 0;
};

}  // namespace reader

