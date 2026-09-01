#pragma once

#include <experimental/filesystem>
#include <string>
#include <vector>

#include "io/resource_locator.h"

namespace reader {

class FileSystemLocator : public IResourceLocator {
public:
  FileSystemLocator();
  ~FileSystemLocator() override = default;

  std::experimental::filesystem::path resolve(const std::string& path) const override;
  void addSearchPath(const std::string& dir) override;

private:
  std::vector<std::experimental::filesystem::path> search_paths_;
};

}  // namespace reader

