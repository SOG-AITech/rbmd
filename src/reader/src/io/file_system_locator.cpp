#include "io/file_system_locator.h"

#include <experimental/filesystem>
#include <stdexcept>

namespace reader {

namespace fs = std::experimental::filesystem;

FileSystemLocator::FileSystemLocator() {
  search_paths_.push_back(fs::current_path());
}

void FileSystemLocator::addSearchPath(const std::string& dir) {
  if (dir.empty()) {
    return;
  }
  fs::path path(dir);
  if (!path.is_absolute()) {
    path = fs::current_path() / path;
  }
  if (fs::exists(path)) {
    search_paths_.push_back(fs::canonical(path));
  }
}

fs::path FileSystemLocator::resolve(const std::string& path) const {
  fs::path candidate(path);
  if (candidate.is_absolute()) {
    if (!fs::exists(candidate)) {
      throw std::runtime_error("File not found: " + candidate.string());
    }
    return fs::canonical(candidate);
  }

  for (const auto& base : search_paths_) {
    auto attempt = base / candidate;
    if (fs::exists(attempt)) {
      return fs::canonical(attempt);
    }
  }

  throw std::runtime_error("Unable to resolve resource: " + path);
}

}  // namespace reader

