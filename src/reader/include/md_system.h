#pragma once

#include <memory>
#include <string>

class MDData;

namespace reader {

class MDDataOwner;

struct LoadOptions {
  bool use_mmap = false;
  std::string atom_style = "full";
  std::string force_field = "LJ";
};

class MDSystem {
public:
  explicit MDSystem(std::shared_ptr<MDDataOwner> owner);

  static MDSystem load(const std::string& file_path, bool use_mmap = false);
  static MDSystem load(const std::string& file_path, const LoadOptions& options);

  std::shared_ptr<MDData> data() const noexcept;
  std::shared_ptr<MDDataOwner> dataOwner() const noexcept { return owner_; }

private:
  std::shared_ptr<MDDataOwner> owner_;
};

}  // namespace reader
