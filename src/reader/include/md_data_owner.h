#pragma once

#include <memory>
#include <utility>
#include <vector>

class MDData;

namespace reader {

class MDDataOwner {
public:
  explicit MDDataOwner(std::shared_ptr<MDData> data)
      : data_(std::move(data)) {}

  std::shared_ptr<MDData> data() const noexcept { return data_; }

  template <typename T, typename... Args>
  T* allocateScalar(Args&&... args) {
    auto holder = std::shared_ptr<T>(new T(std::forward<Args>(args)...),
                                     [](T* ptr) { delete ptr; });
    auto raw = holder.get();
    ephemerals_.push_back(std::move(holder));
    return raw;
  }

  void adopt(std::shared_ptr<void> holder) { ephemerals_.push_back(std::move(holder)); }

  void resetEphemerals() { ephemerals_.clear(); }

private:
  std::shared_ptr<MDData> data_;
  std::vector<std::shared_ptr<void>> ephemerals_;
};

}  // namespace reader
