#pragma once

#include <memory>
#include <string>
#include <vector>

namespace reader {

class IFormatProvider;
class IResourceLocator;
class ISource;

class FormatRegistry {
public:
  void registerProvider(std::shared_ptr<IFormatProvider> provider);

  std::shared_ptr<IFormatProvider> bestMatch(const ISource& src,
                                             const IResourceLocator& locator) const;

  const std::vector<std::shared_ptr<IFormatProvider>>& providers() const noexcept {
    return providers_;
  }

private:
  std::vector<std::shared_ptr<IFormatProvider>> providers_;
};

}  // namespace reader

