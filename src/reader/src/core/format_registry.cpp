#include "core/format_registry.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "core/i_format_provider.h"
#include "io/resource_locator.h"

namespace reader {

void FormatRegistry::registerProvider(std::shared_ptr<IFormatProvider> provider) {
  if (!provider) {
    throw std::invalid_argument("Cannot register null provider");
  }
  const auto it = std::find_if(providers_.begin(), providers_.end(),
                         [&](const auto& existing) {
                           return existing && existing->name() == provider->name();
                         });
  if (it == providers_.end()) {
    providers_.push_back(std::move(provider));
  }
}

std::shared_ptr<IFormatProvider> FormatRegistry::bestMatch(
    const ISource& src, const IResourceLocator& locator) const {
  int best_score = std::numeric_limits<int>::min();
  std::shared_ptr<IFormatProvider> best_provider;
  for (const auto& provider : providers_) {
    if (!provider) {
      continue;
    }
    const int score = provider->probe(src, locator);
    if (score > best_score && score > 0) {
      best_score = score;
      best_provider = provider;
    }
  }
  return best_provider;
}

}  // namespace reader

