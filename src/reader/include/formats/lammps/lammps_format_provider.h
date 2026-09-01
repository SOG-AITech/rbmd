#pragma once

#include <memory>
#include <string>

#include "core/i_format_provider.h"

namespace reader {

class LammpsFormatProvider : public IFormatProvider {
 public:
  LammpsFormatProvider() = default;
  ~LammpsFormatProvider() override = default;

  std::string name() const override;
  int probe(const ISource& src, const IResourceLocator& locator) const override;
  std::shared_ptr<StructureReader> createStructureReader(
      std::shared_ptr<ISource> src,
      std::shared_ptr<IResourceLocator> locator,
      std::string file_path,
      std::shared_ptr<MDDataOwner> owner) const override;
};

}  // namespace reader
