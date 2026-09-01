#pragma once

#include <memory>
#include <string>

namespace reader {

class MDDataOwner;
class StructureReader;
class ISource;
class IResourceLocator;

class IFormatProvider {
public:
  virtual ~IFormatProvider() = default;

  virtual std::string name() const = 0;
  
  virtual int probe(const ISource& src, const IResourceLocator& locator) const = 0;

  virtual std::shared_ptr<StructureReader> createStructureReader(
      std::shared_ptr<ISource> src,
      std::shared_ptr<IResourceLocator> locator,
      std::string file_path,
      std::shared_ptr<MDDataOwner> owner) const = 0;
};

}  // namespace reader
