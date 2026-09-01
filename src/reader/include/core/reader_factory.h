#pragma once

#include <experimental/filesystem>
#include <memory>
#include <string>

namespace reader {

class FormatRegistry;
class IResourceLocator;
class ISource;
class MDDataOwner;
class StructureReader;

class ReaderFactory {
public:
  ReaderFactory();
  ReaderFactory(std::shared_ptr<FormatRegistry> registry,
                std::shared_ptr<IResourceLocator> locator);

  std::shared_ptr<StructureReader> makeStructureReader(
      const std::string& file_path,
      bool use_mmap,
      std::shared_ptr<MDDataOwner> owner) const;

  std::shared_ptr<FormatRegistry> registry() const noexcept { return registry_; }
  std::shared_ptr<IResourceLocator> locator() const noexcept { return locator_; }

private:
  std::shared_ptr<ISource> createSource(const std::experimental::filesystem::path& path,
                                        bool use_mmap) const;

  std::shared_ptr<FormatRegistry> registry_;
  std::shared_ptr<IResourceLocator> locator_;
};

}  // namespace reader
