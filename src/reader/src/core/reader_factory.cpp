#include "core/reader_factory.h"

#include <experimental/filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include "core/format_registry.h"
#include "core/i_format_provider.h"
#include "formats/lammps/lammps_format_provider.h"
#include "io/file_stream_source.h"
#include "io/file_system_locator.h"
#include "io/isource.h"
#include "io/mmap_source.h"
#include "io/resource_locator.h"
#include "md_data_owner.h"

namespace reader {

namespace {

std::shared_ptr<FormatRegistry> MakeDefaultRegistry() {
  auto registry = std::make_shared<FormatRegistry>();
  registry->registerProvider(std::make_shared<LammpsFormatProvider>());
  return registry;
}

std::shared_ptr<IResourceLocator> MakeDefaultLocator() {
  return std::make_shared<FileSystemLocator>();
}

}  // namespace

ReaderFactory::ReaderFactory()
    : registry_(MakeDefaultRegistry()), locator_(MakeDefaultLocator()) {}

ReaderFactory::ReaderFactory(std::shared_ptr<FormatRegistry> registry,
                             std::shared_ptr<IResourceLocator> locator)
    : registry_(std::move(registry)), locator_(std::move(locator)) {
  if (!registry_) {
    registry_ = MakeDefaultRegistry();
  }
  if (!locator_) {
    locator_ = MakeDefaultLocator();
  }
}

std::shared_ptr<StructureReader> ReaderFactory::makeStructureReader(
    const std::string& file_path,
    bool use_mmap,
    std::shared_ptr<MDDataOwner> owner) const {
  if (!owner || !owner->data()) {
    throw std::invalid_argument("makeStructureReader requires MDData storage");
  }
  if (!registry_) {
    throw std::runtime_error("ReaderFactory has no registry");
  }
  if (!locator_) {
    throw std::runtime_error("ReaderFactory has no resource locator");
  }

  auto resolved = locator_->resolve(file_path);
  auto source = createSource(resolved, use_mmap);
  if (!source) {
    throw std::runtime_error("Failed to create source for: " + resolved.string());
  }

  auto provider = registry_->bestMatch(*source, *locator_);
  if (!provider) {
    throw std::runtime_error("No matching reader provider for: " + resolved.string());
  }

  return provider->createStructureReader(std::move(source), locator_, resolved.string(),
                                         std::move(owner));
}

std::shared_ptr<ISource> ReaderFactory::createSource(const std::experimental::filesystem::path& path,
                                                     bool use_mmap) const {
  if (use_mmap) {
    return std::make_shared<MmapSource>(path);
  }
  return std::make_shared<FileStreamSource>(path);
}

}  // namespace reader
