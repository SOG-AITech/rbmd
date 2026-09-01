#include "core/reader_base.h"

#include <utility>

#include "io/resource_locator.h"
#include "io/isource.h"

namespace reader {

ReaderBase::ReaderBase(std::shared_ptr<ISource> source,
                       std::shared_ptr<IResourceLocator> locator,
                       std::string file_path)
    : source_(std::move(source)),
      locator_(std::move(locator)),
      file_path_(std::move(file_path)) {}

}  // namespace reader

