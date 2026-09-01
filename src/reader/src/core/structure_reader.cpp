#include "core/structure_reader.h"

#include <stdexcept>
#include <utility>

#include "io/isource.h"

namespace reader {

StructureReader::StructureReader(std::shared_ptr<ISource> source,
                                 std::shared_ptr<IResourceLocator> locator,
                                 std::shared_ptr<MDDataOwner> owner,
                                 std::string file_path)
    : ReaderBase(std::move(source), std::move(locator), std::move(file_path)),
      owner_(std::move(owner)) {
  if (!owner_ || !owner_->data()) {
    throw std::invalid_argument("StructureReader requires valid MDData owner");
  }
}

int StructureReader::Read() {
  if (!owner_ || !owner_->data()) {
    throw std::runtime_error("StructureReader lost MDData ownership");
  }
  owner_->resetEphemerals();
  resetCursor();
  return ReadData();
}

ICursor& StructureReader::cursor() {
  if (!cursor_) {
    throw std::runtime_error("Cursor not initialised");
  }
  return *cursor_;
}

void StructureReader::resetCursor(std::uint64_t offset) {
  auto src = source();
  if (!src) {
    throw std::runtime_error("Reader has no source");
  }
  cursor_ = src->cursor(offset);
  if (!cursor_) {
    throw std::runtime_error("Unable to create cursor from source");
  }
}

}  // namespace reader
