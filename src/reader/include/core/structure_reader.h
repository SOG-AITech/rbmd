#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "core/reader_base.h"
#include "md_data_owner.h"
#include "io/icursor.h"
class MDData;

namespace reader {


class StructureReader : public ReaderBase {
public:
  StructureReader(std::shared_ptr<ISource> source,
                  std::shared_ptr<IResourceLocator> locator,
                  std::shared_ptr<MDDataOwner> owner,
                  std::string file_path);
  ~StructureReader() override = default;

  int Read() override;

protected:
  virtual int ReadData() = 0;
  virtual void AllocateDataSpace(std::size_t num_atoms) = 0;

  std::shared_ptr<MDData> data() const noexcept { return owner_ ? owner_->data() : nullptr; }
  MDDataOwner& dataOwner() const noexcept { return *owner_; }
  ICursor& cursor();
  bool hasCursor() const noexcept { return static_cast<bool>(cursor_); }
  void resetCursor(std::uint64_t offset = 0);

private:
  std::shared_ptr<MDDataOwner> owner_;
  std::unique_ptr<ICursor> cursor_;
};

}  // namespace reader
