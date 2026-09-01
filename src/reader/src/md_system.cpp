#include "md_system.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "model/md_data.h"
#include "core/reader_factory.h"
#include "core/structure_reader.h"
#include "common/startup_phase_debug.h"
#include "md_data_owner.h"
#include "data_manager.h"

#ifdef READER_ENABLE_MPI
#include "mpi.h"
#include "global_structure_info.h"
#include "rbmd_parallel_until_locator.h"
#endif

namespace reader {

MDSystem::MDSystem(std::shared_ptr<MDDataOwner> owner)
    : owner_(std::move(owner)) {
  if (!owner_ || !owner_->data()) {
    throw std::invalid_argument("MDSystem requires valid MDData owner");
  }
}

MDSystem MDSystem::load(const std::string& file_path, bool use_mmap) {
  LoadOptions options;
  options.use_mmap = use_mmap;
  return load(file_path, options);
}

MDSystem MDSystem::load(const std::string& file_path, const LoadOptions& options) {
  if (file_path.empty()) {
    throw std::invalid_argument("Structure file path must not be empty");
  }

  // 使用 DataManager 中已存在的 MDData 实例，而不是创建新的实例
  // 这样确保 LammpsFullReader 填充的数据能被 MemoryScheduler 正确访问
  auto data = DataManager::getInstance().getMDData();
  if (!data) {
    throw std::runtime_error("DataManager MDData not initialized. Call DataManager::Initialize() first.");
  }

  auto owner = std::make_shared<MDDataOwner>(data);

  ReaderFactory factory;
  auto reader = factory.makeStructureReader(file_path, options.use_mmap, owner);
  if (!reader) {
    throw std::runtime_error("Failed to create structure reader");
  }
  rbmd::debug::StartupPhaseLog("reader.md_system.load.before_reader_read",
                               "file=" + file_path);
  if (reader->Read() != 0) {
    throw std::runtime_error("Failed to parse structure file: " + file_path);
  }
  rbmd::debug::StartupPhaseLog("reader.md_system.load.after_reader_read",
                               "file=" + file_path);
  
  // 在并行环境下，广播全局结构信息到所有进程
#ifdef READER_ENABLE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    rbmd::debug::StartupPhaseLog(
        "reader.md_system.load.before_broadcast_global_structure");
    GlobalStructureInfo& global_info = 
        RbmdParallelUntilLocator::GetInstance().GetRbmdParallelUntil()->_global_structure_info;
    BroadcastGlobalStructureInfo(global_info, 0, MPI_COMM_WORLD);
    rbmd::debug::StartupPhaseLog(
        "reader.md_system.load.after_broadcast_global_structure");
  }
#endif
  
  return MDSystem(std::move(owner));
}

std::shared_ptr<MDData> MDSystem::data() const noexcept {
  return owner_ ? owner_->data() : nullptr;
}

}  // namespace reader
