#include "rbmd_parallel_until_locator.h"

#include "data_manager.h"
#include "model/md_data.h"

void RbmdParallelUntilLocator::Release() {
  this->_rbmd_parallel_until.reset();
}
RbmdParallelUntilLocator& RbmdParallelUntilLocator::GetInstance() {
  static RbmdParallelUntilLocator instance;
  return instance;
}
std::shared_ptr<RbmdParallelUntil> RbmdParallelUntilLocator::GetRbmdParallelUntil() {
  if (nullptr == _rbmd_parallel_until) {
      _rbmd_parallel_until = std::make_shared<RbmdParallelUntil>();
    }
  return _rbmd_parallel_until;
  }