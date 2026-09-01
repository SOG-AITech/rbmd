#pragma once
#include "rbmd_parallel_until.h"
#define GET_RBMD_PARALLEL RbmdParallelUntilLocator::GetInstance().GetRbmdParallelUntil()

class RbmdParallelUntilLocator {
public:
  RbmdParallelUntilLocator(const RbmdParallelUntilLocator &) = delete;

  RbmdParallelUntilLocator &operator=(const RbmdParallelUntilLocator &) = delete;

  void Release();

private:
  RbmdParallelUntilLocator() {}

  ~RbmdParallelUntilLocator() = default;   // bug!

  std::shared_ptr<RbmdParallelUntil> _rbmd_parallel_until = nullptr;

public:
  static RbmdParallelUntilLocator &GetInstance();

  std::shared_ptr<RbmdParallelUntil> GetRbmdParallelUntil();
};
