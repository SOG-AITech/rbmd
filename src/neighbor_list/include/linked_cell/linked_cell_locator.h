#pragma once

#include <memory>

#include "linked_cell.h"

class LinkedCellLocator {
 public:
  LinkedCellLocator(const LinkedCellLocator &) = delete;

  LinkedCellLocator &operator=(const LinkedCellLocator &) = delete;

 private:
  LinkedCellLocator() {}

  ~LinkedCellLocator() = default;

  std::shared_ptr<LinkedCell> _linked_cell = nullptr;
  std::shared_ptr<Box> _box;   // TODO 可能不太适合 待重构
  rbmd::Real _pending_halo_cutoff = 0;
  bool _has_pending_halo_cutoff = false;

 public:
  static LinkedCellLocator &GetInstance();

  std::shared_ptr<LinkedCell> GetLinkedCell();
  void SetHaloCutoff(rbmd::Real halo_cutoff);
};
