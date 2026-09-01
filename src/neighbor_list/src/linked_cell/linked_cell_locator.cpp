#include "linked_cell/linked_cell_locator.h"

#include "data_manager.h"
#include "model/md_data.h"
#include "output/include/Logger.hpp"

LinkedCellLocator& LinkedCellLocator::GetInstance() {
  static LinkedCellLocator instance;
  return instance;
}
std::shared_ptr<LinkedCell> LinkedCellLocator::GetLinkedCell() {
  if (nullptr == _linked_cell) {
    this->_linked_cell = std::make_shared<LinkedCell>();
    if (_has_pending_halo_cutoff) {
      _linked_cell->SetHaloCutoff(_pending_halo_cutoff);
    }
    this->_box = DataManager::getInstance().getMDData()->_box;
    auto short_edge =
        MIN(MIN(_box->_length[0], _box->_length[1]), _box->_length[2]);
    if (this->_linked_cell->_neighbor_cutoff >= short_edge) {
      Logger::Instance().info(
          "\033[31mError: neighbor cutoff must be less than the shortest side "
          "of the box.\033[0m");
      exit(0);
    }
    // RBL
    if (DataManager::getInstance().getConfigData()->Get<std::string>(
            "type", "hyper_parameters", "neighbor") == "RBL") {
      const auto _r_core =
          DataManager::getInstance().getConfigData()->Get<rbmd::Real>(
              "r_core", "hyper_parameters", "neighbor");
      if (_r_core >= _linked_cell->_cutoff) {
        Logger::Instance().error("\033[31mError r_core must be less than  the "
                     "cutoff.\033[0m");
        exit(0);
      }
      _linked_cell->_cell_count_within_cutoff = static_cast<rbmd::Id>(
          std::ceil(
              static_cast<double>(_linked_cell->_neighbor_cutoff / _r_core)));
    }
    _linked_cell->Build();
    _linked_cell->InitializeCells();
  }
  return this->_linked_cell;
}

void LinkedCellLocator::SetHaloCutoff(rbmd::Real halo_cutoff) {
  _pending_halo_cutoff = halo_cutoff;
  _has_pending_halo_cutoff = true;
  if (_linked_cell) {
    _linked_cell->SetHaloCutoff(halo_cutoff);
    _linked_cell->Build();
    _linked_cell->SyncHToD();
    _linked_cell->InitializeCells();
  }
}
