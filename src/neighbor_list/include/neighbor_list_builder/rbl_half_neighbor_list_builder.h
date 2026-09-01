#pragma once
#include "rbl_full_neighbor_list_builder.h"

// 必须采样一圈，只能找出全部邻居
class RblHalfNeighborListBuilder : public RblFullNeighborListBuilder {
  bool SupportsCandidateCache() const override { return false; }
  void EstimateNeighborsList() override;
  rbmd::Id GenerateNeighborsList() override;
};
