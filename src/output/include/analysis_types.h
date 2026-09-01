#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "model/box.h"

struct AnalysisFrame {
  rbmd::Id timestep{};
  Box box_snapshot;
  std::vector<rbmd::Id> ids{};
  std::vector<rbmd::Id> types{};
  std::vector<rbmd::Real> ux{};
  std::vector<rbmd::Real> uy{};
  std::vector<rbmd::Real> uz{};
  std::vector<rbmd::Real> vx{};
  std::vector<rbmd::Real> vy{};
  std::vector<rbmd::Real> vz{};
};

struct AnalysisHostFrame {
  rbmd::Id* h_ids{};
  rbmd::Id* h_types{};
  rbmd::Real* h_ux{};
  rbmd::Real* h_uy{};
  rbmd::Real* h_uz{};
  rbmd::Real* h_vx{};
  rbmd::Real* h_vy{};
  rbmd::Real* h_vz{};
  rbmd::Id timestep{};
  size_t num_atoms{};
  Box box_snapshot;
  EVENT_T copy_complete_event{};
};

struct RdfGroupConfig {
  std::string name{};
  std::vector<rbmd::Id> types{};
};

struct RdfGroupPairConfig {
  std::string lhs{};
  std::string rhs{};
};

struct RdfConfig {
  bool enabled{false};
  rbmd::Id interval{0};
  rbmd::Real radius{0};
  rbmd::Real dr{0};
  rbmd::Id statistics_rdf_steps{0};
  std::vector<std::vector<rbmd::Id>> atoms_pair{};
  std::vector<RdfGroupConfig> groups{};
  std::vector<RdfGroupPairConfig> group_pairs{};
};

struct MsdConfig {
  bool enabled{false};
  rbmd::Id interval{0};
  rbmd::Id start_step{0};
  rbmd::Id end_step{0};
};

struct VacfConfig {
  bool enabled{false};
  rbmd::Id interval{0};
  rbmd::Id start_step{0};
  rbmd::Id end_step{0};
};
