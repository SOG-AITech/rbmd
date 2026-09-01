#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include "core/parsed_snapshot.h"
#include "core/structure_reader.h"

namespace reader {

namespace mpi {
struct StreamCtx;
}  // namespace mpi

class LammpsFullReader : public StructureReader {
 public:
  LammpsFullReader(std::shared_ptr<ISource> src,
                   std::shared_ptr<IResourceLocator> locator,
                   std::string file_path,
                   std::shared_ptr<MDDataOwner> owner);
  ~LammpsFullReader() override = default;

 protected:
  int ReadData() override;
  void AllocateDataSpace(std::size_t num_atoms) override;

 private:
  enum class Section {
    None,
    Masses,
    PairCoeffs,
    BondCoeffs,
    AngleCoeffs,
    DihedralCoeffs,
    ImproperCoeffs,
    Atoms,
    Velocities,
    Bonds,
    Angles,
    Dihedrals,
    Impropers
  };

  int ParseHeader(std::string_view line);
  int ParseMass(std::string_view line);
  int ParsePairCoeff(std::string_view line);
  int ParseBondCoeff(std::string_view line);
  int ParseAngleCoeff(std::string_view line);
  int ParseDihedralCoeff(std::string_view line);
  int ParseImproperCoeff(std::string_view line);
  int ParseAtom(std::string_view line);
  int ParseVelocity(std::string_view line);
  int ParseBond(std::string_view line);
  int ParseAngle(std::string_view line);
  int ParseDihedral(std::string_view line);
  int ParseImproper(std::string_view line);
  Section DetectSection(std::string_view line) const;

  void ResetSnapshot();
  void PopulateDataManagers();
  void PopulateBox();
  void PopulateStructure();
  void PopulateStructureInfo();
#ifdef READER_ENABLE_MPI
  void PopulateGlobalStructureInfo();
  void CalculateAndPopulateTopologyParameters();
#endif
  void PopulateForceField();
  void PopulateSpecialCounts();
  void ValidateRequiredCoeffSectionsOrThrow() const;
  void LoadReplicateFactors();
  void LoadReplicateBondPeriodic();
  void ApplyGaussianVelocitiesIfRequested();
  bool ReplicateEnabled() const;
  std::size_t ReplicateCount() const;
  bool ReplicateRequiresImageFlags() const;
  void ValidateReplicateImageFlagsOrThrow() const;
  void ApplyReplicate();
  void StreamReplicatedSnapshot();

  enum class AtomStyle { Atomic, Charge, Full };

  AtomStyle DetectAtomStyle() const;
  void LoadSpecialBondWeights();
  bool IsEamForceField() const;
  void ResolveEamPotentialFile();

  static std::string Trim(std::string_view value);
  static bool IsComment(std::string_view value);

  Section current_section_ = Section::None;
  std::size_t expected_atoms_ = 0;
  std::size_t expected_bonds_ = 0;
  std::size_t expected_angles_ = 0;
  std::size_t expected_dihedrals_ = 0;
  std::size_t expected_impropers_ = 0;
  ParsedSnapshot snapshot_;
  mpi::StreamCtx* streaming_ctx_ = nullptr;
  bool streaming_mode_ = false;
  AtomStyle atom_style_ = AtomStyle::Full;
  std::string eam_potential_file_;
  std::multimap<rbmd::Id, rbmd::Id> bond_adjacency_;
  std::array<std::size_t, 3> replicate_factors_{1, 1, 1};
  bool replicate_bond_periodic_ = false;
  bool atom_image_flags_seen_ = false;
  bool atom_image_flags_missing_ = false;
  bool triclinic_box_detected_ = false;
  std::size_t parsed_velocity_count_ = 0;
  std::unordered_map<rbmd::Id, std::size_t> atom_index_by_id_;
  std::array<rbmd::Real, 3> special_bond_weights_{
      rbmd::Real{1.0}, rbmd::Real{1.0}, rbmd::Real{1.0}};
};

}  // namespace reader
