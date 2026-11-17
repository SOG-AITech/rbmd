#pragma once
#include <array>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "../../common/rbmd_define.h"
#include "../../common/types.h"
#include "force.h"
#include "kspace_calculator.h" //

#include "model/box.h"
#include "neighbor_list/include/neighbor_list/neighbor_list.h"
#include "neighbor_list/include/neighbor_list_builder/full_neighbor_list_builder.h"
class RblFullNeighborListBuilder;
class CVFF : public Force
{
public:
  CVFF();
  virtual ~CVFF();

  void Init() override;
  void  Execute() override;
  void EvaluatePotentialEnergy() override;

  void ComputeLJCutCoulForce();
  void ComputeLJVerlet();
  void ComputeLJRBL();
  void LJCoulSOGInit();
  void ComputeLJSOG();
  void ComputeLJCoulEnergy();

  void SumForces(STREAM_T stream = nullptr);

  void ComputeBondForce(STREAM_T stream = nullptr,
                        bool defer_finalize = false); //Harmonic
  void ComputeAngleForce(STREAM_T stream = nullptr,
                         bool defer_finalize = false); //Harmonic
  void ComputeDihedralForce(STREAM_T stream = nullptr,
                            bool defer_finalize = false); //
  void DihedralOPLS(STREAM_T stream = nullptr,
                    bool defer_finalize = false);    //OPLS
  void DihedralHarmonic(STREAM_T stream = nullptr,
                        bool defer_finalize = false);//Harmonic
  void DihedralFourier(STREAM_T stream = nullptr,
                       bool defer_finalize = false);//Fourier
  void ComputeImproperForce(STREAM_T stream = nullptr,
                            bool defer_finalize = false); //
  void ImproperHarmonic(STREAM_T stream = nullptr,
                        bool defer_finalize = false);//Harmonic
  void ImproperCVFF(STREAM_T stream = nullptr,
                    bool defer_finalize = false);    //CVFF

private:
  bool UsePerAtomBondPath() const;
  bool UsePerAtomAnglePath() const;
  bool UsePerAtomDihedralPath() const;
  bool UsePerAtomImproperPath() const;
  void InitializeRblCoulombLookup();
  bool ForceOverlapEnabled() const;
  void ConfigureForceOverlap();
  bool CanOverlapVerletRbe() const;
  bool CanOverlapRblRbe() const;
  bool RblEnergyEnabled() const;
  void InitializeForceOverlapResources();
  void DestroyForceOverlapResources();
  void ExecuteVerletRbeOverlap();
  void ExecuteRblRbeOverlap();
  void PrepareLJVerletNeighborList();
  void LaunchLJVerletForce(STREAM_T stream);
  void FinalizeLJVerletForce();
  void PrepareRblNeighborLists();
  void LaunchRblForce(STREAM_T stream);
  void FinalizeRblCorrection();
  void LaunchRblEnergy(STREAM_T stream);
  void FinalizeRblEnergy();
  void FinalizeBondedForces();
  void WriteForceSnapshot() const;
  void LaunchOverlapBondedForces();
  void QueueOverlapPairEnergy();
  void StartOverlapEnergyReduction();
  void FinishOverlapEnergyReduction();
  void StartOverlapRblCorrection();
  void FinishOverlapRblCorrection();
  void RecordOverlapTiming(const char* category, EVENT_T begin,
                           EVENT_T end) const;

  std::shared_ptr<RblFullNeighborListBuilder> _rbl_neighbor_list_builder;
  std::shared_ptr<BaseNeighborListBuilder> _neighbor_list_builder;
  std::shared_ptr<NeighborList> _rbl_list;
  std::shared_ptr<NeighborList> _list;

  //energy
  rbmd::Real _e_vdwl = 0;
  rbmd::Real _e_coul = 0;
  rbmd::Real _e_special_coul = 0;
  rbmd::Real _e_self_energy = 0;
  rbmd::Real _e_kspace = 0;
  rbmd::Real _e_bond = 0;
  rbmd::Real _e_angle = 0;
  rbmd::Real _e_dihedral = 0;
  rbmd::Real _e_improper = 0;
  rbmd::Real _e_pe = 0;

  rbmd::Real _e_vdwl_rbl = 0;
  rbmd::Real _e_coul_rbl = 0;
  rbmd::Real _e_pe_rbl = 0;

  //RBL
  std::string  _energy_rbl_flag = "yes"; //default
  std::string _neighbor_type;
  std::string _coulomb_type;
  rbmd::Real _cut_off;

  rbmd::Real _corr_value_x = 0;
  rbmd::Real _corr_value_y = 0;
  rbmd::Real _corr_value_z = 0;
  rbmd::Id _lj_num_types = 0;
  bool _geometric_lj_mixing = false;
  rbmd::Id _global_num_atoms = 0;
  thrust::device_vector<rbmd::Real> _d_lj_force12;
  thrust::device_vector<rbmd::Real> _d_lj_force6;
  thrust::device_vector<rbmd::Real> _d_rbl_coulomb_coeff;
  thrust::device_vector<rbmd::Real> _d_rbl_core_taylor;
  thrust::device_vector<rbmd::Real> _d_rbl_force_sum;
  thrust::device_vector<rbmd::Real> _d_total_evdwl;
  thrust::device_vector<rbmd::Real> _d_total_ecoul;
  thrust::device_vector<rbmd::Real> _d_total_ebond;
  thrust::device_vector<rbmd::Real> _d_total_eangle;
  thrust::device_vector<rbmd::Real> _d_total_edihedral;
  thrust::device_vector<rbmd::Real> _d_total_eimproper;
  thrust::device_vector<unsigned long long> _d_total_cbond;
  thrust::device_vector<unsigned long long> _d_total_cangle;
  thrust::device_vector<unsigned long long> _d_total_cdihedral;
  thrust::device_vector<unsigned long long> _d_total_cimproper;
  std::array<rbmd::Real, 3> _h_rbl_force_sum{};
  std::array<rbmd::Real, 6> _rbl_core_taylor{};
  rbmd::Real _rbl_table_r2_min = 0;
  rbmd::Real _rbl_table_scale = 0;
  rbmd::Id _rbl_table_intervals = 0;

  //kspace
  rbmd::Real _qqr2e;
  rbmd::Real _alpha;
  rbmd::Real _rbsog_b;
  rbmd::Real _rbsog_sigma;
  rbmd::Real _rbsog_omega;
  rbmd::Real _rbsog_Mmax;
  rbmd::Real _w0 ;
  thrust::device_vector<rbmd::Real> _d_taylor_coeff;
  std::unique_ptr<KSpaceCalculator> _kspace_calculator;

  // Separate pinned regions: correction[3], pair energies[2], bonded energies[4].
  // Each producer stream finishes its D2H copy before its ready event; MPI owns
  // only the corresponding region until that request has completed.
  rbmd::Real* _h_overlap_values = nullptr;
#ifdef USE_MPI
  MPI_Request _correction_request = MPI_REQUEST_NULL;
  MPI_Request _energy_request = MPI_REQUEST_NULL;
#endif
  bool _force_overlap_enabled = false;
  bool _bonded_overlap_enabled = true;
  bool _force_overlap_resources_ready = false;
  STREAM_T _pair_stream{};
  STREAM_T _kspace_stream{};
  STREAM_T _bonded_stream{};
  STREAM_T _angle_stream{};
  STREAM_T _dihedral_stream{};
  STREAM_T _improper_stream{};
  STREAM_T _join_stream{};
  EVENT_T _atoms_ready{};
  EVENT_T _rho_ready{};
  EVENT_T _rbl_correction_ready{};
  EVENT_T _pair_begin{};
  EVENT_T _pair_done{};
  EVENT_T _kspace_begin{};
  EVENT_T _kspace_done{};
  EVENT_T _bonded_begin{};
  EVENT_T _angle_done{};
  EVENT_T _dihedral_done{};
  EVENT_T _improper_done{};
  EVENT_T _bonded_done{};
  EVENT_T _forces_joined{};
};
