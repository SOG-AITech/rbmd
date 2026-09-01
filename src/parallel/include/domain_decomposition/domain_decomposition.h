#pragma once
#include "mpi.h"

#include <memory>

#include "communication_partner.h"
#include "data_manager/include/model/box.h"
#include "halo_region.h"
#include "neighbor_list/include/linked_cell/linked_cell.h"
class IndirectNeighbourCommunicationScheme;
class DomainDecomposition {
 public:
  explicit DomainDecomposition(int current_rank, int total_ranks);
  ~DomainDecomposition() = default;
  MPI_Comm _mpi_comm = MPI_COMM_WORLD;

  int _coords[3]{};                 //!< MPI进程网格中进程的坐标
  std::array<int, 3> _grid_size{};  //!< MPI进程网格的每个维度中的进程数
  //! 当前进程(Host or Device)的编号
  int _current_rank = 0;
  //! 总进程数
  int _total_ranks = 1;

  // local!!!!
  Box* h_box;
  // local!!!!

  void SetCommunicationScheme(const std::string& comm_scheme,
                              const std::string& zonal_method);

  void InitCommunicationPartners(rbmd::Real cutoff);

  void GetSubDomainBox(
      Box* h_box);  // 应该是初始化的时候调用一次  要整理盒子什么时候创建的逻辑
  std::vector<CommunicationPartner> GetNeighboursFromHaloRegion(
      const HaloRegion& haloRegion);  // TODO 放在这里不合适吧

  //! may be public
  std::shared_ptr<Box> _h_global_box{};  // global box
  //! may be public
  std::shared_ptr<IndirectNeighbourCommunicationScheme>
      _neighbour_coneighbour_communication_scheme;

 private:
  // 在代码封装locator的情况下，如何进行MPI的初始化是个问题
  void InitMPIGridDims(rbmd::Real interaction_length);
  void ValidateResolvableGrid(rbmd::Real interaction_length) const;
  std::shared_ptr<DeviceData> _device_data;
  // 处理覆盖整个域的逻辑都放在proctect里面。 KDD也是避免不了这个的


public:
  //! localbox
  void HandleDomainLeavingAtoms(unsigned dim, LinkedCell* linked_cell) const;

  void PopulateHaloLayerWithCopies(unsigned dim, LinkedCell* linked_cell) const;

  void ExchangeMoleculesMPI(LinkedCell*moleculeContainer);

  // 输出所有进程的localbox信息到同一个txt文件
  void OutputLocalBoxInfoToFile();

  // TODO handleForceExchange 比较困难，暂时不实现
};
