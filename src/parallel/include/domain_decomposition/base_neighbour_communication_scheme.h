#pragma once
#include <vector>
#include "common/rbmd_define.h"
#include "neighbor_list/include/linked_cell/linked_cell.h"
#include "communication_partner.h"
#include "zonal_method.h"

class DomainDecomposition;

// TODO DomainDecomposition是否可行呢？
class BaseNeighbourCommunicationScheme {
 public:
  BaseNeighbourCommunicationScheme() = default;

  BaseNeighbourCommunicationScheme(unsigned int commDimms,
                                   ZonalMethod* zonalMethod, bool pushPull);

  virtual ~BaseNeighbourCommunicationScheme();

  BaseNeighbourCommunicationScheme(BaseNeighbourCommunicationScheme const&) =
      delete;

  void operator=(BaseNeighbourCommunicationScheme const& other) = delete;

  virtual void prepareNonBlockingStageImpl(
      LinkedCell* linked_cell, unsigned int stageNumber, MessageType msgType,
      bool removeRecvDuplicates,
      DomainDecomposition*
          domainDecomp) = 0;  //! NOTE 并行暂时不用，应该是用于driect或kdd

  virtual void finishNonBlockingStageImpl(
      LinkedCell* linked_cell, unsigned int stageNumber, MessageType msgType,
      bool removeRecvDuplicates,
      DomainDecomposition*
          domainDecomp) = 0;  //! NOTE 并行暂时不用，应该是用于driect或kdd

  virtual void exchangeMoleculesMPI(LinkedCell* moleculeContainer,
                            DomainDecomposition* domainDecomp) = 0;

  /*!
   * @brief
   * 在特定的并行计算框架下（如基于MPI的领域分解），"_commDimms"变量表示所需进行的连续通信步骤数，同时也是"_neighbours"成员变量（一个数组）的外层维度大小
   * @return
   */
  unsigned int GetCommDims() const { return _commDimms; }

  void setCoverWholeDomain(unsigned int d, bool covers) {
    _coversWholeDomain[d] = covers;
  }

  virtual void initCommunicationPartners(rbmd::Real cutoffRadius,
                                         DomainDecomposition* domainDecomp) = 0;

  virtual std::vector<int> get3StageNeighbourRanks() = 0;

  virtual std::vector<int> getFullShellNeighbourRanks() {
    std::vector<int> neighbourRanks;
    for (auto& _fullShellNeighbour : _fullShellNeighbours) {
      neighbourRanks.push_back(_fullShellNeighbour.getRank());
    }
    return neighbourRanks;
  }

  virtual size_t getDynamicSize() {
    size_t totSize = 0;
    // _fullShellNeighbours
    totSize += sizeof(*this);
    // std::cout << "pre FSN:" << totSize;
    totSize += _fullShellNeighbours.capacity() * sizeof(CommunicationPartner);
    for (CommunicationPartner& neigh : _fullShellNeighbours) {
      totSize += neigh.getDynamicSize();
      // std::cout << "FSN:" << neigh.getDynamicSize();
    }
    // std::cout << "post FSN/pre neigh:" << totSize;
    totSize += (*_neighbours).capacity() * sizeof(CommunicationPartner);
    for (auto& neighList : (*_neighbours)) {
      for (auto& neigh : neighList) {
        totSize += neigh.getDynamicSize();
      }
    }
    return totSize;
  }

  void printCommunicationPartners(std::string filename) const;

  void setSequentialFallback(bool useSequentialFallback) {
    _useSequentialFallback = useSequentialFallback;
  }

 protected:
  std::shared_ptr<Box> _h_local_box{};
  //! vector of neighbours. The first dimension should be of size getCommDims().
  std::vector<std::vector<CommunicationPartner> >* _neighbours;

  // -------------------------------------------------------------------------
  std::vector<std::vector<CommunicationPartner> >*
      _haloExportForceImportNeighbours;
  std::vector<std::vector<CommunicationPartner> >*
      _haloImportForceExportNeighbours;
  std::vector<std::vector<CommunicationPartner> >* _leavingExportNeighbours;
  std::vector<std::vector<CommunicationPartner> >* _leavingImportNeighbours;

  void SelectNeighbours(MessageType msgType, bool import);

  // -------------------------------------------------------------------------

  //! flag, which tells whether a processor covers the whole domain along a
  //! dimension if true, we will use the methods provided by the base class for
  //! handling the respective dimension, instead of packing and unpacking
  //! messages to self
  bool _coversWholeDomain[3];

  unsigned int _commDimms;

  //! zonal method (FullShell, HalfShell, ...)
  ZonalMethod* _zonalMethod;

  //! list of all neighbours (non-squeezed)
  std::vector<CommunicationPartner> _fullShellNeighbours;

  bool _pushPull;

  bool _useSequentialFallback{true};
};