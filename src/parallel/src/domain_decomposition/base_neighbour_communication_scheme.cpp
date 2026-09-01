#include "base_neighbour_communication_scheme.h"

#include <fstream>

#include "data_manager.h"

BaseNeighbourCommunicationScheme::BaseNeighbourCommunicationScheme(
    unsigned int commDimms, ZonalMethod* zonalMethod, bool pushPull)
    : _coversWholeDomain{false, false, false},
      _commDimms(commDimms),
      _zonalMethod(zonalMethod),
      _pushPull(pushPull) {
  this->_h_local_box = DataManager::getInstance().getMDData()->_box;
  if (_pushPull) {
    _haloExportForceImportNeighbours =
        new std::vector<std::vector<CommunicationPartner>>();
    _haloImportForceExportNeighbours =
        new std::vector<std::vector<CommunicationPartner>>();
    _leavingExportNeighbours =
        new std::vector<std::vector<CommunicationPartner>>();
    _leavingImportNeighbours =
        new std::vector<std::vector<CommunicationPartner>>();

    _haloExportForceImportNeighbours->resize(this->GetCommDims());
    _haloImportForceExportNeighbours->resize(this->GetCommDims());
    _leavingExportNeighbours->resize(this->GetCommDims());
    _leavingImportNeighbours->resize(this->GetCommDims());

    _neighbours = nullptr;
  } else {
    _haloExportForceImportNeighbours = nullptr;
    _haloImportForceExportNeighbours = nullptr;
    _leavingExportNeighbours = nullptr;
    _leavingImportNeighbours = nullptr;

    _neighbours = new std::vector<std::vector<CommunicationPartner>>();
    _neighbours->resize(this->GetCommDims());

  }
}

BaseNeighbourCommunicationScheme::~BaseNeighbourCommunicationScheme() {
  if (_pushPull) {
    delete _haloExportForceImportNeighbours;
    delete _haloImportForceExportNeighbours;
    delete _leavingExportNeighbours;
    delete _leavingImportNeighbours;
  } else {
    delete _neighbours;
  }
  delete _zonalMethod;
}

void printNeigbours(std::ofstream& stream,
                    std::vector<std::vector<CommunicationPartner>>& partners) {
  for (size_t i = 0; i < partners.size(); i++) {
    auto& vector = partners[i];
    stream << "neighbor dimension: " << i << std::endl;
    for (size_t j = 0; j < vector.size(); j++) {
      auto& partner = vector[j];
      stream << "Partner: " << j << std::endl;
      partner.print(stream);
    }
  }
}

void BaseNeighbourCommunicationScheme::printCommunicationPartners(
    std::string filename) const {
  std::ofstream checkpointfilestream;
  checkpointfilestream.open(filename.c_str());

  if (_pushPull) {
    checkpointfilestream << "haloExportForceImport:" << std::endl;
    printNeigbours(checkpointfilestream, *_haloExportForceImportNeighbours);
    checkpointfilestream << "haloImportForceExportNeighbours:" << std::endl;
    printNeigbours(checkpointfilestream, *_haloImportForceExportNeighbours);
    checkpointfilestream << "leavingExportNeighbours:" << std::endl;
    printNeigbours(checkpointfilestream, *_leavingExportNeighbours);
    checkpointfilestream << "leavingImportNeighbours:" << std::endl;
    printNeigbours(checkpointfilestream, *_leavingImportNeighbours);
  } else {
    checkpointfilestream << "neighbours:" << std::endl;
    printNeigbours(checkpointfilestream, *_neighbours);
  }
  checkpointfilestream.close();
}

void BaseNeighbourCommunicationScheme::SelectNeighbours(MessageType msgType,
                                                        bool import) {
  switch (msgType) {
    case LEAVING_ONLY:
      // leavingImport / leavingExport
      if (import)
        _neighbours = _leavingImportNeighbours;
      else
        _neighbours = _leavingExportNeighbours;
      break;
    case HALO_COPIES:
      // haloImport / haloExport
      if (import)
        _neighbours = _haloImportForceExportNeighbours;
      else
        _neighbours = _haloExportForceImportNeighbours;
      break;
    case FORCES:
      // forceImport / forceExport
      if (import)
        _neighbours = _haloExportForceImportNeighbours;
      else
        _neighbours = _haloImportForceExportNeighbours;
      break;
    case LEAVING_AND_HALO_COPIES:
      //     Log::global_log->error() << "WRONG type in selectNeighbours - this
      //     should not be used for push-pull-partners "
      //                            "selectNeighbours method"
      //                         << std::endl;
      // Simulation::exit(1);
      exit(1);
      break;
  }
}
