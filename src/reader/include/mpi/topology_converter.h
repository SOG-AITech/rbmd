#pragma once

#include "core/parsed_snapshot.h"
#include "data_manager/include/model/device_data.h"
#include "data_manager/include/structure_data/full_structure_data.h"

namespace reader::mpi {

/// Convert global topology lists into per-atom topology arrays (host side).
/// The output arrays store GID/tag and follow newton_bond=off visibility:
/// - bond: write to atom1 and atom2
/// - angle: write to atom1/atom2/atom3
/// - dihedral/improper: write to atom1/atom2/atom3/atom4
void ConvertGlobalToPerAtomTopology(const ParsedSnapshot& snapshot,
                                    FullStructureData& output,
                                    const DeviceData& device_data);

}  // namespace reader::mpi
