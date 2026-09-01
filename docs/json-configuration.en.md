# RBMD JSON Configuration Reference

[中文](json-configuration.zh_CN.md) · [Build and Run Guide](build.en.md)

This document describes the JSON keys read by the current code. JSON is parsed with `nlohmann::ordered_json` and comments are allowed. Relative paths are generally resolved from the directory that contains the JSON file.

## Quick Takeaways

- If `outputs.trajectory_out` is omitted, trajectory output is disabled by default and `rbmd.trj` is not created or truncated.
- When `outputs.trajectory_out` is present, `enabled` defaults to `true`. When trajectory output is enabled, `interval` is required; omitting it raises an error when it is read during initialization.
- Set `outputs.trajectory_out.enabled = false` explicitly to disable the trajectory module.
- Setting `outputs.trajectory_out.interval` to `0` or a negative number also prevents frames from being written, but this is sampling-layer compatibility behavior; `enabled: false` is clearer.
- If `outputs.rdf_out` is omitted, RDF is not configured and no post-processing is triggered by RDF.
- When `rdf_out`, `msd_out`, or `vacf_out` is present, `PostprocessRunner` calls `scripts/postprocess_freud.py` on the root rank by default, unless `outputs.analysis_postprocess.enabled = false` is set.
- The current `MDApplication` pipeline directly attaches only `TrajectoryOutput`. `AnalysisOutput` exists but is not added to `CompositeOutput` in the main pipeline. RDF/MSD/VACF therefore mainly rely on the final Python post-processing step and `rbmd.trj`.

## Top-level Structure

```json
{
  "init_configuration": {},
  "hyper_parameters": {},
  "execution": {},
  "outputs": {}
}
```

| Top-level key | Type | Required | Description |
| --- | --- | --- | --- |
| `init_configuration` | object | Yes | Initial structure input configuration. |
| `hyper_parameters` | object | Yes | Force field, neighbor-list, Coulomb, and extended-function settings. |
| `execution` | object | Yes | Ensemble, integration, temperature/pressure control, and step settings. |
| `outputs` | object | Yes | Thermodynamic, trajectory, analysis, and post-processing output settings. |

## `init_configuration`

### `init_configuration.read_data`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `file` | string | Yes | `"peo+Li.data"` | Path to a LAMMPS data file. Relative paths are resolved from the JSON file directory. |
| `unit` | string | Yes | `"LJ"`, `"REAL"`, `"METAL"` | Unit system; controls unit conversion. |
| `atom_style` | string | Yes | `"atomic"`, `"charge"`, `"full"` | Atom style; affects structure data and parallel communication fields. |
| `velocity_type` | string | No | `"GAUSS"` | If the data file has no `Velocities` section, Gaussian/Maxwell initial velocities are generated from `execution.temperature[0]`. Existing velocities are not overwritten. |
| `velocity_seed` | int | No | `12345` | Random seed for `velocity_type: "GAUSS"`. The default is `12345`; `seed` is also accepted for compatibility. |
| `replicate` | array<int, 3> | No | `[2, 2, 1]` | Replicates the system in x/y/z after reading the data file. All three values must be positive integers. |
| `replicate_dims` | array<int, 3> | No | `[2, 2, 1]` | Alias for `replicate`. If both are present, `replicate` takes precedence. |
| `replicate_bond_periodic` | bool | No | `true` | Whether bonds are treated periodically while replicating. |
| `bond_periodic` | bool | No | `true` | Compatibility spelling for `replicate_bond_periodic`. |
| `bond/periodic` | bool | No | `true` | Compatibility spelling for `replicate_bond_periodic`. |

Notes:

- Older examples contain `init_configuration.inbuild`, but the current `ReadMDData()` path reads only `read_data.file`; built-in structure initialization is not used.

### `replicate` Support Scope

`replicate` is a reader-level structure replication feature and does not branch on `hyper_parameters.force_field.type`. It first reads the LAMMPS data file, then expands the box, atom coordinates, and related structure data in x/y/z. The selected `force_field.type` determines which resulting data are consumed by force calculations.

| `force_field.type` | Typical `atom_style` | Current replication status | Description |
| --- | --- | --- | --- |
| `CVFF` | `full` | Full topology supported | Replication copies `atoms`, `velocities`, `bonds`, `angles`, `dihedrals`, and `impropers`, then rebuilds 1-2/1-3/1-4 `special_bonds` data. PEO 2×1×1 has been verified with MPI. |
| `LJ/CUT` | `atomic` | Non-bonded systems supported | Replication copies atoms, velocities, box, masses, and pair coefficients. The `LJ/CUT` force path does not consume topology. |
| `LJ/CUT/COUL/LONG` | `charge` or `full` | Charge, pair, and box data supported | Replication copies atoms, velocities, charges, box, masses, and pair coefficients, and updates dimensions required by k-space. This force path currently does not use `bonds`/`angles`/`dihedrals`/`impropers` or apply `special_bonds`. For LAMMPS-style 1-2/1-3/1-4 non-bonded scaling, use the currently topology-enabled `CVFF` path or add special-bond kernel support to `LJ/CUT/COUL/LONG`. |
| `EAM` | `atomic` | Statically checked | EAM consumes only atom type, coordinates, box, neighbor list, and `potential_file`; replication copies these data. `EAM::Init()` resolves `potential_file` relative to the JSON directory. This document does not record a dedicated EAM replication run verification. |

Constraints:

- Only orthogonal boxes are supported for replication. A triclinic box with `xy`/`xz`/`yz` tilt raises an error.
- For `atom_style full` or data with molecular topology, every atom in the `Atoms` section must include image flags `ix iy iz`.
- `atom_style full` only means that the data file includes molecule ID, charge, and topology fields. Whether bonded terms or special-bond scaling are calculated is still determined by the force kernel for `force_field.type`.

## `hyper_parameters`

### `hyper_parameters.force_field`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `type` | string | Yes | `"CVFF"`, `"LJ/CUT"`, `"LJ/CUT/COUL/LONG"`, `"EAM"`, `"Tersoff"` | Force-field type; selects the force implementation and memory scheduler. |
| `bond_type` | string | CVFF-related | `"harmonic"` | CVFF bond potential type. |
| `angle_type` | string | CVFF-related | `"harmonic"` | CVFF angle potential type. |
| `dihedral_type` | string | CVFF-related | `"harmonic"`, `"opls"` | CVFF dihedral type; used when reading data and scheduling CVFF parameters. |
| `improper_type` | string | Optional for CVFF | `"harmonic"`, `"cvff"` | Improper type; read when improper data exist. |
| `potential_file` | string | EAM/Tersoff-related | `"Cu.eam.alloy"` | Potential-file path. EAM and Tersoff resolve it relative to the JSON directory. |
| `potential_elements` | string | Tersoff-related | `"Si"` | Tersoff element mapping string. |

### `hyper_parameters.neighbor`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `type` | string | Yes | `"Verlet-List"`, `"RBL"` | Neighbor-list type. |
| `cut_off` | number | Yes | `12.0` | Cutoff radius; read by force, domain-decomposition, and linked-cell code. |
| `skin` | number | No | `1.0` | Neighbor-list skin. The default is `0`; a positive value enables caching for Verlet-List or RBL. |
| `interval` | integer | Optional for Tersoff | `10` | Tersoff-related update interval. |
| `r_core` | number | Required when `type = "RBL"` | `6.0` | RBL core radius. |
| `neighbor_sample_num` | integer | Required when `type = "RBL"` | `100` | RBL sample count. |
| `energy_rbl_flag` | string | Required when `type = "RBL"` | `"yes"`, `"no"` | Selects the RBL energy path; the code checks this string. |

#### RBL Configuration and `skin` Reuse

RBL supports `skin` reuse. It does not cache a complete shell neighbor list over `cut_off + skin`; instead it caches reference coordinates, linked-cell bins, MPI halos, and the forward plan. A reuse step updates ghost coordinates and regenerates the core and shell from current coordinates:

- Core neighbors with `r <= r_core` are written in full.
- Shell neighbors with `r_core < r <= cut_off` are resampled from the current step and the global atom-ID pair.
- Candidates with `r > cut_off` are ignored.

Consequently, RBL does not freeze the same set of shell samples across timesteps and does not need to retain a complete Verlet shell candidate list. It therefore usually uses less neighbor-list device memory than Verlet-List.

```json
"neighbor": {
  "type": "RBL",
  "cut_off": 9.0,
  "skin": 1.0,
  "r_core": 6.0,
  "neighbor_sample_num": 100,
  "energy_rbl_flag": "no"
}
```

RBL constraints are `skin >= 0`, `0 < r_core < cut_off`, and `neighbor_sample_num > 0`. Neighbors within `r_core` are evaluated completely, while the shell from `r_core` to `cut_off` is sampled. Choose `energy_rbl_flag: "yes"` only when the RBL energy path is required. Setting `skin = 0` or omitting it disables reuse and restores complete communication, binning, and construction each time. With a positive skin, reuse continues only while the maximum accumulated displacement on every MPI rank is at most `skin / 2`; otherwise a global rebuild is performed.

Verlet-List is configured as follows:

```json
"neighbor": {
  "type": "Verlet-List",
  "cut_off": 9.0,
  "skin": 1.0
}
```

For both neighbor-list types, changing particle order, particle count, box, neighbor cutoff, or another cache condition also causes a global rebuild. Use the following environment variables to inspect reuse:

```bash
RBMD_DEBUG_NEIGHBOR_SKIN=1 mpirun -np 4 ./rbmd -j input.json
```

Debug records are written to `logs/debug/neighbor_skin_cache_rank*_pid*.csv`. `action=reuse` indicates reuse; `action=rebuild` indicates rebuilding. Set `RBMD_DEBUG_NEIGHBOR_SKIN_EVERY=10` to reduce the record frequency.

### `hyper_parameters.coulomb`

The complete `hyper_parameters.coulomb` object may be omitted. Omitting it disables long-range Coulomb configuration.

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `type` | string | Required when `coulomb` is present | `"RBE"`, `"RBSOG"`, `"EWALD"` | Coulomb algorithm type. |
| `accuracy` | number | Required for automatic RBE/EWALD alpha | `1.0e-4` | Relative accuracy used to derive alpha automatically. |
| `alpha` | number | Optional for RBE/EWALD | `0.04333893` | Manual alpha. When present, it takes precedence over `accuracy`. |
| `kmax` | array<int, 3> | Optional for RBE energy or EWALD | `[18, 34, 22]` | Manually specified k-space range; must contain exactly three integers. |
| `coulomb_sample_num` | integer | Required for RBE/RBSOG | `2000` | RBE/RBSOG sample count. |
| `energy_rbe_flag` | string | Required for RBE | `"yes"`, `"no"` | Must be defined for RBE. With `"yes"`, k-space energy parameters are prepared. |
| `rbsog_level` | integer | Required for RBSOG | `0` to `4` | RBSOG preset level; only `[0, 4]` is accepted. |

RBSOG notes:

- RBSOG no longer accepts the legacy keys `alpha`, `accuracy`, `rbsog_b`, `rbsog_sigma`, `rbsog_Mmax`, and `rbsog_Kcut`.
- RBSOG accepts only `rbsog_level`; the code derives its internal parameters from this level.
- A higher `rbsog_level` has a lower target error and higher accuracy, but incurs more computation. Use `4` for the highest accuracy.

| `rbsog_level` | Target error | `Mmax` |
| --- | --- | --- |
| `0` | `2.289e-3` | `6` |
| `1` | `1.158e-4` | `16` |
| `2` | `1.142e-5` | `30` |
| `3` | `5.583e-8` | `64` |
| `4` | `3.389e-11` | `102` |

### `hyper_parameters.extend`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `fix_shake` | bool | No | `true` | Enables the SHAKE-related path; read by several integration and temperature/pressure-control modules. |
| `special_bonds` | array<number, 3> | No | `[0, 0, 0]` | CVFF special-bonds scale factors for 1-2/1-3/1-4 relationships. |

### `hyper_parameters.shift_value`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `shift_value` | number | Optional for Tersoff | `0.0` | Displacement/translation-related parameter read by Tersoff, placed directly under `hyper_parameters`. |

## `execution`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `ensemble` | string | Yes | `"NVE"`, `"NVT"`, `"NPT"` | Main ensemble type. |
| `temperature` | array<number> | NVT/NPT-related | `[298.0, 298.0, 100]` | Temperature-control parameters, typically initial temperature, target temperature, and damping/time constant. |
| `pressure` | array<number> | NPT-related | `[1.0, 1.0, 1000]` | Pressure-control parameters, typically initial pressure, target pressure, and damping/time constant. |
| `temp_ctrl_type` | string | NVT/NPT-related | `"RESCALE"`, `"BERENDSEN"`, `"LANGEVIN"`, `"NOSE_HOOVER"` | Thermostat type. NVT supports these four types. |
| `press_ctrl_type` | string | NPT-related | `"BERENDSEN"`, `"NOSE_HOOVER"` | Barostat type. NPT requires matched thermostat/barostat combinations. |
| `timestep` | number | Yes | `1.0` | Time step. |
| `num_steps` | number/integer | Yes | `3000` | Number of simulation steps. The current `KeepGoing()` uses `<= num_steps`, so the loop includes step 0. |
| `integration_type` | string | Yes | `"vv"`, `"bm"` | Integration scheme. `"bm"` additionally reads `par_a` and `par_b`. |
| `par_a` | number | Required when `integration_type = "bm"` | `0.1` | BM integration parameter. |
| `par_b` | number | Required when `integration_type = "bm"` | `0.1` | BM integration parameter. |
| `group` | string | Optional for temperature/pressure control | `"all"` | Berendsen/Nose-Hoover control group. If omitted, the default group logic is used. |
| `com_bias` | string | Optional for temperature/pressure control | `"yes"`, `"no"` | `"yes"` enables center-of-mass velocity bias correction. |
| `momentum_control` | object | No | See below | Enables momentum control. |

### `execution.momentum_control`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `interval` | integer | Yes | `100` | Momentum-control interval. When `0`, the execution function returns immediately. |
| `linear` | array<int, 3> | No | `[1, 1, 1]` | Whether to remove linear momentum in x/y/z. The presence of this key enables the linear branch. |
| `angular` | bool | No | `false` | Whether to remove angular momentum. Defaults to `false`. |
| `rescale` | bool | No | `false` | Whether to rescale velocities to the original kinetic energy after removing momentum. Defaults to `false`. |
| `group` | string | Yes | `"all"` | Target group. An empty string falls back to `"all"`. |

## `outputs`

### `outputs.thermo_out`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `interval` | integer | Yes | `100` | Thermodynamic-output interval; it is read by force and temperature/pressure-control code. |

### `outputs.trajectory_out`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `enabled` | bool | No | `false` | Trajectory-output switch. It applies only when the `trajectory_out` object is present; within the object, the default is `true`. When `false`, `interval` is not read and no trajectory file is created. |
| `interval` | integer | Required when enabled | `10` | Trajectory sampling interval. The output filename is fixed as `rbmd.trj`. |

The shortest way to disable trajectory output is to omit the entire `trajectory_out` object:

```json
"outputs": {
  "thermo_out": {
    "interval": 100
  }
}
```

You can also retain the object and disable it explicitly:

```json
"outputs": {
  "trajectory_out": {
    "enabled": false
  }
}
```

Enable trajectory output and control its sampling interval:

```json
"outputs": {
  "trajectory_out": {
    "enabled": true,
    "interval": 100
  }
}
```

`interval = 1` writes every step; `interval = 100` writes every 100 steps. An `interval` of `0` or less makes `TrajectoryOutput::Execute()` return without writing frames, but omitting `trajectory_out` or using `enabled: false` is the recommended explicit form. Do not disable trajectory output if the later Python post-processing for `rdf_out` is needed, because it reads `rbmd.trj` by default.

### `outputs.rdf_out`

When the full `rdf_out` object is omitted, RDF is disabled and it does not trigger final analysis post-processing.

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `interval` | integer | Yes | `10` | RDF sampling interval; must be greater than `0`. |
| `radius` | number | Yes | `10.0` | Maximum RDF radius; must be greater than `0`. |
| `dr` | number | Yes | `0.1` | RDF bin width; must be greater than `0`. |
| `statistics_rdf_steps` | integer | No | `1000` | RDF statistics/flush interval; defaults to `0`. |
| `atoms_pair` | array<array<int, 2>> | Optional legacy form | `[[1, 1]]` | 1-based atom-type pairs. |
| `groups` | array<object> | Optional group form | See below | RDF type-group definitions. |
| `group_pairs` | array<array<string, 2>> | Optional group form | See below | RDF group-pair definitions. |

`atoms_pair` form:

```json
"rdf_out": {
  "interval": 10,
  "radius": 10,
  "dr": 0.1,
  "statistics_rdf_steps": 1000,
  "atoms_pair": [[1, 1]]
}
```

Group form:

```json
"rdf_out": {
  "interval": 10,
  "radius": 10,
  "dr": 0.1,
  "statistics_rdf_steps": 1000,
  "groups": [
    { "name": "Li", "types": [1] },
    { "name": "O", "types": [4, 9, 11] }
  ],
  "group_pairs": [
    ["Li", "Li"],
    ["Li", "O"],
    ["O", "O"]
  ]
}
```

Notes:

- Type numbers are written as 1-based values and converted to 0-based indices internally.
- If `rdf_out` exists but has invalid parameters, `AnalysisOutput` disables RDF. However, `PostprocessRunner` still attempts post-processing whenever analysis-output configuration is present.

### `outputs.msd_out`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `interval` | integer | Yes | `10` | MSD sampling interval; must be greater than `0`. |
| `start_step` | integer | Yes | `0` | MSD starting step. |
| `end_step` | integer | Yes | `10000` | MSD ending step; must not be less than `start_step`. |

### `outputs.vacf_out`

| Key | Type | Required | Example | Description |
| --- | --- | --- | --- | --- |
| `interval` | integer | Yes | `10` | VACF sampling interval; must be greater than `0`. |
| `start_step` | integer | Yes | `0` | VACF starting step. |
| `end_step` | integer | Yes | `10000` | VACF ending step; must not be less than `start_step`. |

### `outputs.analysis_postprocess`

Whenever `outputs` contains `rdf_out`, `msd_out`, or `vacf_out`, the program attempts final analysis post-processing by default. Use this object to control that behavior explicitly.

| Key | Type | Default | Example | Description |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | `false` | Whether automatic post-processing is enabled. |
| `backend` | string | `"freud"` | `"freud"` | Only `freud` is currently supported. |
| `python` | string | Automatically finds the bundled Python, otherwise `python3` | `"python3"` | Python executable path. |
| `script` | string | `"scripts/postprocess_freud.py"` | `"scripts/postprocess_freud.py"` | Post-processing script path. |
| `trajectory` | string | `"rbmd.trj"` | `"rbmd.trj"` | Trajectory file read by post-processing. |
| `outdir` | string | `"."` | `"analysis"` | Post-processing output directory. |

Disable automatic post-processing:

```json
"outputs": {
  "rdf_out": {
    "interval": 10,
    "radius": 10,
    "dr": 0.1,
    "atoms_pair": [[1, 1]]
  },
  "analysis_postprocess": {
    "enabled": false
  }
}
```

## Minimal Example

```json
{
  "init_configuration": {
    "read_data": {
      "file": "system.data",
      "unit": "LJ",
      "atom_style": "atomic"
    }
  },
  "hyper_parameters": {
    "force_field": {
      "type": "LJ/CUT"
    },
    "neighbor": {
      "type": "Verlet-List",
      "cut_off": 12.0
    }
  },
  "execution": {
    "ensemble": "NVE",
    "timestep": 0.001,
    "num_steps": 1000,
    "integration_type": "vv"
  },
  "outputs": {
    "thermo_out": {
      "interval": 100
    },
    "trajectory_out": {
      "interval": 100
    }
  }
}
```

## Common Configuration Goals

### Do Not Output RDF

Delete the complete `outputs.rdf_out` object:

```json
"outputs": {
  "thermo_out": { "interval": 100 },
  "trajectory_out": { "interval": 100 }
}
```

### Keep RDF Configuration but Disable Automatic Post-processing

```json
"outputs": {
  "rdf_out": {
    "interval": 10,
    "radius": 10,
    "dr": 0.1,
    "atoms_pair": [[1, 1]]
  },
  "analysis_postprocess": {
    "enabled": false
  },
  "trajectory_out": {
    "interval": 10
  }
}
```

### Disable Trajectory Output

```json
"outputs": {
  "thermo_out": { "interval": 100 }
}
```
