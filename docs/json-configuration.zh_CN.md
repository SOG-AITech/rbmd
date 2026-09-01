# RBMD JSON 配置项说明

[English](json-configuration.en.md) · [构建与运行指南](build.zh_CN.md)

本文档按当前代码实际读取的 JSON 字段整理配置项。JSON 使用 `nlohmann::ordered_json` 解析，允许注释；相对路径一般按 JSON 文件所在目录解析。

## 快速结论

- `outputs.trajectory_out` 不写时，轨迹输出默认关闭，不创建/截断 `rbmd.trj`。
- 写了 `outputs.trajectory_out` 时，`enabled` 缺省为 `true`；启用轨迹时，`interval` 是必需项，省略会在初始化读取 `interval` 时抛错。
- 也可以显式设置 `outputs.trajectory_out.enabled = false` 关闭轨迹模块。
- 将 `outputs.trajectory_out.interval` 设为 `0` 或负数，也可以让采样逻辑不写帧；但这只是采样层面的兼容行为，不如 `enabled:false` 明确。
- `outputs.rdf_out` 不写时，不启用 RDF 配置；程序结束后的分析后处理也不会因为 RDF 被触发。
- 如果写了 `rdf_out`、`msd_out` 或 `vacf_out`，`PostprocessRunner` 默认会在 root rank 调用 `scripts/postprocess_freud.py`，除非设置 `outputs.analysis_postprocess.enabled = false`。
- 当前 `MDApplication` 主流程只直接挂载 `TrajectoryOutput`；`AnalysisOutput` 代码存在，但没有在主流程中加入 `CompositeOutput`。因此 RDF/MSD/VACF 主要依赖结束后的 Python 后处理与 `rbmd.trj`。

## 顶层结构

```json
{
  "init_configuration": {},
  "hyper_parameters": {},
  "execution": {},
  "outputs": {}
}
```

| 顶层字段 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `init_configuration` | object | 是 | 初始结构读取配置。 |
| `hyper_parameters` | object | 是 | 力场、邻居表、库仑、扩展功能配置。 |
| `execution` | object | 是 | 系综、积分、温压控制和步数配置。 |
| `outputs` | object | 是 | 热力学、轨迹、分析和后处理输出配置。 |

## `init_configuration`

### `init_configuration.read_data`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `file` | string | 是 | `"peo+Li.data"` | LAMMPS data 文件路径；相对路径按 JSON 文件目录解析。 |
| `unit` | string | 是 | `"LJ"`, `"REAL"`, `"METAL"` | 单位体系；控制单位换算。 |
| `atom_style` | string | 是 | `"atomic"`, `"charge"`, `"full"` | 原子样式；会影响结构数据和并行通信字段。 |
| `velocity_type` | string | 否 | `"GAUSS"` | 当 data 没有 `Velocities` 段时，按 `execution.temperature[0]` 生成 Gaussian/Maxwell 初速度；已有速度时不覆盖。 |
| `velocity_seed` | int | 否 | `12345` | `velocity_type: "GAUSS"` 的随机种子；未设置时默认 `12345`，也兼容读取 `seed`。 |
| `replicate` | array<int, 3> | 否 | `[2, 2, 1]` | 读取 data 后按 x/y/z 复制体系；三个值都必须为正整数。 |
| `replicate_dims` | array<int, 3> | 否 | `[2, 2, 1]` | `replicate` 的别名；两者同时存在时优先使用 `replicate`。 |
| `replicate_bond_periodic` | bool | 否 | `true` | 复制体系时是否按周期键处理。 |
| `bond_periodic` | bool | 否 | `true` | `replicate_bond_periodic` 的兼容写法。 |
| `bond/periodic` | bool | 否 | `true` | `replicate_bond_periodic` 的兼容写法。 |

说明：

- 示例里出现过 `init_configuration.inbuild`，但当前主流程 `ReadMDData()` 实际只读取 `read_data.file`，不会走内建结构初始化。

### `replicate` 支持范围

`replicate` 是 reader 层的结构复制功能，本身不按 `hyper_parameters.force_field.type` 分支。它会先读取 LAMMPS data，再按 x/y/z 扩展 box、原子坐标和相关结构数据；后续哪些数据真正参与力计算，由 `force_field.type` 决定。

| `force_field.type` | 常用 `atom_style` | 当前扩胞支持状态 | 说明 |
| --- | --- | --- | --- |
| `CVFF` | `full` | 支持 full topology | 扩胞会复制 `atoms`、`velocities`、`bonds`、`angles`、`dihedrals`、`impropers`，并重建 1-2/1-3/1-4 `special_bonds` 数据；MPI 下 PEO 2x1x1 已验证。 |
| `LJ/CUT` | `atomic` | 支持非键体系 | 扩胞会复制原子、速度、box、质量和 pair coeff；`LJ/CUT` 力场路径不消费 topology。 |
| `LJ/CUT/COUL/LONG` | `charge` 或 `full` | 支持 charge + pair + box | 扩胞会复制原子、速度、电荷、box、质量和 pair coeff，并更新 kspace 所需体系尺寸；当前该力场路径不使用 `bonds/angles/dihedrals/impropers`，也不应用 `special_bonds`。如果需要 LAMMPS 风格的 1-2/1-3/1-4 非键缩放，应使用当前已接入 topology 的 `CVFF` 路径，或先为 `LJ/CUT/COUL/LONG` 补 special-bond kernel 支持。 |
| `EAM` | `atomic` | 静态检查支持 | EAM 力场只消费原子类型、坐标、box、邻居表和 `potential_file`；扩胞会复制这些结构数据。`EAM::Init()` 会按 JSON 目录解析并读取 `potential_file`。目前未在本文档中记录专门的 EAM replicate 运行验证。 |

约束：

- 扩胞目前只支持正交盒；检测到 triclinic `xy/xz/yz` tilt 会抛错。
- 对 `atom_style full` 或存在 molecular topology 的 data，扩胞要求 `Atoms` 段每个原子都有 image flags `ix iy iz`。
- `atom_style full` 只表示 data 文件包含 molecule id、charge 和 topology 字段；是否计算 bonded terms 或 special-bond 缩放，仍由 `force_field.type` 的 force kernel 决定。

## `hyper_parameters`

### `hyper_parameters.force_field`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `type` | string | 是 | `"CVFF"`, `"LJ/CUT"`, `"LJ/CUT/COUL/LONG"`, `"EAM"`, `"Tersoff"` | 力场类型；决定 force 与 memory scheduler。 |
| `bond_type` | string | CVFF 相关 | `"harmonic"` | CVFF 键势类型。 |
| `angle_type` | string | CVFF 相关 | `"harmonic"` | CVFF 角势类型。 |
| `dihedral_type` | string | CVFF 相关 | `"harmonic"`, `"opls"` | 二面角类型；读 data 与调度 CVFF 参数时会使用。 |
| `improper_type` | string | CVFF 可选 | `"harmonic"`, `"cvff"` | improper 类型；存在 improper 数据时读取。 |
| `potential_file` | string | EAM/Tersoff 相关 | `"Cu.eam.alloy"` | 势文件路径；EAM/Tersoff 会解析为相对 JSON 目录的路径。 |
| `potential_elements` | string | Tersoff 相关 | `"Si"` | Tersoff 元素映射字符串。 |

### `hyper_parameters.neighbor`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `type` | string | 是 | `"Verlet-List"`, `"RBL"` | 邻居表类型。 |
| `cut_off` | number | 是 | `12.0` | 截断半径；force、domain decomposition、linked cell 都会读取。 |
| `skin` | number | 否 | `1.0` | 邻居表 skin；未配置时为 `0`。正数会启用 Verlet-List 或 RBL 的缓存复用。 |
| `interval` | integer | Tersoff 可选 | `10` | Tersoff 中用于控制相关更新间隔。 |
| `r_core` | number | `type = "RBL"` 时需要 | `6.0` | RBL core 半径。 |
| `neighbor_sample_num` | integer | `type = "RBL"` 时需要 | `100` | RBL 采样数。 |
| `energy_rbl_flag` | string | `type = "RBL"` 时需要 | `"yes"`, `"no"` | 控制 RBL 能量相关路径；代码按字符串判断。 |

#### RBL 配置与 skin 复用

RBL 支持 `skin` 复用。它不会缓存 `cut_off + skin` 范围内的完整 shell
邻居表，而是缓存参考坐标、linked-cell 分桶、MPI halo 和 forward plan。
复用步只更新 ghost 坐标，并根据当前坐标重新生成 core 与 shell：

- `r <= r_core` 的 core 邻居完整写入；
- `r_core < r <= cut_off` 的 shell 邻居按当前 step 和全局原子 ID 对重新采样；
- `r > cut_off` 的候选邻居忽略。

因此 RBL 不会在多个时间步冻结同一批 shell 样本，也不需要保存完整的
Verlet shell 候选表，通常仍比 Verlet-List 占用更少的邻居表显存。

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

RBL 参数约束：`skin >= 0`、`0 < r_core < cut_off`、
`neighbor_sample_num > 0`。
`r_core` 内的邻居完整计算，`r_core` 到 `cut_off` 的 shell 邻居进行采样；
`energy_rbl_flag` 是否设为 `"yes"` 应按是否需要 RBL 能量路径选择。
设置 `skin = 0` 或省略该字段会关闭缓存复用，并恢复为每次完整通信、分桶和
构建。设置正数时，所有 MPI rank 上的最大累计位移不超过 `skin / 2` 才会
复用；超过阈值后执行全局重建。

Verlet-List 的配置方式如下：

```json
"neighbor": {
  "type": "Verlet-List",
  "cut_off": 9.0,
  "skin": 1.0
}
```

对于两种邻居表，当粒子顺序、数量、盒子或邻居截断等缓存条件变化时，程序也
会全局重建。可用以下环境变量检查是否真正复用：

```bash
RBMD_DEBUG_NEIGHBOR_SKIN=1 mpirun -np 4 ./rbmd -j input.json
```

调试记录写入 `logs/debug/neighbor_skin_cache_rank*_pid*.csv`；其中
`action=reuse` 表示复用，`action=rebuild` 表示重建。可设置
`RBMD_DEBUG_NEIGHBOR_SKIN_EVERY=10` 降低记录频率。

### `hyper_parameters.coulomb`

`hyper_parameters.coulomb` 整个对象可省略；省略时不启用长程库仑配置。

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `type` | string | 写 `coulomb` 时必需 | `"RBE"`, `"RBSOG"`, `"EWALD"` | 库仑算法类型。 |
| `accuracy` | number | RBE/EWALD 自动 alpha 时必需 | `1.0e-4` | 自动推导 alpha 的相对精度。 |
| `alpha` | number | RBE/EWALD 可选 | `0.04333893` | 手动指定 alpha；存在时优先于 `accuracy`。 |
| `kmax` | array<int, 3> | RBE 能量或 EWALD 可选 | `[18, 34, 22]` | 手动指定 k 空间范围；必须正好 3 个整数。 |
| `coulomb_sample_num` | integer | RBE/RBSOG 必需 | `2000` | RBE/RBSOG 采样数。 |
| `energy_rbe_flag` | string | RBE 必需 | `"yes"`, `"no"` | RBE 下必须定义；为 `"yes"` 时会准备 k 空间能量参数。 |
| `rbsog_level` | integer | RBSOG 必需 | `0` 到 `4` | RBSOG 预设等级；当前只接受 `[0, 4]`。 |

RBSOG 注意事项：

- RBSOG 不再支持旧字段 `alpha`、`accuracy`、`rbsog_b`、`rbsog_sigma`、`rbsog_Mmax`、`rbsog_Kcut`。
- RBSOG 只接受 `rbsog_level`，代码会据此推导内部参数。
- `rbsog_level` 越高，目标误差越小、精度越高，但计算开销也越大；追求最高精度时选择 `4`。

| `rbsog_level` | 目标误差 | `Mmax` |
| --- | --- | --- |
| `0` | `2.289e-3` | `6` |
| `1` | `1.158e-4` | `16` |
| `2` | `1.142e-5` | `30` |
| `3` | `5.583e-8` | `64` |
| `4` | `3.389e-11` | `102` |

### `hyper_parameters.extend`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `fix_shake` | bool | 否 | `true` | 启用 SHAKE 相关路径；多个积分、温压控制模块会读取。 |
| `special_bonds` | array<number, 3> | 否 | `[0, 0, 0]` | CVFF special bonds 缩放系数；用于 1-2/1-3/1-4 关系。 |

### `hyper_parameters.shift_value`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `shift_value` | number | Tersoff 可选 | `0.0` | Tersoff 读取的位移/平移相关参数，直接位于 `hyper_parameters` 下。 |

## `execution`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `ensemble` | string | 是 | `"NVE"`, `"NVT"`, `"NPT"` | 主系综类型。 |
| `temperature` | array<number> | NVT/NPT 相关 | `[298.0, 298.0, 100]` | 温控参数；通常为初温、目标温度、阻尼/时间常数。 |
| `pressure` | array<number> | NPT 相关 | `[1.0, 1.0, 1000]` | 压控参数；通常为初压、目标压力、阻尼/时间常数。 |
| `temp_ctrl_type` | string | NVT/NPT 相关 | `"RESCALE"`, `"BERENDSEN"`, `"LANGEVIN"`, `"NOSE_HOOVER"` | 温控器类型；NVT 支持这四类。 |
| `press_ctrl_type` | string | NPT 相关 | `"BERENDSEN"`, `"NOSE_HOOVER"` | 压控器类型；NPT 要求温控与压控成对一致。 |
| `timestep` | number | 是 | `1.0` | 时间步长。 |
| `num_steps` | number/integer | 是 | `3000` | 模拟步数；当前 `KeepGoing()` 使用 `<= num_steps`，会执行包含 step 0 的循环。 |
| `integration_type` | string | 是 | `"vv"`, `"bm"` | 积分类型；`"bm"` 会额外读取 `par_a` 和 `par_b`。 |
| `par_a` | number | `integration_type = "bm"` 时必需 | `0.1` | BM 积分参数。 |
| `par_b` | number | `integration_type = "bm"` 时必需 | `0.1` | BM 积分参数。 |
| `group` | string | 温压控制可选 | `"all"` | Berendsen/Nose-Hoover 控制组；未配置时使用默认组逻辑。 |
| `com_bias` | string | 温压控制可选 | `"yes"`, `"no"` | 为 `"yes"` 时启用质心速度偏置修正。 |
| `momentum_control` | object | 否 | 见下 | 启用动量控制。 |

### `execution.momentum_control`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `interval` | integer | 是 | `100` | 动量控制执行间隔；为 `0` 时执行函数直接返回。 |
| `linear` | array<int, 3> | 否 | `[1, 1, 1]` | 是否移除 x/y/z 方向线动量；存在该字段即启用 linear 分支。 |
| `angular` | bool | 否 | `false` | 是否移除角动量；缺省为 `false`。 |
| `rescale` | bool | 否 | `false` | 移除动量后是否按原动能重标速度；缺省为 `false`。 |
| `group` | string | 是 | `"all"` | 作用组；空字符串会回退到 `"all"`。 |

## `outputs`

### `outputs.thermo_out`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `interval` | integer | 是 | `100` | 热力学输出间隔；force 与温压控制中都会读取。 |

### `outputs.trajectory_out`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `enabled` | bool | 否 | `false` | 轨迹输出开关；只有写了 `trajectory_out` 对象时才生效，对象内缺省为 `true`。为 `false` 时不读取 `interval`，不创建轨迹文件。 |
| `interval` | integer | 启用时必需 | `10` | 轨迹采样间隔；输出文件名固定为 `rbmd.trj`。 |

关闭轨迹输出时，最简写法是删除整个 `trajectory_out` 对象：

```json
"outputs": {
  "thermo_out": {
    "interval": 100
  }
}
```

也可以保留对象并显式关闭：

```json
"outputs": {
  "trajectory_out": {
    "enabled": false
  }
}
```

启用轨迹并控制输出频率：

```json
"outputs": {
  "trajectory_out": {
    "enabled": true,
    "interval": 100
  }
}
```

`interval = 1` 表示每步输出；`interval = 100` 表示每 100 步输出一次。`interval = 0` 或负数会让 `TrajectoryOutput::Execute()` 因 `interval <= 0` 直接返回，不写轨迹帧；但推荐直接删除 `trajectory_out` 或使用 `enabled:false` 作为正式关闭方式。如果后续要用 `rdf_out` 的 Python 后处理，不建议关闭轨迹，因为后处理默认读取 `rbmd.trj`。

### `outputs.rdf_out`

`rdf_out` 整个对象不写时，RDF 不启用，结束后的分析后处理也不会因 RDF 被触发。

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `interval` | integer | 是 | `10` | RDF 采样间隔；必须大于 `0`。 |
| `radius` | number | 是 | `10.0` | RDF 最大半径；必须大于 `0`。 |
| `dr` | number | 是 | `0.1` | RDF bin 宽度；必须大于 `0`。 |
| `statistics_rdf_steps` | integer | 否 | `1000` | RDF 统计/flush 步数；缺省为 `0`。 |
| `atoms_pair` | array<array<int, 2>> | 旧写法可选 | `[[1, 1]]` | 1-based 原子类型对。 |
| `groups` | array<object> | 分组写法可选 | 见下 | RDF 类型组定义。 |
| `group_pairs` | array<array<string, 2>> | 分组写法可选 | 见下 | RDF 组对定义。 |

`atoms_pair` 写法：

```json
"rdf_out": {
  "interval": 10,
  "radius": 10,
  "dr": 0.1,
  "statistics_rdf_steps": 1000,
  "atoms_pair": [[1, 1]]
}
```

分组写法：

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

说明：

- 类型编号使用 1-based 写法，代码内部会转为 0-based。
- `rdf_out` 存在但参数无效时，`AnalysisOutput` 会禁用 RDF；但结束后 `PostprocessRunner` 仍以“存在分析输出配置”为条件尝试后处理。

### `outputs.msd_out`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `interval` | integer | 是 | `10` | MSD 采样间隔；必须大于 `0`。 |
| `start_step` | integer | 是 | `0` | MSD 起始步。 |
| `end_step` | integer | 是 | `10000` | MSD 结束步；必须不小于 `start_step`。 |

### `outputs.vacf_out`

| 字段 | 类型 | 必需 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `interval` | integer | 是 | `10` | VACF 采样间隔；必须大于 `0`。 |
| `start_step` | integer | 是 | `0` | VACF 起始步。 |
| `end_step` | integer | 是 | `10000` | VACF 结束步；必须不小于 `start_step`。 |

### `outputs.analysis_postprocess`

只要 `outputs` 下存在 `rdf_out`、`msd_out` 或 `vacf_out` 对象，程序结束后默认会尝试运行分析后处理。可用本段显式控制。

| 字段 | 类型 | 默认值 | 示例 | 说明 |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | `false` | 是否启用自动后处理。 |
| `backend` | string | `"freud"` | `"freud"` | 当前只支持 `freud`。 |
| `python` | string | 自动查找内置 Python，否则 `python3` | `"python3"` | Python 可执行文件路径。 |
| `script` | string | `"scripts/postprocess_freud.py"` | `"scripts/postprocess_freud.py"` | 后处理脚本路径。 |
| `trajectory` | string | `"rbmd.trj"` | `"rbmd.trj"` | 后处理读取的轨迹文件。 |
| `outdir` | string | `"."` | `"analysis"` | 后处理输出目录。 |

禁用自动后处理：

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

## 最小示例

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

## 常见配置目的

### 不输出 RDF

删除整个 `outputs.rdf_out` 对象即可：

```json
"outputs": {
  "thermo_out": { "interval": 100 },
  "trajectory_out": { "interval": 100 }
}
```

### 保留 RDF 配置但不自动后处理

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

### 关闭轨迹输出

```json
"outputs": {
  "thermo_out": { "interval": 100 }
}
```
