#!/usr/bin/env python3
"""Build and run the current EAM Direct kernels against GPU baseline kernels."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def definition(source: str, marker: str) -> str:
    start = source.index(marker)
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main() -> None:
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found")
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_120")
    sanitizer_tool = os.environ.get("GPU_SANITIZER")
    sanitizer = shutil.which("compute-sanitizer") if sanitizer_tool else None
    if sanitizer_tool and not sanitizer:
        raise RuntimeError("GPU_SANITIZER requested but compute-sanitizer not found")
    source = (ROOT / "src/force/src/eam_op/rocm/eam_op.hip.cu").read_text()
    markers = [
        "__device__ void  EAMRho(",
        "__device__ void  EAMFp(",
        "__device__ void EAMForce(",
        "__device__ rbmd::Id WrapEAMDirectCellIndex(",
        "__device__ rbmd::Id MapEAMDirectDimension(",
        "__device__ rbmd::Id EAMDirectNeighborCell(",
    ]
    has_iterator = "struct EAMDirectCellIterator" in source
    if has_iterator:
        markers.append("struct EAMDirectCellIterator")
    markers.extend([
        "template <bool COMPUTE_PAIR = false>",
        "__global__ void EAMForceRBLDirect(",
    ])
    definitions = [definition(source, marker) for marker in markers]
    if has_iterator:
        definitions[6] += ";"  # Struct extraction excludes its semicolon.
    production = "namespace op {\n" + "\n\n".join(definitions) + "\n}\n"
    test = ROOT / "tests/eam_direct_gpu_test.cu"
    with tempfile.TemporaryDirectory(prefix="rbmd-eam-direct-gpu-") as directory:
        temp = Path(directory)
        (temp / "production_eam_direct.cuh").write_text(production)
        for precision, id_bits in ((0, 0), (0, 1), (1, 0), (1, 1)):
            executable = temp / f"eam-direct-{precision}-{id_bits}"
            subprocess.run(
                [
                    nvcc,
                    "-std=c++17",
                    "-O2",
                    "-lineinfo",
                    f"-arch={cuda_arch}",
                    "-x",
                    "cu",
                    f"-DUSE_DOUBLE={precision}",
                    f"-DUSE_64BIT_IDS={id_bits}",
                    f"-I{temp}",
                    str(test),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            command = [str(executable)]
            if sanitizer:
                command = [
                    sanitizer,
                    "--tool",
                    sanitizer_tool,
                    "--error-exitcode",
                    "99",
                    str(executable),
                ]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
