"""aicc JIT compiler for the fused-MoE HIPC moe_sorting kernel.

Compiles Hygon ``aicc`` sources under ``moe_sorting_hipc/csrc/`` and caches the
``.so`` under the torch extension build root. Kept next to the kernel because
only the Triton fused MoE path uses it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
from types import ModuleType
from typing import Iterable

import torch
from torch.utils.cpp_extension import get_default_build_root, include_paths, library_paths
from torch.utils.file_baton import FileBaton


def _aicc() -> Path:
    configured = os.environ.get("BOLT_OPS_AICC")
    candidates = [
        configured,
        shutil.which("aicc"),
        "/opt/dtk/bin/aicc",
        shutil.which("hipcc"),
        "/opt/rocm/bin/hipcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise RuntimeError(
        "HIPC kernel compilation requires aicc or hipcc; "
        "set BOLT_OPS_AICC to its path"
    )


def _compiler_identity(compiler: Path) -> bytes:
    proc = subprocess.run(
        [str(compiler), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout


def _digest(
    name: str, sources: Iterable[Path], headers: Iterable[Path], compiler: Path
) -> str:
    value = hashlib.sha256()
    value.update(name.encode())
    value.update(_compiler_identity(compiler))
    value.update(torch.__version__.encode())
    value.update(str(torch.version.hip).encode())
    value.update(str(torch._C._GLIBCXX_USE_CXX11_ABI).encode())
    value.update(f"{sys.version_info[:3]}".encode())
    if torch.cuda.is_available():
        value.update(torch.cuda.get_device_name(torch.cuda.current_device()).encode())
    for path in sorted({*sources, *headers}):
        value.update(path.name.encode())
        value.update(path.read_bytes())
    return value.hexdigest()[:16]


def _compile_flags(module_name: str, include_dirs: Iterable[Path]) -> list[str]:
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    flags = [
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        f"-DTORCH_EXTENSION_NAME={module_name}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
    ]
    flags.extend(f"-I{path}" for path in include_dirs)
    flags.extend(f"-isystem{path}" for path in include_paths())
    python_include = sysconfig.get_path("include")
    if python_include:
        flags.append(f"-isystem{python_include}")
    flags.extend(
        [
            "-isystem/opt/dtk/include",
            "-fPIC",
            "-std=c++20",
            "-O3",
            "-DWITH_HIP",
            "-D__HIP_PLATFORM_AMD__=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-DUSE_ROCM=1",
            "-DHIPBLAS_V2",
            "-DCUDA_HAS_FP16=1",
            "-D__HIP_NO_HALF_OPERATORS__=1",
            "-D__HIP_NO_HALF_CONVERSIONS__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mcmodel=large",
            "-fno-unique-section-names",
            "-ffunction-sections",
            "-fdata-sections",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "--offload-arch=native",
            "-DDTK_ENV",
            "-DTORCH_Float4_e2m1fn_x2",
            "-DUSE_PROF_API=1",
            "-Wno-macro-redefined",
            "-Wno-switch-bool",
            "-Wno-undefined-func-template",
            "-Wno-unused-result",
            "-Wno-vla-cxx-extension",
            "-fgpu-flush-denormals-to-zero",
            "-fno-offload-uniform-block",
            "-mllvm",
            "--amdgpu-kernarg-preload-count=16",
            "-mllvm",
            "--lsr-drop-solution=1",
            "-mllvm",
            "-amdgpu-coerce-illegal-types=1",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
            "-mllvm",
            "-enable-post-misched=0",
            "-fno-gpu-rdc",
        ]
    )
    return flags


def _link_flags() -> list[str]:
    result = [
        "-shared",
        "-mcmodel=large",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wl,--gc-sections",
    ]
    for path in library_paths():
        result.extend((f"-L{path}", f"-Wl,-rpath,{path}"))
    result.extend(
        [
            "-lc10",
            "-lc10_hip",
            "-ltorch_cpu",
            "-ltorch_hip",
            "-ltorch",
            "-ltorch_python",
            "-L/opt/dtk/lib",
            "-Wl,-rpath,/opt/dtk/lib",
            "-lamdhip64",
        ]
    )
    return result


def _write_ninja(
    path: Path,
    module_path: Path,
    sources: list[Path],
    compiler: Path,
    module_name: str,
    include_dirs: list[Path],
) -> None:
    cflags = " ".join(
        shlex.quote(flag) for flag in _compile_flags(module_name, include_dirs)
    )
    ldflags = " ".join(shlex.quote(flag) for flag in _link_flags())
    objects = [path.parent / f"{source.stem}.o" for source in sources]
    lines = [
        "ninja_required_version = 1.3",
        f"aicc = {compiler}",
        "cxx = c++",
        f"cflags = {cflags}",
        f"ldflags = {ldflags}",
        "rule compile",
        "  command = $aicc $cflags -c $in -o $out",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
    ]
    lines.extend(f"build {obj}: compile {source}" for obj, source in zip(objects, sources))
    lines.append(f"build {module_path}: link {' '.join(str(obj) for obj in objects)}")
    lines.append(f"default {module_path}")
    path.write_text("\n".join(lines) + "\n")


def _load(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load HIPC extension from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hipc_extension(
    name: str,
    source_names: Iterable[str],
    *,
    csrc_dir: Path,
) -> ModuleType:
    """Compile and load a HIPC extension from ``csrc_dir``.

    ``source_names`` are filenames relative to ``csrc_dir``. Headers in that
    directory are hashed into the cache key and added to the include path.
    """
    csrc_dir = Path(csrc_dir).resolve()
    sources = [csrc_dir / source for source in source_names]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing HIPC extension sources: {missing}")

    headers = sorted(csrc_dir.glob("*.h"))
    compiler = _aicc()
    digest = _digest(name, sources, headers, compiler)
    module_name = f"_boltops_{name}_{digest}"
    build_dir = Path(get_default_build_root()) / "boltops_hipc" / module_name
    module_path = build_dir / f"{module_name}.so"
    if module_path.is_file():
        return _load(module_name, module_path)

    build_dir.mkdir(parents=True, exist_ok=True)
    baton = FileBaton(str(build_dir / "lock"))
    acquired = baton.try_acquire()
    if acquired:
        try:
            if not module_path.is_file():
                ninja_file = build_dir / "build.ninja"
                _write_ninja(
                    ninja_file,
                    module_path,
                    sources,
                    compiler,
                    module_name,
                    include_dirs=[csrc_dir],
                )
                verbose = os.environ.get("BOLT_OPS_HIPC_VERBOSE", "0") == "1"
                subprocess.run(
                    ["ninja", "-f", str(ninja_file)],
                    cwd=build_dir,
                    check=True,
                    stdout=None if verbose else subprocess.PIPE,
                    stderr=None if verbose else subprocess.STDOUT,
                    text=True,
                )
        except subprocess.CalledProcessError as error:
            output = error.stdout or ""
            raise RuntimeError(f"HIPC extension build failed:\n{output}") from error
        finally:
            baton.release()
    else:
        baton.wait()

    return _load(module_name, module_path)


__all__ = ["load_hipc_extension"]
