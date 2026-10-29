# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Benchmark saved MoE configs and write Aiter-compatible performance CSVs.

Run in a fresh process per config; tune scripts read their environment at import.
Inputs are prepared outside timing. Rows are published only after stable full-MoE
CUDA-graph measurements. This module does not train or modify launch configs.
"""

import argparse
from contextlib import contextmanager
import csv
import ctypes
import ctypes.util
import errno
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import tempfile
import threading
import time


FIELDS = (
    "arch", "quant_type", "indtype", "token", "inter_dim", "model_dim",
    "expert", "topk", "q_size_n", "q_size_k", "sol_type", "sol_id", "time_us",
)
KEY_FIELDS = FIELDS[:-1]
TRITON_SOLUTION_ID = "triton:fused_moe"
FORMATS = {
    "bf16": ("no_quant", "w16a16"),
    "int8_channel": ("int8_w8a8_channel", "w8a8_channel"),
    "fp8_channel": ("f8_w8a8_channel", "w8a8_channel"),
    "int8": ("int8_w8a8_block", "w8a8_group"),
    "fp8": ("f8_w8a8_block", "w8a8_group"),
    "int4": ("int4_w4a16", "w4a16"),
    "int4int8": ("int4_w4a8", "w4a8_group"),
    "int4int8_channel": ("int4_w4a8_channel", "w4a8_channel"),
}


def validate_solution_id(backend, solution_id):
    """Name an implementation entry point, not an executable/import string."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*", solution_id):
        raise ValueError("sol_id must be backend:implementation_name")
    if solution_id.split(":", 1)[0] != backend:
        raise ValueError("sol_id prefix must match sol_type")


def parse_config(path):
    """Only exact top configurations define benchmarkable K/topk values.

    Fuzzy aliases are lookup rules, not measurements of every matched shape.
    """
    path = Path(path)
    match = re.fullmatch(
        r"E=(\d+),N=(\d+),K=(\d+),topk=(\d+),arch=(gfx\w+)"
        r"(?:,dtype=(\w+))?(?:,block_shape=\[(\d+),(\d+)\])?\.json",
        path.name,
    )
    if not match:
        raise ValueError(f"Expected an exact Gemm1 config: {path.name}")
    e, n, k, topk, arch, dtype, bn, bk = match.groups()
    dtype = dtype or ""
    bn, bk = int(bn or 0), int(bk or 0)
    if dtype in ("int8_w8a8", "fp8_w8a8"):
        test_type = dtype.split("_")[0] + ("" if bn and bk else "_channel")
    else:
        test_type = {"": "bf16", "int4_w4a16": "int4",
                     "int4_w4a8": "int4int8" if bk else "int4int8_channel"}[dtype]
    # Block [0,64] is omitted by historical W4A16 filenames.
    if test_type in ("int4", "int4int8") and not bk:
        bk = 64
    bottom_name = path.name.replace(
        ",block_shape=", ",is_bottom=True,block_shape="
    ) if ",block_shape=" in path.name else path.stem + ",is_bottom=True.json"
    return dict(e=int(e), n=int(n), k=int(k), topk=int(topk), arch=arch,
                dtype=dtype or None, block_n=bn, block_k=bk,
                test_type=test_type, top=path, bottom=path.with_name(bottom_name))


def config_pair(shape):
    raw = [shape[side].read_bytes() for side in ("top", "bottom")]
    top, bottom = [json.loads(data) for data in raw]
    if not top or not bottom or not (set(top) & set(bottom)):
        raise ValueError("Gemm1/Gemm2 must contain common M entries")
    for m in set(top) & set(bottom):
        if int(m) <= 0 or top[m]["BLOCK_SIZE_M"] != bottom[m]["BLOCK_SIZE_M"]:
            raise ValueError(f"Invalid/shared BLOCK_SIZE_M mismatch at M={m}")
    digest = hashlib.sha256(b"\0".join(raw)).hexdigest()
    return top, bottom, digest


def unsupported_batch_sizes(shape, dtype, top, bottom):
    """Known dtype restriction of moe_op._n_tile_reuse_error, per M.

    Plain (unquantized) N_TILES_PER_CTA=2 is a weighted BF16 specialization,
    not an FP16 kernel. Do not alter the config to fabricate its FP16 timing.
    """
    if shape["test_type"] != "bf16" or str(dtype) not in ("float16", "torch.float16"):
        return []
    return sorted(int(m) for m in set(top) & set(bottom)
                  if any(config.get("N_TILES_PER_CTA", 1) == 2 for config in (top[m], bottom[m])))


def make_row(shape, m, dtype, time_us):
    quant, _ = FORMATS[shape["test_type"]]
    return dict(zip(FIELDS, (
        shape["arch"], quant, str(dtype), m, shape["n"], shape["k"],
        shape["e"], shape["topk"], shape["block_n"], shape["block_k"],
        "triton", TRITON_SOLUTION_ID, time_us,
    )))


def _atomic_write(path, write):
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def csv_write_lock(path, timeout=120):
    """Cross-host exclusion, including mounts without working remote flock.

    Never steal an old lock: a paused writer may still own it. After an abrupt
    kill, an operator must verify owner.json before removing this small lock.
    """
    lock_dir = path.with_suffix(".writer-lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"CSV writer lock still held: {lock_dir}")
            time.sleep(0.1)
    owner = lock_dir / "owner.json"
    try:
        owner.write_text(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), time=time.time())))
        # Keep the historical lock too, for compatibility with older writers.
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
    finally:
        owner.unlink(missing_ok=True)
        lock_dir.rmdir()


def upsert_row(directory, test_type, row):
    # Shared mounts can return ESTALE after another host atomically replaces a
    # CSV. Reopen under the same persistent lock; never retry a partial append.
    for attempt in range(5):
        try:
            return _upsert_row(directory, test_type, row)
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == 4:
                raise
            time.sleep(1)


def _upsert_row(directory, test_type, row):
    """Lock + atomic replacement: concurrent devices must not lose rows."""
    if set(row) != set(FIELDS):
        raise ValueError("Unexpected performance CSV schema")
    if not math.isfinite(float(row["time_us"])) or float(row["time_us"]) <= 0:
        raise ValueError("time_us must be finite and positive")
    if row["quant_type"] != FORMATS[test_type][0]:
        raise ValueError("Inconsistent provider/quantization")
    validate_solution_id(row["sol_type"], row["sol_id"])
    for field in ("token", "inter_dim", "model_dim", "expert", "topk"):
        if int(row[field]) <= 0:
            raise ValueError(f"{field} must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tuned_fmoe_{row['sol_type']}_{FORMATS[test_type][1]}.csv"
    row = {key: str(value) for key, value in row.items()}
    row["time_us"] = f"{float(row['time_us']):.6f}"
    key = lambda item: tuple(item[field] for field in KEY_FIELDS)
    with csv_write_lock(path):
        rows = {}
        if path.exists():
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != FIELDS:
                    raise ValueError(f"Unexpected existing header: {path}")
                rows = {key(item): item for item in reader}
        rows[key(row)] = row

        def write_csv(stream):
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows.values(), key=lambda item: (
                item["arch"], item["quant_type"], item["indtype"],
                *(int(item[field]) for field in (
                    "expert", "inter_dim", "model_dim", "topk", "q_size_n", "q_size_k", "token"
                )),
            )))
        _atomic_write(path, write_csv)
    return path


def spread_fraction(samples):
    return (max(samples) - min(samples)) / statistics.median(samples)


def _display_error(error):
    """Format a log copy only; never use this text as a runtime identifier."""
    return re.sub(r"amdgpu|amd|xgmi", "[backend]", str(error), flags=re.IGNORECASE)


class DeviceGuard:
    """Detect other containers through KFD sysfs (container-local ps is insufficient).

    Caller must initialize its CUDA queue first. A sole KFD process with queues
    on this GPU is then ours. Monitoring is conservative: even an idle foreign
    queue invalidates the measurement. Missing observability fails closed.
    """

    def __init__(self, device):
        try:
            hip = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
        except OSError as error:
            with tempfile.NamedTemporaryFile(
                mode="w", prefix="boltops-hip-load-", suffix=".log",
                encoding="utf-8", delete=False,
            ) as log:
                log.write(_display_error(error))
            raise OSError(
                f"Cannot load the HIP runtime for device monitoring; details: {log.name}"
            ) from None
        bus = ctypes.create_string_buffer(32)
        get_bus = hip.hipDeviceGetPCIBusId
        get_bus.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        get_bus.restype = ctypes.c_int
        if get_bus(bus, len(bus), device) != 0:
            raise RuntimeError("Cannot resolve HIP device PCI address")
        domain, bus_id, slot = bus.value.decode().split(":")
        slot, function = slot.split(".")
        location = (int(bus_id, 16) << 8) | (int(slot, 16) << 3) | int(function, 16)
        self.gpu_id = None
        for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
            props = dict(line.split() for line in (node / "properties").read_text().splitlines())
            if int(props.get("location_id", -1)) == location and int(props.get("domain", -1)) == int(domain, 16):
                gpu_id = int((node / "gpu_id").read_text())
                if gpu_id:
                    self.gpu_id = gpu_id
                    break
        if self.gpu_id is None:
            raise RuntimeError("Cannot map HIP device to KFD topology; not exporting unchecked timing")
        self.stop = threading.Event()
        self.contaminated = False

    def processes(self):
        result = set()
        for process in Path("/sys/class/kfd/kfd/proc").iterdir():
            try:
                for queue in (process / "queues").iterdir():
                    if int((queue / "gpuid").read_text()) == self.gpu_id:
                        result.add(int(process.name))
            except FileNotFoundError:
                # A process or queue exited while reading sysfs.
                continue
        return result

    def __enter__(self):
        self.owner = self.processes()
        if len(self.owner) != 1:
            raise RuntimeError(f"GPU is not exclusive: KFD processes={sorted(self.owner)}")
        self.contaminated = False
        self.stop.clear()

        def monitor():
            while not self.stop.is_set():
                try:
                    current = self.processes()
                    if current != self.owner:
                        self.contaminated = True
                except OSError:
                    self.contaminated = True
                self.stop.wait(0.1)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def wait_until_exclusive(self, timeout=600):
        deadline = time.monotonic() + timeout
        while True:
            owners = self.processes()
            if len(owners) == 1:
                return
            print(f"CSV_RETRY waiting for exclusive GPU: KFD processes={sorted(owners)}", flush=True)
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU remained occupied; leave this shape pending")
            time.sleep(min(10, max(0, deadline - time.monotonic())))

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        if self.processes() != self.owner:
            self.contaminated = True


def _from_environment(test_type):
    from boltops.fused_moe.triton.moe_config_utils import (
        _TRITON_CONFIGS_DIR, get_config_file_name,
    )
    from boltops.utility.triton_capability import get_triton_capabilities

    # Read actual script defaults, rather than guessing K from the file name.
    module = importlib.import_module(f"{__package__}.tune_moe_{test_type}")

    dtype = {"bf16": None, "int8": "int8_w8a8", "int8_channel": "int8_w8a8",
             "fp8": "fp8_w8a8", "fp8_channel": "fp8_w8a8", "int4": "int4_w4a16",
             "int4int8": "int4_w4a8", "int4int8_channel": "int4_w4a8"}[test_type]
    block = None
    if test_type in ("int8", "fp8"):
        block = [int(x) for x in os.environ.get("MOE_BLOCK_SHAPE", "128,128").split(",")]
    if test_type in ("int4", "int4int8"):
        block = [0, 64]
    return _TRITON_CONFIGS_DIR / get_triton_capabilities().config_dir_name / get_config_file_name(
        int(os.environ["MOE_NUM_EXPERTS"]), int(os.environ["MOE_HIDDEN_SIZE"]),
        dtype, block, K=module.FFN_HIDDEN_SIZES[0], top_k=module.TOP_KS[0],
    )


def _prepare_environment(device):
    # Tune scripts cache these values at import, including _from_environment().
    os.environ.update({
        "TRAINING_MODE": "0", "BENCHMARK_MODE": "0", "TRITON_COMPILE_ONLY": "0",
        "EP_SIZE": "1", "MOE_NUM_SHARED_EXPERTS": "0", "MOE_ACTIVATION": "silu",
        "MOE_IS_GATED": "1", "AUTOTUNE_DEVICE_START_ID": str(device),
        "AUTOTUNE_DEVICE_COUNT": "1", "NUM_GROUPS_RUN": "1", "CUR_GROUP": "0",
    })
    os.environ.pop("TOPK_IDS_DIR", None)


def benchmark(args):
    _prepare_environment(args.device)
    import torch
    import triton
    import triton.testing
    from boltops.fused_moe.triton import moe_config_utils
    from boltops.utility.arch_info import get_arch
    from boltops.utility.triton_capability import get_triton_capabilities

    torch.cuda.set_device(args.device)
    # Initialize our queue before identifying its host-visible KFD PID.
    torch.empty(1, device="cuda").zero_()
    torch.cuda.synchronize()
    guard = DeviceGuard(args.device)
    path = Path(args.config).resolve() if args.config else _from_environment(args.test_type)
    if args.allow_missing_config and not path.exists():
        print(f"CSV export skipped: no exact saved JSON for this benchmark: {path}", flush=True)
        return
    shape = parse_config(path)
    if shape["arch"] != get_arch():
        raise ValueError(f"Config arch {shape['arch']} != device {get_arch()}")
    if shape["arch"] == "gfx936" and shape["test_type"].startswith("fp8"):
        raise ValueError("gfx936 does not support FP8")
    if path.parent.name != get_triton_capabilities().config_dir_name:
        raise ValueError("Benchmark must use the config's Triton capability family")
    top, bottom, digest = config_pair(shape)
    common_m = set(top) & set(bottom)
    missing_m = sorted(map(int, set(top) ^ set(bottom)))
    if missing_m:
        print(f"CSV_SKIP missing matching Gemm1/Gemm2 M entries: {missing_m}", flush=True)
    batches = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else sorted(map(int, common_m))
    if args.allow_missing_config:
        skipped = [m for m in batches if str(m) not in common_m]
        if skipped:
            print(f"CSV_SKIP M values without an exact saved pair: {skipped}", flush=True)
        batches = [m for m in batches if str(m) in common_m]
        if not batches:
            return
    if not batches or any(str(m) not in common_m for m in batches):
        raise ValueError("Every requested M must exist in the saved config (no nearest-M fallback)")
    # Verify normal production lookup, not a monkeypatched launch or 3.2 fallback.
    for is_bottom, expected in ((False, top), (True, bottom)):
        actual = moe_config_utils.get_moe_configs(
            shape["e"], shape["n"], shape["dtype"], shape["block_n"],
            shape["block_k"], is_bottom, K=shape["k"], top_k=shape["topk"],
        )
        if actual != {int(m): cfg for m, cfg in expected.items()}:
            raise ValueError("Production lookup does not match the requested config pair")

    os.environ.update({
        "MOE_NUM_EXPERTS": str(shape["e"]),
        "MOE_HIDDEN_SIZE": str(shape["n"]), "MOE_FFN_HIDDEN_SIZE": str(shape["k"]),
        "MOE_TOP_K": str(shape["topk"]), "MOE_BATCH_SIZES": ",".join(map(str, batches)),
    })
    if shape["block_n"] and shape["block_k"]:
        os.environ["MOE_BLOCK_SHAPE"] = f"{shape['block_n']},{shape['block_k']}"
    module = importlib.import_module(f"{__package__}.tune_moe_{shape['test_type']}")
    input_helper = module.input_helper

    def seeded_input_helper(*inputs, **kwargs):
        if "seed" in inspect.signature(input_helper).parameters:
            kwargs["seed"] = args.seed
        return input_helper(*inputs, **kwargs)

    module.input_helper = seeded_input_helper
    raw = module.fused_experts_impl_benchmark.__wrapped__
    callbacks = [value.fn for name, value in vars(module).items()
                 if name.startswith("bench_fused_moe_") and hasattr(value, "fn")]
    if len(callbacks) != 1:
        raise ValueError("Expected one tune-script performance callback")
    callback = callbacks[0]
    samples = []

    def measure(*inputs, **kwargs):
        # Data generation, routing and weight quantization are outside timing;
        # activation quantization, sorting, both GEMMs and sum stay inside.
        fn = lambda: raw(*inputs, **kwargs)
        output = fn()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite output: no performance row will be published")
        torch.cuda.synchronize()
        for attempt in range(args.max_attempts):
            try:
                guard.wait_until_exclusive()
                with guard:
                    measured = [float(triton.testing.do_bench_cudagraph(fn, rep=args.rep_ms)) * 1000
                                for _ in range(args.repeats)]
                spread = spread_fraction(measured)
                if guard.contaminated or spread > args.max_spread:
                    raise RuntimeError(f"unstable: spread={spread:.2%}, foreign_queue={guard.contaminated}")
                samples[:] = measured
                break
            except RuntimeError as exc:
                print(f"CSV_RETRY attempt={attempt + 1}/{args.max_attempts}: {_display_error(exc)}", flush=True)
                if attempt + 1 == args.max_attempts:
                    raise RuntimeError(_display_error(exc)) from None
                time.sleep(args.retry_delay)
        return output, statistics.median(samples)

    module.fused_experts_impl_benchmark = measure
    dtypes = [getattr(torch, name) for name in args.dtypes.split(",")] if args.dtypes else module.DTYPES
    for dtype in dtypes:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Only float16/bfloat16 activation dtypes are supported")
        unsupported = unsupported_batch_sizes(shape, dtype, top, bottom)
        if unsupported:
            print(f"CSV_UNSUPPORTED dtype={dtype} M={unsupported}: N_TILES_PER_CTA=2 requires weighted BF16", flush=True)
        for m in batches:
            if m in unsupported:
                continue
            # Check before allocating a shape's weights, not only while timing.
            guard.wait_until_exclusive()
            torch.manual_seed(args.seed)
            kwargs = dict(BATCH_SIZE=m, HIDDEN_SIZE=shape["n"], FFN_HIDDEN_SIZE=shape["k"],
                          NUM_EXPERTS=shape["e"], TOP_K=shape["topk"],
                          provider="triton", dtype=dtype)
            params = inspect.signature(callback).parameters
            kwargs["NUM_SHARED_EXPERTS" if "NUM_SHARED_EXPERTS" in params else "EP_SIZE"] = (
                0 if "NUM_SHARED_EXPERTS" in params else 1
            )
            if "block_size" in params:
                kwargs["block_size"] = [shape["block_n"], shape["block_k"]]
            with torch.inference_mode():
                time_us = callback(**kwargs) * 1000
            if config_pair(shape)[2] != digest:
                raise RuntimeError("Config changed during measurement; refusing stale CSV row")
            csv_path = upsert_row(args.output_dir or path.parent, shape["test_type"],
                                  make_row(shape, m, dtype, time_us))
            print(f"CSV_RESULT M={m} dtype={dtype} time_us={time_us:.6f} file={csv_path}", flush=True)
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Exact Gemm1 JSON; normal runtime lookup must match")
    source.add_argument("--test-type", help="Resolve exact JSON using tune CLI environment")
    parser.add_argument("--batch-sizes", default=None)
    parser.add_argument("--dtypes", default=None, help="float16,bfloat16; default: tune script dtype")
    parser.add_argument("--device", type=int, default=int(os.environ.get("AUTOTUNE_DEVICE_START_ID", "0")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rep-ms", type=float, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-missing-config", action="store_true",
                        help="Benchmark-only: skip unsaved/fuzzy-only shapes instead of failing")
    parser.add_argument("--max-spread", type=float, default=0.05,
                        help="Maximum (max-min)/median across repeats; default 5%%")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10)
    args = parser.parse_args()
    if args.repeats < 3 or args.rep_ms <= 0 or args.max_spread <= 0 or args.max_attempts < 1 or args.retry_delay < 0:
        parser.error("Need >=3 repeats and positive timing/stability limits")
    if args.test_type:
        # Aiter's 13 columns cannot distinguish these operation semantics.
        if (os.environ.get("EP_SIZE", "1") != "1"
                or os.environ.get("MOE_NUM_SHARED_EXPERTS", "0") != "0"
                or os.environ.get("MOE_ACTIVATION", "silu") != "silu"
                or os.environ.get("MOE_IS_GATED", "1").lower() in ("0", "false")):
            print("CSV export skipped: schema supports only gated SiLU, EP=1, no shared experts")
            return
        args.batch_sizes = args.batch_sizes or os.environ.get("MOE_BATCH_SIZES")
    benchmark(args)


if __name__ == "__main__":
    main()
