# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Optional


def random_int8_weights(shape, device):
    """Preserve rand->scale->truncate values without full-sized FP32 copies.

    Called separately for each weight: peak storage is one FP32 weight plus
    its INT8 result, not both GEMMs' FP32 weights and expression temporaries.
    The random draw shape/order and FP32 operation order remain unchanged.
    """
    import torch

    values = torch.rand(shape, dtype=torch.float32, device=device)
    return values.sub_(0.5).mul_(2).mul_(127).clamp_(-128, 127).to(torch.int8)


def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value!r}") from exc


def parse_env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    token = raw.strip().lower()
    if token == "":
        return None
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 仅支持 true/false (或 1/0)，当前值: {raw!r}")


def parse_env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是浮点数，当前值: {raw!r}") from exc


def resolve_moe_activation_and_gate(
    *,
    activation_env: str = "MOE_ACTIVATION",
    is_gated_env: str = "MOE_IS_GATED",
    default_activation: str = "silu",
) -> tuple[str, bool]:
    from boltops.fused_moe.triton.moe_activation import _normalize_activation_and_gate

    return _normalize_activation_and_gate(
        os.environ.get(activation_env, default_activation),
        parse_env_optional_bool(is_gated_env),
    )


def resolve_situ_parameters(
    activation_name: str,
    *,
    beta_env: str = "MOE_ACTIVATION_SITU_BETA",
    linear_beta_env: str = "MOE_ACTIVATION_SITU_LINEAR_BETA",
) -> tuple[float | None, float | None]:
    beta = parse_env_optional_float(beta_env)
    linear_beta = parse_env_optional_float(linear_beta_env)
    if activation_name != "situ":
        return beta, linear_beta
    if beta is None:
        raise ValueError(f"activation='situ' 时必须设置 {beta_env}")
    if beta <= 0:
        raise ValueError(f"{beta_env} 必须大于 0")
    if linear_beta is not None and linear_beta <= 0:
        raise ValueError(f"{linear_beta_env} 必须大于 0")
    return beta, linear_beta


def apply_activation_ref(
    x,
    *,
    activation_name: str,
    is_gated: bool,
    activation_situ_beta: float | None = None,
    activation_situ_linear_beta: float | None = None,
):
    import torch
    import torch.nn.functional as F

    if is_gated:
        gate, up = torch.chunk(x, 2, dim=-1)
        if activation_name == "silu":
            act = F.silu(gate)
        elif activation_name == "gelu":
            act = F.gelu(gate, approximate="none")
        elif activation_name == "gelu_tanh":
            act = F.gelu(gate, approximate="tanh")
        elif activation_name == "situ":
            if activation_situ_beta is None:
                raise ValueError("SITU reference requires activation_situ_beta")
            act = (
                activation_situ_beta
                * torch.tanh(gate / activation_situ_beta)
                * torch.sigmoid(gate)
            )
            if activation_situ_linear_beta is not None:
                up = activation_situ_linear_beta * torch.tanh(
                    up / activation_situ_linear_beta
                )
        elif activation_name == "swigluoai":
            gate_clamp = torch.clamp(gate, max=7.0)
            up_clamp = torch.clamp(up, min=-7.0, max=7.0)
            act = gate_clamp * torch.sigmoid(1.702 * gate_clamp)
            return act * (up_clamp + 1.0)
        elif activation_name == "swiglustep":
            silu_gate = F.silu(gate)
            act = torch.clamp(silu_gate, max=7.0)
            up_clamp = torch.clamp(up, min=-7.0, max=7.0)
            return act * up_clamp
        else:
            raise ValueError(f"Unsupported gated activation for ref: {activation_name}")
        return act * up

    if activation_name in {"silu", "silu_no_mul"}:
        return F.silu(x)
    if activation_name in {"gelu", "gelu_no_mul"}:
        return F.gelu(x, approximate="none")
    if activation_name in {"gelu_tanh", "gelu_tanh_no_mul"}:
        return F.gelu(x, approximate="tanh")
    if activation_name in {"relu2", "relu2_no_mul"}:
        return torch.where(x > 0, x * x, torch.zeros_like(x))
    raise ValueError(f"Unsupported non-gated activation for ref: {activation_name}")


@dataclass(frozen=True)
class ScalarType:
    """Minimal ScalarType for int quantization used by fused_moe int4 paths."""

    size_bits: int
    signed: bool
    bias: int = 0

    @classmethod
    def int_(cls, size_bits: int, bias: Optional[int] = None) -> "ScalarType":
        return cls(size_bits=size_bits, signed=True, bias=(bias or 0))

    @classmethod
    def uint(cls, size_bits: int, bias: Optional[int] = None) -> "ScalarType":
        return cls(size_bits=size_bits, signed=False, bias=(bias or 0))

    def is_integer(self) -> bool:
        return True

    def is_signed(self) -> bool:
        return self.signed

    def has_bias(self) -> bool:
        return self.bias != 0

    def min(self) -> int:
        raw_min = -(1 << (self.size_bits - 1)) if self.signed else 0
        return raw_min - self.bias

    def max(self) -> int:
        raw_max = (1 << (self.size_bits - 1)) - 1 if self.signed else (1 << self.size_bits) - 1
        return raw_max - self.bias


class scalar_types:
    int4 = ScalarType.int_(4, None)
    uint4 = ScalarType.uint(4, None)
    int8 = ScalarType.int_(8, None)
    uint8 = ScalarType.uint(8, None)

    # "gptq" style aliases used by int4 autotune tests
    uint4b8 = ScalarType.uint(4, 8)
    uint8b128 = ScalarType.uint(8, 128)


def quantize_weights(
    w,
    quant_type: ScalarType,
    group_size: Optional[int],
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    import torch

    assert quant_type.is_integer(), "Only integer quantization is supported"
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided (-1 means channelwise)"
    )

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape
    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [group_size, -1] for grouped quantization.
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values
    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.tensor([1.0], device=w.device)
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
        else:
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    if group_size is not None and group_size < size_k:

        def reshape_w(t):
            t = t.reshape((group_size, -1, size_n))
            t = t.permute(1, 0, 2)
            return t.reshape((size_k, size_n)).contiguous()

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous().to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


@contextmanager
def noop_patched_environment():
    yield


def resolve_patched_environment():
    """Resolve patched_environment with explicit precedence.

    The local migrated patch module is preferred; noop is the fallback.
    """
    candidates = [
        "boltops.tools.fused_moe_triton_tune.autotune_patches",
    ]
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, "patched_environment", None)
            if callable(fn):
                return fn, module_name, False, errors
            errors.append(f"{module_name}: missing callable patched_environment")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    return noop_patched_environment, "noop", True, errors


def enforce_training_patch_contract(
    *,
    training_mode: int,
    benchmark_mode: int,
    patch_is_noop: bool,
    patch_source: str,
    import_errors: list[str] | None = None,
) -> None:
    # Contract: training mode must have real patch; benchmark mode can tolerate noop.
    if training_mode == 1 and patch_is_noop:
        details = ""
        if import_errors:
            details = "\n".join(f"  - {item}" for item in import_errors)
        raise RuntimeError(
            "TRAINING_MODE=1 但未加载到 autotune_patches（当前为 noop）。\n"
            f"patch_source={patch_source}\n"
            "请确认 boltops.tools.fused_moe_triton_tune.autotune_patches 可导入\n"
            + (f"\n导入错误详情:\n{details}" if details else "")
        )
    if benchmark_mode == 1 and patch_is_noop:
        print(
            f"警告: BENCHMARK_MODE=1 使用 noop patch（patch_source={patch_source}），按设计允许继续。",
            file=sys.stderr,
        )


def check_patched_environment(strict_training: bool = False) -> int:
    training_mode = get_env_int("TRAINING_MODE", 0)
    benchmark_mode = get_env_int("BENCHMARK_MODE", 0)
    patch_fn, patch_source, patch_is_noop, import_errors = resolve_patched_environment()
    print(f"patch_source={patch_source}")
    print(f"patch_is_noop={patch_is_noop}")
    _ = patch_fn  # keep symbol for future debug extension
    if strict_training:
        try:
            enforce_training_patch_contract(
                training_mode=training_mode,
                benchmark_mode=benchmark_mode,
                patch_is_noop=patch_is_noop,
                patch_source=patch_source,
                import_errors=import_errors,
            )
        except Exception as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1
    return 0


def ensure_op_tests_namespace(
    *,
    fused_moe_dir: str,
    namespace_root: str,
) -> tuple[str, bool]:
    fused_path = Path(fused_moe_dir).resolve()
    if not fused_path.is_dir():
        raise ValueError(f"fused_moe_dir 不存在或不是目录: {fused_path}")

    root = Path(namespace_root).resolve()
    op_tests_dir = root / "op_tests"
    triton_dir = op_tests_dir / "triton_autotune"
    target = triton_dir / "fused_moe"

    triton_dir.mkdir(parents=True, exist_ok=True)
    (op_tests_dir / "__init__.py").touch(exist_ok=True)
    (triton_dir / "__init__.py").touch(exist_ok=True)

    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == fused_path:
            return str(target), False
        raise ValueError(
            f"目标路径已存在且不匹配: {target} -> {target.resolve() if target.exists() else 'N/A'}"
        )

    target.symlink_to(fused_path, target_is_directory=True)
    return str(target), True


def parse_batch_sizes_from_env(default_batches: list[int], env_name: str = "MOE_BATCH_SIZES") -> list[int]:
    raw = os.environ.get(env_name, "").strip()
    if raw == "":
        return default_batches

    batch_sizes: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{env_name} 包含非法整数: {token!r}") from exc
        if value <= 0:
            raise ValueError(f"{env_name} 必须全部 > 0，当前值: {value}")
        batch_sizes.append(value)

    if not batch_sizes:
        raise ValueError(f"{env_name} 解析后为空，请至少提供一个 batch，如 1,2,4")
    return sorted(set(batch_sizes))


def resolve_device_settings(total_cuda_devices: int) -> tuple[int, int, int]:
    device_start_id = get_env_int("AUTOTUNE_DEVICE_START_ID", 0)
    if device_start_id < 0:
        raise ValueError(f"AUTOTUNE_DEVICE_START_ID must be >= 0, got {device_start_id}")

    if total_cuda_devices > 0 and device_start_id < total_cuda_devices:
        default_device_count = total_cuda_devices - device_start_id
    else:
        default_device_count = 1

    device_count = get_env_int("AUTOTUNE_DEVICE_COUNT", default_device_count)
    if device_count <= 0:
        raise ValueError(f"AUTOTUNE_DEVICE_COUNT must be >= 1, got {device_count}")

    if total_cuda_devices > 0 and device_start_id + device_count > total_cuda_devices:
        raise ValueError(
            f"Invalid device range: start={device_start_id}, count={device_count}, total={total_cuda_devices}"
        )

    # 简化规则：run group 数量永远等于 device_count。
    num_groups_run = device_count

    return device_start_id, device_count, num_groups_run


def _parse_int(value: str, name: str, *, allow_zero: bool = True, min_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值: {value!r}") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} 必须 >= {min_value}，当前值: {parsed}")
    if not allow_zero and parsed == 0:
        raise ValueError(f"{name} 不能为 0")
    return parsed


def get_total_cuda_devices() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"无法导入 torch 以检测 CUDA 设备数量: {exc}") from exc

    total_cuda_devices = int(torch.cuda.device_count())
    if total_cuda_devices <= 0:
        raise ValueError("当前机器未检测到 CUDA 设备")
    return total_cuda_devices


def parse_hy_smi_cards(hy_smi_text: str) -> list[tuple[int, float, float]]:
    cards: list[tuple[int, float, float]] = []
    for raw_line in hy_smi_text.splitlines():
        line = raw_line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            idx = int(parts[0])
            vram_pct = float(parts[5].rstrip("%"))
            hcu_pct = float(parts[6].rstrip("%"))
        except ValueError:
            continue
        cards.append((idx, vram_pct, hcu_pct))
    cards.sort(key=lambda x: x[0])
    return cards


def pick_best_contiguous_cards(
    cards: list[tuple[int, float, float]],
    desired_count: int,
    max_usage_pct: float,
) -> tuple[int, int] | None:
    if not cards:
        return None

    card_map = {idx: (vram, hcu) for idx, vram, hcu in cards}
    indices = [idx for idx, _, _ in cards]
    max_len = min(desired_count, len(indices))

    best: tuple[float, float, int, int] | None = None
    for length in range(max_len, 0, -1):
        candidates: list[tuple[float, float, int, int]] = []
        for start in range(indices[0], indices[-1] - length + 2):
            window = list(range(start, start + length))
            if not all(i in card_map for i in window):
                continue
            vals = [card_map[i] for i in window]
            # 规则：VRAM/HCU 任一指标 >= 阈值都视为不可用，只有都 < 阈值才可用。
            if all(vram < max_usage_pct and hcu < max_usage_pct for vram, hcu in vals):
                avg_vram = sum(vram for vram, _ in vals) / length
                avg_hcu = sum(hcu for _, hcu in vals) / length
                candidates.append((avg_vram, avg_hcu, start, length))
        if candidates:
            best = min(candidates)
            break

    if best is None:
        return None
    _, _, start, length = best
    return start, length


def resolve_launcher_env() -> tuple[dict[str, int], list[str]]:
    notes: list[str] = []
    total_cuda_devices = get_total_cuda_devices()

    start_raw = os.environ.get("AUTOTUNE_DEVICE_START_ID", "")
    count_raw = os.environ.get("AUTOTUNE_DEVICE_COUNT", "")
    auto_select_hy_smi = os.environ.get("AUTO_SELECT_DEVICES_BY_HY_SMI", "1")

    hy_smi_max_usage_pct = float(os.environ.get("HY_SMI_MAX_VRAM_PCT", "30"))

    if auto_select_hy_smi == "1" and start_raw == "" and count_raw == "":
        if shutil.which("hy-smi") is None:
            notes.append("警告: 未找到 hy-smi，回退到默认设备选择")
        else:
            desired_count_raw = os.environ.get("AUTOTUNE_DEVICE_COUNT", "4")
            desired_count = 4
            try:
                desired_count = _parse_int(
                    desired_count_raw,
                    "AUTOTUNE_DEVICE_COUNT",
                    allow_zero=False,
                    min_value=1,
                )
            except ValueError:
                desired_count = 4

            desired_count = min(desired_count, total_cuda_devices)
            try:
                hy_smi_output = subprocess.check_output(["hy-smi"], text=True)
            except Exception as exc:
                raise RuntimeError(f"执行 hy-smi 失败: {exc}") from exc

            cards = parse_hy_smi_cards(hy_smi_output)
            if not cards:
                raise ValueError("hy-smi 输出解析失败，请检查 hy-smi 输出格式")

            best = pick_best_contiguous_cards(
                cards=cards,
                desired_count=desired_count,
                max_usage_pct=hy_smi_max_usage_pct,
            )
            if best is None:
                raise ValueError(
                    f"hy-smi 检测不到满足条件的连续空闲卡（要求 VRAM/HCU 均 < {hy_smi_max_usage_pct}%）"
                )

            start_id, device_count = best
            start_raw = str(start_id)
            count_raw = str(device_count)
            notes.append(
                f"hy-smi 自动选择设备: start={start_id}, count={device_count}, "
                f"阈值(vram/hcu < {hy_smi_max_usage_pct}%)"
            )

    if start_raw == "":
        start_raw = "0"
    device_start_id = _parse_int(
        start_raw,
        "AUTOTUNE_DEVICE_START_ID",
        allow_zero=True,
        min_value=0,
    )
    if device_start_id >= total_cuda_devices:
        raise ValueError(
            f"AUTOTUNE_DEVICE_START_ID={device_start_id} 超过设备上限 {total_cuda_devices}"
        )

    if count_raw == "":
        count_raw = str(total_cuda_devices - device_start_id)
    device_count = _parse_int(
        count_raw,
        "AUTOTUNE_DEVICE_COUNT",
        allow_zero=False,
        min_value=1,
    )
    if device_start_id + device_count > total_cuda_devices:
        raise ValueError(
            f"设备范围越界 start={device_start_id}, count={device_count}, total={total_cuda_devices}"
        )

    # 简化规则：run group 数量永远等于 device_count。
    num_groups_run = device_count

    resolved = {
        "TOTAL_CUDA_DEVICES": total_cuda_devices,
        "AUTOTUNE_DEVICE_START_ID": device_start_id,
        "AUTOTUNE_DEVICE_COUNT": device_count,
        "AUTOTUNE_NUM_GROUPS_RUN": num_groups_run,
    }
    return resolved, notes


def print_launcher_env_exports() -> int:
    try:
        resolved, notes = resolve_launcher_env()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    for note in notes:
        print(note, file=sys.stderr)

    for key, value in resolved.items():
        print(f"export {key}={value}")
    return 0


def build_compile_group_ids(
    num_groups_compile: int,
    device_start_id: int,
    device_count: int,
    compile_batches: list[int],
) -> dict[int, tuple[int, list[int]]]:
    group_ids: dict[int, tuple[int, list[int]]] = {}
    for group_id in range(num_groups_compile):
        device_id = device_start_id + (group_id % device_count)
        group_ids[group_id] = (device_id, compile_batches.copy())
    return group_ids


def _collect_batches_in_range(
    full_batches: list[int],
    min_batch: int | None = None,
    max_batch: int | None = None,
) -> list[int]:
    return [
        batch
        for batch in full_batches
        if (min_batch is None or batch >= min_batch) and (max_batch is None or batch <= max_batch)
    ]


def _split_fallback_evenly(full_batches: list[int], group_count: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(group_count)]
    for idx, batch in enumerate(full_batches):
        # 按索引均匀切分，确保所有 batch 都有归属
        group_idx = min(idx * group_count // len(full_batches), group_count - 1)
        groups[group_idx].append(batch)
    return [group for group in groups if group]


# 直观规则：
# 1 组：全量
# 2 组：小 batch(<=2048) / 大 batch(>2048)
# 3 组：小(<=512) / 中(1024~8192) / 大(>8192)
# 4 组：小(<=512) / 中(1024~8192) / 大(16384) / 超大(>16384)
RUN_BATCH_SPLIT_RULES: dict[int, list[tuple[int | None, int | None]]] = {
    1: [(None, None)],
    2: [(None, 2048), (2049, None)],
    3: [(None, 512), (1024, 8192), (8193, None)],
    4: [(None, 512), (1024, 8192), (16384, 16384), (16385, None)],
}


def split_run_batches(full_batches: list[int], group_count: int) -> list[list[int]]:
    rules = RUN_BATCH_SPLIT_RULES.get(group_count)
    if rules is not None:
        groups = [
            _collect_batches_in_range(
                full_batches,
                min_batch=min_batch,
                max_batch=max_batch,
            )
            for min_batch, max_batch in rules
        ]
        if all(groups):
            return groups

    return _split_fallback_evenly(full_batches, group_count)


def build_run_group_ids(
    device_start_id: int,
    device_count: int,
    num_groups_run: int,
    full_batches: list[int],
    start_group: int | None = None,
) -> dict[int, tuple[int, list[int]]]:
    if start_group is None:
        start_group = get_env_int("AUTOTUNE_START_GROUP", 0)
    if start_group < 0:
        raise ValueError(f"AUTOTUNE_START_GROUP 必须是非负整数，当前值: {start_group}")

    if not full_batches:
        raise ValueError("full_batches 不能为空")

    effective_groups = min(num_groups_run, len(full_batches))
    if effective_groups != num_groups_run:
        print(
            f"警告: num_groups_run={num_groups_run} 大于 batch 数量={len(full_batches)}，"
            f"自动收敛为 {effective_groups}",
            file=sys.stderr,
        )

    run_batches = split_run_batches(full_batches, effective_groups)
    if len(run_batches) != effective_groups:
        raise ValueError(
            f"Unexpected run batch split result: expected {effective_groups}, got {len(run_batches)}"
        )

    group_ids: dict[int, tuple[int, list[int]]] = {}
    for local_group_id in range(effective_groups):
        global_group_id = start_group + local_group_id
        device_id = device_start_id + (local_group_id % device_count)
        group_ids[global_group_id] = (device_id, run_batches[local_group_id])
    return group_ids


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "print-launcher-env":
        sys.exit(print_launcher_env_exports())
    if len(sys.argv) >= 2 and sys.argv[1] == "check-patched-environment":
        strict_training = "--strict-training" in sys.argv[2:]
        sys.exit(check_patched_environment(strict_training=strict_training))
    if len(sys.argv) >= 2 and sys.argv[1] == "ensure-op-tests-namespace":
        # Usage:
        #   python common.py ensure-op-tests-namespace /zhenxin/fused_moe /zhenxin
        if len(sys.argv) < 4:
            print(
                "错误: ensure-op-tests-namespace 需要两个参数: <fused_moe_dir> <namespace_root>",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            target, created = ensure_op_tests_namespace(
                fused_moe_dir=sys.argv[2],
                namespace_root=sys.argv[3],
            )
        except Exception as exc:
            print(f"错误: {exc}", file=sys.stderr)
            sys.exit(1)
        if created:
            print(f"已创建兼容命名空间链接: {target}")
        else:
            print(f"兼容命名空间已存在: {target}")
        sys.exit(0)
    print(
        "Usage:\n"
        "  python common.py print-launcher-env\n"
        "  python common.py check-patched-environment [--strict-training]\n"
        "  python common.py ensure-op-tests-namespace <fused_moe_dir> <namespace_root>",
        file=sys.stderr,
    )
    sys.exit(2)
