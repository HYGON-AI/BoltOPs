#!/usr/bin/env python3
import argparse
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from boltops.utility.triton_capability import get_triton_capabilities


VALID_DEVICE_NAMES = {"zd", "bmz", "nmz", "yy"}
# Site code → GPU ISA used in MoE JSON filenames (arch=...).
DEVICE_ARCH_MAP = {
    "zd": "gfx928",
    "bmz": "gfx936",
    "nmz": "gfx938",
    "yy": "gfx92a",
}
VALID_TEST_TYPES = {
    "int4",
    "int8",
    "int8_channel",
    "int4int8",
    "int4int8_channel",
    "fp8",
    "fp8_channel",
    "bf16",
}
VALID_TP_VALUES = {1, 2, 4, 8}
PRESET_CONFIGS: dict[str, dict[str, object]] = {
    "full": {
        "tp_list": "1,2,4,8",
        "num_experts": 256,
        "hidden_size_default": 384,
        "ep_size": 1,
        "skip_compile_only": False,
        "num_groups_compile": 8,
        "auto_select_devices_by_hy_smi": True,
        "hy_smi_max_vram_pct": 30.0,
    }
}
DEFAULT_BATCH_SIZES = "1,2,4,8,16,24,32,64,128,256,512,1024,2048,4096,8192,16384,32768"
TEST_TYPES_SUPPORT_SHARED_EXPERTS = {"bf16", "fp8", "fp8_channel", "int8", "int8_channel"}
TEST_TYPES_SUPPORT_BLOCK_SHAPE = {"fp8", "int8"}
FIXED_GROUP_SIZE_BY_TEST: dict[str, int] = {
    "int4": 64,
    "int4int8": 64,
}
DEFAULT_BLOCK_SHAPE_BY_TEST: dict[str, tuple[int, int]] = {
    "fp8": (128, 128),
    "int8": (128, 128),
}
DEFAULT_FFN_HIDDEN_SIZE_BY_TEST: dict[str, int] = {
    "bf16": 7168,
    "fp8": 7168,
    "fp8_channel": 7168,
    "int8": 7168,
    "int8_channel": 7168,
    "int4": 7168,
    "int4int8": 7168,
    "int4int8_channel": 7168,
}
VALID_MOE_ACTIVATIONS = {
    "silu",
    "gelu",
    "gelu_tanh",
    "gelu_pytorch_tanh",
    "situ",
    "swigluoai",
    "swiglustep",
    "silu_no_mul",
    "gelu_no_mul",
    "gelu_tanh_no_mul",
    "gelu_pytorch_tanh_no_mul",
    "relu2_no_mul",
    "relu2",
}


def parse_int_list(raw: str, name: str) -> list[int]:
    tokens = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]
    if not tokens:
        raise ValueError(f"{name} 不能为空")

    values: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"{name} 包含非法整数: {token!r}") from exc
        values.append(value)
    return values


def parse_tp_list(raw: str) -> list[int]:
    values = parse_int_list(raw, "tp_list")
    for value in values:
        if value not in VALID_TP_VALUES:
            raise ValueError(f"TP 仅支持 1,2,4,8，当前值: {value}")
    return values


def parse_block_shape(raw: str, name: str = "block_shape") -> tuple[int, int]:
    tokens = [token.strip() for token in re.split(r"[,xX ]+", raw.strip()) if token.strip()]
    if len(tokens) != 2:
        raise ValueError(f"{name} 格式必须是 M,K（如 128,128），当前值: {raw!r}")
    try:
        block_m = int(tokens[0])
        block_k = int(tokens[1])
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值: {raw!r}") from exc
    if block_m <= 0 or block_k <= 0:
        raise ValueError(f"{name} 必须 > 0，当前值: {(block_m, block_k)}")
    return block_m, block_k


def validate_batch_sizes(raw: str) -> str:
    values = parse_int_list(raw, "batch_sizes")
    for value in values:
        if value <= 0:
            raise ValueError(f"batch_sizes 必须全部 > 0，当前值: {value}")
    return ",".join(str(x) for x in values)


def parse_bool(raw: str, name: str) -> bool:
    token = raw.strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} 仅支持 true/false (或 1/0)，当前值: {raw!r}")


def print_effective_config(config_items: list[tuple[str, object]]) -> None:
    print("================================")
    print("Effective Config:")
    for key, value in config_items:
        if key == "expected_config_jsons" and isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    {item}")
            continue
        print(f"  {key}={value}")
    print("================================")


def configure_output_streams() -> None:
    # 当 stdout/stderr 重定向到文件（如 log.training）时，开启行缓冲，保证步骤日志及时落盘。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def resolve_runtime_dirs() -> tuple[Path, Path, str]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    in_installed_layout = any(part in {"site-packages", "dist-packages"} for part in script_dir.parts)
    runtime_layout = "installed" if in_installed_layout else "develop"
    # Wheel and editable installs share the same package layout. Executable
    # helpers live beside this module; generated logs belong in the caller's cwd.
    return script_dir, cwd, runtime_layout


def build_expected_config_json_paths(
    *,
    log_root_dir: Path,
    device_name: str,
    test_type: str,
    tp_list: list[int],
    num_experts: int,
    num_shared_experts: int,
    ep_size: int,
    hidden_size_default: int,
    block_shape: tuple[int, int] | None,
) -> list[str]:
    paths: list[str] = []
    local_e = (num_experts // ep_size) + num_shared_experts
    for tp in tp_list:
        if hidden_size_default % tp != 0:
            paths.append(f"tp{tp}: <invalid hidden_size_default/tp>")
            continue
        local_n = hidden_size_default // tp
        try:
            expected_json0, expected_json1 = expected_json_names(
                device_name=device_name,
                test_type=test_type,
                local_e=local_e,
                local_n=local_n,
                block_shape=block_shape,
            )
        except ValueError as exc:
            paths.append(f"tp{tp}: <error {exc}>")
            continue
        log_dir = log_root_dir / f"logs_{device_name}_{test_type}_tp{tp}"
        paths.append(f"tp{tp} top={log_dir / expected_json0}")
        paths.append(f"tp{tp} bottom={log_dir / expected_json1}")
    return paths


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path | None = None,
    check: bool = True,
) -> int:
    if log_file is None:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return proc.returncode

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as fp:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def check_patch_contract(
    *,
    script_dir: Path,
    env: dict[str, str],
    strict_training: bool,
) -> tuple[bool, str]:
    cmd = [sys.executable, str(script_dir / "common.py"), "check-patched-environment"]
    if strict_training:
        cmd.append("--strict-training")
    proc = subprocess.run(
        cmd,
        cwd=str(script_dir),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, output


def remove_dir_with_retries(path: Path, *, retries: int = 5, base_sleep_s: float = 0.2) -> None:
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep_s * (attempt + 1))


def run_benchmark_step(
    *,
    script_dir: Path,
    test_script_name: str,
    tp_env: dict[str, str],
    tp: int,
    ep_size: int,
    break_on_error: bool,
) -> int:
    print("--------------------------------")
    print("Step4: Run benchmark/correctness")
    print("--------------------------------")
    bench_env = tp_env.copy()
    bench_env["TRAINING_MODE"] = "0"
    bench_env["BENCHMARK_MODE"] = "1"

    bench_rc = run_cmd(
        ["pytest", "-v", "-s", test_script_name],
        cwd=script_dir,
        env=bench_env,
        check=False,
    )
    if bench_rc != 0:
        message = f"TP={tp} benchmark/correctness 失败 (rc={bench_rc})，请查看终端输出"
        if break_on_error:
            print(f"错误: {message}", file=sys.stderr)
            return bench_rc
        print(f"警告: {message}")

    print("--------------------------------")
    print(f"All steps for TP={tp}, EP_SIZE={ep_size} completed")
    print("--------------------------------")
    return 0


def kill_moe_processes() -> tuple[list[int], list[int]]:
    # 仅匹配 fused_moe 相关训练/调度进程，避免误杀其他任务。
    patterns = [
        re.compile(r"\bpytest\b.*\btune_moe_.*\.py\b"),
        re.compile(r"\btune_moe_(bf16|fp8|fp8_channel|int4|int4int8|int4int8_channel|int8|int8_channel)\.py\b"),
        re.compile(r"\btune_moe_cli\.py\b"),
        re.compile(r"\brun_test_moe_training\.sh\b"),
    ]

    try:
        ps_output = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception as exc:
        raise RuntimeError(f"读取进程列表失败: {exc}") from exc

    self_pid = os.getpid()
    targets: list[int] = []
    for line in ps_output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_str, cmd = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if any(p.search(cmd) for p in patterns):
            targets.append(pid)

    targets = sorted(set(targets))
    if not targets:
        return [], []

    term_sent: list[int] = []
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
            term_sent.append(pid)
        except (ProcessLookupError, PermissionError):
            continue

    time.sleep(1.0)
    kill_sent: list[int] = []
    for pid in term_sent:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            kill_sent.append(pid)
        except (ProcessLookupError, PermissionError):
            continue

    return term_sent, kill_sent


def run_parallel_pytest_groups(
    *,
    script_dir: Path,
    test_script_name: str,
    env_base: dict[str, str],
    group_ids: list[int],
    log_dir: Path,
    log_name_builder,
    triton_compile_only: bool,
    break_on_error: bool,
    startup_sleep_sec: float = 2.0,
) -> bool:
    processes: list[tuple[int, subprocess.Popen, object, Path]] = []
    failed_groups: list[tuple[int, int, Path]] = []

    for group_id in group_ids:
        env = env_base.copy()
        env["TRAINING_MODE"] = "1"
        env["CUR_GROUP"] = str(group_id)
        env["TRITON_COMPILE_ONLY"] = "1" if triton_compile_only else "0"

        log_file = log_dir / log_name_builder(group_id)
        log_fp = log_file.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            ["pytest", "-v", "-s", test_script_name],
            cwd=str(script_dir),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        processes.append((group_id, proc, log_fp, log_file))
        print(f"Started group {group_id} (PID: {proc.pid}) -> {log_file}")
        time.sleep(startup_sleep_sec)

    for group_id, proc, log_fp, log_file in processes:
        return_code = proc.wait()
        log_fp.close()
        if return_code != 0:
            failed_groups.append((group_id, return_code, log_file))

    if failed_groups:
        if triton_compile_only:
            summary = ", ".join(f"group={gid}, rc={rc}" for gid, rc, _ in failed_groups)
            print(f"警告: compile-only 分组存在失败，继续后续步骤: {summary}")
            return False

        summary = ", ".join(f"group={gid}, rc={rc}" for gid, rc, _ in failed_groups)
        message = f"pytest groups failed: {summary}"
        if break_on_error:
            raise RuntimeError(message)
        print(f"警告: {message}")
        return False
    return True


def merge_group_logs(
    *,
    log_dir: Path,
    merged_log_file: Path,
    group_ids: list[int],
    log_name_builder,
) -> None:
    merged_log_file.parent.mkdir(parents=True, exist_ok=True)
    with merged_log_file.open("w", encoding="utf-8") as out_fp:
        for group_id in group_ids:
            log_file = log_dir / log_name_builder(group_id)
            if not log_file.exists():
                print(f"警告: 缺少 group 日志 {log_file}")
                continue
            out_fp.write(log_file.read_text(encoding="utf-8", errors="replace"))


def normalize_kernel_name_in_log(log_file: Path, test_type: str) -> None:
    if not log_file.exists():
        return
    text = log_file.read_text(encoding="utf-8", errors="replace")
    if test_type == "int4":
        text = text.replace("fused_moe_kernel_gptq_awq", "fused_moe_kernel")
    elif test_type == "int4int8":
        text = text.replace("fused_moe_kernel_gptq_awq_w4a8", "fused_moe_kernel")
    elif test_type == "int4int8_channel":
        text = text.replace("fused_moe_kernel_gptq_awq_w4a8_channelwise_bot", "fused_moe_kernel")
        text = text.replace("fused_moe_kernel_gptq_awq_w4a8_channelwise", "fused_moe_kernel")
    else:
        text = re.sub(r"fused_moe\w*_kernel", "fused_moe_kernel", text)
    log_file.write_text(text, encoding="utf-8")


def resolve_launcher_env(script_dir: Path, env: dict[str, str]) -> dict[str, str]:
    cmd = [sys.executable, str(script_dir / "common.py"), "print-launcher-env"]
    proc = subprocess.run(cmd, cwd=str(script_dir), env=env, text=True, capture_output=True, check=False)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError("解析/选择设备失败，请检查 AUTOTUNE 设备参数与 hy-smi 状态")

    resolved = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        payload = line[len("export ") :]
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        resolved[key] = value
    return resolved


def expected_json_names(
    *,
    device_name: str,
    test_type: str,
    local_e: int,
    local_n: int,
    block_shape: tuple[int, int] | None = None,
) -> tuple[str, str]:
    arch = DEVICE_ARCH_MAP[device_name]

    if test_type == "bf16":
        suffix0 = ""
        suffix1 = ",is_bottom=True"
    elif test_type == "int8":
        if block_shape is None:
            block_shape = DEFAULT_BLOCK_SHAPE_BY_TEST["int8"]
        suffix0 = f",dtype=int8_w8a8,block_shape=[{block_shape[0]},{block_shape[1]}]"
        suffix1 = f",dtype=int8_w8a8,is_bottom=True,block_shape=[{block_shape[0]},{block_shape[1]}]"
    elif test_type == "int8_channel":
        suffix0 = ",dtype=int8_w8a8"
        suffix1 = ",dtype=int8_w8a8,is_bottom=True"
    elif test_type == "int4":
        suffix0 = ",dtype=int4_w4a16"
        suffix1 = ",dtype=int4_w4a16,is_bottom=True"
    elif test_type in {"int4int8", "int4int8_channel"}:
        suffix0 = ",dtype=int4_w4a8"
        suffix1 = ",dtype=int4_w4a8,is_bottom=True"
    elif test_type == "fp8":
        if block_shape is None:
            block_shape = DEFAULT_BLOCK_SHAPE_BY_TEST["fp8"]
        suffix0 = f",dtype=fp8_w8a8,block_shape=[{block_shape[0]},{block_shape[1]}]"
        suffix1 = f",dtype=fp8_w8a8,is_bottom=True,block_shape=[{block_shape[0]},{block_shape[1]}]"
    elif test_type == "fp8_channel":
        suffix0 = ",dtype=fp8_w8a8"
        suffix1 = ",dtype=fp8_w8a8,is_bottom=True"
    else:  # pragma: no cover
        raise ValueError(f"未知 test_type: {test_type}")

    base = f"E={local_e},N={local_n},arch={arch}"
    return f"{base}{suffix0}.json", f"{base}{suffix1}.json"


def parse_args() -> argparse.Namespace:
    full = PRESET_CONFIGS["full"]
    parser = argparse.ArgumentParser(
        description="MoE autotune/training CLI (Python replacement for run_test_moe_training.sh)"
    )
    parser.add_argument("device_name", choices=sorted(VALID_DEVICE_NAMES))
    parser.add_argument("test_type", choices=sorted(VALID_TEST_TYPES))
    moe_group = parser.add_argument_group("核心 MoE 参数（常用）")
    runtime_group = parser.add_argument_group("运行与调度参数（高级）")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS.keys()),
        default="full",
        help=argparse.SUPPRESS,
    )

    moe_group.add_argument(
        "--tp-list",
        default=str(full["tp_list"]),
        help=f"TP 列表，逗号或空格分隔。默认: {full['tp_list']}",
    )
    moe_group.add_argument(
        "--num-experts",
        type=int,
        default=int(full["num_experts"]),
        help=f"MoE 总 experts 数。默认: {full['num_experts']}",
    )
    moe_group.add_argument(
        "--hidden-size-default",
        type=int,
        default=int(full["hidden_size_default"]),
        help=f"基准 hidden size，实际按 hidden_size_default/TP。默认: {full['hidden_size_default']}",
    )
    moe_group.add_argument(
        "--ep-size",
        type=int,
        default=int(full["ep_size"]),
        help=f"EP 大小。默认: {full['ep_size']}",
    )
    moe_group.add_argument(
        "--num-shared-experts",
        type=int,
        default=0,
        help="共享 experts 数。默认: 0（仅 bf16/fp8/fp8_channel/int8/int8_channel 支持）",
    )
    moe_group.add_argument("--top-k", type=int, default=None, help="覆盖 MOE_TOP_K。默认: 脚本内部默认")
    moe_group.add_argument(
        "--ffn-hidden-size",
        type=int,
        default=None,
        help="覆盖 MOE_FFN_HIDDEN_SIZE。默认: 脚本内部默认",
    )
    moe_group.add_argument(
        "--block-shape",
        type=str,
        default=None,
        help="覆盖 MOE_BLOCK_SHAPE，格式 M,K（如 128,128）。int8/fp8 默认: 128,128",
    )
    moe_group.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help=f"覆盖 MOE_BATCH_SIZES。默认: {DEFAULT_BATCH_SIZES}",
    )
    moe_group.add_argument(
        "--activation",
        type=str,
        default=None,
        help=(
            "覆盖 MOE_ACTIVATION。支持: "
            "silu/gelu/gelu_tanh/situ/swigluoai/swiglustep/"
            "silu_no_mul/gelu_no_mul/gelu_tanh_no_mul/relu2(_no_mul)"
        ),
    )
    moe_group.add_argument(
        "--is-gated",
        type=str,
        default=None,
        help="覆盖 MOE_IS_GATED。支持 true/false (或 1/0)，默认由 activation 自动推断",
    )
    moe_group.add_argument(
        "--activation-situ-beta",
        type=float,
        default=None,
        help="SITU 的 beta；--activation situ 时必填且必须大于 0",
    )
    moe_group.add_argument(
        "--activation-situ-linear-beta",
        type=float,
        default=None,
        help="SITU 可选的 linear_beta；设置时必须大于 0",
    )

    runtime_group.add_argument(
        "--skip-compile-only",
        action="store_true",
        default=bool(full["skip_compile_only"]),
        help=f"跳过 compile-only 阶段。默认: {full['skip_compile_only']}",
    )
    runtime_group.add_argument(
        "--num-groups-compile",
        type=int,
        default=int(full["num_groups_compile"]),
        help=f"compile-only 分组数。默认: {full['num_groups_compile']}",
    )

    runtime_group.add_argument("--device-start-id", type=int, default=None, help="起始设备 ID。默认: 自动选择")
    runtime_group.add_argument("--device-count", type=int, default=None, help="连续设备数量。默认: 自动选择")
    runtime_group.add_argument(
        "--disable-auto-select-devices-by-hy-smi",
        action="store_true",
        default=not bool(full["auto_select_devices_by_hy_smi"]),
        help=f"禁用 hy-smi 自动选卡。默认自动选卡: {full['auto_select_devices_by_hy_smi']}",
    )
    runtime_group.add_argument(
        "--hy-smi-max-vram-pct",
        type=float,
        default=float(full["hy_smi_max_vram_pct"]),
        help=f"hy-smi 自动选卡使用率阈值(%%，VRAM/HCU 共用)。默认: {full['hy_smi_max_vram_pct']}",
    )

    runtime_group.add_argument(
        "--break-on-error",
        nargs="?",
        const="true",
        default="true",
        help="遇到错误立即退出。默认: true；如需继续后续 TP 可传 false",
    )
    runtime_group.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="仅执行 benchmark(pytest)，不执行训练/编译/解析步骤。默认: 关闭",
    )
    runtime_group.add_argument(
        "--kill",
        action="store_true",
        default=False,
        help="立即杀死所有 MoE autotune/训练相关进程并退出",
    )
    runtime_group.add_argument("--dry-run", action="store_true", default=False, help="仅打印参数与计划，不执行训练")
    return parser.parse_args()


def main() -> int:
    configure_output_streams()
    args = parse_args()
    if args.kill:
        try:
            term_sent, kill_sent = kill_moe_processes()
        except Exception as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1

        if not term_sent:
            print("未发现可结束的 MoE 相关进程。")
            return 0

        print(f"已发送 SIGTERM 到 {len(term_sent)} 个进程: {term_sent}")
        if kill_sent:
            print(f"仍存活进程已发送 SIGKILL: {kill_sent}")
        return 0

    try:
        script_dir, log_root_dir, runtime_layout = resolve_runtime_dirs()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    test_script_name = f"tune_moe_{args.test_type}.py"
    test_script_path = script_dir / test_script_name
    if not test_script_path.exists():
        print(f"错误: 未找到测试脚本 {test_script_path}", file=sys.stderr)
        return 1

    if args.ep_size <= 0:
        print(f"错误: --ep-size 必须 > 0，当前值: {args.ep_size}", file=sys.stderr)
        return 1
    if args.num_experts <= 0:
        print(f"错误: --num-experts 必须 > 0，当前值: {args.num_experts}", file=sys.stderr)
        return 1
    if args.hidden_size_default <= 0:
        print(f"错误: --hidden-size-default 必须 > 0，当前值: {args.hidden_size_default}", file=sys.stderr)
        return 1
    if args.num_groups_compile <= 0:
        print(f"错误: --num-groups-compile 必须 > 0，当前值: {args.num_groups_compile}", file=sys.stderr)
        return 1
    if args.num_shared_experts < 0:
        print(f"错误: --num-shared-experts 必须 >= 0，当前值: {args.num_shared_experts}", file=sys.stderr)
        return 1

    try:
        break_on_error = parse_bool(args.break_on_error, "--break-on-error")
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    try:
        tp_list = parse_tp_list(args.tp_list)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    supports_shared_experts = args.test_type in TEST_TYPES_SUPPORT_SHARED_EXPERTS
    supports_block_shape = args.test_type in TEST_TYPES_SUPPORT_BLOCK_SHAPE

    if args.num_shared_experts > 0 and not supports_shared_experts:
        print(
            f"错误: test_type={args.test_type} 不支持 --num-shared-experts，当前值: {args.num_shared_experts}",
            file=sys.stderr,
        )
        return 1

    if args.block_shape is not None and not supports_block_shape:
        print(
            f"错误: test_type={args.test_type} 不支持 --block-shape，当前值: {args.block_shape}",
            file=sys.stderr,
        )
        return 1

    try:
        if supports_block_shape:
            if args.block_shape is not None:
                resolved_block_shape = parse_block_shape(args.block_shape, "--block-shape")
            else:
                resolved_block_shape = DEFAULT_BLOCK_SHAPE_BY_TEST[args.test_type]
        else:
            resolved_block_shape = None
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.batch_sizes is not None:
        try:
            batch_sizes_value = validate_batch_sizes(args.batch_sizes)
        except ValueError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1
    else:
        # 默认始终注入统一 batch 列表，避免各 tune 脚本内部默认不一致。
        batch_sizes_value = DEFAULT_BATCH_SIZES

    activation_value: str | None = None
    is_gated_value: bool | None = None
    if args.activation is not None:
        activation_value = args.activation.strip().lower()
        if activation_value == "":
            print("错误: --activation 不能为空字符串", file=sys.stderr)
            return 1
        if activation_value not in VALID_MOE_ACTIVATIONS:
            print(
                f"错误: --activation 不支持 {args.activation!r}，可选: {sorted(VALID_MOE_ACTIVATIONS)}",
                file=sys.stderr,
            )
            return 1

    if args.is_gated is not None:
        try:
            is_gated_value = parse_bool(args.is_gated, "--is-gated")
        except ValueError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1

    if args.activation_situ_beta is not None and args.activation_situ_beta <= 0:
        print("错误: --activation-situ-beta 必须大于 0", file=sys.stderr)
        return 1
    if (
        args.activation_situ_linear_beta is not None
        and args.activation_situ_linear_beta <= 0
    ):
        print("错误: --activation-situ-linear-beta 必须大于 0", file=sys.stderr)
        return 1
    if activation_value == "situ" and args.activation_situ_beta is None:
        print("错误: --activation situ 时必须设置 --activation-situ-beta", file=sys.stderr)
        return 1

    env = os.environ.copy()
    # develop：把仓库根加入 PYTHONPATH，便于未 editable 安装时导入 boltops。
    # installed：依赖 site-packages 中的 boltops，勿把 parents[2] 误加进 PYTHONPATH。
    if runtime_layout == "develop":
        repo_root = script_dir.parents[2]
        inherited_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_root) + (
            os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
        )
    # 避免继承外部残留环境变量导致行为漂移：默认按 CLI 参数/预设执行。
    for key in [
        "MOE_BATCH_SIZES",
        "MOE_TOP_K",
        "MOE_FFN_HIDDEN_SIZE",
        "MOE_NUM_SHARED_EXPERTS",
        "MOE_BLOCK_SHAPE",
        "MOE_ACTIVATION",
        "MOE_IS_GATED",
        "MOE_ACTIVATION_SITU_BETA",
        "MOE_ACTIVATION_SITU_LINEAR_BETA",
        "AUTOTUNE_DEVICE_START_ID",
        "AUTOTUNE_DEVICE_COUNT",
        "AUTOTUNE_NUM_GROUPS_RUN",
        "AUTO_SELECT_DEVICES_BY_HY_SMI",
        "HY_SMI_MAX_VRAM_PCT",
    ]:
        env.pop(key, None)

    env["AUTOTUNE_TEST_TYPE"] = args.test_type
    # 先固定 split-k=0，避免 CLI 过于复杂；后续若需要可再恢复高级参数。
    env["SPLITK_SIZE"] = "0"

    if args.top_k is not None:
        env["MOE_TOP_K"] = str(args.top_k)
    if args.ffn_hidden_size is not None:
        env["MOE_FFN_HIDDEN_SIZE"] = str(args.ffn_hidden_size)
    env["MOE_BATCH_SIZES"] = batch_sizes_value
    if supports_shared_experts:
        env["MOE_NUM_SHARED_EXPERTS"] = str(args.num_shared_experts)
    if resolved_block_shape is not None:
        env["MOE_BLOCK_SHAPE"] = f"{resolved_block_shape[0]},{resolved_block_shape[1]}"
    if activation_value is not None:
        env["MOE_ACTIVATION"] = activation_value
    if is_gated_value is not None:
        env["MOE_IS_GATED"] = "1" if is_gated_value else "0"
    if args.activation_situ_beta is not None:
        env["MOE_ACTIVATION_SITU_BETA"] = str(args.activation_situ_beta)
    if args.activation_situ_linear_beta is not None:
        env["MOE_ACTIVATION_SITU_LINEAR_BETA"] = str(
            args.activation_situ_linear_beta
        )

    if args.device_start_id is not None:
        env["AUTOTUNE_DEVICE_START_ID"] = str(args.device_start_id)
    if args.device_count is not None:
        env["AUTOTUNE_DEVICE_COUNT"] = str(args.device_count)

    auto_select_devices = not args.disable_auto_select_devices_by_hy_smi
    env["AUTO_SELECT_DEVICES_BY_HY_SMI"] = "1" if auto_select_devices else "0"
    env["HY_SMI_MAX_VRAM_PCT"] = str(args.hy_smi_max_vram_pct)

    try:
        triton_capabilities = get_triton_capabilities()
    except Exception as exc:
        print(f"错误: 无法检测 Triton capability: {exc}", file=sys.stderr)
        return 1

    try:
        resolved_env = resolve_launcher_env(script_dir, env)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    env.update(resolved_env)

    # 当 batch 数量小于运行分组数时，自动收敛运行分组，避免收集阶段报错。
    # 例如 --batch-sizes 1 时，只需要 1 个 run group。
    try:
        batch_count = len(parse_int_list(env.get("MOE_BATCH_SIZES", ""), "MOE_BATCH_SIZES"))
        run_group_count = int(env.get("AUTOTUNE_NUM_GROUPS_RUN", "1"))
        if run_group_count > batch_count:
            print(
                "警告: 运行分组数大于 batch 数量，自动收敛 "
                f"AUTOTUNE_NUM_GROUPS_RUN: {run_group_count} -> {batch_count}"
            )
            env["AUTOTUNE_NUM_GROUPS_RUN"] = str(batch_count)
    except Exception:
        # 这里不影响主流程；参数已在前文校验，理论上不会触发。
        pass

    patch_check_ok, patch_check_output = check_patch_contract(
        script_dir=script_dir,
        env=env,
        strict_training=(not args.benchmark and not args.dry_run),
    )
    if patch_check_output:
        print(patch_check_output)
    if not patch_check_ok:
        print(
            "错误: autotune patch 环境检查失败。可尝试：\n"
            "  1) 使用统一入口命令: boltops-fused-moe-tune-cli <device_name> <test_type>\n"
            "  2) 检查 boltops 包可导入: python -c \"import boltops; print(boltops.__file__)\"\n"
            "  3) 检查 patch: python -c \"import boltops.tools.fused_moe_triton_tune.autotune_patches\"",
            file=sys.stderr,
        )
        return 1

    expected_config_jsons = build_expected_config_json_paths(
        log_root_dir=log_root_dir,
        device_name=args.device_name,
        test_type=args.test_type,
        tp_list=tp_list,
        num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts if supports_shared_experts else 0,
        ep_size=args.ep_size,
        hidden_size_default=args.hidden_size_default,
        block_shape=resolved_block_shape,
    )

    effective_items: list[tuple[str, object]] = [
        ("runtime_layout", runtime_layout),
        ("script_dir", script_dir),
        ("log_root_dir", log_root_dir),
        ("triton_package_version", triton_capabilities.package_version),
        ("triton_capability", triton_capabilities.version),
        ("triton_backend", triton_capabilities.backend),
        ("triton_config_dir", triton_capabilities.config_dir_name),
        (
            "use_block_pingpong",
            triton_capabilities.supports("use_block_pingpong"),
        ),
        ("device_name", args.device_name),
        ("test_type", args.test_type),
        ("test_script_name", test_script_name),
        ("tp_list", tp_list),
        ("num_experts", args.num_experts),
        ("hidden_size_default", args.hidden_size_default),
        ("ep_size", args.ep_size),
        ("num_shared_experts", env.get("MOE_NUM_SHARED_EXPERTS", "<unsupported_or_default_0>")),
        ("top_k", env.get("MOE_TOP_K", "<script_default>")),
        ("ffn_hidden_size", env.get("MOE_FFN_HIDDEN_SIZE", "<script_default>")),
        ("block_shape", env.get("MOE_BLOCK_SHAPE", "<script_default_or_unused>")),
        ("activation", env.get("MOE_ACTIVATION", "<script_default>")),
        ("is_gated", env.get("MOE_IS_GATED", "<auto_by_activation>")),
        ("activation_situ_beta", env.get("MOE_ACTIVATION_SITU_BETA", "<unset>")),
        (
            "activation_situ_linear_beta",
            env.get("MOE_ACTIVATION_SITU_LINEAR_BETA", "<unset>"),
        ),
        ("fixed_group_size", FIXED_GROUP_SIZE_BY_TEST.get(args.test_type, "<n/a>")),
        ("batch_sizes", env.get("MOE_BATCH_SIZES")),
        ("splitk_size", env.get("SPLITK_SIZE")),
        ("skip_compile_only", args.skip_compile_only),
        ("num_groups_compile", args.num_groups_compile),
        ("device_start_id", env.get("AUTOTUNE_DEVICE_START_ID")),
        ("device_count", env.get("AUTOTUNE_DEVICE_COUNT")),
        ("num_groups_run", env.get("AUTOTUNE_NUM_GROUPS_RUN")),
        ("auto_select_devices_by_hy_smi", auto_select_devices),
        ("hy_smi_max_usage_pct", args.hy_smi_max_vram_pct),
        ("break_on_error", break_on_error),
        ("benchmark", args.benchmark),
        ("dry_run", args.dry_run),
        ("expected_config_jsons", expected_config_jsons),
    ]
    print_effective_config(effective_items)

    if args.dry_run:
        print("Dry-run mode: 仅展示参数，不执行训练。")
        return 0

    # 训练模式默认先清理已有 MoE 相关进程，避免并发任务互相干扰。
    if not args.benchmark:
        try:
            term_sent, kill_sent = kill_moe_processes()
        except Exception as exc:
            print(f"错误: 自动清理已有 MoE 进程失败: {exc}", file=sys.stderr)
            return 1
        if term_sent:
            print(f"检测到已有 MoE 进程，已发送 SIGTERM 清理: {term_sent}")
            if kill_sent:
                print(f"仍存活进程已发送 SIGKILL: {kill_sent}")
        else:
            print("未检测到运行中的 MoE 相关进程。")

    # 每次训练任务启动前清理本次 tp-list 对应目录，避免历史日志干扰；
    # benchmark-only 模式需要复用已有 config，故不清理。
    if not args.benchmark:
        for tp in tp_list:
            tp_log_dir = log_root_dir / f"logs_{args.device_name}_{args.test_type}_tp{tp}"
            if tp_log_dir.exists():
                try:
                    remove_dir_with_retries(tp_log_dir)
                except OSError as exc:
                    print(
                        f"错误: 清理日志目录失败: {tp_log_dir} ({exc})。"
                        "请先确认没有并发训练进程占用该目录，或先执行 --kill。",
                        file=sys.stderr,
                    )
                    return 1

    for tp in tp_list:
        if args.hidden_size_default % tp != 0:
            print(
                f"错误: hidden_size_default={args.hidden_size_default} 不能被 tp={tp} 整除",
                file=sys.stderr,
            )
            return 1

        current_hidden_size = args.hidden_size_default // tp
        current_ffn_hidden_size = args.ffn_hidden_size
        if current_ffn_hidden_size is None:
            current_ffn_hidden_size = DEFAULT_FFN_HIDDEN_SIZE_BY_TEST.get(args.test_type)

        if resolved_block_shape is not None:
            block_m, block_k = resolved_block_shape
            if current_hidden_size % block_k != 0:
                print(
                    f"错误: test_type={args.test_type} 默认/当前 block_shape=[{block_m},{block_k}]，"
                    f"要求 MOE_HIDDEN_SIZE 可被 {block_k} 整除；当前 TP={tp} 时为 {current_hidden_size}",
                    file=sys.stderr,
                )
                return 1
            if current_ffn_hidden_size is not None and current_ffn_hidden_size % block_k != 0:
                print(
                    f"错误: test_type={args.test_type} 默认/当前 block_shape=[{block_m},{block_k}]，"
                    f"要求 MOE_FFN_HIDDEN_SIZE 可被 {block_k} 整除；当前值: {current_ffn_hidden_size}",
                    file=sys.stderr,
                )
                return 1

        fixed_group_size = FIXED_GROUP_SIZE_BY_TEST.get(args.test_type)
        if fixed_group_size is not None:
            if current_hidden_size % fixed_group_size != 0:
                print(
                    f"错误: test_type={args.test_type} 固定 group_size={fixed_group_size}，"
                    f"要求 MOE_HIDDEN_SIZE 可被 {fixed_group_size} 整除；当前 TP={tp} 时为 {current_hidden_size}",
                    file=sys.stderr,
                )
                return 1
            if current_ffn_hidden_size is not None and current_ffn_hidden_size % fixed_group_size != 0:
                print(
                    f"错误: test_type={args.test_type} 固定 group_size={fixed_group_size}，"
                    f"要求 MOE_FFN_HIDDEN_SIZE 可被 {fixed_group_size} 整除；当前值: {current_ffn_hidden_size}",
                    file=sys.stderr,
                )
                return 1

        tp_env = env.copy()
        tp_env["EP_SIZE"] = str(args.ep_size)
        tp_env["MOE_NUM_EXPERTS"] = str(args.num_experts)
        tp_env["MOE_HIDDEN_SIZE"] = str(current_hidden_size)
        tp_env["MOE_TP"] = str(tp)

        print("================================")
        print(f"Start TP={tp}: MOE_HIDDEN_SIZE={current_hidden_size}")
        if not args.benchmark:
            print(f"LOG_DIR={log_root_dir / f'logs_{args.device_name}_{args.test_type}_tp{tp}'}")
        print("================================")

        if args.benchmark:
            # benchmark 模式只跑 pytest benchmark，不执行/复用 Step1/2/3。
            bench_ret = run_benchmark_step(
                script_dir=script_dir,
                test_script_name=test_script_name,
                tp_env=tp_env,
                tp=tp,
                ep_size=args.ep_size,
                break_on_error=break_on_error,
            )
            if bench_ret != 0:
                return bench_ret
            continue

        # training + parse + benchmark (默认流程)
        log_dir = log_root_dir / f"logs_{args.device_name}_{args.test_type}_tp{tp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        local_e = (args.num_experts // args.ep_size) + (
            args.num_shared_experts if supports_shared_experts else 0
        )
        local_n = current_hidden_size
        expected_json0, expected_json1 = expected_json_names(
            device_name=args.device_name,
            test_type=args.test_type,
            local_e=local_e,
            local_n=local_n,
            block_shape=resolved_block_shape,
        )
        out_json0 = log_dir / expected_json0
        out_json1 = log_dir / expected_json1
        tp_env["NUM_GROUPS_COMPILE"] = str(args.num_groups_compile)

        # Step1
        if args.skip_compile_only:
            print("--------------------------------")
            print("Step1: Skip compile only")
            print("--------------------------------")
        else:
            print("--------------------------------")
            print("Step1: Run compile only")
            print("--------------------------------")
            compile_group_ids = list(range(args.num_groups_compile))
            compile_ok = run_parallel_pytest_groups(
                script_dir=script_dir,
                test_script_name=test_script_name,
                env_base=tp_env,
                group_ids=compile_group_ids,
                log_dir=log_dir,
                log_name_builder=lambda gid: f"autotune_v12_{args.device_name}_{args.ep_size}_{args.test_type}_compile_only_g{gid}.log",
                triton_compile_only=True,
                break_on_error=break_on_error,
            )
            if not compile_ok:
                print(f"警告: TP={tp} compile-only 存在失败分组")

        # Step2
        print("--------------------------------")
        print("Step2: Run autotune")
        print("--------------------------------")
        num_groups_run = int(tp_env["AUTOTUNE_NUM_GROUPS_RUN"])
        run_group_ids = list(range(num_groups_run))
        print(f"Run groups: {run_group_ids}")

        run_ok = run_parallel_pytest_groups(
            script_dir=script_dir,
            test_script_name=test_script_name,
            env_base=tp_env,
            group_ids=run_group_ids,
            log_dir=log_dir,
            log_name_builder=lambda gid: f"autotune_v12_{args.device_name}_{args.ep_size}_{args.test_type}_g{gid}.log",
            triton_compile_only=False,
            break_on_error=break_on_error,
        )
        if not run_ok:
            print(f"警告: TP={tp} autotune 分组存在失败")

        merged_log_file = log_dir / f"autotune_v12_{args.device_name}_{args.ep_size}_{args.test_type}.log"
        print(f"Merging logs to: {merged_log_file}")
        merge_group_logs(
            log_dir=log_dir,
            merged_log_file=merged_log_file,
            group_ids=run_group_ids,
            log_name_builder=lambda gid: f"autotune_v12_{args.device_name}_{args.ep_size}_{args.test_type}_g{gid}.log",
        )
        normalize_kernel_name_in_log(merged_log_file, args.test_type)

        # Step3
        print("--------------------------------")
        print("Step3: Parse logs")
        print("--------------------------------")
        print(f"Output config json: {out_json0}")
        print(f"Output config json(bottom): {out_json1}")

        try:
            run_cmd(
                [
                    sys.executable,
                    str(script_dir / "moe_log_parser.py"),
                    str(merged_log_file),
                    str(out_json0),
                    str(out_json1),
                ],
                cwd=script_dir,
                env=tp_env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            message = f"TP={tp} 日志解析失败 (rc={exc.returncode})"
            if break_on_error:
                print(f"错误: {message}", file=sys.stderr)
                return exc.returncode or 1
            print(f"警告: {message}，继续后续步骤")

        bench_ret = run_benchmark_step(
            script_dir=script_dir,
            test_script_name=test_script_name,
            tp_env=tp_env,
            tp=tp,
            ep_size=args.ep_size,
            break_on_error=break_on_error,
        )
        if bench_ret != 0:
            return bench_ret

    print(f"All TP runs completed! TP_LIST={tp_list}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
