# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
import re
import ast
from decimal import Decimal
from textwrap import shorten
import json
import sys
import os
import shutil
from pathlib import Path

from boltops.utility.triton_capability import (
    get_triton_capabilities,
    get_triton_config_dir,
)

# 定义特殊键值映射表
SPECIAL_KEY_MAP = {
    1: 1,    # 第1个是1
    2: 2,    # 第2个是2
    3: 4,    # 第3个是4
    4: 8,    # 第4个是8
    5: 16,   # 第5个是16
    6: 24,   # 第6个是24
    7: 32,   # 第7个是32
    8: 64    # 第8个是64
}

_SAVED_CONFIG_KEYS = (
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "N_TILES_PER_CTA",
    "GROUP_SIZE_M",
    "COMBINE_SCALE_LOAD",
    "USE_MLS_LOAD",
    "a_sorted",
    "bottom_a_use_mls_load",
    "instruction_sched_variant",
    "schedule_hint",
    "sched_latency",
    "kpack",
    "use_block_pingpong",
    "buffer_cache_swizzle",
    "num_warps",
    "num_stages",
)


def _parse_config_value(raw_value):
    raw_value = raw_value.strip()
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def _parse_autotune_config(config_text):
    return {
        key: _parse_config_value(value)
        for key, value in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,\n]+)", config_text
        )
    }


def _parse_log_cases(log_content):
    # 匹配测试用例起始点（Config 1: 且包含fused_moe_kernel）
    test_case_pattern = r'(?=Config 1:.*?fused_moe_kernel:)'
    test_cases = re.split(test_case_pattern, log_content, flags=re.DOTALL)

    all_results = []

    for case in test_cases[1:]:  # 跳过第一个空匹配
        case = case.strip()
        if not case:
            continue

        # 提取当前测试用例的所有Config块
        config_blocks = re.findall(
            r'(Config (\d+):\s*({.*?})\s*.*?)(?=Config \d+:|$)',
            case,
            re.DOTALL
        )

        case_best = None  # (config_num, config_params, kernel1_best_id, kernel2_best_id, total_time, kernel1_params, kernel2_params)

        for full_block, config_num_str, config_params in config_blocks:
            config_num = int(config_num_str)

            # 提取两个kernel块
            kernel_blocks = re.findall(
                r'fused_moe_kernel:(.*?)(?=(?:fused_moe_kernel:|Config \d+:|$))',
                full_block,
                re.DOTALL
            )
            if len(kernel_blocks) < 2:
                continue

            # 解析时间数据
            kernel_best_times = []  # [(best_id, best_time, best_params), ...]

            for block in kernel_blocks[:2]:  # 只处理前两个kernel块
                block_times = {}
                block_params = {}
                best_autotune_id = None

                for m in re.finditer(
                    r'^\s*(\*?)\s*(\d+)\s*:\s*(.*?)\n\s*:\s*times\([^)]+\):\s*([\d.eE+-]+)\s*ms',
                    block,
                    re.MULTILINE,
                ):
                    is_best = m.group(1) == '*'
                    autotune_id = int(m.group(2))
                    config_values = _parse_autotune_config(m.group(3))
                    time_val = float(m.group(4))

                    required = {
                        "BLOCK_SIZE_N",
                        "BLOCK_SIZE_K",
                        "GROUP_SIZE_M",
                        "num_warps",
                        "num_stages",
                    }
                    if not required.issubset(config_values):
                        continue

                    block_times[autotune_id] = time_val
                    block_params[autotune_id] = {
                        key: config_values[key]
                        for key in _SAVED_CONFIG_KEYS
                        if key in config_values
                    }

                    if is_best:
                        best_autotune_id = autotune_id

                if best_autotune_id is None:
                    continue

                # 获取最好的autotune ID及其相关信息
                best_time = block_times[best_autotune_id]
                best_params = block_params[best_autotune_id]
                kernel_best_times.append((best_autotune_id, best_time, best_params))
            # 有效性检查
            if len(kernel_best_times) != 2:
                continue

            # 计算总时间
            total_time = kernel_best_times[0][1] + kernel_best_times[1][1]

            # 更新测试用例最佳配置
            current_entry = (
                config_num,
                config_params.strip(),
                kernel_best_times[0][0],  # kernel1 best autotune id
                kernel_best_times[1][0],  # kernel2 best autotune id
                total_time,
                kernel_best_times[0][2],  # kernel1 best params (warps, stages)
                kernel_best_times[1][2]   # kernel2 best params (warps, stages)
            )

            if (case_best is None) or (total_time < case_best[4]):
                case_best = current_entry

        if case_best:
            all_results.append(case_best)

    return all_results


def parse_log_file(log_content):
    batch_matches = list(
        re.finditer(
            r'^(?:Test:|Test Count \d+:)\s*m=(\d+),',
            log_content,
            re.MULTILINE,
        )
    )
    if not batch_matches:
        # Keep compatibility with older standalone logs that do not print m.
        return _parse_log_cases(log_content)

    all_results = []
    for idx, match in enumerate(batch_matches):
        section_end = (
            batch_matches[idx + 1].start()
            if idx + 1 < len(batch_matches)
            else len(log_content)
        )
        section_results = _parse_log_cases(
            log_content[match.start():section_end]
        )
        batch_size = int(match.group(1))
        all_results.extend((*result, batch_size) for result in section_results)
    return all_results


def get_special_key(idx):
    """根据索引获取特殊键值"""
    if idx in SPECIAL_KEY_MAP:
        return SPECIAL_KEY_MAP[idx]
    else:
        # 超过64后，使用2的幂次：128, 256, 512, 1024, ...
        return 2 ** (idx - 8 + 6)  # 从2^7(128)开始


def _result_key_and_fields(idx, result):
    if len(result) == 8:
        return result[-1], result[:-1]
    return get_special_key(idx), result


def print_results(results):
    print("┌──────────────┬────────────┬──────────────┬──────────────┬──────────────┬───────────────────────────────────────────────────────────────────────────────────────┐")
    print("│ 特殊键值     │ Config ID  │ Kernel1 ID   │ Kernel2 ID   │ 总耗时 (ms)  │ 配置参数                                                                                │")
    print("├──────────────┼────────────┼──────────────┼──────────────┼──────────────┼───────────────────────────────────────────────────────────────────────────────────────┤")
    for idx, result in enumerate(results, 1):
        special_key, fields = _result_key_and_fields(idx, result)
        cfg_num, params, k1_id, k2_id, total, k1_params, k2_params = fields

        # 构建参数字符串
        full_params = (
            f"{params}, "
            f"K1({k1_params}), "
            f"K2({k2_params})"
        )
        shortened_params = shorten(full_params, width=160, placeholder="...")
        print(f"│ {special_key:12d} │ {cfg_num:10d} │ {k1_id:12d} │ {k2_id:12d} │ {total:12.4f} │ {shortened_params:160} │")
    print("└──────────────┴────────────┴──────────────┴──────────────┴──────────────┴───────────────────────────────────────────────────────────────────────────────────────┘")

def save_results_to_json(results, k1_filename, k2_filename):
    json_data_k1 = {}
    json_data_k2 = {}

    for idx, result in enumerate(results, 1):
        special_key, fields = _result_key_and_fields(idx, result)
        cfg_num, params, k1_id, k2_id, total, k1_params, k2_params = fields
        config_dict = ast.literal_eval(params)
        block_size_m = config_dict["BLOCK_SIZE_M"]

        entry_k1 = {
            "BLOCK_SIZE_M": block_size_m,
            **{
                key: value
                for key, value in k1_params.items()
                if key not in ("a_sorted", "bottom_a_use_mls_load")
            },
        }
        entry_k2 = {
            "BLOCK_SIZE_M": block_size_m,
            **{key: value for key, value in k2_params.items() if key != "a_sorted"},
        }
        # Keep the bridge layout and the actual Gemm2 load choice separate.
        # Legacy configs without a_sorted still use bottom_a_use_mls_load
        # for both; new configs preserve the independently tuned switches.
        bottom_a_sorted = k2_params.get(
            "a_sorted", config_dict.get("bottom_a_use_mls_load")
        )
        if bottom_a_sorted is not None:
            entry_k2["a_sorted"] = bool(bottom_a_sorted)
        if "bottom_a_use_mls_load" not in entry_k2 and bottom_a_sorted is not None:
            entry_k2["bottom_a_use_mls_load"] = bool(bottom_a_sorted)

        json_data_k1[f"{special_key}"] = entry_k1
        json_data_k2[f"{special_key}"] = entry_k2

    json_data_k1 = dict(sorted(json_data_k1.items(), key=lambda item: int(item[0])))
    json_data_k2 = dict(sorted(json_data_k2.items(), key=lambda item: int(item[0])))

    # 保存两个文件
    with open(k1_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data_k1, f, ensure_ascii=False, indent=4)

    with open(k2_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data_k2, f, ensure_ascii=False, indent=4)


def resolve_boltops_moe_config_dir() -> Path:
    config_root = (
        Path(__file__).resolve().parents[2]
        / "fused_moe"
        / "triton"
        / "configs"
    )
    return get_triton_config_dir(config_root, get_triton_capabilities())

def main():
    # 默认使用autotune_v5.log作为日志文件和默认输出JSON文件名
    log_file = "autotune_v5.log"
    output_json1 = "E=256,N=256,K=7168,topk=8,arch=gfx936.json"
    output_json2 = "E=256,N=256,K=7168,topk=8,arch=gfx936,is_bottom=True.json"

    # 解析命令行参数
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    if len(sys.argv) >=4:
        output_json1 = sys.argv[2]
        output_json2 = sys.argv[3]

    if len(sys.argv) > 4:
        print(f"错误: 输入参数过多")
        sys.exit(1)

    print(f"正在解析日志文件: {log_file}")
    print(f"输出JSON文件1: {output_json1}")
    print(f"输出JSON文件2: {output_json2}")

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()

        parsed_results = parse_log_file(log_content)
        print_results(parsed_results)

        save_results_to_json(parsed_results, output_json1, output_json2)
        print(f"解析完成，结果已保存到 {output_json1} 和 {output_json2}")

        # 将生成的JSON文件拷贝到指定目录
        try:
            config_dir = resolve_boltops_moe_config_dir()
            config_dir.mkdir(parents=True, exist_ok=True)
            print(f"目标配置目录: {config_dir}")

            for filename in [output_json1, output_json2]:
                src = Path(filename).resolve()
                shutil.copy(src, config_dir)
                print(f"已拷贝: {src} -> {config_dir}")
            print(f"所有文件已成功拷贝到 {config_dir}")
        except PermissionError:
            print(f"错误: 没有权限写入目录 {config_dir}，请使用sudo运行或检查权限")
        except Exception as e:
            print(f"错误: 拷贝文件时出错: {str(e)}")
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 '{log_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 解析日志文件时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
