import re
from decimal import Decimal
from textwrap import shorten
import json
import sys
import os
import shutil
from pathlib import Path

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


def parse_log_file(log_content):
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

                # for m in re.finditer(
                #       r'^\s*(\*?)\s*(\d+)\s*:\s*BLOCK_SIZE_N:\s*(\d+),\s*BLOCK_SIZE_K:\s*(\d+),\s*GROUP_SIZE_M:\s*(\d+)[\s\S]*?COMBINE_SCALE_LOAD:\s*([a-zA-Z0-9-]+)[\s\S]*?instruction_sched_variant:\s*([a-zA-Z0-9-]+)[\s\S]*?num_warps:\s*(\d+)[\s\S]*?num_stages:\s*(\d+)[\s\S]*?times\([^)]+\):\s*([\d.eE+-]+)\s*ms',
                #       block,
                #       re.MULTILINE
                # ):
                for m in re.finditer(
                      r'^\s*(\*?)\s*(\d+)\s*:\s*BLOCK_SIZE_N:\s*(\d+),\s*BLOCK_SIZE_K:\s*(\d+),\s*GROUP_SIZE_M:\s*(\d+)[\s\S]*?COMBINE_SCALE_LOAD:\s*([a-zA-Z0-9-]+)[\s\S]*?USE_MLS_LOAD:\s*([a-zA-Z0-9-]+)[\s\S]*?instruction_sched_variant:\s*([a-zA-Z0-9-]+)[\s\S]*?sched_latency:\s*([a-zA-Z0-9-]+)[\s\S]*?kpack:\s*(\d+)[\s\S]*?num_warps:\s*(\d+)[\s\S]*?num_stages:\s*(\d+)[\s\S]*?times\([^)]+\):\s*([\d.eE+-]+)\s*ms',
                      block,
                      re.MULTILINE
                ):
                    is_best = m.group(1) == '*'
                    autotune_id = int(m.group(2))
                    block_n = int(m.group(3))  # BLOCK_SIZE_N
                    block_k = int(m.group(4))  # BLOCK_SIZE_K
                    group_m = int(m.group(5))  # GROUP_SIZE_M
                    combine_scale_load = True if str(m.group(6)) == "True" else False
                    use_mls_load = True if str(m.group(7)) == "True" else False
                    instruction_sched_variant = str(m.group(8))  # instruction_sched_variant
                    sched_latency = str(m.group(9))
                    kpack = int(m.group(10))
                    num_warps = int(m.group(11))
                    num_stages = int(m.group(12))
                    time_val = float(m.group(13))

                    block_times[autotune_id] = time_val
                    # 保存所有参数到元组中
                    block_params[autotune_id] = (block_n, block_k, group_m, combine_scale_load, use_mls_load, instruction_sched_variant, sched_latency, kpack, num_warps, num_stages)

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

def get_special_key(idx):
    """根据索引获取特殊键值"""
    if idx in SPECIAL_KEY_MAP:
        return SPECIAL_KEY_MAP[idx]
    else:
        # 超过64后，使用2的幂次：128, 256, 512, 1024, ...
        return 2 ** (idx - 8 + 6)  # 从2^7(128)开始

def print_results(results):
    print("┌──────────────┬────────────┬──────────────┬──────────────┬──────────────┬───────────────────────────────────────────────────────────────────────────────────────┐")
    print("│ 特殊键值     │ Config ID  │ Kernel1 ID   │ Kernel2 ID   │ 总耗时 (ms)  │ 配置参数                                                                                │")
    print("├──────────────┼────────────┼──────────────┼──────────────┼──────────────┼───────────────────────────────────────────────────────────────────────────────────────┤")
    for idx, (cfg_num, params, k1_id, k2_id, total, k1_params, k2_params) in enumerate(results, 1):
        # 生成特殊键名
        special_key = get_special_key(idx)

        # 解包参数
        k1_block_n, k1_block_k, k1_group_m, k1_combine_scale_load, k1_use_mls_load, k1_instruction_sched_variant, k1_sched_latency, k1_kpack, k1_warps, k1_stages = k1_params
        k2_block_n, k2_block_k, k2_group_m, k2_combine_scale_load, k2_use_mls_load, k2_instruction_sched_variant, k2_sched_latency, k2_kpack, k2_warps, k2_stages = k2_params

        # 构建参数字符串
        full_params = (
            f"{params}, "
            f"K1(BLOCK_SIZE_N:{k1_block_n}, BLOCK_SIZE_K:{k1_block_k}, GROUP_SIZE_M:{k1_group_m}, COMBINE_SCALE_LOAD:{k1_combine_scale_load}, USE_MLS_LOAD:{k1_use_mls_load}, instruction_sched_variant:{k1_instruction_sched_variant}, sched_latency:{k1_sched_latency}, kpack:{k1_kpack}, num_warps:{k1_warps}, num_stages:{k1_stages}), "
            f"K2(BLOCK_SIZE_N:{k2_block_n}, BLOCK_SIZE_K:{k2_block_k}, GROUP_SIZE_M:{k2_group_m}, COMBINE_SCALE_LOAD:{k2_combine_scale_load}, USE_MLS_LOAD:{k2_use_mls_load}, instruction_sched_variant:{k2_instruction_sched_variant}, sched_latency:{k2_sched_latency}, kpack:{k2_kpack}, num_warps:{k2_warps}, num_stages:{k2_stages})"
        )
        shortened_params = shorten(full_params, width=160, placeholder="...")
        print(f"│ {special_key:12d} │ {cfg_num:10d} │ {k1_id:12d} │ {k2_id:12d} │ {total:12.4f} │ {shortened_params:160} │")
    print("└──────────────┴────────────┴──────────────┴──────────────┴──────────────┴───────────────────────────────────────────────────────────────────────────────────────┘")

def save_results_to_json(results, k1_filename, k2_filename):
    json_data_k1 = {}
    json_data_k2 = {}

    for idx, (cfg_num, params, k1_id, k2_id, total, k1_params, k2_params) in enumerate(results, 1):
        config_dict = eval(params)  # 注意：实际使用中建议用更安全的方式解析

        # 解析两个kernel的参数
        k1_block_n, k1_block_k, k1_group_m, k1_combine_scale_load, k1_use_mls_load, k1_instruction_sched_variant, k1_sched_latency, k1_kpack, k1_warps, k1_stages = k1_params
        k2_block_n, k2_block_k, k2_group_m, k2_combine_scale_load, k2_use_mls_load, k2_instruction_sched_variant, k2_sched_latency, k2_kpack, k2_warps, k2_stages = k2_params
        block_size_m = config_dict["BLOCK_SIZE_M"]

        # 构建Kernel1条目
        entry_k1 = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": k1_block_n,
            "BLOCK_SIZE_K": k1_block_k,
            "GROUP_SIZE_M": k1_group_m,
            "COMBINE_SCALE_LOAD": k1_combine_scale_load,
            "USE_MLS_LOAD": k1_use_mls_load,
            "instruction_sched_variant": k1_instruction_sched_variant,
            "sched_latency": k1_sched_latency,
            "kpack": k1_kpack,
            "num_warps": k1_warps,
            "num_stages": k1_stages
        }

        # 构建Kernel2条目
        entry_k2 = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": k2_block_n,
            "BLOCK_SIZE_K": k2_block_k,
            "GROUP_SIZE_M": k2_group_m,
            "COMBINE_SCALE_LOAD": k2_combine_scale_load,
            "USE_MLS_LOAD": k2_use_mls_load,
            "instruction_sched_variant": k2_instruction_sched_variant,
            "sched_latency": k2_sched_latency,
            "kpack": k2_kpack,
            "num_warps": k2_warps,
            "num_stages": k2_stages
        }

        # 生成特殊键名
        special_key = get_special_key(idx)

        json_data_k1[f"{special_key}"] = entry_k1
        json_data_k2[f"{special_key}"] = entry_k2

    # 保存两个文件
    with open(k1_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data_k1, f, ensure_ascii=False, indent=4)

    with open(k2_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data_k2, f, ensure_ascii=False, indent=4)


def resolve_boltops_moe_config_dir() -> Path:
    # Prefer the active boltops installation, with a source-tree fallback.
    try:
        import boltops

        return (
            Path(boltops.__file__).resolve().parent
            / "fused_moe"
            / "triton"
            / "configs"
            / "moe"
        )
    except Exception:
        script_dir = Path(__file__).resolve().parent
        fallback_candidates = [
            (
                script_dir.parents[2]
                / "boltops"
                / "fused_moe"
                / "triton"
                / "configs"
                / "moe"
            ).resolve(),
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                return candidate
        return fallback_candidates[0]

def main():
    # 默认使用autotune_v5.log作为日志文件和默认输出JSON文件名
    log_file = "autotune_v5.log"
    output_json1 = "E=256,N=256,arch=gfx936.json"
    output_json2 = "E=256,N=256,arch=gfx936,is_bottom=True.json"

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
