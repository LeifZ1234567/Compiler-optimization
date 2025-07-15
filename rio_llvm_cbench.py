import os
import sys
import time
import random
import numpy as np
import argparse # 导入命令行参数库

# 导入你的工具类
from util import Util, cbench # 假设 cbench 列表在 util.py 中

# --- 脚本配置 ---
iters = 180          # 每个程序跑多少次随机配置评估
begin2end = 5        # 对每个程序，完整跑几轮随机搜索以统计

random.seed(123)     # 设置随机种子

# --- 实例化你的工具类并获取 LLVM 编译选项 ---
util = Util()
# 使用 util.py 中定义的 LLVM 编译选项 (flags)
options = util.gain_flags() # 或者 util.gcc_flags，取决于你在 util.py 中如何命名
n_flags = len(options)
print(f"加载了 {n_flags} 个 LLVM 编译选项用于随机搜索。")

# --- 移除硬编码的 LLVM Pass 相关命令 ---
# cmd0, cmd1, cmd2, cmd3, cmd4, cmd5 不再需要

# --- 帮助函数 (generate_opts 可能不再需要) ---
# 如果 util.run_procedure 直接接受二进制向量，这个函数可以删除
# def generate_opts(independent):
#     # ... (保留以防万一，但可能不用)

# --- 目标函数 (核心修改) ---
def get_objective_score(independent, program_name):
    """
    评估给定的编译选项配置（二进制向量）对特定程序的效果。
    使用 util.run_procedure 处理编译、运行和加速比计算。

    Args:
        independent (list or np.array): 代表编译选项选择的二进制向量 (0 或 1)。
        program_name (str): 要评估的 Cbench 程序名 (例如 'automotive_susan_c')。

    Returns:
        float: 相对于基准 (clang -O3) 的负加速比。随机搜索的目标是最小化这个值。
               如果评估失败，返回一个很差的值（例如 0.0）。
    """
    # 确保 'independent' 是 run_procedure 期望的格式 (例如 list of ints)
    flags_binary_vector = list(map(int, independent))

    print(f"\n评估程序 {program_name} 的配置...")
    # print(f"二进制配置: {flags_binary_vector}") # 可选的调试输出

    # 调用你的工具函数，它负责所有事情：
    # 更新 Makefile, make clean, make, 运行, 获取时间, 对比 O3, 返回 -speedup
    try:
        # 调用 run_procedure，它计算相对于 O3 的加速比并返回负值
        negative_speedup = util.run_procedure(program_name, flags_binary_vector)

        # 处理 run_procedure 可能失败的情况
        if negative_speedup is None or not isinstance(negative_speedup, (int, float)):
             print(f"警告: 程序 {program_name} 评估失败。返回差评。")
             # 返回一个非常差的分数 (0 加速比对应无穷差的负加速比，这里用 0.0)
             # 注意：如果你的 run_procedure 返回的是 0 表示失败，这里逻辑是对的。
             # 如果 run_procedure 返回 None 表示失败，上面的判断已经处理了。
             return 0.0

        print(f"程序 {program_name} 的评估结果 (负加速比): {negative_speedup:.4f}")
        return negative_speedup

    except Exception as e:
        print(f"程序 {program_name} 评估过程中发生错误: {e}")
        # 返回一个非常差的分数
        return 0.0

# --- 主随机搜索循环 ---
def main(program_name):
    """对指定的程序执行随机搜索优化过程。"""
    evaluated_configs_int = set() # 存储已经评估过的配置的整数表示，避免重复
    results_indep = [] # 存储评估过的配置 (二进制向量)
    results_dep = []   # 存储对应的目标分数 (-speedup)
    results_ts = []    # 存储评估完成的时间戳

    print(f"开始对程序 {program_name} 进行 {iters} 次随机配置评估...")
    b = time.time() # 开始计时

    evaluated_count = 0
    attempts = 0
    max_attempts = iters * 5 # 设置尝试上限，防止因配置重复而死循环

    while evaluated_count < iters and attempts < max_attempts:
        attempts += 1
        # 1. 生成一个随机配置
        # 用随机整数代表一个配置组合
        x = random.randint(0, 2 ** n_flags - 1) # 确保覆盖所有组合

        # 2. 检查配置是否已经评估过
        if x not in evaluated_configs_int:
            evaluated_configs_int.add(x) # 标记为已尝试

            # 将整数 x 转换为二进制向量
            comb = bin(x).replace('0b', '')
            comb = '0' * (n_flags - len(comb)) + comb
            conf_vector = [int(s) for s in comb]

            # 3. 评估这个新配置
            score = get_objective_score(conf_vector, program_name)

            # 4. 记录结果 (仅当评估成功时)
            # 假设 0.0 是表示评估失败的分数
            if score != 0.0:
                results_indep.append(conf_vector)
                results_dep.append(score)
                results_ts.append(time.time() - b)
                evaluated_count += 1
                print(f"已评估 {evaluated_count}/{iters} 个有效配置。")
            else:
                 print("跳过失败的配置评估。")
        # else: # 如果需要，可以打印重复信息
        #     print("跳过重复配置。")

    total_time = time.time() - b
    print(f'随机搜索完成，耗时: {total_time:.2f} 秒。成功评估 {evaluated_count} 个配置。')
    if attempts >= max_attempts:
        print(f"警告：达到了最大尝试次数 {max_attempts}，可能存在大量重复配置或评估失败。")

    # 返回评估得到的分数列表和时间戳列表
    return results_dep, results_ts

# --- 脚本执行入口 ---
if __name__ == '__main__':
    # --- 添加命令行参数解析 ---
    parser = argparse.ArgumentParser(description="使用随机搜索为 Cbench 程序调优 LLVM 编译选项。")
    parser.add_argument("program_name", help="要调优的 Cbench 程序名 (例如 automotive_susan_c)。", choices=cbench) # 使用 util.py 中的 cbench 列表
    args = parser.parse_args()

    program_to_tune = args.program_name
    print(f"=== 开始对程序 {program_to_tune} 进行随机搜索调优 ===")

    init_time = time.time() # 记录开始时间
    # --- 多次运行以获取统计数据 ---
    all_stats = [] # 存储每次运行得到的分数 (-speedup) 列表
    all_times = [] # 存储每次运行得到的时间戳列表

    for i in range(begin2end):
        print(f"\n--- 第 {i+1}/{begin2end} 轮随机搜索 ---")
        run_dep, run_ts = main(program_to_tune)
        if run_dep and run_ts: # 仅当运行成功且有结果时记录
            print(f'第 {i+1} 轮搜索完成，得到 {len(run_dep)} 个有效分数。')
            all_stats.append(run_dep)
            all_times.append(run_ts)
        else:
            print(f"第 {i+1} 轮搜索失败或未产生有效结果。")

    # --- 处理并打印最终结果 ---
    if not all_stats:
        print("\n所有运行均未成功完成。无法计算统计数据。退出。")
        sys.exit(1)

    end_time = time.time() # 记录结束时间
    total_time = end_time - init_time
    print(f"\n总耗时: {total_time:.2f} 秒。")
    print("\n\n=== 最终结果统计 ===")
    print(f"目标程序: {program_to_tune}")
    print(f"每次运行评估次数: {iters}")
    print(f"运行总轮数: {len(all_stats)}")

    # 计算每轮搜索找到的最佳分数 (-speedup)
    best_scores_per_run = [min(run_data) for run_data in all_stats if run_data]

    if best_scores_per_run:
        # 找到的最佳负加速比的平均值和标准差
        avg_best_score = np.mean(best_scores_per_run)
        std_best_score = np.std(best_scores_per_run)
        # 转换为加速比更容易理解
        avg_best_speedup = -avg_best_score

        print(f"\n平均找到的最佳加速比: {avg_best_speedup:.4f}")
        print(f"最佳负加速比的标准差: {std_best_score:.4f}")

        # 如果需要，可以打印每次运行的最佳分数列表
        # print(f"每轮找到的最佳负加速比: {[round(s, 4) for s in best_scores_per_run]}")

        # 也可以分析所有评估过的分数，但这对于随机搜索意义不大，关键是找到的最好值
        # all_evaluated_scores = [score for run_data in all_stats for score in run_data]
        # print(f"\n所有评估过的配置的平均负加速比: {np.mean(all_evaluated_scores):.4f}")
    else:
        print("\n没有有效的最佳分数可供统计。")

    print("\n=== 调优完成 ===")