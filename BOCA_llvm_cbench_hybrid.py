import os
import sys
import time
import random
import numpy as np
# import copy # 似乎未使用
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import math
import argparse

try:
    from util import Util, cbench
except ImportError:
    print("错误: 无法从 util.py 导入 'Util' 或 'cbench'。")
    print("请确保 util.py 与此脚本在同一目录且定义正确。")
    sys.exit(1)

# --- BOCA 配置 ---
random.seed(456)
iters = 60
begin2end = 5
fnum = int(os.environ.get('FNUM', 8))
decay = float(os.environ.get('DECAY', 0.5))
scale = float(os.environ.get('SCALE', 10))
offset = int(os.environ.get('OFFSET', 20)) # offset 应该是整数迭代次数
rnum0 = int(os.environ.get('RNUM', 2 ** 8))

# --- 编译选项序列 (与 OpenTuner 修改一致) ---
KNOWN_SEQUENCES = [
    ['-sroa', '-jump-threading'],
    ['-mem2reg', '-gvn', '-instcombine'],
    ['-mem2reg', '-gvn', '-prune-eh'],
    ['-mem2reg', '-gvn', '-dse'],
    ['-mem2reg', '-loop-sink', '-loop-distribute'],
    ['-early-cse-memssa', '-instcombine'],
    ['-early-cse-memssa', '-dse'],
    ['-lcssa', '-loop-unroll'],
    ['-licm', '-gvn', '-instcombine'],
    ['-licm', '-gvn', '-prune-eh'],
    ['-licm', '-gvn', '-dse'],
    ['-memcpyopt', '-loop-distribute']
]
n_sequences = len(KNOWN_SEQUENCES)
print(f"已加载 {n_sequences} 个预定义的编译选项序列。")

# --- 全局工具实例和单个独立编译选项 ---
try:
    util = Util()
    # 这里 individual_options 可以基于聚类或通用列表
    # 为了简化，我们先用通用的 gain_flags()
    # 如果需要根据程序聚类，可以在主逻辑中针对特定程序进行调整
    individual_options = util.gain_flags() # 从 util.py 获取 LLVM 单个编译选项列表
    if not individual_options or not isinstance(individual_options, list):
        print("错误: util.gain_flags() 未返回有效的单个选项列表。")
        sys.exit(1)
    n_individual_flags = len(individual_options)
    print(f"已加载 {n_individual_flags} 个独立的 LLVM 编译选项。")
    
    # 总的编码维度 = 单个选项数 + 序列数
    total_dimensions = n_individual_flags + n_sequences
    print(f"混合编码的总维度: {total_dimensions}")

except Exception as e:
    print(f"初始化 Util 类或获取编译选项时出错: {e}")
    sys.exit(1)


# --- 解码函数：将混合编码向量转换为实际编译选项列表 ---
def decode_hybrid_config(hybrid_config_vector):
    """
    将混合编码的二进制向量解码为实际的编译选项列表。
    """
    if len(hybrid_config_vector) != total_dimensions:
        raise ValueError(f"混合编码向量长度 ({len(hybrid_config_vector)}) 与总维度 ({total_dimensions}) 不匹配。")

    active_flags_set = set()

    # 1. 处理单个独立选项
    for i in range(n_individual_flags):
        if hybrid_config_vector[i] == 1:
            active_flags_set.add(individual_options[i])

    # 2. 处理序列选项
    for i in range(n_sequences):
        if hybrid_config_vector[n_individual_flags + i] == 1:
            active_flags_set.update(KNOWN_SEQUENCES[i])
    
    return list(active_flags_set)


# --- 目标函数：评估混合编码的编译选项的效果 ---
def get_objective_score(hybrid_flags_vector, program_name_to_eval):
    """
    评估给定的混合编码编译选项组合对特定程序的效果。
    """
    hybrid_flags_binary_representation = list(map(int, hybrid_flags_vector))
    
    try:
        actual_compiler_flags = decode_hybrid_config(hybrid_flags_binary_representation)
    except ValueError as ve:
        print(f"解码混合编码时出错: {ve}")
        return 0.0 # 解码失败，返回差分数

    if not actual_compiler_flags:
        print(f"警告: 为 {program_name_to_eval} 解码后无有效编译选项。返回差分数。")
        return 0.0 # 没有选项，可能导致编译失败或默认行为

    print(f"\n正在评估 {program_name_to_eval} 的配置...")
    # print(f"混合编码向量: {hybrid_flags_binary_representation}") # 调试时取消注释
    # print(f"实际编译选项: {' '.join(actual_compiler_flags)}") # 调试时取消注释

    try:
        # util.run_procedure 现在接收实际的选项列表，而不是二进制向量
        # 我们需要修改 util.run_procedure 或者在这里适配
        # 假设 util.run_procedure 的第二个参数是标志列表
        # 如果 util.run_procedure 期望的是二进制向量和 util.gcc_flags,
        # 那么我们需要将 actual_compiler_flags 转换回相对于 util.gcc_flags 的二进制向量
        # 为了简单起见，我们假设 util.run_procedure2 可以接收实际的flag字符串列表
        # （根据你提供的 EnhancedHybridBA.py，它调用 run_procedure2(self.compile_files, options)）
        
        # 使用 run_procedure_runtime 来获取原始运行时间，然后计算speedup
        # 首先获取基准时间（如果尚未获取）
        # 注意：这里为了简化，每次都获取O3，实际应用中可以缓存
        baseline_time = util.run_procedure_runtime(program_name_to_eval, ["-O3"]) # 假设 ["-O3"] 是有效的
        if baseline_time is None or baseline_time <= 0:
            print(f"警告: 无法获取 {program_name_to_eval} 的 -O3 基准时间。")
            return 0.0

        current_time = util.run_procedure_runtime(program_name_to_eval, actual_compiler_flags)
        if current_time is None or current_time <= 0:
             print(f"警告: {program_name_to_eval} 的评估失败或返回无效时间。返回差分数。")
             return 0.0
        
        speedup = baseline_time / current_time
        negative_speedup = -speedup # BOCA 最小化负加速比

        print(f"{program_name_to_eval} 的评估结果 (负加速比): {negative_speedup:.4f}")
        return negative_speedup

    except Exception as e:
        print(f"为 {program_name_to_eval} 执行 util.run_procedure_runtime 时出错: {e}")
        return 0.0

# --- 配置生成：将整数转换为混合编码二进制向量 ---
def generate_conf(integer_representation):
    # 注意：这里的整数范围是 0 到 2**total_dimensions - 1
    binary_string = bin(integer_representation).replace('0b', '')
    padded_binary_string = '0' * (total_dimensions - len(binary_string)) + binary_string
    config_vector = [int(bit) for bit in padded_binary_string]
    return config_vector

# --- BOCA 算法的辅助类和函数 (大部分保持不变，但作用于 total_dimensions) ---
class get_exchange(object):
    def __init__(self, incumbent_important_features):
        self.incumbent = incumbent_important_features # [(feature_index, value), ...]

    def to_next(self, randomly_selected_feature_indices):
        next_config = [0] * total_dimensions # 使用 total_dimensions
        for f_idx in randomly_selected_feature_indices:
            next_config[f_idx] = random.randint(0,1) # 随机0/1扰动
        for f_idx, val in self.incumbent:
            next_config[f_idx] = val
        return next_config

def do_search(training_configurations, model_labeled, best_score_so_far_eta, current_rnum):
    try:
        feature_importances = model_labeled.feature_importances_
        if len(feature_importances) != total_dimensions:
             print(f"警告: 模型特征重要性长度 ({len(feature_importances)}) 与总维度 ({total_dimensions}) 不匹配。使用等权重。")
             feature_importances = np.ones(total_dimensions) / total_dimensions
    except AttributeError:
        print("警告: 模型没有 feature_importances_ 属性。使用等权重。")
        feature_importances = np.ones(total_dimensions) / total_dimensions

    sorted_features = sorted([[i, imp] for i, imp in enumerate(feature_importances)], key=lambda x: x[1], reverse=True)
    
    num_important_features_to_consider = min(fnum, total_dimensions)
    selected_important_features = sorted_features[:num_important_features_to_consider]
    all_feature_indices = list(range(total_dimensions)) # 使用 total_dimensions

    neighborhood_centers = []
    # 确保 2**num_important_features_to_consider 不会过大
    max_centers_to_generate = 2**min(num_important_features_to_consider, 10) # 例如，最多2^10个中心
    for i in range(max_centers_to_generate):
        binary_combination = bin(i).replace('0b', '').zfill(num_important_features_to_consider)
        current_center_config_for_important = []
        # 当 max_centers_to_generate < 2**num_important_features_to_consider 时，
        # binary_combination 对应的其实不是所有的重要特征组合，这里需要修正
        # 或者限制 num_important_features_to_consider 的上限
        if len(binary_combination) > num_important_features_to_consider: # 不应该发生
            continue

        for k, bit_char in enumerate(binary_combination):
            if k < len(selected_important_features): # 确保索引有效
                feature_index = selected_important_features[k][0]
                value = int(bit_char)
                current_center_config_for_important.append((feature_index, value))
        if current_center_config_for_important: # 仅当成功构建时添加
            neighborhood_centers.append(get_exchange(current_center_config_for_important))
    
    if not neighborhood_centers and selected_important_features: # 如果没有中心但有重要特征，至少创建一个
         neighborhood_centers.append(get_exchange([(sf[0], random.randint(0,1)) for sf in selected_important_features]))


    neighbor_configs_to_evaluate = []
    for center_generator in neighborhood_centers:
        for _ in range(max(1, int(current_rnum))):
            num_to_sample = random.randint(0, total_dimensions) # 使用 total_dimensions
            randomly_perturbed_indices = random.sample(all_feature_indices, num_to_sample)
            neighbor_config = center_generator.to_next(randomly_perturbed_indices)
            neighbor_configs_to_evaluate.append(neighbor_config)

    if not neighbor_configs_to_evaluate:
      return []

    predicted_scores_from_estimators = []
    np_neighbors = np.array(neighbor_configs_to_evaluate)
    if np_neighbors.shape[1] != total_dimensions:
        print(f"警告: 邻域配置维度 ({np_neighbors.shape[1]}) 与总维度 ({total_dimensions}) 不匹配。跳过预测。")
        return [[config, 0.0] for config in neighbor_configs_to_evaluate]

    try:
        for estimator_tree in model_labeled.estimators_:
            predicted_scores_from_estimators.append(estimator_tree.predict(np_neighbors))
        
        acquisition_values = get_ei(predicted_scores_from_estimators, best_score_so_far_eta)
        return [[config, ei] for ei, config in zip(acquisition_values, neighbor_configs_to_evaluate)]
    except Exception as e:
        print(f"do_search 中预测/EI计算出错: {e}")
        return [[config, 0.0] for config in neighbor_configs_to_evaluate]

def get_ei(predictions_from_trees, best_observed_score_eta):
    pred_array = np.array(predictions_from_trees).T
    if pred_array.size == 0: return np.array([]) # 处理空预测

    mean_predictions = np.mean(pred_array, axis=1)
    std_dev_predictions = np.std(pred_array, axis=1)
    std_dev_predictions[std_dev_predictions == 0] = 1e-9

    z_score = (best_observed_score_eta - mean_predictions) / std_dev_predictions
    expected_improvement = (best_observed_score_eta - mean_predictions) * norm.cdf(z_score) + \
                           std_dev_predictions * norm.pdf(z_score)
    expected_improvement[np.isclose(std_dev_predictions, 1e-9)] = 0.0
    return expected_improvement

def get_next_candidate_config(evaluated_configs_train_indep,
                              evaluated_scores_train_dep,
                              best_score_so_far_eta,
                              current_rnum_for_search):
    model = RandomForestRegressor(random_state=456, n_estimators=100)
    
    np_evaluated_configs = np.array(evaluated_configs_train_indep)
    if np_evaluated_configs.ndim == 1 and np_evaluated_configs.size > 0 : # 单个样本的情况
        np_evaluated_configs = np_evaluated_configs.reshape(1, -1)
    elif np_evaluated_configs.size == 0: # 没有已评估样本
        print("警告: 没有已评估的配置来训练模型。返回随机未见配置。")
        while True:
            x = random.randint(0, 2 ** total_dimensions -1)
            conf = generate_conf(x)
            # 确保 evaluated_configs_train_indep 是列表的列表或空的
            if not any(np.array_equal(conf, देखा_config) for देखा_config in evaluated_configs_train_indep if isinstance(देखा_config, (list, np.ndarray))):
                return conf, 0.0
                
    if np_evaluated_configs.shape[1] != total_dimensions and np_evaluated_configs.size > 0:
        print(f"警告: 训练数据维度 ({np_evaluated_configs.shape[1]}) 与总维度 ({total_dimensions}) 不匹配。返回随机。")
        # (代码同上一个if块的while True)
        while True:
            x = random.randint(0, 2 ** total_dimensions -1) # 使用 total_dimensions
            conf = generate_conf(x)
            if not any(np.array_equal(conf, देखा_config) for देखा_config in evaluated_configs_train_indep if isinstance(देखा_config, (list, np.ndarray))):
                return conf, 0.0

    try:
        model.fit(np_evaluated_configs, np.array(evaluated_scores_train_dep))
    except ValueError as ve:
        print(f"拟合模型出错: {ve}。可能是因为训练数据不足或有问题。")
        while True:
            x = random.randint(0, 2 ** total_dimensions -1) # 使用 total_dimensions
            conf = generate_conf(x)
            if not any(np.array_equal(conf, देखा_config) for देखा_config in evaluated_configs_train_indep if isinstance(देखा_config, (list, np.ndarray))):
                return conf, 0.0
    
    candidate_configs_with_ei = do_search(evaluated_configs_train_indep, model, best_score_so_far_eta, current_rnum_for_search)

    if not candidate_configs_with_ei:
        while True:
            x = random.randint(0, 2 ** total_dimensions -1) # 使用 total_dimensions
            conf = generate_conf(x)
            if not any(np.array_equal(conf, देखा_config) for देखा_config in evaluated_configs_train_indep if isinstance(देखा_config, (list, np.ndarray))):
                return conf, 0.0

    candidate_configs_with_ei.sort(key=lambda item: item[1], reverse=True)

    for config_vec, ei_val in candidate_configs_with_ei:
        is_new_config = True
        for existing_config in evaluated_configs_train_indep:
            if np.array_equal(config_vec, existing_config):
                is_new_config = False
                break
        if is_new_config:
            return config_vec, ei_val

    while True:
        x = random.randint(0, 2 ** total_dimensions -1) # 使用 total_dimensions
        conf = generate_conf(x)
        is_new_config = True
        for existing_config in evaluated_configs_train_indep:
             if np.array_equal(conf, existing_config):
                 is_new_config = False
                 break
        if is_new_config:
             return conf, 0.0

# --- 主 BOCA 优化循环函数 ---
def run_boca_optimization(target_program_name, time_limit_seconds=None):
    boca_run_start_time = time.time()
    evaluated_configs = []
    evaluated_scores = []
    evaluation_timestamps = []
    # best_config = None # 将在循环外找到全局最佳
    # best_score = float('inf') # 将使用 current_best_score
    
    # 初始随机采样数量
    num_initial_random_samples = max(5, min(10, total_dimensions // 2 if total_dimensions > 0 else 5))
    
    safe_decay_rate = min(decay, 0.999)
    if safe_decay_rate <= 0: safe_decay_rate = 0.5
    sigma_squared_for_decay = -scale ** 2 / (2 * math.log(safe_decay_rate)) if safe_decay_rate < 1 and safe_decay_rate > 0 else float('inf')

    print(f"为 {target_program_name} 生成 {num_initial_random_samples} 个初始随机样本...")
    initial_sampling_start_time_ref = time.time()

    while len(evaluated_configs) < num_initial_random_samples:
        if time_limit_seconds is not None and (time.time() - boca_run_start_time) > time_limit_seconds:
            print(f"初始采样期间达到时间限制 ({time_limit_seconds}s)。停止。")
            break
        
        random_int_repr = random.randint(0, 2 ** total_dimensions -1) # 使用 total_dimensions
        current_config = generate_conf(random_int_repr)
        
        is_config_new = True
        for existing_cfg in evaluated_configs:
            if np.array_equal(current_config, existing_cfg):
                is_config_new = False
                break
        
        if is_config_new:
            score = get_objective_score(current_config, target_program_name)
            if score != 0.0: # 假设 0.0 是失败标记
                evaluated_configs.append(current_config)
                evaluated_scores.append(score)
                evaluation_timestamps.append(time.time() - boca_run_start_time)
            else:
                 # 评估失败时也记录一个占位，防止死循环，但后续处理时要小心
                 # 或者选择不添加，但需要有其他机制防止死循环
                 # 这里选择不添加失败的评估，依靠时间限制和迭代预算
                 pass
        
        if time.time() - initial_sampling_start_time_ref > 3600 and not evaluated_configs:
            print(f"错误: {target_program_name} 1小时后无成功初始样本。")
            return [], [], None # 返回空结果和None for best_config

    if not evaluated_scores:
        print(f"错误: {target_program_name} 的初始采样未产生有效结果 (可能由于时间限制)。")
        return [], [], None

    current_best_score = min(evaluated_scores)
    best_config_so_far = evaluated_configs[np.argmin(evaluated_scores)] # 初始最佳配置
    print(f"{target_program_name} 初始随机采样后最佳分数: {current_best_score:.4f}")

    total_evaluation_budget = iters 
    while len(evaluated_configs) < total_evaluation_budget:
        elapsed_time_total = time.time() - boca_run_start_time
        if time_limit_seconds is not None and elapsed_time_total > time_limit_seconds:
            print(f"{target_program_name} 达到时间限制 ({time_limit_seconds}s)。停止优化循环。")
            print(f"已完成 {len(evaluated_configs)}/{total_evaluation_budget} 次评估。")
            break

        print(f"\n--- {target_program_name} 迭代 {len(evaluated_configs) - num_initial_random_samples + 1}/{total_evaluation_budget - num_initial_random_samples} (总评估: {len(evaluated_configs)}) (已用时: {elapsed_time_total:.0f}s) ---")

        if sigma_squared_for_decay > 0 and not math.isinf(sigma_squared_for_decay):
             exp_val = -max(0, len(evaluated_configs) - offset) ** 2 / (2 * sigma_squared_for_decay)
             current_rnum = rnum0 * math.exp(exp_val)
        else: # 线性衰减或固定rnum0
             current_rnum = rnum0 if len(evaluated_configs) <= offset else rnum0 * (decay**(len(evaluated_configs)-offset)) # 确保衰减

        next_config_to_eval, predicted_ei = get_next_candidate_config(
            evaluated_configs, evaluated_scores, current_best_score, current_rnum
        )

        score_of_next_config = get_objective_score(next_config_to_eval, target_program_name)
        
        # 只有成功评估的才加入记录
        if score_of_next_config != 0.0:
            evaluation_timestamps.append(time.time() - boca_run_start_time) # 对应成功评估的时间戳
            evaluated_configs.append(next_config_to_eval)
            evaluated_scores.append(score_of_next_config)
            
            if score_of_next_config < current_best_score:
                best_config_so_far = next_config_to_eval
                current_best_score = score_of_next_config
                print(f"*** {target_program_name} 新的最佳分数: {current_best_score:.4f} ***")
        # else:
            # 如果评估失败，不将其添加到 evaluated_configs，这样迭代计数不会增加，
            # 但总迭代次数的上限 (total_evaluation_budget) 和时间限制仍然有效。

    print(f"\n{target_program_name} 的优化完成。")
    # 返回的是整个运行过程中记录的所有分数和时间戳，以及最终找到的最佳配置
    return evaluated_scores, evaluation_timestamps, best_config_so_far


# --- 脚本主执行逻辑 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 BOCA (混合编码) 调优 Cbench 程序的 LLVM 编译选项。")
    valid_programs = cbench if 'cbench' in globals() and isinstance(cbench, list) and cbench else ["program_placeholder"]
    if valid_programs == ["program_placeholder"]:
        print("警告: util.py 中未找到 'cbench' 列表或列表为空。请在 'choices' 中添加程序名。")

    parser.add_argument("program_name", help="要调优的 Cbench 程序名。", choices=valid_programs)
    parser.add_argument("--time_limit", type=int, default=None, 
                        help="可选的每次 BOCA 运行的总时间限制 (秒，例如 2000)。")
    parser.add_argument("--iters", type=int, default=iters,
                        help=f"每次 BOCA 运行的目标评估次数 (默认: {iters})。")
    parser.add_argument("--runs", type=int, default=begin2end,
                        help=f"用于统计的独立 BOCA 运行次数 (默认: {begin2end})。")
    
    args = parser.parse_args()

    target_program = args.program_name
    time_limit_per_run_s = args.time_limit
    iters = args.iters
    begin2end = args.runs

    # 如果程序特定地需要调整单个选项列表（例如基于聚类）
    # 可以在这里进行，并重新计算 total_dimensions
    # 例如:
    # cluster_idx = util.get_cluster_index(target_program)
    # individual_options = util.gain_flags_cluster(cluster_idx)
    # n_individual_flags = len(individual_options)
    # total_dimensions = n_individual_flags + n_sequences
    # print(f"针对 {target_program} 更新了单个选项，新总维度: {total_dimensions}")

    print(f"=== 开始为 {target_program} 进行 BOCA (混合编码) 调优 ===")
    print(f"=== 每次运行目标评估次数: {iters} ===")
    print(f"=== 独立运行次数: {begin2end} ===")
    if time_limit_per_run_s:
        print(f"=== 每次运行时间限制: {time_limit_per_run_s} 秒 ===")

    all_runs_scores = []
    all_runs_timestamps = []
    all_runs_best_configs = [] # 存储每次运行找到的最佳配置向量

    for i in range(begin2end):
        print(f"\n--- {target_program} 的 BOCA 运行 {i+1}/{begin2end} ---")
        scores_one_run, timestamps_one_run, best_config_this_run = run_boca_optimization(
            target_program, time_limit_seconds=time_limit_per_run_s
        )
        
        if scores_one_run: # 确保运行有结果
            all_runs_scores.append(scores_one_run)
            all_runs_timestamps.append(timestamps_one_run)
            if best_config_this_run is not None:
                 all_runs_best_configs.append(best_config_this_run)
            # else:
                 # all_runs_best_configs.append(None) # 或者用一个占位符
        else:
            print(f"{target_program} 的运行 {i+1} 失败或未产生结果。")
    
    if not all_runs_scores:
        print(f"\n{target_program} 没有成功的 BOCA 运行。正在退出。")
        sys.exit(1)

    print(f"\n\n=== {target_program} 的最终结果 ===")
    print(f"成功 BOCA 运行次数: {len(all_runs_scores)}")

    best_scores_over_iterations_all_runs = []
    actual_evals_per_run = []
    for run_scores_list in all_runs_scores:
        actual_evals_per_run.append(len(run_scores_list))
        current_best = float('inf')
        run_best_progression = []
        for score_val in run_scores_list:
            current_best = min(current_best, score_val)
            run_best_progression.append(current_best)
        best_scores_over_iterations_all_runs.append(run_best_progression)
    
    print(f"每次运行实际完成的评估次数: {actual_evals_per_run}")

    max_evals_across_runs = 0
    if best_scores_over_iterations_all_runs:
         max_evals_across_runs = max(len(run_prog) for run_prog in best_scores_over_iterations_all_runs if run_prog) if best_scores_over_iterations_all_runs else 0
    
    padded_best_scores_all_runs = []
    if best_scores_over_iterations_all_runs:
        for run_prog in best_scores_over_iterations_all_runs:
            if run_prog:
                last_val = run_prog[-1]
                padded_run = run_prog + [last_val] * (max_evals_across_runs - len(run_prog))
                padded_best_scores_all_runs.append(padded_run)

    if padded_best_scores_all_runs:
        mean_best_neg_speedup_progression = np.mean(padded_best_scores_all_runs, axis=0) # 直接是负加速比的均值
        std_dev_best_neg_speedup_progression = np.std(padded_best_scores_all_runs, axis=0)

        print("\n平均最佳 (-加速比) (随迭代次数变化，最多到实际评估次数):")
        print([round(s, 4) for s in mean_best_neg_speedup_progression])
        
        print("\n最佳 (-加速比) 的标准差 (随迭代次数变化):")
        print([round(s, 4) for s in std_dev_best_neg_speedup_progression])
    else:
        print("\n无有效运行数据计算分数聚合统计。")

    padded_timestamps_all_runs = []
    if all_runs_timestamps:
        for run_ts_list in all_runs_timestamps:
            if run_ts_list:
                last_ts_val = run_ts_list[-1]
                padded_ts_run = run_ts_list + [last_ts_val] * (max_evals_across_runs - len(run_ts_list))
                padded_timestamps_all_runs.append(padded_ts_run)

    if padded_timestamps_all_runs:
        mean_cumulative_time_progression = np.mean(padded_timestamps_all_runs, axis=0)
        print("\n平均累积时间 (秒) (在每个评估步骤，最多到实际评估次数):")
        print([round(t, 2) for t in mean_cumulative_time_progression])
    else:
        print("\n无有效运行数据计算时间戳聚合统计。")

    global_best_overall_score = float('inf')
    global_best_hybrid_config_vector = None

    for r_idx, best_config_from_run in enumerate(all_runs_best_configs):
        if best_config_from_run is not None:
            # 获取该配置对应的分数 (即该次运行的最终最佳分数)
            # 注意：all_runs_scores[r_idx] 是该次运行的所有分数列表
            # 需要找到 best_config_from_run 对应的实际分数，或者直接取该次运行的最小值
            if all_runs_scores[r_idx]:
                score_for_this_config = min(all_runs_scores[r_idx]) # 近似为其在该次运行的最终最佳分数
                if score_for_this_config < global_best_overall_score:
                    global_best_overall_score = score_for_this_config
                    global_best_hybrid_config_vector = best_config_from_run

    if global_best_hybrid_config_vector is not None:
        print("\n=== 全局最佳配置 (混合编码) ===")
        print(f"程序: {target_program}")
        print(f"对应的 (-加速比): {global_best_overall_score:.4f}")
        
        final_actual_flags = decode_hybrid_config(global_best_hybrid_config_vector)
        print("激活的编译选项:")
        for flag in final_actual_flags:
            print(f"  {flag}")
        
        print("\n重新测量最终性能...")
        try:
            # 再次获取-O3基准时间
            baseline_O3_time = util.run_procedure_runtime(target_program, ["-O3"])
            if baseline_O3_time is None or baseline_O3_time <= 0:
                print("无法获取-O3基准时间进行最终比较。")
            else:
                final_perf_time = util.run_procedure_runtime(target_program, final_actual_flags)
                if final_perf_time is not None and final_perf_time > 0:
                    final_speedup = baseline_O3_time / final_perf_time
                    print(f"最终运行时间: {final_perf_time:.4f}s") # 注意这里不是负加速比
                    print(f"相对于 -O3 的加速比: {final_speedup:.3f}x")
                else:
                    print("未能测量最终性能。")
        except Exception as e:
            print(f"测量最终配置时出错: {e}")
    else:
        print("\n未找到有效的全局最佳配置。")

    print(f"\n=== {target_program} 的调优完成 ===")