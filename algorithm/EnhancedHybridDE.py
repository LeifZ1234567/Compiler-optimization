import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import util
import sys
import time
import os
# parent_dir = "/home/work/zjq/eatunr_enhanba"
# sys.path.append(parent_dir)
import util

import numpy as np
from tqdm import tqdm
import random
import util
import sys
import time

class EnhancedHybridDE(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 CR=0.9,
                 F=0.5,
                 known_sequences=None,
                 early_stop_patience=10,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.CR = CR
        self.F = F
        self.curve = np.full(n_gen, np.inf)  # 初始化为正无穷
        
        # 编码空间定义
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)
        self.index  = self.get_cluster_index(self.compile_files)
        self.new_flags = self.gain_flags_cluster(self.index)
        # self.n_base_flags = len(self.gcc_flags)
        self.n_base_flags = len(self.new_flags)
        self.total_dims = self.n_sequences + self.n_base_flags

         # 新增早停参数
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold

    def init_population(self):
        """强先验知识注入的连续值初始化"""
        population = np.random.uniform(-3, 3, (self.n_pop, self.total_dims))
        # 强制设置前n_sequences个个体激活对应序列
        for i in range(min(self.n_pop, self.n_sequences)):
            population[i, i] = 10.0  # 确保转换后为1
        return population

    def binary_conversion(self, X):
        """鲁棒的二进制转换（核心修改点）"""
        # 增大Sigmoid陡度 + 减小噪声
        prob = 1 / (1 + np.exp(-5 * X))
        prob += np.random.normal(0, 0.05, prob.shape)
        binary = (prob > 0.5).astype(int)
        
        # 强制至少一个激活选项
        if np.sum(binary) == 0:
            activate_dim = np.random.choice([i for i in range(self.n_sequences)])
            binary[activate_dim] = 1
        return binary
    def evaluate_individual(self, individual_bin):
        """评估个体适应度"""
        options = self.decode_solution(individual_bin)
        
        try:
            raw_speedup = self.run_procedure2(self.compile_files, options)
            
            # 异常值检测
            if abs(raw_speedup) < -5:  # 加速比绝对值超过5视为异常
                print(f"异常加速比: {raw_speedup:.2f}x | 选项: {' '.join(options)}")
                return float('inf')  # 返回极大适应度值，促使算法淘汰该解 
            return raw_speedup  # 正常返回负加速比
        
        except Exception as e:
            print(f"评估失败: {str(e)} | 选项: {' '.join(options)}")
            return float('inf')  # 返回极大适应度值

    def decode_solution(self, individual):
        """解码二进制编码到编译选项"""
        # 激活的序列选项
        activated_seqs = [
            self.known_sequences[i] 
            for i in range(self.n_sequences) 
            if individual[i] == 1
        ]
        
        # 展开序列选项
        seq_options = []
        for seq in activated_seqs:
            seq_options.extend(seq)
            
        # 基础选项
        base_options = [
            # self.gcc_flags[i - self.n_sequences] 
            self.new_flags[i - self.n_sequences] 
            for i in range(self.n_sequences, self.total_dims)
            if individual[i] == 1
        ]
        
        # 冲突解决（后出现的选项优先）
        combined = seq_options + base_options
        seen = {}
        for opt in reversed(combined):
            key = opt.split('=')[0]  # 提取选项主名称
            if key not in seen:
                seen[key] = opt
        return list(reversed(seen.values()))


    def differential_evolution(self):
        """增强型差分进化算法主流程（完整版）"""
        # 初始化种群
        X = self.init_population()
        X_bin = self.binary_conversion(X)
        
        # 初始适应度计算
        fit = np.array([self.evaluate_individual(x) for x in X_bin])
        best_idx = np.argmin(fit)
        X_best = X[best_idx].copy()
        self.curve[0] = fit[best_idx]
          # 新增早停相关变量
        best_fitness_history = [self.curve[0]]  # 记录历史最佳
        no_improve_count = 0

        # 优化主循环（关键修改：分离全局最优更新）
        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            # 必须保留的个体更新步骤
            for i in range(self.n_pop):
                # 1. 变异操作（必须保留）
                candidates = [idx for idx in range(self.n_pop) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = X[r1] + self.F * (X[r2] - X[r3])
                mutant = np.clip(mutant, -5, 5)

                # 2. 交叉操作（必须保留）
                cross_points = np.random.rand(self.total_dims) < self.CR
                if not np.any(cross_points):  # 强制至少一个维度交叉
                    cross_points[np.random.randint(0, self.total_dims)] = True
                trial = np.where(cross_points, mutant, X[i])

                # 3. 评估试验向量（必须保留）
                trial_bin = self.binary_conversion(trial.reshape(1, -1))[0]
                trial_fit = self.evaluate_individual(trial_bin)

                # 4. 贪婪选择（必须保留）
                if trial_fit <= fit[i]:
                    X[i] = trial
                    fit[i] = trial_fit

            # 新增的全局最优更新（关键修改点）
            current_best_idx = np.argmin(fit)
            current_best_fit = fit[current_best_idx]
            if current_best_fit < self.curve[t-1]:
                self.curve[t] = current_best_fit
                X_best = X[current_best_idx].copy()
            else:
                self.curve[t] = self.curve[t-1]
            
            # === 新增早停检测逻辑 ===
            # 计算相对改进量（注意符号处理）
            current_best_fitness = self.curve[t]
            previous_best = best_fitness_history[-1]
            improvement = (previous_best - current_best_fitness) / abs(previous_best) 

            # 更新最佳历史记录
            if improvement > self.early_stop_threshold:
                best_fitness_history.append(current_best_fitness)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # 早停检查
            if no_improve_count >= self.early_stop_patience:
                print(f"\n 早停触发：连续 {no_improve_count} 代未显著改进")
                self.curve = self.curve[:t+1]  # 截断结果曲线
                break


        # 最终解码
        best_bin = self.binary_conversion(X_best.reshape(1, -1))[0]
        return self.decode_solution(best_bin),  self.curve[-1]
    def start(self):
        best_bin, raw_fitness = self.differential_evolution()
        return best_bin, raw_fitness  # 保持负值输出


# if __name__ == "__main__":
#     known_seqs = [
#         ['-sroa', '-jump-threading'],
#         ['-mem2reg', '-gvn', '-instcombine'],
#         ['-mem2reg', '-gvn', '-prune-eh'],
#         ['-mem2reg', '-gvn', '-dse'],
#         ['-mem2reg', '-loop-sink', '-loop-distribute'],
#         ['-early-cse-memssa', '-instcombine'],
#         ['-early-cse-memssa', '-dse'],
#         ['-lcssa', '-loop-unroll'],
#         ['-licm', '-gvn', '-instcombine'],
#         ['-licm', '-gvn', '-prune-eh'],
#         ['-licm', '-gvn', '-dse'],
#         ['-memcpyopt', '-loop-distribute']
#     ]
    
#     benchmark = [
#         "security_sha", "office_stringsearch1", 
#         "consumer_tiff2bw", "consumer_tiff2rgba",
#         "consumer_tiffdither", "consumer_tiffmedian"
#     ]
    
#     # for program in benchmark:
#     program = benchmark[0]
#     print(f"\n当前优化目标：{program}")
#     start_time = time.time()
    
#     optimizer = EnhancedHybridDE(
#         compile_files=program,
#         known_sequences=known_seqs,
#         n_pop=10,
#         n_gen=1,
#         CR=0.85,
#         F=0.7
#     )
    
#     best_solution, best_fitness = optimizer.differential_evolution()
#     best_options = optimizer.decode_solution(best_solution)
#     elapsed = time.time() - start_time
    
#     print(f"最优编译选项：{' '.join(best_options)}")
#     print(f"最佳加速比：{best_fitness:.2f}x")
#     print(f"耗时：{elapsed:.2f}秒")
#     plt.plot(optimizer.curve, label=program)

# #     plt.xlabel("Generation")
# #     plt.ylabel("Best Fitness")
# #     plt.title("Convergence Curve")
# #     plt.legend()
# #     plt.show()