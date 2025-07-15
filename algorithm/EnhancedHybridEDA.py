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

class EnhancedHybridEDA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 CR=0.9,
                 MR=0.01,
                 SL=0.5,
                 known_sequences=None,
                 early_stop_patience=10,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.CR = CR    # 交叉率
        self.MR = MR    # 变异率
        self.SL = SL    # 选择比例
        self.curve = np.zeros(n_gen, dtype=float)
        
        # 编码空间定义
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)
        self.index  = self.get_cluster_index(self.compile_files)
        self.new_flags = self.gain_flags_cluster(self.index)
        self.n_base_flags = len(self.new_flags)
        # self.n_base_flags = len(self.gcc_flags)
        
        self.total_dims = self.n_sequences + self.n_base_flags  # 总维度=序列数+基础选项数
        self.n_sel = int(SL * self.n_pop)

         # 新增早停参数
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold

    def init_population(self):
        """初始化种群并注入先验知识"""
        population = np.random.randint(0, 2, (self.n_pop, self.total_dims))
        # 确保至少一个有效序列被激活
        for i in range(min(self.n_pop, self.n_sequences)):
            population[i, i] = 1
        return population

    def decode_solution(self, individual):
        """解码二进制编码到编译选项（与HybridBA/DE相同）"""
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
            key = opt.split('=')[0]
            if key not in seen:
                seen[key] = opt
        return list(reversed(seen.values())) or ["-O1"]  # 默认选项保护

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

    def cal_prob(self, X):
        """计算概率模型（针对混合编码优化）"""
        prob = np.zeros(self.total_dims)
        for i in range(self.total_dims):
            cnt = np.sum(X[:, i])
            p = cnt / self.n_sel
            # 对序列开关位施加更强约束
            if i < self.n_sequences:
                prob[i] = np.clip(p, 0.1, 0.9)  # 保证序列位有基础概率
            else:
                prob[i] = np.clip(p, 0.01, 0.99)
        return prob

    def generate_offspring(self, prob):
        """生成子代（带精英保留）"""
        # 精英保留前10%
        elite_num = max(1, int(0.1 * self.n_pop))
        elite = self.population[:elite_num].copy()
        
        # 生成新种群
        new_pop = []
        for _ in range(self.n_pop - elite_num):
            individual = [1 if random.random() < p else 0 for p in prob]
            new_pop.append(individual)
        return np.vstack([elite, np.array(new_pop)])

    def estimation_distribution_algorithm(self):
        """增强型分布估计算法主流程"""
        # 初始化
        self.population = self.init_population()
        fit = np.array([self.evaluate_individual(ind) for ind in self.population])
        best_idx = np.argmin(fit)
        best_solution = self.population[best_idx].copy()
        self.curve[0] = fit[best_idx]

         # 新增早停相关变量
        best_fitness_history = [self.curve[0]]  # 记录历史最佳
        no_improve_count = 0

        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            # 选择
            ranked_idx = np.argsort(fit)
            selected = self.population[ranked_idx[:self.n_sel]]
            
            # 概率模型更新
            prob = self.cal_prob(selected)
            
            # 生成子代
            self.population = self.generate_offspring(prob)
            
            # 变异操作
            for i in range(self.n_pop):
                for d in range(self.total_dims):
                    if random.random() < self.MR:
                        self.population[i, d] ^= 1  # 位翻转
            
            # 强制至少一个序列激活
            for i in range(self.n_pop):
                if np.sum(self.population[i, :self.n_sequences]) == 0:
                    activate_idx = random.randint(0, self.n_sequences-1)
                    self.population[i, activate_idx] = 1
            
            # 评估
            fit = np.array([self.evaluate_individual(ind) for ind in self.population])
            
            # 更新最优
            current_best = np.argmin(fit)
            if fit[current_best] < self.curve[t-1]:
                best_solution = self.population[current_best].copy()
                self.curve[t] = fit[current_best]
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

        return self.decode_solution(best_solution),  self.curve[-1]

    def start(self):
        return self.estimation_distribution_algorithm()

# if __name__ == "__main__":
#     known_seqs =[
#             ['-sroa', '-jump-threading'],
#             ['-mem2reg', '-gvn', '-instcombine'],
#             ['-mem2reg', '-gvn', '-prune-eh'],
#             ['-mem2reg', '-gvn', '-dse'],
#             ['-mem2reg', '-loop-sink', '-loop-distribute'],
#             ['-early-cse-memssa', '-instcombine'],
#             ['-early-cse-memssa', '-dse'],
#             ['-lcssa', '-loop-unroll'],
#             ['-licm', '-gvn', '-instcombine'],
#             ['-licm', '-gvn', '-prune-eh'],
#             ['-licm', '-gvn', '-dse'],
#             ['-memcpyopt', '-loop-distribute']
#     ]
#     cbench = ["automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian"]
#     for pro in cbench:
#         print(pro)
#         init_time = time.time()
#         optimizer = EnhancedHybridEDA(
#             compile_files=pro,
#             known_sequences=known_seqs,
#             n_pop=10,
#             n_gen=1
#         )

#         best_solution, best_fitness = optimizer.estimation_distribution_algorithm()
#         best_options = optimizer.decode_solution(best_solution)
#         cost_time = time.time() - init_time
#         print(f"最优编译选项：{' '.join(best_options)}")
#         print(f"最佳加速比：{best_fitness:.2f}x")
#         print(f"耗时：{cost_time:.2f}x")