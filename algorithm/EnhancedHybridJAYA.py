import numpy as np
from tqdm import tqdm
import random
import util
import sys

class EnhancedHybridJAYA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 known_sequences=None,
                 early_stop_patience=5,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01
                 ):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.curve = np.zeros(n_gen+1, dtype=float)
        
        # 编码空间定义
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)
        self.n_base_flags = len(self.gcc_flags)
        self.total_dims = self.n_sequences + self.n_base_flags
        # 新增早停参数
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold

    def init_population(self):
        """初始化混合编码种群"""
        pop = np.random.uniform(-5, 5, (self.n_pop, self.total_dims))
        
        # 先验知识注入：至少激活一个有效序列
        for i in range(min(self.n_pop, self.n_sequences)):
            pop[i, i] = 5.0  # 高初始值保证激活
        return pop

    def binary_conversion(self, X):
        """带扰动的Sigmoid转换"""
        prob = 1 / (1 + np.exp(-X))
        prob += np.random.normal(0, 0.1, prob.shape)  # 增加探索性
        return (prob > 0.5).astype(int)

    def decode_solution(self, individual_bin):
        """解码二进制编码到编译选项"""
        # 激活的序列选项
        activated_seqs = [
            self.known_sequences[i] 
            for i in range(self.n_sequences) 
            if individual_bin[i] == 1
        ]
        
        # 展开序列选项
        seq_options = []
        for seq in activated_seqs:
            seq_options.extend(seq)
            
        # 基础选项
        base_options = [
            self.gcc_flags[i - self.n_sequences] 
            for i in range(self.n_sequences, self.total_dims)
            if individual_bin[i] == 1
        ]
        
        # 冲突解决（后出现的选项优先）
        combined = seq_options + base_options
        seen = {}
        for opt in reversed(combined):
            key = opt.split('=')[0]
            seen[key] = opt
        return list(seen.values())

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

    def jaya_algorithm(self):
        """增强型混合JAYA算法"""
        X = self.init_population()
        X_bin = self.binary_conversion(X)
        fitness = np.array([self.evaluate_individual(x) for x in X_bin])
        best_idx = np.argmin(fitness)
        best_solution = X[best_idx].copy()
        self.curve[0] = fitness[best_idx]

        # 早停跟踪变量
        best_fitness = self.curve[0]
        no_improve = 0
        
        for t in tqdm(range(self.n_gen), file=sys.stdout):
            self.current_gen = t  # 用于噪声调整
            # 动态调整参数
            r_factor = 1 - (t / self.n_gen)**0.5
            
            # 获取最优最差解
            worst_idx = np.argmax(fitness)
            X_w = X[worst_idx].copy()
            X_b = best_solution.copy()

            new_X = np.zeros_like(X)
            for i in range(self.n_pop):
                for d in range(self.total_dims):
                    # JAYA核心更新公式
                    r1, r2 = random.random(), random.random()
                    new_X[i,d] = X[i,d] + r1*(X_b[d] - abs(X[i,d])) - r2*r_factor*(X_w[d] - abs(X[i,d]))
                    new_X[i,d] = np.clip(new_X[i,d], -5, 5)  # 边界约束

            # 评估新种群
            new_bin = self.binary_conversion(new_X)
            new_fitness = np.array([self.evaluate_individual(x) for x in new_bin])
            
            # 贪婪选择
            improved = new_fitness < fitness
            X[improved] = new_X[improved]
            fitness[improved] = new_fitness[improved]
            
            # 更新最优解
            current_best = np.argmin(fitness)
            current_fitness = fitness[current_best]  # 新增此行
            
            if current_fitness < self.curve[t]:
                best_solution = X[current_best].copy()
                self.curve[t+1] = current_fitness
            else:
                self.curve[t+1] = self.curve[t]

            # 早停检测
            improvement = (best_fitness - current_fitness) / abs(best_fitness)
            if improvement > self.early_stop_threshold:
                best_fitness = current_fitness
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= self.early_stop_patience:
                print(f"\n早停触发：第{t}代 | 最佳适应度 {abs(best_fitness):.2f}x")
                self.curve = self.curve[:t+2]
                break

        # 最终解码
        best_bin = self.binary_conversion(best_solution.reshape(1, -1))[0]
        return self.decode_solution(best_bin),  self.curve[-1]

    def start(self):
        return self.jaya_algorithm()
