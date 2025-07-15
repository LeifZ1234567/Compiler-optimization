import numpy as np
import random
import math
import util
import sys
from tqdm import tqdm
import time

class EnhancedHybridFPA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 P=0.8,
                 beta=1.5,
                 known_sequences=None):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.P = P          # 全局授粉概率
        self.beta = beta    # Lévy分布参数
        self.curve = np.zeros(n_gen, dtype=float)
        
        # 编码空间定义
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)
        self.n_base_flags = len(self.gcc_flags)
        self.total_dims = self.n_sequences + self.n_base_flags

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
        return self.run_procedure2(self.compile_files, options)

    def levy_flight(self):
        """生成Lévy飞行步长"""
        nume = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        deno = math.gamma((1 + self.beta)/2) * self.beta * 2**((self.beta-1)/2)
        sigma = (nume / deno) ** (1/self.beta)
        
        u = np.random.normal(0, sigma, self.total_dims)
        v = np.abs(np.random.normal(0, 1, self.total_dims))
        return 0.01 * u / (v ** (1/self.beta))

    def flower_pollination_algorithm(self):
        """增强型混合花授粉算法"""
        # 初始化
        X = self.init_population()
        X_bin = self.binary_conversion(X)
        fitness = np.array([self.evaluate_individual(x) for x in X_bin])
        best_idx = np.argmin(fitness)
        best_solution = X[best_idx].copy()
        self.curve[0] = fitness[best_idx]

        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            new_X = X.copy()
            
            for i in range(self.n_pop):
                if random.random() < self.P:  # 全局授粉
                    step = self.levy_flight()
                    new_X[i] += step * (X[i] - best_solution)
                else:  # 局部授粉
                    j, k = np.random.choice(self.n_pop, 2, replace=False)
                    eps = random.random()
                    new_X[i] += eps * (X[j] - X[k])
                
                # 边界约束
                new_X[i] = np.clip(new_X[i], -5, 5)
            
            # 评估新种群
            new_bin = self.binary_conversion(new_X)
            new_fitness = np.array([self.evaluate_individual(x) for x in new_bin])
            
            # 贪婪选择
            improved = new_fitness < fitness
            X[improved] = new_X[improved]
            fitness[improved] = new_fitness[improved]
            
            # 更新最优解
            current_best = np.argmin(fitness)
            if fitness[current_best] < self.curve[t-1]:
                best_solution = X[current_best].copy()
            self.curve[t] = fitness[current_best]

        # 最终解码
        best_bin = self.binary_conversion(best_solution.reshape(1, -1))[0]
        return self.decode_solution(best_bin),  self.curve[-1]

    def start(self):
        return self.flower_pollination_algorithm()
# # 使用示例
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
    

    

#     # for program in benchmarks:
#     program = "consumer_tiffmedian"
#     print(f"\n当前优化目标：{program}")
#     start_time = time.time()
    
#     optimizer = EnhancedHybridFPA(
#     compile_files=program,
#     known_sequences=known_seqs,
#     n_pop=1,
#     n_gen=1,
#     P=0.85,
#     beta=1.8
#     )
    
#     best_solution, best_fitness = optimizer.flower_pollination_algorithm()
#     elapsed = time.time() - start_time
#     print(f"最优编译选项：{' '.join(optimizer.decode_solution(best_solution))}")
#     print(f"最佳加速比：{best_fitness:.2f}x")
#     print(f"耗时：{elapsed:.2f}秒")