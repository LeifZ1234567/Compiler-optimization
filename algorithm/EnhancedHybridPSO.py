import numpy as np
from tqdm import tqdm
import random
import util
import sys

class EnhancedHybridPSO(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 w=0.8,
                 c1=1.5,
                 c2=1.5,
                 known_sequences=None):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.w = w      # 惯性权重
        self.c1 = c1    # 个体学习因子
        self.c2 = c2    # 社会学习因子
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

    def init_velocity(self):
        """初始化速度矩阵"""
        V_max = (1 - 0) / 2 * np.ones(self.total_dims)
        V_min = -V_max
        V = np.random.uniform(V_min, V_max, (self.n_pop, self.total_dims))
        return V, V_max, V_min

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

    def particle_swarm_optimization(self):
        """增强型混合粒子群算法"""
        # 初始化
        X = self.init_population()
        V, V_max, V_min = self.init_velocity()
        
        # 个体最佳和全局最佳
        pbest = X.copy()
        pbest_fit = np.full(self.n_pop, np.inf)
        gbest = np.zeros(self.total_dims)
        gbest_fit = np.inf

        # 初始评估
        X_bin = self.binary_conversion(X)
        for i in range(self.n_pop):
            current_fit = self.evaluate_individual(X_bin[i])
            pbest_fit[i] = current_fit
            if current_fit < gbest_fit:
                gbest = X[i].copy()
                gbest_fit = current_fit
        self.curve[0] = gbest_fit

        # 优化循环
        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            # 动态惯性权重衰减
            self.w *= 0.98
            
            # 更新速度和位置
            for i in range(self.n_pop):
                r1, r2 = random.random(), random.random()
                V[i] = (self.w * V[i] + 
                       self.c1 * r1 * (pbest[i] - X[i]) + 
                       self.c2 * r2 * (gbest - X[i]))
                V[i] = np.clip(V[i], V_min, V_max)
                X[i] += V[i]
                X[i] = np.clip(X[i], -5, 5)
            
            # 评估新种群
            X_bin = self.binary_conversion(X)
            for i in range(self.n_pop):
                current_fit = self.evaluate_individual(X_bin[i])
                # 更新个体最佳
                if current_fit < pbest_fit[i]:
                    pbest[i] = X[i].copy()
                    pbest_fit[i] = current_fit
                # 更新全局最佳
                if current_fit < gbest_fit:
                    gbest = X[i].copy()
                    gbest_fit = current_fit
            self.curve[t] = gbest_fit

        # 最终解码
        best_bin = self.binary_conversion(gbest.reshape(1, -1))[0]
        return self.decode_solution(best_bin),  self.curve[-1]

    def start(self):
        return self.particle_swarm_optimization()

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
    
#     benchmarks = [
#         "security_sha", "office_stringsearch1",
#         "consumer_tiff2bw", "consumer_tiff2rgba",
#         "consumer_tiffdither", "consumer_tiffmedian"
#     ]
    
#     # for program in benchmarks:
#     program = "consumer_tiffmedian"
#     print(f"\n当前优化目标：{program}")
#     optimizer = EnhancedHybridPSO(
#         compile_files=program,
#         known_sequences=known_seqs,
#         n_pop=1,
#         n_gen=1,
#         w=0.9,
#         c1=1.8,
#         c2=1.8
#     )
#     best_solution, best_fitness = optimizer.particle_swarm_optimization()
#     print(f"最优编译选项：{' '.join(optimizer.decode_solution(best_solution))}")
#     print(f"最佳加速比：{best_fitness:.2f}x")