import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import util
import sys
import time

class EnhancedHybridFA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 alpha=1.0,
                 beta0=1.0,
                 gamma=1.0,
                 theta=0.97,
                 known_sequences=None):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.alpha = alpha    # 步长因子
        self.beta0 = beta0    # 初始吸引度
        self.gamma = gamma    # 光吸收系数
        self.theta = theta    # 衰减系数
        self.curve = np.full(n_gen, np.inf)  # 初始化为正无穷
        
        # 编码空间定义
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)
        self.n_base_flags = len(self.gcc_flags)
        self.total_dims = self.n_sequences + self.n_base_flags

    def init_population(self):
        """强先验知识注入的连续值初始化"""
        population = np.random.uniform(-5, 5, (self.n_pop, self.total_dims))
        # 强制前n_sequences个个体激活对应序列
        for i in range(min(self.n_pop, self.n_sequences)):
            population[i, i] = 10.0  # 确保转换后为1
        return population

    def binary_conversion(self, X):
        """鲁棒的二进制转换（核心修改点）"""
        # 增大Sigmoid陡度 + 减小噪声
        prob = 1 / (1 + np.exp(-5 * X))
        prob += np.random.normal(0, 0.05, prob.shape)
        binary = (prob > 0.5).astype(int)
        
        # 强制至少激活一个序列
        for i in range(binary.shape[0]):
            if np.sum(binary[i, :self.n_sequences]) == 0:
                activate_idx = random.randint(0, self.n_sequences-1)
                binary[i, activate_idx] = 1
        return binary

    def decode_solution(self, individual):
        """解码二进制编码到编译选项（与DE/CS相同）"""
        activated_seqs = [
            self.known_sequences[i] 
            for i in range(self.n_sequences) 
            if individual[i] == 1
        ]
        
        seq_options = []
        for seq in activated_seqs:
            seq_options.extend(seq)
            
        base_options = [
            self.gcc_flags[i - self.n_sequences] 
            for i in range(self.n_sequences, self.total_dims)
            if individual[i] == 1
        ]
        
        combined = seq_options + base_options
        seen = {}
        for opt in reversed(combined):
            key = opt.split('=')[0]
            if key not in seen:
                seen[key] = opt
        return list(reversed(seen.values())) or ["-O1"]  # 默认选项保护

    def evaluate_individual(self, individual):
        """评估个体适应度"""
        options = self.decode_solution(individual)
        return self.run_procedure2(self.compile_files, options)

    def firefly_algorithm(self):
        """增强型混合萤火虫算法主流程"""
        # 初始化
        X = self.init_population()
        X_bin = self.binary_conversion(X)
        fitness = np.array([self.evaluate_individual(x) for x in X_bin])
        best_idx = np.argmin(fitness)
        best_solution = X[best_idx].copy()
        self.curve[0] = fitness[best_idx]

        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            # 参数衰减
            self.alpha *= self.theta
            
            # 按亮度排序
            ranked_idx = np.argsort(fitness)
            X_sorted = X[ranked_idx].copy()
            fitness_sorted = fitness[ranked_idx]
            
            # 萤火虫移动
            for i in range(self.n_pop):
                for j in range(self.n_pop):
                    # 仅向更亮的个体移动
                    if fitness_sorted[i] > fitness_sorted[j]:
                        # 计算距离
                        r = np.linalg.norm(X_sorted[i] - X_sorted[j])
                        
                        # 计算吸引度
                        beta = self.beta0 * math.exp(-self.gamma * r**2)
                        
                        # 更新位置
                        delta = beta * (X_sorted[j] - X_sorted[i])
                        noise = self.alpha * (np.random.rand(self.total_dims) - 0.5)
                        X_sorted[i] += delta + noise
                        
                        # 边界处理
                        X_sorted[i] = np.clip(X_sorted[i], -10, 10)
                        
                        # 转换并评估新位置
                        new_bin = self.binary_conversion(X_sorted[i].reshape(1, -1))[0]
                        new_fit = self.evaluate_individual(new_bin)
                        
                        # 更新个体
                        if new_fit < fitness_sorted[i]:
                            fitness_sorted[i] = new_fit
                            X_sorted[i] = X_sorted[i]  # 位置已更新
                            
                            # 更新全局最优
                            if new_fit < self.curve[t-1]:
                                best_solution = X_sorted[i].copy()
                                self.curve[t] = new_fit
                            else:
                                self.curve[t] = self.curve[t-1]
            
            # 保持种群更新
            X = X_sorted.copy()
            fitness = fitness_sorted.copy()

        # 最终解码
        best_bin = self.binary_conversion(best_solution.reshape(1, -1))[0]
        return self.decode_solution(best_bin),  self.curve[-1]

    def start(self):
        return self.firefly_algorithm()

# if __name__ == "__main__":
    # known_seqs = [
    #     ['-sroa', '-jump-threading'],
    #     ['-mem2reg', '-gvn', '-instcombine'],
    #     ['-mem2reg', '-gvn', '-prune-eh'],
    #     ['-mem2reg', '-gvn', '-dse'],
    #     ['-mem2reg', '-loop-sink', '-loop-distribute'],
    #     ['-early-cse-memssa', '-instcombine'],
    #     ['-early-cse-memssa', '-dse'],
    #     ['-lcssa', '-loop-unroll'],
    #     ['-licm', '-gvn', '-instcombine'],
    #     ['-licm', '-gvn', '-prune-eh'],
    #     ['-licm', '-gvn', '-dse'],
    #     ['-memcpyopt', '-loop-distribute']
    # ]
    
    # benchmarks = [
    #     "security_sha", "office_stringsearch1",
    #     "consumer_tiff2bw", "consumer_tiff2rgba",
    #     "consumer_tiffdither", "consumer_tiffmedian"
    # ]
    
    # # 参数配置
    # config = {
    #     "n_pop": 1,
    #     "n_gen": 1,
    #     "alpha": 1.0,
    #     "beta0": 1.0,
    #     "gamma": 0.5,
    #     "theta": 0.95
    # }
    
    # # 运行测试
    # plt.figure(figsize=(10,6))
    # # for program in benchmarks:
    # program = "consumer_tiffmedian"
    # print(f"\n当前优化目标：{program}")
    # start_time = time.time()
    
    # optimizer = EnhancedHybridFA(
    #     compile_files=program,
    #     known_sequences=known_seqs,
    #     **config
    # )
    
    # best_solution, best_fitness = optimizer.firefly_algorithm()
    # best_options = optimizer.decode_solution(best_solution)
    # elapsed = time.time() - start_time
    
    # print(f"最优编译选项：{' '.join(best_options)}")
    # print(f"最佳加速比：{-best_fitness:.2f}x")  # 注意负号转换
    # print(f"耗时：{elapsed:.2f}秒")