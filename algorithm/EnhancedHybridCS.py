import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import util
import sys
import time

class EnhancedHybridCS(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 Pa=0.25,
                 alpha=1.0,
                 beta=1.5,
                 known_sequences=None,
                 early_stop_patience=5,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.Pa = Pa       # 发现概率
        self.alpha = alpha  # 步长因子
        self.beta = beta    # Lévy分布参数
        self.curve = np.full(n_gen, np.inf)  # 初始化为正无穷
        
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
        population = np.random.uniform(-5, 5, (self.n_pop, self.total_dims))
        
        # 先验知识注入：前n_sequences个个体激活对应序列
        for i in range(min(self.n_pop, self.n_sequences)):
            population[i, i] = 10.0  # 确保转换后为1
        return population

    def binary_conversion(self, X):
        """鲁棒的二进制转换（核心修改点）"""
        prob = 1 / (1 + np.exp(-5 * X))  # 增大陡度系数
        prob += np.random.normal(0, 0.05, prob.shape)  # 减小噪声
        binary = (prob > 0.5).astype(int)
        
        # 强制至少激活一个序列
        for i in range(binary.shape[0]):
            if np.sum(binary[i, :self.n_sequences]) == 0:
                activate_idx = random.randint(0, self.n_sequences-1)
                binary[i, activate_idx] = 1
        return binary

    def decode_solution(self, individual):
        """解码二进制编码到编译选项（与DE相同）"""
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
        return list(reversed(seen.values())) or ["-O1"]

    # def evaluate_individual(self, individual):
    #     """评估个体适应度"""
    #     options = self.decode_solution(individual)
    #     return self.run_procedure2(self.compile_files, options)
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


    def levy_flight(self):
        """生成Lévy飞行步长"""
        nume = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        deno = math.gamma((1 + self.beta)/2) * self.beta * 2**((self.beta-1)/2)
        sigma = (nume / deno) ** (1/self.beta)
        
        u = np.random.normal(0, sigma**2, self.total_dims)
        v = np.abs(np.random.normal(0, 1, self.total_dims))
        step = u / (v ** (1/self.beta))
        return 0.01 * step * self.alpha

    def cuckoo_search(self):
        """增强型混合布谷鸟搜索主流程"""
        # 初始化
        X = self.init_population()
        X_bin = self.binary_conversion(X)
        fitness = np.array([self.evaluate_individual(x) for x in X_bin])
        best_idx = np.argmin(fitness)
        best_solution = X[best_idx].copy()
        self.curve[0] = fitness[best_idx]

         # 新增早停相关变量
        best_fitness_history = [self.curve[0]]  # 记录历史最佳
        no_improve_count = 0

        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            # Lévy飞行阶段
            new_X = X.copy()
            for i in range(self.n_pop):
                step = self.levy_flight()
                new_X[i] += step * (X[i] - best_solution)
                new_X[i] = np.clip(new_X[i], -10, 10)
            
            # 二进制转换和评估
            new_bin = self.binary_conversion(new_X)
            new_fitness = np.array([self.evaluate_individual(x) for x in new_bin])
            
            # 贪婪选择
            improved = new_fitness <= fitness
            X[improved] = new_X[improved]
            fitness[improved] = new_fitness[improved]
            
            # 发现阶段（自适应变异）
            for i in range(self.n_pop):
                if random.random() < self.Pa:
                    j, k = np.random.choice(self.n_pop, 2, replace=False)
                    mutation = np.random.normal(0, 1, self.total_dims)
                    X[i] += mutation * (X[j] - X[k])
                    X[i] = np.clip(X[i], -10, 10)
            
            # 更新全局最优
            current_best = np.argmin(fitness)
            if fitness[current_best] < self.curve[t-1]:
                best_solution = X[current_best].copy()
                self.curve[t] = fitness[current_best]
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
        best_bin = self.binary_conversion(best_solution.reshape(1, -1))[0]
        return self.decode_solution(best_bin),  self.curve[-1]

    def start(self):
        return self.cuckoo_search()

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
    
    # # 参数设置
    # config = {
    #     "n_pop": 2,
    #     "n_gen": 1,
    #     "Pa": 0.3,
    #     "alpha": 1.2,
    #     "beta": 1.8
    # }
    
    # # for program in benchmarks:
    # program = "consumer_tiffmedian"
    # print(f"\n当前优化目标：{program}")
    # start_time = time.time()
    
    # optimizer = EnhancedHybridCS(
    #     compile_files=program,
    #     known_sequences=known_seqs,
    #     **config
    # )
    
    # best_solution, best_fitness = optimizer.cuckoo_search()
    # best_options = optimizer.decode_solution(best_solution)
    # elapsed = time.time() - start_time
    
    # print(f"最优编译选项：{' '.join(best_options)}")
    # print(f"最佳加速比：{-best_fitness:.2f}x")  # 注意负号转换
    # print(f"耗时：{elapsed:.2f}秒")