import numpy as np
from tqdm import tqdm
import random
import util
import sys

class EnhancedHybridSCA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 alpha=2.0,
                 known_sequences=None, 
                 early_stop_patience=5,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.alpha = alpha    # 控制参数
        self.curve = np.zeros(n_gen, dtype=float)
        
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
        """带噪声的Sigmoid转换"""
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

    def sine_cosine_algorithm(self):
        """增强型混合SCA算法"""
        # 初始化
        X = self.init_population()
        X_best = np.zeros(self.total_dims)
        fitness = np.full(self.n_pop, np.inf)
        best_fitness = np.inf

         # 新增早停相关变量
        best_fitness_history = [self.curve[0]]  # 记录历史最佳
        no_improve_count = 0

        for t in tqdm(range(self.n_gen), file=sys.stdout):
            # 动态参数衰减
            r1 = self.alpha - t * (self.alpha / self.n_gen)
            
            # 评估和更新最优解
            X_bin = self.binary_conversion(X)
            for i in range(self.n_pop):
                current_fit = self.evaluate_individual(X_bin[i])
                if current_fit < best_fitness:
                    best_fitness = current_fit
                    X_best = X[i].copy()
            self.curve[t] = best_fitness

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

            # 位置更新
            for i in range(self.n_pop):
                for d in range(self.total_dims):
                    r2 = 2 * np.pi * random.random()
                    r3 = 2 * random.random()
                    r4 = random.random()
                    
                    # 更新规则
                    if r4 > 0.5:
                        X[i,d] += r1 * np.sin(r2) * abs(r3 * X_best[d] - X[i,d])
                    else:
                        X[i,d] += r1 * np.cos(r2) * abs(r3 * X_best[d] - X[i,d])
                    
                    # 边界约束
                    X[i,d] = np.clip(X[i,d], -5, 5)

        # 最终解码
        best_bin = self.binary_conversion(X_best.reshape(1, -1))[0]
        return self.decode_solution(best_bin), best_fitness

    def start(self):
        return self.sine_cosine_algorithm()

