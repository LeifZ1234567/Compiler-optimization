import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import util

class UnifiedBatOptimizer(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=20,
                 n_gen=50,
                 f_max=2,
                 f_min=0,
                 alpha=0.9,
                 gamma=0.9,
                 A_max=2,
                 r0_max=1,
                 known_sequences=None,
                 binary_ratio=0.3):
        """
        统一编码蝙蝠优化器
        :param known_sequences: 候选子序列列表
        :param binary_ratio: 初始种群中已知组合的激活比例
        """
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.f_max = f_max
        self.f_min = f_min
        self.alpha = alpha
        self.gamma = gamma
        self.A_max = A_max
        self.r0_max = r0_max
        self.curve = np.zeros(self.n_gen)
        
        # 编码空间结构
        self.sequence_flags = known_sequences or []
        self.n_sequences = len(self.sequence_flags)
        self.n_base_flags = len(self.gcc_flags)
        self.total_dims = self.n_sequences + self.n_base_flags
        
        # 混合策略参数
        self.binary_ratio = binary_ratio
        self.conflict_resolve_mode = 'last'  # 冲突解决策略

    def init_unified_population(self):
        """初始化统一编码种群"""
        population = np.zeros((self.n_pop, self.total_dims), dtype=int)
        
        # 注入已知组合
        num_known = int(self.n_pop * self.binary_ratio)
        for i in range(num_known):
            if self.sequence_flags:
                # 随机激活1-3个子序列
                seq_mask = np.random.choice([0,1], size=self.n_sequences, 
                                           p=[0.7, 0.3])
                population[i,:self.n_sequences] = seq_mask
                
            # 基础选项部分激活优质组合
            base_mask = np.random.binomial(1, 0.4, self.n_base_flags)
            population[i, self.n_sequences:] = base_mask
        
        # 补充随机个体
        for i in range(num_known, self.n_pop):
            population[i] = np.random.randint(0, 2, self.total_dims)
            
        return population

    def decode_solution(self, individual):
        """解码二进制编码到编译选项"""
        activated_seqs = [seq for seq, mask in zip(self.sequence_flags, individual[:self.n_sequences]) if mask]
        
        # 展开子序列
        seq_options = []
        for seq in activated_seqs:
            seq_options.extend(seq)
            
        # 基础选项
        base_options = [flag for flag, mask in zip(self.gcc_flags, individual[self.n_sequences:]) if mask]
        
        # 冲突解决（保留最后出现的选项）
        combined = seq_options + base_options
        seen = {}
        for opt in reversed(combined):
            if opt not in seen:
                seen[opt] = True
        return list(reversed(seen.keys()))

    def adaptive_mutation(self, individual):
        """自适应变异策略"""
        mutation_prob = np.clip(0.1 + 0.2 * (1 - np.mean(individual)), 0.05, 0.3)
        mutation_mask = np.random.rand(self.total_dims) < mutation_prob
        return np.where(mutation_mask, 1 - individual, individual)

    def evaluate(self, population):
        """批量评估种群适应度"""
        fitness = np.zeros(self.n_pop)
        for i in range(self.n_pop):
            options = self.decode_solution(population[i])
            fitness[i] = self.run_procedure2(self.compile_files, options)
        return fitness

    def unified_bat_algorithm(self):
        """统一编码蝙蝠算法主流程"""
        # 初始化
        population = self.init_unified_population()
        velocity = np.random.uniform(-1, 1, (self.n_pop, self.total_dims))
        
        # 评估初始种群
        fitness = self.evaluate(population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.curve[0] = best_fitness
        
        # 算法参数
        A = np.random.uniform(1, self.A_max, self.n_pop)
        r = np.random.uniform(0, self.r0_max, self.n_pop)
        
        for t in tqdm(range(1, self.n_gen)):
            new_pop = population.copy()
            
            # 频率调节因子
            freq = self.f_min + (self.f_max - self.f_min) * np.random.rand(self.n_pop)
            
            # 速度更新（包含全局引导和惯性项）
            global_guide = best_solution - population
            velocity = velocity + global_guide * freq[:,None] + np.random.normal(0,0.1, (self.n_pop, self.total_dims))
            
            # 位置更新（带sigmoid变换）
            prob = 1 / (1 + np.exp(-velocity))
            new_pop = (prob > np.random.rand(*prob.shape)).astype(int)
            
            # 变异操作
            for i in range(self.n_pop):
                new_pop[i] = self.adaptive_mutation(new_pop[i])
                
            # 贪婪选择
            new_fitness = self.evaluate(new_pop)
            update_mask = (new_fitness <= fitness) & (np.random.rand(self.n_pop) < A)
            population[update_mask] = new_pop[update_mask]
            fitness[update_mask] = new_fitness[update_mask]
            
            # 更新全局最优
            current_best = np.argmin(fitness)
            if fitness[current_best] < best_fitness:
                best_solution = population[current_best].copy()
                best_fitness = fitness[current_best]
                
            # 参数衰减
            A *= self.alpha
            r = self.r0_max * (1 - np.exp(-self.gamma * t))
            
            self.curve[t] = best_fitness
        
        return self.decode_solution(best_solution), best_fitness

    def start(self):
        return self.unified_bat_algorithm(), self.times

# 示例用法
if __name__ == "__main__":
    candidate_sequences = [
        ['-sroa', '-jump-threading'],
        ['-mem2reg', '-gvn', '-instcombine'],
        ['-licm', '-gvn', '-dse']
    ]
    
    optimizer = UnifiedBatOptimizer(
        compile_files="automotive_bitcount",
        known_sequences=candidate_sequences,
        binary_ratio=0.4
    )
    
    best_options, performance = optimizer.unified_bat_algorithm()
    print(f"最优编译选项: {best_options}")
    print(f"性能提升: {-performance:.2f}x")