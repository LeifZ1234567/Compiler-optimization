import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import util
import sys
import time

class EnhancedHybridGA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 CR=0.85,
                 MR=0.02,
                 known_sequences=None,
                 early_stop_patience=10,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01):
        super().__init__()
        self.compile_files = compile_files
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.CR = CR  # 交叉概率
        self.MR = MR  # 变异概率
        self.curve = np.zeros(self.n_gen, dtype=float)
        
        # 编码空间定义
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)
        # self.n_base_flags = len(self.gcc_flags)
        self.index  = self.get_cluster_index(self.compile_files)
        self.new_flags = self.gain_flags_cluster(self.index)
        # self.n_base_flags = len(self.gcc_flags)
        self.n_base_flags = len(self.new_flags)
        self.total_dims = self.n_sequences + self.n_base_flags

        # 新增早停参数
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold

    def init_population(self):
        """初始化混合编码种群"""
        pop = np.random.randint(0, 2, (self.n_pop, self.total_dims))
        
        # 先验知识注入：至少激活一个有效序列
        for i in range(min(self.n_pop, self.n_sequences)):
            pop[i, i] = 1
        return pop

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
            self.gcc_flags[i - self.n_sequences] 
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


    def tournament_selection(self, population, fitness, k=3):
        """锦标赛选择"""
        selected = []
        for _ in range(self.n_pop):
            candidates = np.random.choice(len(population), k)
            best_idx = candidates[np.argmin(fitness[candidates])]
            selected.append(population[best_idx])
        return np.array(selected)

    def adaptive_mutation(self, individual):
        """自适应变异"""
        mutation_rate = self.MR * (1 - np.mean(individual))  # 多样性越低变异率越高
        return np.where(np.random.rand(self.total_dims) < mutation_rate, 
                        1 - individual, individual)

    def genetic_algorithm(self):
        """增强型混合遗传算法"""
        # 初始化种群
        population = self.init_population()
        fitness = np.array([self.evaluate_individual(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        self.curve[0] = fitness[best_idx]

         # 新增早停相关变量
        best_fitness_history = [self.curve[0]]  # 记录历史最佳
        no_improve_count = 0

        for t in tqdm(range(1, self.n_gen), file=sys.stdout):
            # 选择
            selected = self.tournament_selection(population, fitness)
            
            # 交叉
            offspring = []
            for i in range(0, self.n_pop, 2):
                p1, p2 = selected[i], selected[(i+1)%self.n_pop]
                if random.random() < self.CR:
                    pt = random.randint(1, self.total_dims-1)
                    c1 = np.concatenate([p1[:pt], p2[pt:]])
                    c2 = np.concatenate([p2[:pt], p1[pt:]])
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.extend([c1, c2])
            
            # 变异
            population = np.array([self.adaptive_mutation(ind) for ind in offspring])
            
            # 精英保留
            population[0] = best_solution
            
            # 评估适应度
            fitness = np.array([self.evaluate_individual(ind) for ind in population])
            
            # 更新最优解
            current_best = np.argmin(fitness)
            if fitness[current_best] < self.curve[t-1]:
                best_solution = population[current_best].copy()
            self.curve[t] = fitness[current_best]

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
        return self.genetic_algorithm()


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
    
#     for program in benchmarks:
#         print(f"\n当前优化目标：{program}")
#         start_time = time.time()
        
#         optimizer = EnhancedHybridGA(
#             compile_files=program,
#             known_sequences=known_seqs,
#             n_pop=20,
#             n_gen=50,
#             CR=0.85,
#             MR=0.05
#         )
        
#         best_solution, best_fitness = optimizer.genetic_algorithm()
#         best_options = optimizer.decode_solution(best_solution)
#         elapsed = time.time() - start_time
        
#         print(f"最优编译选项：{' '.join(best_options)}")
#         print(f"最佳加速比：{best_fitness:.2f}x")
#         print(f"耗时：{elapsed:.2f}秒")
#         plt.plot(optimizer.curve, label=program)

#     plt.xlabel("Generation")
#     plt.ylabel("Best Fitness")
#     plt.title("GA Convergence Curve")
#     plt.legend()
#     plt.show()


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
#     program = "consumer_tiffmedian"
#     # for program in benchmarks:
#     print(f"\n当前优化目标：{program}")
#     start_time = time.time()
    
#     optimizer = EnhancedHybridEDA(
#         compile_files=program,
#         known_sequences=known_seqs,
#         n_pop=1,
#         n_gen=1,
#         MR=0.05,
#         SL=0.4
#     )
    
#     best_solution, best_fitness = optimizer.estimation_distribution_algorithm()
#     best_options = optimizer.decode_solution(best_solution)
#     elapsed = time.time() - start_time
    
#     print(f"最优编译选项：{' '.join(best_options)}")
#     print(f"最佳加速比：{best_fitness:.2f}x")
#     print(f"耗时：{elapsed:.2f}秒")
  