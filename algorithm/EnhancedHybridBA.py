import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import util
import time
class EnhancedHybridBA(util.Util):
    def __init__(self, 
                 compile_files="automotive_bitcount",
                 n_pop=10,
                 n_gen=20,
                 f_max=2,
                 f_min=0,
                 alpha=0.9,
                 gamma=0.9,
                 A_max=2,
                 r0_max=1,
                 known_sequences=None,
                 early_stop_patience=10,    # 新增：连续未改进代数阈值
                 early_stop_threshold=0.01):
        """
        增强型混合算法：序列和选项均采用开关控制
        :param known_sequences: 已知有效序列列表，每个序列对应一个开关位
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
        self.curve = np.zeros(self.n_gen, dtype=float)
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        
        # 编码空间定义
        
        self.known_sequences = known_sequences or []
        self.n_sequences = len(self.known_sequences)

        self.index  = self.get_cluster_index(self.compile_files)
        self.new_flags = self.gain_flags_cluster(self.index)
        # self.n_base_flags = len(self.gcc_flags)
        self.n_base_flags = len(self.new_flags)
        self.total_dims = self.n_sequences + self.n_base_flags  # 总维度=序列数+基础选项数

    def init_population(self):
        """初始化种群：序列开关和基础选项开关随机初始化"""
        population = np.random.randint(0, 2, (self.n_pop, self.total_dims))
        
        # 注入先验知识：至少保留一个有效序列
        for i in range(min(self.n_pop, self.n_sequences)):
            population[i, i] = 1  # 对角线注入已知序列
        return population

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

    def evaluate_individual(self, individual):
        """评估个体适应度"""
        options = self.decode_solution(individual)
        
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

    # def binary_conversion(self, velocity):
    #     """带噪声的二进制转换"""
    #     prob = 1 / (1 + np.exp(-velocity))
    #     prob += np.random.normal(0, 0.1, prob.shape)  # 增加探索
    #     return (prob > 0.5).astype(int)
    def binary_conversion(self, X):
        prob = 1 / (1 + np.exp(-3 * X))  # 增大陡度系数
        prob += np.random.normal(0, 0.05, prob.shape)  # 减小噪声
        return (prob > 0.5).astype(int)

    def mutate_binary(self, binary):
        """自适应变异"""
        mutation_rate = 0.1 + 0.2 * (1 - np.mean(binary))  # 多样性越低，变异率越高
        mutation_mask = np.random.rand(self.total_dims) < mutation_rate
        return np.where(mutation_mask, 1 - binary, binary)

    def enhanced_bat_algorithm(self):
        """增强型蝙蝠算法主流程"""
        # 初始化
        
        population = self.init_population()
        velocity = np.random.uniform(-1, 1, (self.n_pop, self.total_dims))
        fitness = np.array([self.evaluate_individual(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        self.curve[0] = fitness[best_idx]
        
        # 蝙蝠参数
        A = np.random.uniform(1, self.A_max, self.n_pop)
        r = np.random.uniform(0, self.r0_max, self.n_pop)

         # 新增早停相关变量
        best_fitness_history = [self.curve[0]]  # 记录历史最佳
        no_improve_count = 0

        for t in tqdm(range(1, self.n_gen)):
            # 生成新解
            new_pop = np.zeros_like(population)
            for i in range(self.n_pop):
                freq = self.f_min + (self.f_max - self.f_min) * random.random()
                velocity[i] += (best_solution - population[i]) * freq
                velocity[i] += np.random.normal(0, 0.1, self.total_dims)
                new_pop[i] = self.binary_conversion(population[i] + velocity[i])
            
            # 变异操作
            new_pop = np.array([self.mutate_binary(ind) for ind in new_pop])
            
            # 评估新解
            new_fitness = np.array([self.evaluate_individual(ind) for ind in new_pop])
            
            # 贪婪选择
            improved = new_fitness <= fitness
            population[improved] = new_pop[improved]
            fitness[improved] = new_fitness[improved]
            
            # 更新全局最优
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
             # 参数衰减
            A *= self.alpha
            # r = self.r0_max * (1 - np.exp(-self.gamma * t))
            # 改为个体独立衰减
            r = [ri * (1 - np.exp(-self.gamma * t)) for ri in r]


        return self.decode_solution(best_solution), self.curve[-1]

    def start(self):
        return self.enhanced_bat_algorithm()


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
#     # cbench =["security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian"]
#     # ['automotive_susan_c', 'automotive_susan_e', 'automotive_susan_s', 'automotive_bitcount', 'bzip2d', 'office_rsynth', 'telecom_adpcm_c', 'telecom_adpcm_d', 'security_blowfish_d', 'security_blowfish_e', 'bzip2e', 'telecom_CRC32', 'network_dijkstra', 'consumer_jpeg_c', 'consumer_jpeg_d', 'network_patricia', 'automotive_qsort1', 'security_rijndael_d']
#     print(len(cbench))
    
#     for pro in cbench:
#         print(pro)
#         init_time = time.time()
#         optimizer = EnhancedHybridBA(
#             compile_files=pro,
#             known_sequences=known_seqs,
#             n_pop=20,
#             n_gen=50
#         )
    
#         best_solution, best_fitness = optimizer.enhanced_bat_algorithm()
#         best_options = optimizer.decode_solution(best_solution)
#         cost_time = time.time() - init_time
#         print(f"最优编译选项：{' '.join(best_options)}")
#         print(f"最佳加速比：{best_fitness:.2f}x")
#         print(f"耗时：{cost_time:.2f}x")


