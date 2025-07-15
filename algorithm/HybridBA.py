import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

class HybridBA(util.Util):
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
                 seq_length=3,
                 binary_ratio=0.5):
        """
        混合蝙蝠算法
        :param known_sequences: 已知有效序列列表（例如 [["-O3", "-flto"], ["-march=native"]]）
        :param seq_length: 序列最大长度
        :param binary_ratio: 初始种群中已知序列的比例
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
        
        # 混合策略参数
        self.known_sequences = known_sequences or []
        self.seq_length = seq_length
        self.binary_ratio = binary_ratio

    def init_hybrid_population(self):
        """初始化种群：序列固定，二进制全0"""
        population = []
        
        # 1. 直接注入所有已知序列
        for seq in self.known_sequences:
            population.append({
                'binary': np.zeros(self.n_flags, dtype=int),  # 全0初始化
                'sequence': seq.copy()  # 固定序列
            })
        
        # 2. 补充随机个体（保持序列部分仍使用已知序列）
        remaining = self.n_pop - len(self.known_sequences)
        if remaining > 0:
            for _ in range(remaining):
                # 仍然使用已知序列中的一个
                seq = random.choice(self.known_sequences).copy() 
                population.append({
                    'binary': np.zeros(self.n_flags, dtype=int),
                    'sequence': seq
                })
                
        return population
    def binary_conversion(self, velocity):
        """带噪声的二进制转换"""
        # Sigmoid概率转换
        prob = 1 / (1 + np.exp(-velocity))
        # 添加高斯噪声增强探索
        prob += np.random.normal(0, 0.1, prob.shape)
        return (prob > 0.5).astype(int)

    def mutate_binary(self, binary):
        """二进制位翻转变异"""
        mutation_mask = np.random.rand(len(binary)) < 0.2  # 20%变异概率
        return np.where(mutation_mask, 1 - binary, binary)
    def generate_random_sequence(self):
        """生成随机选项序列"""
        available_flags = [f for f in self.gcc_flags if f not in ['-O1', '-O2', '-O3']]
        return random.sample(available_flags, min(self.seq_length, len(available_flags)))

    def mutate_sequence(self, seq):
        """序列变异操作"""
        """不修改已知序列"""
        if seq in self.known_sequences:
            return seq.copy()  # 直接返回原始序列的副本
        if len(seq) == 0:
            return self.generate_random_sequence()
            
        mutation_type = random.choice(['insert', 'delete', 'swap'])
        
        # 插入操作
        if mutation_type == 'insert' and len(seq) < self.seq_length*2:
            new_opt = random.choice(self.gcc_flags)
            pos = random.randint(0, len(seq))
            seq.insert(pos, new_opt)
            
        # 删除操作
        elif mutation_type == 'delete' and len(seq) > 1:
            pos = random.randint(0, len(seq)-1)
            del seq[pos]
            
        # 交换操作
        elif mutation_type == 'swap' and len(seq) >= 2:
            i, j = random.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
            
        return seq[:self.seq_length*2]  # 限制最大长度

    def crossover_sequences(self, seq1, seq2):
        """不修改已知序列"""
        if seq1 in self.known_sequences or seq2 in self.known_sequences:
            return seq1.copy(), seq2.copy()
        """序列交叉（顺序交叉）"""
        min_len = min(len(seq1), len(seq2))
        if min_len < 2:
            return seq1, seq2
            
        # 随机选择切割点
        start, end = sorted(random.sample(range(min_len), 2))
        
        # 保留父代片段
        child1 = seq1[start:end]
        child2 = seq2[start:end]
        
        # 填充剩余元素
        for opt in seq2:
            if opt not in child1:
                child1.append(opt)
        for opt in seq1:
            if opt not in child2:
                child2.append(opt)
                
        return child1[:self.seq_length], child2[:self.seq_length]

    # def binary_to_options(self, binary):
    #     """将01数组转换为实际编译选项"""
    #     binary = np.asarray(binary).flatten()
    #     print("Binary shape:", binary.shape)  # 应为 (n_flags,)
    #     print(binary)
    #     return [self.gcc_flags[i] for i, val in enumerate(binary) if val == 1]
    def binary_to_options(self, binary):
        binary = np.asarray(binary).flatten()
        # print("Binary content:", binary)  # 打印实际内容
        valid_flags = []
        for i, val in enumerate(binary):
            # 检查元素是否为标量
            if not np.isscalar(val):
                raise ValueError(f"binary[{i}] 是数组，值={val}")
            if val == 1:
                valid_flags.append(self.gcc_flags[i])
        return valid_flags
    def evaluate_individual(self, individual):
        """评估个体适应度"""

        full_options = individual['sequence'] + self.binary_to_options(individual['binary'])
        print(full_options)
        return self.run_procedure2(self.compile_files, full_options)

    def hybrid_bat_algorithm(self):
        """混合蝙蝠算法主流程（修改版：序列固定，仅二进制变异）"""
        # 初始化种群（序列固定，binary全0）
        population = self.init_hybrid_population()
        
        best_fitness = float('inf')
        best_individual = None
        
        # 计算初始适应度
        print("fitness start")
        fitness = np.zeros(self.n_pop)
        for i in range(self.n_pop):
            fitness[i] = self.evaluate_individual(population[i])
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_individual = population[i]
        self.curve[0] = best_fitness
        print("fitness end")
        
        # 蝙蝠算法参数
        A = np.random.uniform(1, self.A_max, self.n_pop)
        r0 = np.random.uniform(0, self.r0_max, self.n_pop)
        r = r0.copy()
        V_binary = np.random.uniform(-1, 1, (self.n_pop, self.n_flags))  # 初始随机速度
        
        # 进化循环
        for t in tqdm(range(1, self.n_gen)):
            new_population = []
            
            for i in range(self.n_pop):
                # 1. 生成频率（保持原有机制）
                beta = random.random()
                freq = self.f_min + (self.f_max - self.f_min) * beta
                
                # 2. 二进制部分更新（核心修改）
                current_binary = population[i]['binary'].astype(float)
                
                # 速度更新（向最佳个体移动）
                if best_individual is not None:
                    velocity = (best_individual['binary'] - current_binary) * freq
                    V_binary[i] += velocity
                
                # 添加随机扰动
                V_binary[i] += np.random.normal(0, 0.1, self.n_flags)
                
                # 生成新二进制（带sigmoid转换）
                new_binary = self.binary_conversion(current_binary + V_binary[i])
                
                # 3. 强制二进制变异（新增）
                new_binary = self.mutate_binary(new_binary)
                
                # 4. 保持序列完全不变（删除所有序列操作）
                new_population.append({
                    'binary': new_binary,
                    'sequence': population[i]['sequence'].copy()  # 直接复制原序列
                })
            
            # 贪婪选择
            for i in range(self.n_pop):
                new_fitness = self.evaluate_individual(new_population[i])
                
                # 接受条件（保持原有逻辑）
                if (new_fitness <= fitness[i]) and (random.random() < A[i]):
                    population[i] = new_population[i]
                    fitness[i] = new_fitness
                    A[i] *= self.alpha
                    r[i] = r0[i] * (1 - np.exp(-self.gamma * t))
                    
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_individual = population[i]
                    
            self.curve[t] = best_fitness
        
        # 返回最佳组合
        best_flags = best_individual['sequence'] + self.binary_to_options(best_individual['binary'])
        return best_flags, best_fitness

    def start(self):
        """统一入口"""
        return self.hybrid_bat_algorithm(), self.times

# 示例用法
if __name__ == "__main__":
    # 已知有效序列示例
    cbench = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian",
    ]

    known_seqs =[
            ['-sroa', '-jump-threading'],
            ['-mem2reg', '-gvn', '-instcombine'],
            ['-mem2reg', '-gvn', '-prune-eh'],
            ['-mem2reg', '-gvn', '-dse'],
            ['-mem2reg', '-loop-sink', '-loop-distribute'],
            ['-early-cse-memssa', '-instcombine'],
            ['-early-cse-memssa', '-dse'],
            ['-lcssa', '-loop-unroll'],
            ['-licm', '-gvn', '-instcombine'],
            ['-licm', '-gvn', '-prune-eh'],
            ['-licm', '-gvn', '-dse'],
            ['-memcpyopt', '-loop-distribute']
    ]
    print(known_seqs)
    
    # 初始化算法
    for pro in cbench:
        optimizer = HybridBA(
            compile_files= pro,
            known_sequences=known_seqs,
        )
        print("Program{} Start".format(pro))
        # 运行优化
        best_flags, best_fitness = optimizer.hybrid_bat_algorithm()
        
        print("\n优化结果:")
        print(f"最佳编译选项: {' '.join(best_flags)}")
        print(f"最佳加速比: {-best_fitness:.2f}x")