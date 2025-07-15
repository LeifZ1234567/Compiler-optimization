import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys

# 樽海鞘算法
class SSA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')


    def salp_swarm_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)

        X_best = np.zeros([self.n_flags],dtype='float')
        

        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')

        for t in tqdm(range(self.n_gen), file=sys.stdout):
            X_bin = self.binary_conversion(X)
            X_temp = self.gain_index()
            if(len(X_temp) > self.n_pop):
                X_bin = X_temp[:self.n_pop]
            else:
                X_bin[:len(X_temp)] = X_temp
            for i in range(self.n_pop):
                fit[i] = self.run_procedure(self.compile_files,X_bin[i])
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]
            
            self.curve[t] = fit_best

            c1 = 2 * np.exp(-(4 * t / self.n_gen) ** 2) # 第一轮初始值为2
            # print(c1)   
            for i in range(self.n_pop):   
                # print(X[i])       
                # First leader update
                if i == 0:  
                    for d in range(self.n_flags):
                        c2 = random.random() *0.3 # 如果不乘系数的话对于后续的影响太大
                        c3 = random.random()

                        if c3 >= 0.5:
                            # X[i,d] = Xf[0,d] + c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                            X[i,d] = X_best[d] + c1 * ((1 - 0) * c2 + 0)
                        else:
                            X[i,d] = X_best[d] - c1 * ((1 - 0) * c2 + 0)
                        X[i,d] = self.boundary(X[i,d])
                        # Salp update
                else:
                    for d in range(self.n_flags):
                        # Salp update by following front salp

                        X[i,d] = (X[i,d] + X[i-1, d]) / 2
                        X[i,d] = self.boundary(X[i,d])
                
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.salp_swarm_algorithm(),self.times


# ssa = SSA(n_pop=10,n_gen=10)
# ssa.salp_swarm_algorithm()
                
                        
