
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys

# 花授粉算法
class FPA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,P = 0.8 , beta = 1.5):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen+1],dtype='float')

        self.beta = beta # Levy component
        self.P = P # 转换概率   

    # Levy Flight
    def levy_distribution(self):
        # Sigma     
        nume  = math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        deno  = math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2)
        sigma = (nume / deno) ** (1 / self.beta) 
        # Parameter u & v 
        u     = np.random.randn(self.n_flags) * sigma
        v     = np.random.randn(self.n_flags)
        # Step 
        step  = u / abs(v) ** (1 / self.beta)
        LF    = 0.01 * step
        
        return LF

    def flower_pollination_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_bin = self.binary_conversion(X)
        X_best = np.zeros([self.n_flags],dtype='float')
        X_temp = self.gain_index()
        if(len(X_temp) > self.n_pop):
            X_bin = X_temp[:self.n_pop]
        else:
            X_bin[:len(X_temp)] = X_temp

        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')

        for i in range(self.n_pop):
            fit[i] = self.run_procedure(self.compile_files,X_bin[i])
            if fit[i] < fit_best:
                fit_best = fit[i]
                X_best = X[i]

        self.curve[0] = fit_best.copy()

        for t in tqdm(range(self.n_gen-1), file=sys.stdout):
            X_new = np.zeros([self.n_pop,self.n_flags],dtype='float')

            for i in range(self.n_pop):
                # 全局散粉  
                if random.random() < self.P:
                    L = self.levy_distribution()
                    for d in range(self.n_flags):
                        X_new[i,d] = X[i,d] + L[d] * (X[i,d] - X_best[d])
                        X_new[i,d] = self.boundary(X_new[i,d])
                # 局部撒粉
                else:
                    # j、k  不是同一个值
                    R = np.random.permutation(self.n_pop)
                    J,K = R[0] , R[1]

                    eps = random.random()
                    for d in range(self.n_flags):
                        X_new[i,d] = X[i,d] + eps*(X[J,d] - X[K,d])
                        X_new[i,d] = self.boundary(X_new[i,d])
            X_bin = self.binary_conversion(X_new)

            # 贪婪选择
            for i in range(self.n_pop):
                fit_new = self.run_procedure(self.compile_files,X_bin[i])
                if fit_new < fit[i]:
                    fit[i] = fit_new
                    X[i] = X_new[i]
                
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]
            self.curve[t+1] = fit_best
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.flower_pollination_algorithm(),self.times

# fpa = FPA(n_pop=10, n_gen=10)
# fpa.flower_pollination_algorithm()




    