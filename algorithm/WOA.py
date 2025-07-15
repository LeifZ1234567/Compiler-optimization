import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys

# 白鲸优化算法
class WOA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20, b = 1):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.b = b       # constant

    def whale_optimization_algorithm(self):
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
            a = 2 - t * (2 / self.n_gen)

            for i in range(self.n_pop):
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                p = random.random()
                l = -1 + 2 * random.random()  

                # 白鲸位置更新
                if p < 0.5:
                    # Encircling prey
                    if abs(A) < 1:
                        for d in range(self.n_flags):
                            Dx     = abs(C * X_best[d] - X[i,d])
                            X[i,d] = X_best[d] - A * Dx
                            X[i,d] = self.boundary(X[i,d])
                    # Search for prey
                    elif abs(A) >= 1:
                        for d in range(self.n_flags):
                            k = np.random.randint(low = 0, high = self.n_pop)
                            Dx = abs(C * X[k,d] - X[i,d])

                            X[i,d] = X[k,d] - A * Dx
                            X[i,d] = self.boundary(X[i,d])
                # Bubble-net attacking 
                elif p >= 0.5:
                    for d in range(self.n_flags):
                        # Distance of whale to prey
                        dist   = abs(X_best[d] - X[i,d])
                        X[i,d] = dist * np.exp(self.b * l) * np.cos(2 * np.pi * l) + X_best[d] 
                        X[i,d] = self.boundary(X[i,d])
            X_bin = self.binary_conversion(X)   

            for i in range(self.n_pop):
                fit[i] = self.run_procedure(self.compile_files,X_bin[i])
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i] 

            self.curve[t+1] = fit_best
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.whale_optimization_algorithm(),self.times

# woa = WOA(n_pop=10,n_gen=10)
# woa.whale_optimization_algorithm()