
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys

class SCA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,alpha = 2 ):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.alpha = alpha

    def sine_cosine_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_best = np.zeros([self.n_flags],dtype='float')
        
        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')

        for t in tqdm(range(self.n_gen), file=sys.stdout) :
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

            # r1, decreases linearly from alpha to 0 (3.4)
            r1 = self.alpha - t * (self.alpha / self.n_gen)

            for i in range(self.n_pop):
                for d in range(self.n_flags):
                    # Random parameter r2 & r3 & r4
                    r2 = (2 * np.pi) * random.random()
                    r3 = 2 * random.random()
                    r4 = random.random()

                    # Position update
                    if r4 > 0.5: # 这里的大于小于号控制求极大和极小
                        # Sine update 修改了绝对值后效果变好了，很奇怪...
                        X[i,d] = X[i,d] + r1 * np.sin(r2) * abs(r3 *( X_best[d] - X[i,d])) 
                    else:
                        # Cosine update 
                        X[i,d] = X[i,d] + r1 * np.cos(r2) * abs(r3 * (X_best[d] - X[i,d]))
                    
                    # Boundary
                    X[i,d] = self.boundary(X[i,d]) 
                # print(X[i])
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.sine_cosine_algorithm(),self.times

# sca = SCA(n_pop = 10 , n_gen = 10)
# sca.sine_cosine_algorithm()