import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import math
import sys

class FA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20, alpha = 1, beta0 = 1, gamma = 1, theta = 0.97):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.alpha  = alpha       # constant
        self.beta0  = beta0       # light amplitude
        self.gamma  = gamma       # absorbtion coefficient
        self.theta  = theta    # control alpha



    def firefly_algorithm(self):
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
            self.alpha *= self.theta
            # Rank firefly based on their light intensity
            index   = np.argsort(fit, axis=0)
            fit_temp    = fit.copy()
            X_temp    = X.copy()
            for i in range(self.n_pop):
                fit[i] = fit_temp[index[i]]
                X[i]   = X_temp[index[i]]

            for i in range(self.n_pop):
                # The attractiveness parameter
                for j in range(self.n_pop):
                    # Update moves if firefly j brighter than firefly i   
                    if fit[i] > fit[j]: 
                        # Compute Euclidean distance 
                        r = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        for d in range(self.n_flags):
                            # Update position
                            eps    = random.random() - 0.5
                            X[i,d] = X[i,d] + beta * (X[j,d] - X[i,d]) + self.alpha * eps 
                            X[i,d] = self.boundary(X[i,d])

                        # Binary conversion
                        temp = np.zeros([self.n_flags], dtype='float')
                        temp = X[i]  
                        X_bin = self.binary_conversion(np.array([temp]))[0]
                        
                        # fitness
                        fit[i]  = self.run_procedure(self.compile_files,X_bin)
                        
                        # best update        
                        if fit[i] < fit_best:
                            X_best = X[i]
                            fit_best = fit[i]
            self.curve[t+1] = fit_best
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.firefly_algorithm(),self.times

# fa = FA(n_pop=10,n_gen=10)
# fa.firefly_algorithm()

   

                        

