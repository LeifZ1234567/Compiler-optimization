import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys

class CS(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,Pa = 0.25 , alpha = 1):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.Pa     = Pa     # discovery rate
        self.alpha  = alpha        # constant
 
        self.beta = 1.5 # levy component


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

    def cuckoo_search(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_bin = self.binary_conversion(X)
        X_best = np.zeros([self.n_flags],dtype='float')
        

        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')
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

        self.curve[0] = fit_best.copy()

        for t in tqdm(range(self.n_gen-1), file=sys.stdout):
            X_new = np.zeros([self.n_pop,self.n_flags],dtype='float')

            # Random walk/Levy flight phase
            for i in range(self.n_pop):
                L = self.levy_distribution()
                for d in range(self.n_flags):
                    X_new[i,d] = X[i,d] + self.alpha * L[d] * (X[i,d] - X_best[d]) 
                    X_new[i,d] = self.boundary(X_new[i,d])
            
            X_bin = self.binary_conversion(X_new)

            for i in range(self.n_pop):
                fit_new = self.run_procedure(self.compile_files,X_bin[i])
                if fit_new < fit[i]:
                    fit[i] = fit_new
                    X[i] = X_new[i]
                
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]

            #  Discovery and abandon worse nests phase
            J  = np.random.permutation(self.n_pop)
            K  = np.random.permutation(self.n_pop)
            X_j = np.zeros([self.n_pop, self.n_flags], dtype='float')
            X_k = np.zeros([self.n_pop, self.n_flags], dtype='float')
            for i in range(self.n_pop):
                X_j[i] = X[J[i]]
                X_k[i] = X[K[i]]

            X_new = np.zeros([self.n_pop,self.n_flags],dtype='float')

            for i in range(self.n_pop): 
                X_new[i] = X[i]
                r = random.random()
                for d in range(self.n_flags):
                    # A fraction of worse nest is discovered with a probability
                    if random.random() < self.Pa:
                        X_new[i,d] = X[i,d] + r * (X_j[i,d] - X_k[i,d])

                    X_new[i,d] = self.boundary(X_new[i,d])
            
            X_bin = self.binary_conversion(X_new)
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
        return self.cuckoo_search(),self.times
# cs = CS(n_pop=10,n_gen=10)
# cs.cuckoo_search()

        