import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import math
import sys

# 灰狼优化算法
class GWO(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20):
        super().__init__()

        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')


    def count_fitness(self, x):
        return  self.run_procedure( self.compile_files,x)

    def grey_wolf_optimizer(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_bin = self.binary_conversion(X)
        
        X_temp = self.gain_index()
        if(len(X_temp) > self.n_pop):
            X_bin = X_temp[:self.n_pop]
        else:
            X_bin[:len(X_temp)] = X_temp
        # 适应度
        fit = np.zeros([self.n_pop], dtype='float')

        alpha = np.zeros([self.n_flags], dtype='float')
        beta = np.zeros([self.n_flags], dtype='float')
        delta = np.zeros([self.n_flags], dtype='float')

        alpha_min = float('inf')
        beta_min = float('inf')
        delta_min = float('inf')


        for i in range(self.n_pop):
            fit[i] = self.count_fitness(X_bin[i])

            if fit[i] < alpha_min:
                alpha = X[i]
                alpha_min = fit[i]

            if fit[i] > alpha_min and fit[i] < beta_min:
                beta = X[i]
                beta_min = fit[i]

            if fit[i] > alpha_min and fit[i] > beta_min and fit[i] < delta_min:
                delta = X[i]
                delta_min = fit[i]

        self.curve[0] = alpha_min.copy() # 这里已经算第一次迭代了

        for t in tqdm(range(self.n_gen-1), file=sys.stdout):
            a = 2 - t * (2/self.n_gen)

            for i in range(self.n_pop):
                for j in range(self.n_flags):
                    c1 = 2 * random.random()
                    c2 = 2 * random.random()
                    c3 = 2 * random.random()

                    alpha_temp = abs(c1*alpha[j] - X[i, j])
                    beta_temp = abs(c2*beta[j] - X[i, j])
                    delta_temp = abs(c3*delta[j] - X[i, j])

                    a1 = 2 * a * random.random() - a
                    a2 = 2 * a * random.random() - a
                    a3 = 2 * a * random.random() - a

                    pops1 = alpha[j] - a1*alpha_temp
                    pops2 = beta[j] - a2*beta_temp
                    pops3 = delta[j] - a3*delta_temp

                    X[i,j] =  (pops1 + pops2 + pops3 ) /3
                    X[i,j] = self.boundary(X[i,j]) 
                
            X_bin = self.binary_conversion(X)
            


            for i in range(self.n_pop):
                fit[i] = self.count_fitness(X_bin[i])

                if fit[i] < alpha_min:
                    alpha = X[i]
                    alpha_min = fit[i]

                if fit[i] > alpha_min and fit[i] < beta_min:
                    beta = X[i]
                    beta_min = fit[i]

                if fit[i] > alpha_min and fit[i] > beta_min and fit[i] < delta_min:
                    delta = X[i]
                    delta_min = fit[i]

            self.curve[t+1] = alpha_min.copy()
        best_flags = self.binary_conversion(np.array([alpha]))[0]

        return best_flags,alpha_min

    def start(self):
        return self.grey_wolf_optimizer(),self.times
# gwo = GWO( n_gen = 10)
# gwo.grey_wolf_optimizer()