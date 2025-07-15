
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

# 粒子群算法
class PSO(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,w = 0.9 , c1 = 2 , c2 = 2):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.w = w
        self.c1 = c1
        self.c2 = c2

    def init_velocity(self, N, dim):
        V    = np.zeros([N, dim], dtype='float')
        V_max = np.zeros([dim], dtype='float')
        V_min = np.zeros([dim], dtype='float')
        # Maximum & minimum velocity
        for d in range(dim):
            V_max[d] = (1 - 0) / 2
            V_min[d] = -V_max[d]
            
        for i in range(N):
            for d in range(dim):
                V[i,d] = V_min[d] + (V_max[d] - V_min[d]) * random.random()
            
        return V, V_max, V_min

    def particle_swarm_optimization(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_best = np.zeros([self.n_flags],dtype='float')
        

        # 速度  
        V ,V_max, V_min = self.init_velocity(self.n_pop,self.n_flags)

        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')
    
        X_gd = np.zeros([self.n_gen,self.n_flags],dtype='float')
        fit_gd = float('inf') * np.ones([self.n_gen],dtype='float')

        for t in tqdm(range(self.n_gen), file=sys.stdout):
            X_bin = self.binary_conversion(X)

            X_temp = self.gain_index()
            if(len(X_temp) > self.n_pop):
                X_bin = X_temp[:self.n_pop]
            else:
                X_bin[:len(X_temp)] = X_temp
            for i in range(self.n_pop):
                fit[i] = self.run_procedure(self.compile_files,X_bin[i])
                if fit[i] < fit_gd[i]:
                    fit_gd[i] = fit[i]
                    X_gd = X[i]

                if fit_gd[i] < fit_best:
                    fit_best = fit_gd[i]
                    X_best = X_gd

            self.curve[t] = fit_best.copy()

            for i in range(self.n_pop):
                for d in range(self.n_flags):
                    r1 = random.random()
                    r2 = random.random()

                    V[i,d] = self.w * V[i,d] + self.c1 * r1 *(X_gd[d]-X[i,d] + self.c2 *r2*(X_gd[d]-X[i,d]))
                    V[i,d] = self.boundary(V[i,d],V_max[d],V_min[d])

                    X[i,d] += V[i,d]
                    X[i,d] = self.boundary(X[i,d])
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.particle_swarm_optimization(),self.times


# pso = PSO(n_pop=10, n_gen=10)
# pso.particle_swarm_optimization()
        