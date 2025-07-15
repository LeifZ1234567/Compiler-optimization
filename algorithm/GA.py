import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

# 遗传算法
class GA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,CR = 0.9 , MR = 0.01):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.CR = CR # crossover rate
        self.MR = MR # mutation rate
        self.curve = np.zeros([self.n_gen],dtype='float')
        self.compile_files = compile_files

    def roulette_wheel(self,prob):
        num = len(prob)
        C   = np.cumsum(prob)
        P   = random.random()
        for i in range(num):
            if C[i] > P:
                index = i
                break
        return index


    def genetic_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X = self.binary_conversion(X)
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
            fit[i] = self.run_procedure(self.compile_files,X[i])
            if fit[i] < fit_best:
                fit_best = fit[i]
                X_best = X[i]
        
        self.curve[0] = fit_best.copy()

        for t in tqdm(range(self.n_gen-1), file=sys.stdout):
            fit_ivt = 1 / (1+fit)
            prob = fit_ivt / np.sum(fit_ivt)

            Nc = 0 # number of crossovers
            for i in range(self.n_pop):
                if random.random() < self.CR:
                    Nc += 1
            
            x1 = np.zeros([Nc,self.n_flags],dtype='int')
            x2 = np.zeros([Nc,self.n_flags],dtype='int')
            for i in range(Nc):
                # 选择父母
                k1 = self.roulette_wheel(prob)
                k2 = self.roulette_wheel(prob)
                p1 = X[k1].copy()
                p2 = X[k2].copy()

                index = np.random.randint(1,self.n_flags-1)

                # 交叉
                x1[i] = np.concatenate((p1[0:index],p2[index:]))
                x2[i] = np.concatenate((p2[0:index],p1[index:]))

                #变异
                for d in range(self.n_flags):
                    if random.random() < self.MR:
                        x1[i,d] = 1- x1[i,d] 
                    if random.random() < self.MR:
                        x2[i,d] = 1-x2[i,d]
            
            # 组成新的种群
            X_new = np.concatenate((x1,x2),axis=0)

            fit_new = np.zeros([2*Nc],dtype='float')
            for i in range(2*Nc):
                fit_new[i] = self.run_procedure(self.compile_files,X_new[i])
                if fit_new[i] < fit_best:
                    fit_best = fit_new[i]
                    X_best = X_new[i]
            
            self.curve[t+1] = fit_best.copy()


        best_flags = self.binary_conversion(np.array([X_best]))[0]
        # print(X_best)
        return best_flags,fit_best

    def start(self):
        return self.genetic_algorithm(),self.times
                 

# ga = GA( n_gen = 10)
# ga.genetic_algorithm()


