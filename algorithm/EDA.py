import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

class EDA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,CR = 0.9 , MR = 0.01 ,SL = 0.5):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.CR = CR # crossover rate
        self.MR = MR # mutation rate
        self.SL = SL # select rat

        self.n_sel = int(SL * self.n_pop)

    # calculate adapt level
    def cal_prob(self,X):
        upper = 1.0 - 1.0/self.n_flags
        lower = 1.0/self.n_flags
        # dimensions of population

        prob = np.zeros([self.n_flags])
        for i in range(self.n_flags):  # calculate the fitness of  each flag
            sum = 0
            for j in range(self.n_sel): # Count the number of flags selected
                sum += X[j,i]
            p = sum/self.n_sel # Count the precent of flags selected in total
            if p > upper:
                prob[i] = upper
            elif p < lower:
                prob[i] = lower
            else:
                prob[i] = p
        return prob


    # 种群二值化
    def binary_conversion_withprob(self,pops,prob):
        size = len(pops)
        # print(pops,size)
        X = np.zeros([size, self.n_flags], dtype='int')
        # print(pops,prob)
        for i in range(size):
            for d in range(self.n_flags):
                if pops[i,d] < prob[i]:
                    X[i,d] = 1
                else:
                    X[i,d] = 0
        return X

    def estimation_distribution_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X = self.binary_conversion(X)
        X_best = np.zeros([self.n_flags],dtype='float')
        X_temp = self.gain_index()
        if(len(X_temp) > self.n_pop):
            X = X_temp[:self.n_pop]
        else:
            X[:len(X_temp)] = X_temp
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
            rank_idx = np.argsort(fit, axis=0)
            X_temp = np.zeros([self.n_sel,self.n_flags],dtype='int')
            for i in range(self.n_sel):
                X_temp[i] = X[rank_idx[i]]
            #     print(fit[index[i]])
            # print(fit)


            prob = self.cal_prob(X_temp)
            # X = np.random.random((self.n_pop, self.n_flags))
            # X = self.binary_conversion_withprob(X,prob)

            
            X = [[ 1 if random.random()<prob[i] else 0 for i in range(self.n_flags)] for _ in range(self.n_pop)]
            X = np.array(X)
    

            #变异
            for i in range(self.n_sel):
                for d in range(self.n_flags):
                    if random.random() < self.MR:
                        X[i,d] = 1- X[i,d] 
            
            for i in range(self.n_pop):
                fit[i] = self.run_procedure(self.compile_files,X[i])
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]
            
            self.curve[t+1] = fit_best
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.estimation_distribution_algorithm(),self.times

# eda = EDA(n_gen=10 ,n_pop=10)
# eda.estimation_distribution_algorithm()       