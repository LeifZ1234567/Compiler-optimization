import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

class BA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,f_max =2,f_min = 0,alpha = 0.9,gamma = 0.9,A_max = 2,r0_max = 1,):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.f_max = f_max
        self.f_min = f_min
        self.alpha = alpha
        self.gamma = gamma
        self.A_max = A_max # 响度最大值 
        self.r0_max = r0_max # 脉冲率最大值
        self.curve = np.zeros([self.n_gen],dtype='float')
        self.compile_files = compile_files


    def bat_inspired_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_bin = self.binary_conversion(X)
        
        X_best = np.zeros([self.n_flags],dtype='float')
        X_temp = self.gain_index()
        if(len(X_temp) > self.n_pop):
            X_bin = X_temp[:self.n_pop]
        else:
            X_bin[:len(X_temp)] = X_temp
        # 速度  
        V = np.zeros([self.n_pop,self.n_flags],dtype='float')

        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')

        for i in range(self.n_pop):
            fit[i] = self.run_procedure(self.compile_files,X_bin[i])
            if fit[i] < fit_best:
                fit_best = fit[i]
                X_best = X[i]
        
        self.curve[0] = fit_best

        A = np.random.uniform(1,self.A_max,self.n_pop)
        r0 = np.random.uniform(0,self.r0_max,self.n_pop)
        r = r0.copy()

        for t in tqdm(range(self.n_gen-1)):
            X_new = np.zeros([self.n_pop,self.n_flags],dtype='float')
            
            for i in range(self.n_pop):
                beta = random.random()
                freq = self.f_min + (self.f_max - self.f_min) * beta
                
                for d in range(self.n_flags):
                    V[i,d] += (X[i,d] - X_best[d]) * freq
                    X_new[i,d] = X[i,d] + V[i,d]
                    X_new[i,d] = self.boundary(X_new[i,d])

                #  根据最佳解决方案生成解决方案
                if random.random() > r[i]:
                    for d in range(self.n_flags):
                        esp = -1 +2 *random.random() # -1<= esp <= 1
                        X_new[i,d] = X_best[d] + esp*np.mean(A)
                        X_new[i,d] = self.boundary(X_new[i,d])

            X_bin = self.binary_conversion(X_new)

            # 贪婪选择
            for i in range(self.n_pop):
                fit_new = self.run_procedure(self.compile_files,X_bin[i])
                if random.random() < A[i] and fit_new <= fit[i]:
                    X[i] = X_new[i]
                    fit[i] = fit_new

                    # 更新响度
                    A[i] *= self.alpha
                    # 更新脉冲发射率
                    r[i] = r0[i] *(1 - np.exp(-self.gamma * t))
                
                
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]
            self.curve[t+1] = fit_best
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best
    
    def start(self):
        return self.bat_inspired_algorithm(),self.times
# ba = BA( n_gen = 10)
# ba.bat_inspired_algorithm()
        
