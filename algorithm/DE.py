import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

class DE(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,CR = 0.9 , F = 0.5):
        super().__init__()

        self.n_pop = n_pop
        self.n_gen = n_gen

        self.CR = CR # cross rate
        self.F = F # factor
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

    def differential_evolution(self):
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

        # 初始种群适应度计算
        for i in range(self.n_pop):
            fit[i] = self.run_procedure(self.compile_files,X_bin[i])
            if fit[i] < fit_best:
                fit_best = fit[i]
                X_best = X[i]
        
        # 优化曲线
        self.curve[0] = fit_best.copy()


        for t in tqdm(range(self.n_gen-1), file=sys.stdout):
            V = np.zeros([self.n_pop, self.n_flags], dtype='float')
            U = np.zeros([self.n_pop, self.n_flags], dtype='float')

            for i in range(self.n_pop):
                # 随机选r1,r2,r3  且都不为i
                R = np.random.permutation(self.n_pop)

                for j in range(self.n_pop):
                    if R[j] == i:
                        R = np.delete(R,j)
                        break
                
                r1,r2,r3 = R[0],R[1],R[2]

                # 变异
                for d in range(self.n_flags):
                    V[i,d] = X[r1,d] + self.F * (X[r2,d] - X[r3,d])
                    V[i,d] = self.boundary(V[i,d])

                index = np.random.randint(0,self.n_flags)

                # 交叉
                for d in range(self.n_flags):
                    if random.random() <= self.CR or d == index:
                        U[i,d] = V[i,d]
                    else :
                        U[i,d] = X[i,d]

            U_bin = self.binary_conversion(U)

            #选择
            for i in range(self.n_pop):
                fit_u = self.run_procedure(self.compile_files,U_bin[i])
                if fit_u <= fit[i]:
                    X[i] = U[i]
                    fit[i] = fit_u
                
                if fit[i] < fit_best:
                    X_best = X[i]
                    fit_best = fit[i]
            
            # 优化曲线
            self.curve[t+1] = fit_best.copy()
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.differential_evolution(),self.times


# de = DE(n_gen = 10)
# de.differential_evolution()
        



        