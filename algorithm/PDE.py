import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

class DE(util.Util):
    def __init__(self, compile_files="network_dijkstra",n_pop = 10 , n_gen = 20,CR = 0.9 , F = 0.5):
        super().__init__()

        self.n_pop = n_pop
        self.n_gen = n_gen

        self.CR = CR # cross rate
        self.F = F # factor
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

    def differential_evolution(self):
        # 种群初始化
        # x = np.array([42.4, 44.4, 51.6, 39.6, 55.6, 50.0, 69.6, 43.2, 58.8, 46.4, 47.2, 46.8, 57.6, 59.2, 52.0, 48.4, 38.4, 42.8, 58.8, 36.8, 57.6, 43.6, 48.8, 60.8, 57.2, 48.0, 30.0, 54.4, 50.4, 47.6, 53.2, 61.2, 63.6, 67.6, 54.0, 28.8, 52.0, 50.4, 61.2, 69.2, 44.8, 56.4, 49.6, 56.4, 38.8, 54.0, 46.8, 44.8, 46.8, 68.8, 57.2])
        # X = np.array([x for _ in range(self.n_pop)])
        X = self.init_position(self.n_pop)
        X_bin = self.binary_conversion(X)
        X_best = np.zeros([self.n_flags],dtype='float')
        
        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')

        # 初始种群适应度计算
        for i in range(self.n_pop):
            fit[i] = self.run_procedure(X_bin[i], self.compile_files)
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
                fit_u = self.run_procedure(U_bin[i], self.compile_files)
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

if __name__ == "__main__":

    de = DE(n_gen = 10)
    de.differential_evolution()
        



        