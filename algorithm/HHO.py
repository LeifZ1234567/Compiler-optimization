
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys

# 哈里斯鹰优化算法
class HHO(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen],dtype='float')

        self.beta = 1.5


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

    def harris_hawks_optimization(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X_best = np.zeros([self.n_flags],dtype='float')
        
        X_temp = self.gain_index()
        if(len(X_temp) > self.n_pop):
            X_bin = X_temp[:self.n_pop]
        else:
            X_bin[:len(X_temp)] = X_temp

        # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')

        for t in tqdm(range(self.n_gen), file=sys.stdout):
            X_bin = self.binary_conversion(X)

            # 适应度计算    
            for i in range(self.n_pop):
                fit[i] = self.run_procedure(X_bin[i],self.compile_files)
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]

            self.curve[t] = fit_best.copy()

            # 鹰的位置🦅
            X_mean= np.zeros([self.n_flags],dtype='float')
            X_mean = np.mean(X,axis=0)

            for i in range(self.n_pop):
                E0 = -1 + 2* random.random() # E0 in [-1,1]
                E  = 2 * E0 * (1 - (t / self.n_gen)) 

                # 探测
                if abs(E) >= 1:
                # Define q in [0,1]
                    q = random.random()
                    if q >= 0.5:
                        # Random select a hawk k
                        k  = np.random.randint(low = 0, high = self.n_pop)
                        r1 = random.random()
                        r2 = random.random()
                        for d in range(self.n_flags):
                            # Position update (1)
                            X[i,d] = X[k,d] - r1 * abs(X[k,d] - 2 * r2 * X[i,d])
                            # Boundary
                            X[i,d] = self.boundary(X[i,d])
                    else:
                        r3 = random.random()
                        r4 = random.random()
                        for d in range(self.n_flags):
                            # Update Hawk (1)
                            X[i,d] = (X_best[d] - X_mean[d]) - r3 * (0 + r4 * (1 - 0))
                            # Boundary
                            X[i,d] = self.boundary(X[i,d])
                else: # abs(E) < 1:
                    # Jump strength 
                    J = 2 * (1 - random.random())
                    r = random.random()
                    # {1} Soft besiege
                    if r >= 0.5 and abs(E) >= 0.5:
                        for d in range(self.n_flags):
                            # Delta X (5)
                            DX     = X_best[d] - X[i,d]
                            # Position update (4)
                            X[i,d] = DX - E * abs(J * X_best[d] - X[i,d])
                            # Boundary
                            X[i,d] = self.boundary(X[i,d])
                    # {2} hard besiege
                    elif r >= 0.5 and abs(E) < 0.5:
                        for d in range(self.n_flags):
                            # Delta X (5)
                            DX     = X_best[d] - X[i,d]
                            # Position update (6)
                            X[i,d] = X_best[d] - E * abs(DX)    
                            # Boundary
                            X[i,d] = self.boundary(X[i,d])

                    # {3} Soft besiege with progressive rapid dives
                    elif r < 0.5 and abs(E) >= 0.5:
                        # Levy distribution (9)
                        LF = self.levy_distribution() 
                        Y  = np.zeros([ self.n_flags], dtype='float')
                        Z  = np.zeros([ self.n_flags], dtype='float')

                        for d in range(self.n_flags):
                            # Compute Y (7)
                            Y[d] = X_best[d] - E * abs(J * X_best[d] - X[i,d])
                            # Boundary
                            Y[d] = self.boundary(Y[d])

                        for d in range(self.n_flags):
                            # Compute Z (8)
                            Z[d] = Y[d] + random.random() * LF[d]
                            # Boundary
                            Z[d] = self.boundary(Z[d])   

                        Y = np.array([Y])
                        Z = np.array([Z])
                        # Binary conversion
                        Y_bin = self.binary_conversion(Y)[0]
                        Z_bin = self.binary_conversion(Z)[0]
                        # fitness
                        fit_Y = self.run_procedure(self.compile_files,Y_bin)
                        fit_Z = self.run_procedure(self.compile_files,Z_bin)

                        # Greedy selection (10)
                        if fit_Y < fit[i]:
                            fit[i]  = fit_Y 
                            X[i]    = Y
                        if fit_Z < fit[i]:
                            fit[i]  = fit_Z
                            X[i]    = Z                             
                    # {4} Hard besiege with progressive rapid dives
                    elif r < 0.5 and abs(E) < 0.5:
                        # Levy distribution (9)
                        LF = self.levy_distribution() 
                        Y  = np.zeros([self.n_flags], dtype='float')
                        Z  = np.zeros([self.n_flags], dtype='float')

                        for d in range(self.n_flags):
                            # Compute Y (12)
                            Y[d] = X_best[d] - E * abs(J * X_best[d] - X_mean[d])
                            # Boundary
                            Y[d] = self.boundary(Y[d])
                    
                        for d in range(self.n_flags):
                            # Compute Z (13)
                            Z[d] = Y[d] + random.random() * LF[d]
                            # Boundary
                            Z[d] = self.boundary(Z[d])    
                        

                        Y = np.array([Y])
                        Z = np.array([Z])

                        # Binary conversion
                        Y_bin = self.binary_conversion(Y)[0]
                        Z_bin = self.binary_conversion(Z)[0]
                        # fitness
                        fit_Y = self.run_procedure(Y_bin,self.compile_files)
                        fit_Z = self.run_procedure(Z_bin,self.compile_files)

                        # Greedy selection (10)
                        if fit_Y < fit[i]:
                            fit[i]  = fit_Y 
                            X[i]    = Y
                        if fit_Z < fit[i]:
                            fit[i]  = fit_Z
                            X[i]    = Z        
        best_flags = self.binary_conversion(np.array([X_best]))[0]

        return best_flags,fit_best

    def start(self):
        return self.harris_hawks_optimization(),self.times

# hho = HHO(n_pop = 10 , n_gen = 10)
# hho.harris_hawks_optimization()

