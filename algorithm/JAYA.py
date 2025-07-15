
import numpy as np
from tqdm import tqdm  # 进度条设置
import random
import util
import sys

class JAYA(util.Util):
    def __init__(self, compile_files="automotive_bitcount",n_pop = 10 , n_gen = 20,P = 0.8 , beta = 1.5):
        super().__init__()
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.compile_files = compile_files
        self.curve = np.zeros([self.n_gen+1],dtype='float')

    def jaya_algorithm(self):
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

        for t in tqdm(range(self.n_gen), file=sys.stdout):
            X_new = np.zeros([self.n_pop, self.n_flags], dtype='float') 

            idx_max = np.argmax(fit)
            X_w      = X[idx_max].copy()
            idx_min = np.argmin(fit)
            X_b      = X[idx_min].copy()  

            for i in range(self.n_pop):
                for d in range(self.n_flags):
                    # Random numbers
                    r1 = random.random()
                    r2 = random.random()
                    # Position update (1)
                    X_new[i,d] = X[i,d] + r1 * (X_b[d] - abs(X[i,d])) - r2 * (X_w[d] - abs(X[i,d])) 
                    # Boundary
                    X_new[i,d] = self.boundary(X_new[i,d])

            # Binary conversion
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
        return self.jaya_algorithm(),self.times

# jaya = JAYA(n_pop = 10 , n_gen = 10)
# jaya.jaya_algorithm()

