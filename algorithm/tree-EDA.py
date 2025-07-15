import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条设置
import random
import math
import util
import sys


class EDA_tree(util.Util):
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

    def cal_bivariate_prob(self,sel_pop):
        upper = 1.0 - 1.0/self.n_flags
        lower = 1.0/self.n_flags
        # dimensions of population

        # n_sel_pop = len(sel_pop)

        probs = np.zeros([self.n_flags,self.n_flags])
        for i in range(self.n_flags):
            for j in range(self.n_flags):
                pr = 0
                if i != j:
                    sum_b = 0
                    for k in range(self.n_sel):
                        sum_b += int(sel_pop[k][i] and sel_pop[k][j])
                    
                    pr = sum_b / self.n_sel
                if(pr > upper):
                    probs[i][j]  = upper
                elif pr < lower:
                    probs[i][j] = lower
                else:
                    probs[i][j] = pr
        # print(probs)
        return probs

    def cal_mutual_information(self,univ,biv):
        mutual_info = np.zeros([self.n_flags,self.n_flags])
        for i in range(self.n_flags):
            for j in range(self.n_flags):
                # i and j
                pij = biv[i][j]
                pi = univ[i]
                pj = univ[j]
                # print(pij,pi,pj)
                if pij/(pi*pj) <= 0:
                    mutual_info[i][j] = 0 
                    continue
                mutual_info[i][j] += pij * math.log(pij/(pi*pj))

                # i and not j
                pij = univ[i] - biv[i][j]
                pi = univ[i]
                pj = 1 - univ[j]
                # print(pij,pi,pj)
                if pij/(pi*pj) <= 0:
                    mutual_info[i][j] = 0
                    continue
                mutual_info[i][j] += pij * math.log(pij/(pi*pj)) 
                # not i and j
                pij = univ[j] - biv[i][j]
                pi = 1 - univ[i]
                pj = univ[j]
                # print(pij,pi,pj)
                if pij/(pi*pj) <= 0:
                    mutual_info[i][j] = 0
                    continue
                mutual_info[i][j] += pij * math.log(pij/(pi*pj))

                # not i and not j
                pij = 1 - (univ[i] + univ[j] -  biv[i][j])
                pi = 1 - univ[i]
                pj = 1 - univ[j]
                # print(pij,pi,pj)
                if pij/(pi*pj) <= 0:
                    mutual_info[i][j] = 0
                    continue
                mutual_info[i][j] += pij * math.log(pij/(pi*pj)) 

                # if mutual_info[i][j] < 0  or np.isnan(mutual_info[i][j]):
                #     mutual_info[i][j] = 0; 
        return mutual_info

    def calc_max_weight_spanning_tree(self,mutual_info):
        tree = {}
        added = []
        root = random.randint(0,self.n_flags-1)
        added.append(root)

        best_match = [root for i in range(self.n_flags)]

        while(len(added) < self.n_flags):
            parent = -1
            to_add = -1
            maximun = -1

            temp = []

            for i in range(self.n_flags):
                if added.count(i) == 0:
                    temp.append(mutual_info[i][best_match[i]])
                    if mutual_info[i][best_match[i]] > maximun:
                        maximun = mutual_info[i][best_match[i]]
                        parent = best_match[i]
                        to_add = i

            if parent in tree:
                tree[parent].append(to_add)
            else:
                tree[parent] = [to_add]

            added.append(to_add)          

            for i in range(self.n_flags):
                if added.count(i) == 0:
                    if(mutual_info[i][to_add] > mutual_info[i][best_match[i]]):
                        best_match[i] = to_add       

        tree[-1] = [root]
        return tree

    def get_parents(self,tree):
        # print(tree)
        # input()
        result = {}

        for parent in tree:
            for son in tree[parent]:
                result[son] = parent
        return result

    def tree_sampling(self,tree,univ,biv):
        pop_flags = [[0 for _ in univ] for _ in range(self.n_pop)]

        root = tree[-1][0]
        parents = self.get_parents(tree)

        for i in range(self.n_pop):
            pop_flags[i][root] = np.random.binomial(1,univ[root],1)[0]
            to_sample = tree[root]  

            while(len(to_sample) != 0):
                node = to_sample[-1]
                parent = parents[node]

                if pop_flags[i][parent]:
                    intersection = biv[parent][node] 
                    condition = univ[parent] 
                else:
                    intersection = univ[node] - biv[parent][node] 
                    condition = 1-univ[parent]

                pop_flags[i][node] = np.random.binomial(1,intersection/condition,1)[0]

                to_sample.pop(-1)

                if list(tree.keys()).count(node):
                    to_add = tree[node]
                    for add in to_add:
                        to_sample.append(add)
        return pop_flags

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




    def tree_estimation_distribution_algorithm(self):
        # 种群初始化
        X = self.init_position(self.n_pop)
        X = self.binary_conversion(X)
        X_best = np.zeros([self.n_flags],dtype='float')
        
                # 适应度
        fit = np.zeros([self.n_pop] , dtype='float')
        fit_best = float('inf')


        for t in tqdm(range(self.n_gen), file=sys.stdout):
            
            for i in range(self.n_pop):
                fit[i] = self.run_procedure(self.compile_files,X[i])
                if fit[i] < fit_best:
                    fit_best = fit[i]
                    X_best = X[i]

            self.curve[t] = fit_best
            best_flags = self.binary_conversion(np.array([X_best]))[0]
            
            # select
            rank_idx = np.argsort(fit, axis=0)
            X_temp = np.zeros([self.n_sel,self.n_flags],dtype='int')
            for i in range(self.n_sel):
                X_temp[i] = X[rank_idx[i]]


            univ_probs = self.cal_prob(X_temp)
            bri_probs = self.cal_bivariate_prob(X_temp)
            mutual_info = self.cal_mutual_information(univ_probs,bri_probs)
            print(mutual_info)
            tree = self.calc_max_weight_spanning_tree(mutual_info)
            X = self.tree_sampling(tree,univ_probs,bri_probs)
            
            X = np.array(X)
            #变异
            for i in range(self.n_sel):
                for d in range(self.n_flags):
                    if random.random() < self.MR:
                        X[i,d] = 1- X[i,d] 

        return best_flags,fit_best

    def start(self):
        return self.tree_estimation_distribution_algorithm(),self.times




eda = EDA_tree(n_pop=5,n_gen=20)
eda.tree_estimation_distribution_algorithm()
print(eda.curve)


    