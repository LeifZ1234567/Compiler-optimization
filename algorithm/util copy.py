from random import randint
from os import system
import numpy as np
import random
import os
import pandas as pd
import pyRAPL
cbench = [
   ]
polybench = []

class Util(object):
    def __init__(self) -> None:
        self.gcc_flags = self.gain_flags()
        self.baseline = 10
        self.n_flags = len(self.gcc_flags)
        self.times = 0

    def get_cluster_index(self,program_name):
        # 定义每个索引对应的程序名列表
        cluster_0_programs = []
        cluster_1_programs = []
        cluster_2_programs = []
        

        # 判断程序名属于哪个索引
        if program_name in cluster_0_programs:
            return 0
        elif program_name in cluster_1_programs:
            return 1
        else:
            return 2
    def gain_flags(self):
        #2024.11.26 原来的
        flags = []
        return flags
    def gain_flags_cluster(self, index):
        if index == 0:
            #分支型编译选项
            flags_branch = []
            #页错误编译选项
            flags_falt = []
            flags = flags_branch + flags_falt
        elif index == 1:
            #计算密集型程序编译选项 
            flags_compute = []
            # 针对内存访问和 dTLB_store_misses
            flags_io = []
            flags = flags_compute + flags_io
       
        else:
            # 内存带宽/延迟密集型程序 (Memory-Bound)。 性能瓶颈在于数据从内存层次结构（L1, L2, LLC, 主存）传输到 CPU 的速度。
            # 针对数据缓存和内存访问优化:
            flags_mem =[]
            #针对指令缓存优化 (L1_icache_load_misses):
            # 与 Cluster 0 中“针对代码大小”的 Pass 很多是重叠的，因为小代码通常 i-cache 表现更好：
            flags_icache = []
            flags = flags_mem + flags_icache
        return flags
    
      

    def gain_baseline(self,suite_name):
        path = self.testsuite_path(suite_name)
        self.update_makefile(path," ","-O1 ","./data/Makefile2.llvm")
        O1 = self.get_runtime(path,suite_name)

        self.update_makefile(path," ","-O2 ","./data/Makefile2.llvm" )
        O2 = self.get_runtime(path,suite_name)

        self.update_makefile(path," ","-O3 ","./data/Makefile2.llvm" )
        O3 = self.get_runtime(path,suite_name)

        print("{} {} {} {}".format(suite_name,O1,O2,O3))
        
    def gain_baseline2(self, suite_name):
            """修改后的基准测试函数，用于保存基准程序数据"""
            path = self.testsuite_path(suite_name)
            
            # 测试O1
            self.update_makefile(path, " ", "-O1 ", "./data/Makefile2.llvm")
            O1 = self.get_runtime(path, suite_name)
            
            # 测试O2
            self.update_makefile(path, " ", "-O2 ", "./data/Makefile2.llvm")
            O2 = self.get_runtime(path, suite_name)
            
            # 测试O3
            self.update_makefile(path, " ", "-O3 ", "./data/Makefile2.llvm")
            O3 = self.get_runtime(path, suite_name)
            print("{} {} {} {}".format(suite_name,O1,O2,O3))
            
            return {'Program': suite_name, 'O1': O1, 'O2': O2, 'O3': O3}

    # 原来越界修改 250312注释
    # def boundary(self,x,n_max=1,n_min=0):
    #     if x < n_min:
    #         return n_min
    #     if x > n_max:
    #         return n_max
    #     return x

    # 新越界修改 250312修改
    def boundary(self, x, n_max=1, n_min=0):
        return np.clip(x, n_min, n_max)  # 直接截断到 [n_min, n_max] 区间
    # 种群初始化
    def init_position(self,N):

        seed = np.random.RandomState(456)
        # seed = np.random.RandomState(8)
        X = seed.random((N, self.n_flags))

        return X

    # 种群二值化 原始 250312注释
    # def binary_conversion(self,pops,thres = 0.5):
    #     size = len(pops)
    #     # print(pops,size)
    #     pop_bin = np.zeros([size, self.n_flags], dtype='int')

    #     for i in range(size):
    #         for d in range(self.n_flags):
    #             if pops[i,d] > thres:
    #                 pop_bin[i,d] = 1
    #             else:
    #                 pop_bin[i,d] = 0
    #     return pop_bin
    
     # 种群二值化 250312修改
    def binary_conversion(self, pops, thres=0.5):
        # 统一转换为二维数组处理（兼容单一个体）
        pops = np.array(pops)
        if pops.ndim == 1:
            pops = pops.reshape(1, -1)  # 一维转二维：[1, n_flags]
        
        n_pop, n_flags = pops.shape
        pop_bin = np.zeros((n_pop, n_flags), dtype=int)
        
        # 向量化操作替代循环（提速 100 倍）
        pop_bin[pops > thres] = 1
        return pop_bin.squeeze()  # 去除冗余维度（如一维输入返回一维）

    # polybench需要链接自己的库
    def testsuite_option(self,file_folder):
        if file_folder in polybench:
            return  " -I. -I../utilities ../utilities/polybench.c "
        if file_folder == "consumer_lame":
            return " -DLAMESNDFILE -DHAVEMPGLIB -DLAMEPARSE "

        return ""

    # testsuite的路径游些许不同
    def testsuite_path(self,file_folder):
        if file_folder in polybench:
            path = "./testsuite/" + file_folder
        else:
            path = "./testsuite/" + file_folder + "/src"
        return path


    def update_makefile(self,path,option,opt_level,makefile="./data/Makefile.llvm"):
        f = open(os.path.join(path, "Makefile"), "w")
        # print(os.path.join(path, "Makefile"))
        with open(os.path.join(makefile), "r") as g:
            while 1:
                line = g.readline()
                if line == "":
                    break
                elif "CCC_OPTS_ADD =" in line:
                    line = line.strip("\n") + option + opt_level + " \n"
                elif "CC_OPTS =" in line:
                    line = line.strip("\n")  + " \n"
                f.writelines(line)
        f.close()
  

    # if flags is None -> baseline evaluate
    def run_procedure(self,suite_name,flags=None):

        path = self.testsuite_path(suite_name)
        option = self.testsuite_option(suite_name)
        if flags is not None :
            opt_level = ""
            for i, flag in enumerate(flags):
                if flag:
                    opt_level += self.gcc_flags[i] + " "
        else:
            opt_level = "-O3 "
            self.update_makefile(path,option,opt_level,"./data/Makefile2.llvm")
            return self.get_runtime(path,suite_name)

        speedups = []
        for _ in range(1):
            # print(opt_level)
            # 更新makefile，编译testsuite
            self.update_makefile(path,option,opt_level)
            # 获取运行时间
            run_time = self.get_runtime(path,suite_name)
            #和O3比还是和O0比
            # self.update_makefile(path,option,"-O0 " )
            self.update_makefile(path,option,"-O3 " )
            baseline = self.get_runtime(path,suite_name)
            # df = pd.read_excel("benchmark_stats.xlsx", sheet_name = 'Max')
            # print("state")
            # df = df.set_index('Unnamed: 0').rename_axis('Program')
            # baseline = df.loc[suite_name, 'O3']
            # print(run_time,baseline)
            speedups.append(baseline/run_time)
        # print("option={}",option)
        print("Speedup={}".format(np.median(speedups)))
        # print(np.median(speedups))
        speedup = -np.median(speedups)
  
        # print(speedup)

        return speedup
    def run_procedure2(self,suite_name,flags=None):

        path = self.testsuite_path(suite_name)
        option = self.testsuite_option(suite_name)
        # print(flags)
        # print(flags)
        if flags is not None :
            opt_level = ""
            # opt_level = " ".join(flags) + " "
            opt_level = " ".join(map(str, flags)) + " "
            # for flag in flags:
            #     opt_level += flag + " "
        else:
            opt_level = "-O3 "
            self.update_makefile(path,option,opt_level,"./data/Makefile2.llvm")
            return self.get_runtime(path,suite_name)
        print("opt_level = {}".format(opt_level))
        speedups = []
        for _ in range(1):
            # print(opt_level)
            # 更新makefile，编译testsuite
            self.update_makefile(path,option,opt_level)
            # 获取运行时间
            run_time = self.get_runtime(path,suite_name)
            #和O3比还是和O0比
            # self.update_makefile(path,option,"-O0 " )
            self.update_makefile(path,option,"-O3 " )
            baseline = self.get_runtime(path,suite_name)
            # print("stat")
            # df = pd.read_excel("benchmark_stats.xlsx", sheet_name = 'Max')
            # df = df.set_index('Unnamed: 0').rename_axis('Program')
            # baseline = df.loc[suite_name, 'O3']
            # print(run_time,baseline)/home/work/zjq/eatuner_120/algorithm
            speedups.append(baseline/run_time)
        # print("option={}",option)
        print("Speedup={}".format(np.median(speedups)))
        # print(np.median(speedups))
        speedup = -np.median(speedups)
  
        return speedup
    def run_procedure_runtime(self,suite_name,flags=None):
    
        path = self.testsuite_path(suite_name)
        option = self.testsuite_option(suite_name)
        # print(flags)
        # print(flags)
        if flags is not None :
            opt_level = ""
            # opt_level = " ".join(flags) + " "
            opt_level = " ".join(map(str, flags)) + " "
            # for flag in flags:
            #     opt_level += flag + " "
        else:
            opt_level = "-O3 "
            self.update_makefile(path,option,opt_level,"./data/Makefile2.llvm")
            return self.get_runtime(path,suite_name)
        print("opy_level = {}".format(opt_level))
        speedups = []
        run_times = []
        for _ in range(1):
            # print(opt_level)
            # 更新makefile，编译testsuite
            self.update_makefile(path,option,opt_level)
            # 获取运行时间
            run_time = self.get_runtime(path,suite_name)
            #和O3比还是和O0比
            # self.update_makefile(path,option,"-O0 " )
            # self.update_makefile(path,option,"-O3 " )
            # baseline = self.get_runtime(path,suite_name)
            # print("stat")
            # df = pd.read_excel("benchmark_stats.xlsx", sheet_name = 'Max')
            # df = df.set_index('Unnamed: 0').rename_axis('Program')
            # baseline = df.loc[suite_name, 'O3']
            # print(run_time,baseline)/home/work/zjq/eatuner_120/algorithm
            # speedups.append(baseline/run_time)
            run_times.append(run_time)
        # print("option={}",option)
        print("run_time={}".format(np.median(run_times)))
        # print("Speedup={}".format(np.median(speedups)))
        # print(np.median(speedups))
        
        # speedup = -np.median(speedups)
  
        return run_time


    def run_procedure_multi_objective(self,suite_name,flags=None):

        path = self.testsuite_path(suite_name)
        option = self.testsuite_option(suite_name)
        # print(flags)
        if flags is not None :
            #####直接加文字版编译选项的我
            opt_level = ""
            # opt_level = " ".join(map(str, flags)) + " "
            for i, flag in enumerate(flags):
                if flag:
                    opt_level += self.gcc_flags[i] + " "
        else:
            opt_level = "-O3 "
            self.update_makefile(path,option,opt_level,"./data/Makefile2.llvm")
            run_times, energys = self.get_runtime_energy(path,suite_name)
            return run_times, energys
        # print("opy_level = {}".format(opt_level))
        speedups = []
        energyups = []
        for _ in range(1):
            # print(opt_level)
            # 更新makefile，编译testsuite
            self.update_makefile(path,option,opt_level)
            # 获取运行时间
            run_time,energy = self.get_runtime_energy(path,suite_name)
            #和O3比还是和O0比
            print("现在的运行时间{},能耗{}".format(run_time,energy))
            # self.update_makefile(path,option,"-O0 " )
            self.update_makefile(path,option,"-O3 " )
            baseline,base_energy = self.get_runtime_energy(path,suite_name)
            print("O3的运行时间{},能耗{}".format(baseline,base_energy))
            # print(run_time,baseline)/home/work/zjq/eatuner_120/algorithm
            speedups.append(baseline/run_time)
            energyups.append(base_energy/energy)
        # print("option={}",option)
        
        # print(np.median(speedups))
        speedup = -np.median(speedups)
        energyup = -np.median(energyups)
        print("加速比={}, 能量比={}".format(-speedup,-energyup))
        return speedup, energyup
    def get_runtime(self,path,suite_name):
        run_time = 0

        if os.path.exists(os.path.join(path, "a.out") ) or os.path.exists(os.path.join(path, "tmp.bc") ) :
            os.system("cd {} && make clean".format(path))
        os.system("cd {} && make ".format(path))

        # cmd = 'sudo /bin/bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
        # os.system(command=cmd)
        # print("clean...")

        if suite_name in polybench:
            command = "cd {} && chmod +x a.out && " \
            "bash -c '(TIMEFORMAT='%3R'; time ./a.out  > output.txt) &> time.txt'".format(path)  # runtime will be wrote in "time.txt"
            os.system(command=command)
            with open(path + "/time.txt", "r") as file:
                run_time = float(file.read().split('\n')[0])
        else : # for cBench
            command = "cd {} && chmod +x a.out && chmod +x ./__run  &&" \
            "bash -c '(TIMEFORMAT='%3R'; time ./__run 1  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"
            os.system(command=command)
            with open(path + "/time.txt", "r") as file:
                run_time = float(file.read().split('\n')[-2])

        return run_time

    def get_runtime_energy(self,path,suite_name):
        run_time = 0

        if os.path.exists(os.path.join(path, "a.out") ) or os.path.exists(os.path.join(path, "tmp.bc") ) :
            os.system("cd {} && make clean".format(path))
        os.system("cd {} && make ".format(path))

        # cmd = 'sudo /bin/bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
        # os.system(command=cmd)
        # print("clean...")
        pyRAPL.setup()
        meter = pyRAPL.Measurement('bar')
        
        if suite_name in polybench:
            meter.begin()
            command = "cd {} && chmod +x a.out && " \
            "bash -c '(TIMEFORMAT='%3R'; time ./a.out  > output.txt) &> time.txt'".format(path)  # runtime will be wrote in "time.txt"
            os.system(command=command)
            meter.end()
            with open(path + "/time.txt", "r") as file:
                run_time = float(file.read().split('\n')[0])
        else : # for cBench
            meter.begin()
            command = "cd {} && chmod +x a.out && chmod +x ./__run  &&" \
            "bash -c '(TIMEFORMAT='%3R'; time ./__run 1  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"
            os.system(command=command)
            meter.end()
            with open(path + "/time.txt", "r") as file:
                run_time = float(file.read().split('\n')[-2])
        print(meter.result)
        cpu_energy = sum(meter.result.pkg)
        ram_energy = sum(meter.result.dram)
        total_energy = cpu_energy + ram_energy
        print("function get_tuntime {}".format(run_time))
        print(total_energy)
        return run_time,total_energy

