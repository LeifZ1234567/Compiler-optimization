from random import randint
from os import system
import numpy as np
import random
import os
import pandas as pd
import pyRAPL
cbench = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_rijndael_e", "security_sha", "office_stringsearch1","consumer_lame", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian",
]
polybench = ["correlation","covariance","2mm","3mm","atax","bicg","doitgen","mvt","gemm","gemver","gesummv","symm","syr2k","syrk","trmm","cholesky","durbin","gramschmidt","lu","ludcmp","trisolv","deriche","floyd-warshall","adi","fdtd-2d","heat-3d","jacobi-1d","jacobi-2d","seidel-2d"]


class Util(object):
    def __init__(self) -> None:
        # self.gcc_flags = ["-falign-labels", "-fcaller-saves", "-fcode-hoisting", "-fcrossjumping", "-fcse-follow-jumps", "-fdevirtualize", "-fdevirtualize-speculatively", "-fexpensive-optimizations", "-fgcse", "-fhoist-adjacent-loads", "-findirect-inlining", "-finline-small-functions", "-fipa-bit-cp", "-fipa-cp", "-fipa-icf", "-fipa-icf-functions", "-fipa-icf-variables", "-fipa-ra", "-fipa-sra", "-fipa-vrp", "-fisolate-erroneous-paths-dereference", "-flra-remat", "-foptimize-sibling-calls", "-foptimize-strlen", "-fpartial-inlining", "-fpeephole2", "-free", "-freorder-blocks-and-partition", "-freorder-functions", "-frerun-cse-after-loop", "-fschedule-insns2", "-fstore-merging", "-fstrict-aliasing", "-fstrict-overflow", "-fthread-jumps", "-ftree-pre", "-ftree-switch-conversion", "-ftree-tail-merge", "-ftree-vrp",]
        self.gcc_flags = self.gain_flags()
        self.baseline = 10
        self.n_flags = len(self.gcc_flags)
        self.times = 0

    def get_cluster_index(self, program_name):
        # 定义每个索引对应的程序名列表
        cluster_0_programs = ['automotive_qsort1', 'automotive_susan_e', 'automotive_susan_s', 'bzip2d', 'bzip2e', 'consumer_jpeg_c', 'consumer_jpeg_d', 'consumer_tiff2bw', 'consumer_tiff2rgba', 'consumer_tiffdither', 'consumer_tiffmedian', 'office_rsynth', 'office_stringsearch1', 'telecom_adpcm_d']
        cluster_1_programs = ['automotive_bitcount', 'automotive_susan_c', 'network_dijkstra', 'network_patricia', 'security_blowfish_d', 'security_blowfish_e', 'security_sha', 'telecom_adpcm_c', 'telecom_CRC32']
        cluster_2_programs = ['security_rijndael_d', 'security_rijndael_e']

        # 判断程序名属于哪个索引
        if program_name in cluster_0_programs:
            return 0
        elif program_name in cluster_1_programs:
            return 1
        else:
            return 2
    def gain_flags(self):
        #2024.11.26 原来的
        flags = ['-tti', '-tbaa', '-scoped-noalias-aa', '-assumption-cache-tracker', '-targetlibinfo', '-verify', '-lower-expect', '-simplifycfg', '-domtree', '-sroa', '-early-cse', '-profile-summary-info', '-annotation2metadata', '-forceattrs', '-inferattrs', '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-globalopt', '--mem2reg', '-deadargelim', '-basic-aa', '-aa', '-loops', '-lazy-branch-prob', '-lazy-block-freq', '-opt-remark-emitter', '-instcombine', '-basiccg', '-globals-aa', '-prune-eh', '-inline', '-openmp-opt-cgscc', '-function-attrs', '-argpromotion', '-memoryssa', '-early-cse-memssa', '-speculative-execution', '-lazy-value-info', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-libcalls-shrinkwrap', '-postdomtree', '-branch-prob', '-block-freq', '-pgo-memop-opt', '-tailcallelim', '-reassociate', '-loop-simplify', '-lcssa-verification', '-lcssa', '-scalar-evolution', '-licm', '-loop-rotate', '-loop-unswitch', '-loop-idiom', '-indvars', '-loop-deletion', '-loop-unroll', '-mldst-motion', '-phi-values', '-memdep', '-gvn', '-sccp', '-demanded-bits', '-bdce', '-adce', '-memcpyopt', '-dse', '-barrier', '-elim-avail-extern', '-rpo-function-attrs', '-globaldce', '-float2int', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-inject-tli-mappings', '-loop-vectorize', '-loop-load-elim', '-slp-vectorizer', '-vector-combine', '-transform-warning', '-alignment-from-assumptions', '-strip-dead-prototypes', '-constmerge', '-cg-profile', '-loop-sink', '-instsimplify', '-div-rem-pairs', '-annotation-remarks']
        # 原来的除去公共
        # flags = ['-tti', '-tbaa', '-scoped-noalias-aa', '-assumption-cache-tracker', '-targetlibinfo', '-verify', '-lower-expect', '-simplifycfg', '-domtree', '-early-cse', '-profile-summary-info', '-annotation2metadata', '-forceattrs', '-inferattrs', '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-globalopt', '-deadargelim', '-basic-aa', '-aa', '-loops', '-lazy-branch-prob', '-lazy-block-freq', '-opt-remark-emitter', '-basiccg', '-globals-aa', '-inline', '-openmp-opt-cgscc', '-function-attrs', '-argpromotion', '-memoryssa', '-speculative-execution', '-lazy-value-info', '-correlated-propagation', '-aggressive-instcombine', '-libcalls-shrinkwrap', '-postdomtree', '-branch-prob', '-block-freq', '-pgo-memop-opt', '-tailcallelim', '-reassociate', '-loop-simplify', '-lcssa-verification', '-scalar-evolution', '-loop-rotate', '-loop-unswitch', '-loop-idiom', '-indvars', '-loop-deletion', '-mldst-motion', '-phi-values', '-memdep', '-sccp', '-demanded-bits', '-bdce', '-adce', '-barrier', '-elim-avail-extern', '-rpo-function-attrs', '-globaldce', '-float2int', '-lower-constant-intrinsics', '-loop-accesses', '-inject-tli-mappings', '-loop-vectorize', '-loop-load-elim', '-slp-vectorizer', '-vector-combine', '-transform-warning', '-alignment-from-assumptions', '-strip-dead-prototypes', '-constmerge', '-cg-profile', '-instsimplify', '-div-rem-pairs', '-annotation-remarks']
        # flags = ['-scoped-noalias-aa', '-assumption-cache-tracker', '-verify', '-lower-expect', '-sroa', '-early-cse', '-profile-summary-info', '-forceattrs', '-mem2reg', '-deadargelim', '-loops', '-lazy-branch-prob', '-instcombine', '-prune-eh', '-inline', '-function-attrs', '-early-cse-memssa', '-speculative-execution', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-postdomtree', '-block-freq', '-pgo-memop-opt', '-lcssa', '-licm', '-loop-rotate', '-loop-unroll', '-mldst-motion', '-gvn', '-adce', '-memcpyopt', '-dse', '-barrier', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-vector-combine', '-transform-warning', '-strip-dead-prototypes', '-cg-profile', '-loop-sink', '-div-rem-pairs', '-annotation-remarks']
        # print(flags)
        return flags
    def gain_flags_cluster(self, index):
        flags_all = ['-scoped-noalias-aa', '-assumption-cache-tracker', '-verify', '-lower-expect', '-sroa', '-early-cse', '-profile-summary-info', '-forceattrs', '-mem2reg', '-deadargelim', '-loops', '-lazy-branch-prob', '-instcombine', '-prune-eh', '-inline', '-function-attrs', '-early-cse-memssa', '-speculative-execution', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-postdomtree', '-block-freq', '-pgo-memop-opt', '-lcssa', '-licm', '-loop-rotate', '-loop-unroll', '-mldst-motion', '-gvn', '-adce', '-memcpyopt', '-dse', '-barrier', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-vector-combine', '-transform-warning', '-strip-dead-prototypes', '-cg-profile', '-loop-sink', '-div-rem-pairs', '-annotation-remarks']
        # flags_al = ['-tti', '-tbaa', '-scoped-noalias-aa', '-assumption-cache-tracker', '-targetlibinfo', '-verify', '-lower-expect', '-simplifycfg', '-domtree', '-sroa', '-early-cse', '-profile-summary-info', '-annotation2metadata', '-forceattrs', '-inferattrs', '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-globalopt', '--mem2reg', '-deadargelim', '-basic-aa', '-aa', '-loops', '-lazy-branch-prob', '-lazy-block-freq', '-opt-remark-emitter', '-instcombine', '-basiccg', '-globals-aa', '-prune-eh', '-inline', '-openmp-opt-cgscc', '-function-attrs', '-argpromotion', '-memoryssa', '-early-cse-memssa', '-speculative-execution', '-lazy-value-info', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-libcalls-shrinkwrap', '-postdomtree', '-branch-prob', '-block-freq', '-pgo-memop-opt', '-tailcallelim', '-reassociate', '-loop-simplify', '-lcssa-verification', '-lcssa', '-scalar-evolution', '-licm', '-loop-rotate', '-loop-unswitch', '-loop-idiom', '-indvars', '-loop-deletion', '-loop-unroll', '-mldst-motion', '-phi-values', '-memdep', '-gvn', '-sccp', '-demanded-bits', '-bdce', '-adce', '-memcpyopt', '-dse', '-barrier', '-elim-avail-extern', '-rpo-function-attrs', '-globaldce', '-float2int', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-inject-tli-mappings', '-loop-vectorize', '-loop-load-elim', '-slp-vectorizer', '-vector-combine', '-transform-warning', '-alignment-from-assumptions', '-strip-dead-prototypes', '-constmerge', '-cg-profile', '-loop-sink', '-instsimplify', '-div-rem-pairs', '-annotation-remarks']
     
        if index == 0:
            #分支型编译选项
            flags_branch = ['-branch-prob', '-jump-threading','-tailcallelim']
            #页错误编译选项
            # flags_falt = ['-globalopt','-deadargelim','-dce','-globaldce','-function-attrs','-instcombine','-strip-dead-prototypes','-mergereturn','-mergefunc']
            # additional_flags = ['-ipsccp', '-bdce', '-demanded-bits','-loop-unswitch', '-loop-reroll','-licm']
            
            flags_add = ['-mem2reg', '-prune-eh', '-licm', '-gvn', '-dse', '-tailcallelim', '-globalopt', '-instcombine', '-mergereturn', '-mergefunc', '-verify', '-sroa', '-early-cse', '-profile-summary-info', '-inline', '-function-attrs', '-early-cse-memssa', '-jump-threading', '-correlated-propagation', '-postdomtree', '-block-freq', '-lcssa', '-loop-unroll', '-mldst-motion', '-adce', '-memcpyopt', '-loop-distribute', '-vector-combine', '-transform-warning', '-cg-profile', '-annotation-remarks']
        
            flags = flags_branch + flags_add
         
        elif index == 1:
            #计算密集型程序编译选项 
            
            # flags_ipc = ['-early-cse-memssa', '-instcombine', '-sccp', '-speculative-execution', '-gvn']
            # flags_cache = ['-loop-distribute', '-licm', '-slp-vectorizer', '-loop-unroll']
            # flags_branch = ['-jump-threading', '-tailcallelim', '-loop-unswitch', '-lazy-branch-prob']
            # flags_TLB  = ['-loop-load-elim', '-mldst-motion']
            # flags_men = ['-dse', '-memcpyopt', '-adce']
            # flags_page_fault = ['-inline', '-reassociate']
            # flags_zonghe  = ['-aggressive-instcombine', '-loop-deletion']
            # 整理成列表
            flag_add = ['-tti', '-tbaa', '-scoped-noalias-aa', '-verify', '-domtree', '-profile-summary-info', '-annotation2metadata', '-forceattrs', '-inferattrs', '-ipsccp', '-called-value-propagation', '-mem2reg', '-basic-aa', '-aa', '-loops', '-lazy-branch-prob', '-basiccg', '-globals-aa', '-argpromotion', '-speculative-execution', '-lazy-value-info', '-correlated-propagation', '-aggressive-instcombine', '-libcalls-shrinkwrap', '-reassociate', '-lcssa', '-scalar-evolution', '-loop-unswitch', '-indvars', '-loop-deletion', '-loop-unroll', '-phi-values', '-memdep', '-sccp', '-bdce', '-adce', '-dse', '-elim-avail-extern', '-globaldce', '-float2int', '-lower-constant-intrinsics', '-loop-distribute', '-loop-vectorize', '-loop-load-elim', '-slp-vectorizer', '-alignment-from-assumptions', '-constmerge', '-div-rem-pairs']
            
            # flags = flags_ipc + flags_cache + flags_branch + flags_TLB + flags_men + flags_page_fault + flags_zonghe + flags_all
            # set一下flag，不要重复
            # flags = list(set(flags))
            flags = flag_add
       
        else:
            # 内存带宽/延迟密集型程序 (Memory-Bound)。 性能瓶颈在于数据从内存层次结构（L1, L2, LLC, 主存）传输到 CPU 的速度。
            # 针对数据缓存和内存访问优化:
            flags_mem =['-licm', '-loop-idiom', '-loop-interchange', '-loop-unroll-and-jam', '-sroa', '-tailcallopt']
            #针对指令缓存优化 (L1_icache_load_misses):
            # 与 Cluster 0 中“针对代码大小”的 Pass 很多是重叠的，因为小代码通常 i-cache 表现更好：
            flags_icache = ['-globalopt', '-dce', '-globaldce', '-instcombine', '-mergefunc', '-mergereturn', '-hotcoldsplit', '-partial-inliner', '-inline']
            additional_flags = ['-ipsccp', '-bdce', '-demanded-bits','-loop-unswitch', '-loop-reroll']
            flags = flags_mem + flags_icache + flags_all + additional_flags
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
            print("opt run_time")
            print(run_time)
            
            #和O3比还是和O0比
            # self.update_makefile(path,option,"-O0 " )
            self.update_makefile(path,option,"-O3 " )
            baseline = self.get_runtime(path,suite_name)

            print("O3 run_time")
            print(baseline)
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

