from random import randint
from os import system
import numpy as np
import random
import os

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

    def gain_index(self):
        # 阈值为13
        result = [
            ['-sroa', '-jump-threading'],
            ['-mem2reg', '-gvn', '-instcombine'],
            ['-mem2reg', '-gvn', '-prune-eh'],
            ['-mem2reg', '-gvn', '-dse'],
            ['-mem2reg', '-loop-sink', '-loop-distribute'],
            ['-early-cse-memssa', '-instcombine'],
            ['-early-cse-memssa', '-dse'],
            ['-lcssa', '-loop-unroll'],
            ['-licm', '-gvn', '-instcombine'],
            ['-licm', '-gvn', '-prune-eh'],
            ['-licm', '-gvn', '-dse'],
            ['-memcpyopt', '-loop-distribute']
        ]
        matrix = np.zeros((len(result),self.n_flags))
        for i, sublist in enumerate(result):
            for value in sublist:
                j = self.gcc_flags.index(value)
                matrix[i][j] = 1;
        return matrix

    def gain_flags(self):
        # command = "sh ./data/diff.sh"
        # 本来想自动获取的，后来好像出了点问题，注释掉了，直接写死了
        # # res = os.system(command)
        # res = os.popen(command)
        # flags = res.read().split()
        # flags = ["-scoped-noalias-aa", "-ipsccp", "-mem2reg", "-loops", "-div-rem-pairs", "-early-cse", "-speculative-execution", "-strip-dead-prototypes", "-loop-simplify", "-slp-vectorizer", "-early-cse-memssa", "-mldst-motion", "-inferattrs", "-bdce", "-dse", "-rpo-function-attrs", "-loop-deletion", "-float2int", "-loop-distribute", "-aa",   "-adce", "-cg-profile", "-sroa", "-licm", "-simplifycfg", "-inject-tli-mappings", "-loop-vectorize", "-globals-aa", "-phi-values", "-sccp", "-lower-constant-intrinsics", "-called-value-propagation", "-tailcallelim", "-loop-idiom", "-function-attrs", "-correlated-propagation", "-aggressive-instcombine", "-memoryssa", "-callsite-splitting", "-basic-aa", "-reassociate", "-globalopt", "-constmerge", "-loop-rotate",  "-branch-prob", "-indvars",   "-openmp-opt-cgscc", "-lazy-value-info", "-jump-threading", "-memcpyopt", "-alignment-from-assumptions", "-memdep", "-scalar-evolution", "-pgo-memop-opt", "-block-freq", "-libcalls-shrinkwrap", "-lcssa", "-elim-avail-extern", "-lower-expect", "-loop-load-elim", "-gvn", "-postdomtree", "-vector-combine", "-instsimplify", "-loop-unroll", "-loop-sink", "-transform-warning", "-tbaa", "-domtree", "-globaldce", "-argpromotion", "-inline", "-deadargelim", "-demanded-bits", "-instcombine"]
        # O3_flags = "-tti -tbaa -scoped-noalias-aa -assumption-cache-tracker -targetlibinfo -verify -lower-expect -simplifycfg -domtree -sroa -early-cse -targetlibinfo -tti -tbaa -scoped-noalias-aa -assumption-cache-tracker -profile-summary-info -annotation2metadata -forceattrs -inferattrs -domtree -callsite-splitting -ipsccp -called-value-propagation -globalopt -domtree -mem2reg -deadargelim -domtree -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -inline -openmp-opt-cgscc -function-attrs -argpromotion -domtree -sroa -basic-aa -aa -memoryssa -early-cse-memssa -speculative-execution -aa -lazy-value-info -jump-threading -correlated-propagation -simplifycfg -domtree -aggressive-instcombine -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -libcalls-shrinkwrap -loops -postdomtree -branch-prob -block-freq -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -pgo-memop-opt -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -tailcallelim -simplifycfg -reassociate -domtree -basic-aa -aa -memoryssa -loops -loop-simplify -lcssa-verification -lcssa -scalar-evolution -lazy-branch-prob -lazy-block-freq -licm -loop-rotate -licm -loop-unswitch -simplifycfg -domtree -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-idiom -indvars -loop-deletion -loop-unroll -sroa -aa -mldst-motion -phi-values -aa -memdep -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -gvn -sccp -demanded-bits -bdce -basic-aa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -lazy-value-info -jump-threading -correlated-propagation -postdomtree -adce -basic-aa -aa -memoryssa -memcpyopt -loops -dse -loop-simplify -lcssa-verification -lcssa -aa -scalar-evolution -lazy-branch-prob -lazy-block-freq -licm -simplifycfg -domtree -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -barrier -elim-avail-extern -basiccg -rpo-function-attrs -globalopt -globaldce -basiccg -globals-aa -domtree -float2int -lower-constant-intrinsics -loops -loop-simplify -lcssa-verification -lcssa -basic-aa -aa -scalar-evolution -loop-rotate -loop-accesses -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-distribute -postdomtree -branch-prob -block-freq -scalar-evolution -basic-aa -aa -loop-accesses -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -inject-tli-mappings -loop-vectorize -loop-simplify -scalar-evolution -aa -loop-accesses -lazy-branch-prob -lazy-block-freq -loop-load-elim -basic-aa -aa -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -domtree -loops -scalar-evolution -basic-aa -aa -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -inject-tli-mappings -slp-vectorizer -vector-combine -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-unroll -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -memoryssa -loop-simplify -lcssa-verification -lcssa -scalar-evolution -lazy-branch-prob -lazy-block-freq -licm -opt-remark-emitter -transform-warning -alignment-from-assumptions -strip-dead-prototypes -globaldce -constmerge -cg-profile -domtree -loops -postdomtree -branch-prob -block-freq -loop-simplify -lcssa-verification -lcssa -basic-aa -aa -scalar-evolution -block-freq -loop-sink -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instsimplify -div-rem-pairs -simplifycfg -annotation-remarks -verify -domtree -targetlibinfo -domtree -loops -postdomtree -branch-prob -block-freq -targetlibinfo -domtree -loops -postdomtree -branch-prob -block-freq -targetlibinfo -domtree -loops -lazy-branch-prob -lazy-block-freq"
        # O3_flags = O3_flags.split()
        # flags = set()
        # for flag in O3_flags:
        #     flags.add(flag)
        #2024.11.26 原来的
        # flags = ['-tti', '-tbaa', '-scoped-noalias-aa', '-assumption-cache-tracker', '-targetlibinfo', '-verify', '-lower-expect', '-simplifycfg', '-domtree', '-sroa', '-early-cse', '-profile-summary-info', '-annotation2metadata', '-forceattrs', '-inferattrs', '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-globalopt', '-mem2reg', '-deadargelim', '-basic-aa', '-aa', '-loops', '-lazy-branch-prob', '-lazy-block-freq', '-opt-remark-emitter', '-instcombine', '-basiccg', '-globals-aa', '-prune-eh', '-inline', '-openmp-opt-cgscc', '-function-attrs', '-argpromotion', '-memoryssa', '-early-cse-memssa', '-speculative-execution', '-lazy-value-info', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-libcalls-shrinkwrap', '-postdomtree', '-branch-prob', '-block-freq', '-pgo-memop-opt', '-tailcallelim', '-reassociate', '-loop-simplify', '-lcssa-verification', '-lcssa', '-scalar-evolution', '-licm', '-loop-rotate', '-loop-unswitch', '-loop-idiom', '-indvars', '-loop-deletion', '-loop-unroll', '-mldst-motion', '-phi-values', '-memdep', '-gvn', '-sccp', '-demanded-bits', '-bdce', '-adce', '-memcpyopt', '-dse', '-barrier', '-elim-avail-extern', '-rpo-function-attrs', '-globaldce', '-float2int', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-inject-tli-mappings', '-loop-vectorize', '-loop-load-elim', '-slp-vectorizer', '-vector-combine', '-transform-warning', '-alignment-from-assumptions', '-strip-dead-prototypes', '-constmerge', '-cg-profile', '-loop-sink', '-instsimplify', '-div-rem-pairs', '-annotation-remarks']
        # flags = ['-scoped-noalias-aa', '-assumption-cache-tracker', '-verify', '-lower-expect', '-sroa', '-early-cse', '-profile-summary-info', '-forceattrs', '-mem2reg', '-deadargelim', '-loops', '-lazy-branch-prob', '-instcombine', '-prune-eh', '-inline', '-function-attrs', '-early-cse-memssa', '-speculative-execution', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-postdomtree', '-block-freq', '-pgo-memop-opt', '-lcssa', '-licm', '-loop-rotate', '-loop-unroll', '-mldst-motion', '-gvn', '-adce', '-memcpyopt', '-dse', '-barrier', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-vector-combine', '-transform-warning', '-strip-dead-prototypes', '-cg-profile', '-loop-sink', '-div-rem-pairs', '-annotation-remarks']
        flags = ['-scoped-noalias-aa', '-assumption-cache-tracker', '-verify', '-lower-expect', '-sroa', '-early-cse', '-profile-summary-info', '-forceattrs', '-mem2reg', '-deadargelim', '-loops', '-lazy-branch-prob', '-instcombine', '-prune-eh', '-inline', '-function-attrs', '-early-cse-memssa', '-speculative-execution', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-postdomtree', '-branch-prob', '-block-freq', '-pgo-memop-opt', '-lcssa', '-licm', '-loop-rotate', '-loop-unroll', '-mldst-motion', '-gvn', '-adce', '-memcpyopt', '-dse', '-barrier', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-vector-combine', '-transform-warning', '-cg-profile', '-loop-sink', '-div-rem-pairs', '-annotation-remarks']  
        
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


    # 越界修改
    def boundary(self,x,n_max=1,n_min=0):
        if x < n_min:
            return n_min
        if x > n_max:
            return n_max
        return x

    # 种群初始化
    def init_position(self,N):

        seed = np.random.RandomState(456)
        # seed = np.random.RandomState(8)
        X = seed.random((N, self.n_flags))

        return X

    # 种群二值化
    def binary_conversion(self,pops,thres = 0.5):
        size = len(pops)
        # print(pops,size)
        pop_bin = np.zeros([size, self.n_flags], dtype='int')

        for i in range(size):
            for d in range(self.n_flags):
                if pops[i,d] > thres:
                    pop_bin[i,d] = 1
                else:
                    pop_bin[i,d] = 0
        return pop_bin

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
            opt_level = " ".join(flags) + " "
            # for flag in flags:
            #     opt_level += flag + " "
        else:
            opt_level = "-O0 "
            self.update_makefile(path,option,opt_level,"./data/Makefile2.llvm")
            return self.get_runtime(path,suite_name)
        print("opy_level = {}".format(opt_level))
        speedups = []
        for _ in range(1):
            # print(opt_level)
            # 更新makefile，编译testsuite
            self.update_makefile(path,option,opt_level)
            # 获取运行时间
            run_time = self.get_runtime(path,suite_name)
            #和O3比还是和O0比
            # self.update_makefile(path,option,"-O0 " )
            self.update_makefile(path,option,"-O0 " )
            baseline = self.get_runtime(path,suite_name)
            # print(run_time,baseline)/home/work/zjq/eatuner_120/algorithm
            speedups.append(baseline/run_time)
        # print("option={}",option)
        print("Speedup={}".format(np.median(speedups)))
        # print(np.median(speedups))
        speedup = -np.median(speedups)
  
        # print(speedup)

        return speedup

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

    #注释掉的是一些之前在GCC上跑的代码

    # def print_min(self,min_runtime, min_flags):
    #     timer = min_runtime[0]
    #     gen = 0
    #     flags = min_flags[0]
    #     for i in range(len(min_runtime)):
    #         if timer > min_runtime[i] :
    #             timer = min_runtime[i]
    #             flags = min_flags[i]
    #             gen = i

    #     print("best-> time:{}, gen:{}, flags{}".format(timer,gen,flags))

    # def gain_baseline_cbench(self,file_folder,opt):
    #     path = "./cBench_V1.1/" + file_folder + "/src"
    #     f = open(os.path.join(path, "Makefile"), "w")
    #     with open(os.path.join("./data/Makefile.gcc"), "r") as g:
    #         while 1:
    #             line = g.readline()
    #             if line == "":
    #                 break
    #             elif "CCC_OPTS_ADD =" in line:
    #                 line = line.strip("\n") + " \n"
    #             elif "CC_OPTS =" in line:
    #                 line = line.strip("\n")  + opt + " \n"
    #             f.writelines(line)
    #     f.close()

    #     if os.path.exists(os.path.join(path, "*.o")):
    #         os.system("cd {} && make clean".format(path))
    #     os.system("cd {} && make".format(path))


    #     command = "cd {} && chmod +x a.out && chmod +x ./__run  &&" \
    #         "bash -c '(TIMEFORMAT='%3R'; time ./__run 1  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"

    #     baselines = []
    #     for _ in range(10):
    #         os.system(command=command)
    #         with open(path + "/time.txt","r") as file:
    #             baselines.append(float(file.read().split('\n')[-2]))
    #         # print(baselines[-1])
    #     self.baseline = min(baselines)
    #     # return self.baseline
    #     # print("file_folder:{},opt:{},baseline: {}".format(file_folder,opt,self.baseline))

    # def gain_baseline(self,file_folder,opt):
    #     path = "./polybench/" + file_folder
    #     f = open(os.path.join(path, "Makefile"), "w")
    #     with open(os.path.join("./data/Makefile.gcc"), "r") as g:
    #         while 1:
    #             line = g.readline()
    #             if line == "":
    #                 break
    #             elif "CCC_OPTS_ADD =" in line:
    #                 line = line.strip("\n") + " \n"
    #             elif "CC_OPTS =" in line:
    #                 line = line.strip("\n")  + opt + "  -I. -I../utilities ../utilities/polybench.c\n"
    #             f.writelines(line)
    #     f.close()

    #     if os.path.exists(os.path.join(path, "*.o")):
    #         os.system("cd {} && make clean".format(path))
    #     os.system("cd {} && make".format(path))


    #     command = "cd {} && chmod +x a.out  &&" \
    #         "bash -c '(TIMEFORMAT='%3R'; time ./a.out  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"

    #     baselines = []
    #     for _ in range(10):
    #         os.system(command=command)
    #         # input()
    #         with open(path + "/time.txt","r") as file:
    #             baselines.append(float(file.read().split('\n')[0]))
    #         # print(baselines[-1])
    #     self.baseline = min(baselines)


    # # GCC
    # def run_procedure_GCC(self,flags,file_folder,run_opts=""):
    #     self.times += 1
    #     compile_flags = ""
    #     for i,flag in enumerate(flags):
    #         if flag:
    #             compile_flags += self.gcc_flags[i] + " "
    #     # print(compile_flags)
    #     # path = "./cBench_V1.1/automotive_bitcount/src"
    #     path = "./cBench_V1.1/" + file_folder + "/src"
    #     f = open(os.path.join(path, "Makefile"), "w")
    #     with open(os.path.join("./data/Makefile.gcc"), "r") as g:
    #         while 1:
    #             line = g.readline()
    #             if line == "":
    #                 break
    #             elif "CCC_OPTS_ADD =" in line:
    #                 line = line.strip("\n") + " {}\n".format(compile_flags)
    #             elif "CC_OPTS =" in line:
    #                 line = line.strip("\n")  + "-O1 \n"
    #             f.writelines(line)
    #     f.close()

    #     if os.path.exists(os.path.join(path, "*.o")):
    #         os.system("cd {} && make clean".format(path))
    #     os.system("cd {} && make ".format(path))


    #     command = "cd {} && chmod +x a.out && chmod +x ./__run  &&" \
    #         "bash -c '(TIMEFORMAT='%3R'; time ./__run 1  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"
    #     os.system(command=command)
    #     # input()
    #     with open(path + "/time.txt","r") as file:
    #         # print(str(file.read()).split('\n'))
    #         run_time = float(file.read().split('\n')[-2])
    #     # print(run_time)
    #     return run_time

    # # GCC ploy
    # def run_procedure(self,flags,file_folder,run_opts=""):
    #     self.times += 1
    #     compile_flags = ""
    #     for i,flag in enumerate(flags):
    #         if flag:
    #             compile_flags += self.gcc_flags[i] + " "

    #     path = "./polybench/" + file_folder
    #     f = open(os.path.join(path, "Makefile"), "w")
    #     with open(os.path.join("./data/Makefile.gcc"), "r") as g:
    #         while 1:
    #             line = g.readline()
    #             if line == "":
    #                 break
    #             elif "CCC_OPTS_ADD =" in line:
    #                 line = line.strip("\n") + " {}\n".format(compile_flags)
    #             elif "CC_OPTS =" in line:
    #                 line = line.strip("\n")  + "-O1  -I. -I../utilities ../utilities/polybench.c\n"
    #             f.writelines(line)
    #     f.close()

    #     if os.path.exists(os.path.join(path, "*.o")):
    #         os.system("cd {} && make clean".format(path))
    #     os.system("cd {} && make ".format(path))

    #     cmd = 'sudo /bin/bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
    #     os.system(command=cmd)

    #     command = "cd {} && chmod +x a.out  &&" \
    #         "bash -c '(TIMEFORMAT='%3R'; time ./a.out  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"
    #     os.system(command=command)
    #     # input()
    #     with open(path + "/time.txt","r") as file:
    #         run_time = float(file.read().split('\n')[0])
    #     # print(run_time)
    #     return run_time


    # def run_procedure3(self,flags,file_folder,run_opts=""):
    #     compile_flags = ""
    #     for i,flag in enumerate(flags):
    #         if flag:
    #             compile_flags += self.gcc_flags[i] + " "
    #     # print(compile_flags)
    #     # path = "./cBench_V1.1/automotive_bitcount/src"
    #     path = "./MiBench2/" + file_folder + "/"
    #     f = open(os.path.join(path, "Makefile"), "w")
    #     with open(os.path.join("./data/Makefile.gcc"), "r") as g:
    #         while 1:
    #             line = g.readline()
    #             if line == "":
    #                 break
    #             elif "CCC_OPTS_ADD =" in line:
    #                 line = line.strip("\n") + " {}\n".format(compile_flags)
    #             elif "CC_OPTS =" in line:
    #                 line = line.strip("\n")  + "-O1 \n"
    #             f.writelines(line)
    #     f.close()

    #     if os.path.exists(os.path.join(path, "*.o")):
    #         os.system("cd {} && make clean".format(path))
    #     os.system("cd {} && make ".format(path))


    #     command = "cd {} && chmod +x a.out && chmod +x ./a.out  &&" \
    #         "bash -c '(TIMEFORMAT='%3R'; time ./a.out  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"
    #     os.system(command=command)
    #     # input()
    #     with open(path + "/time.txt","r") as file:
    #         # print(str(file.read()).split('\n'))
    #         run_time = float(file.read().split('\n')[-2])
    #     # print(run_time)
    #     return run_time


    # def run_procedure2(self,flags,file_folder,run_opts=""):
    #     compile_flags = ""
    #     for i,flag in enumerate(flags):
    #         if flag:
    #             compile_flags += self.gcc_flags[i] + " "
    #     # print(compile_flags)
    #     # path = "./cBench_V1.1/automotive_bitcount/src"
    #     path = "./cpu2006/benchspec/CPU2006/" + file_folder + ""
    #     f = open(os.path.join(path, "src/Makefile"), "w")
    #     with open(os.path.join("./data/Makefile.spec.gcc"), "r") as g:
    #         while 1:
    #             line = g.readline()
    #             if line == "":
    #                 break
    #             elif "CCC_OPTS_ADD =" in line:
    #                 line = line.strip("\n") + " {}\n".format(compile_flags)
    #             elif "CC_OPTS =" in line:
    #                 line = line.strip("\n")  + "-O1 \n"

    #             f.writelines(line)
    #     f.close()

    #     if os.path.exists(os.path.join(path, "*.o")):
    #         os.system("cd {} && make clean".format(path+ '/src'))
    #     os.system("cd {} && make ".format(path+ '/src'))

    #     # temp.sh >>
    #     # rm input.program.bz2
    #     # chmod +x ./a.out
    #     # bash -c '(TIMEFORMAT='%3R'; time ./a.out ./input.program -k > output.txt) &> time.txt'

    #     command = "cd {} && sh temp.sh".format(path + '/exe') # runtime will be wrote in "time.txt"
    #     os.system(command=command)
    #     # input()
    #     with open(path + "/exe/time.txt","r") as file:
    #         # print(str(file.read()).split('\n'))
    #         run_time = float(file.read().split('\n')[-2])
    #     # print(run_time)
    #     return run_time




    #     def run_procedure4(self,flags,file_folder,run_opts=""):
    #         compile_flags = ""
    #         for i,flag in enumerate(flags):
    #             if flag:
    #                 compile_flags += self.gcc_flags[i] + " "
    #         # print(compile_flags)
    #         # path = "./cBench_V1.1/automotive_bitcount/src"
    #         path = "./cBench_V1.1/" + file_folder + "/src"
    #         f = open(os.path.join(path, "Makefile"), "w")
    #         with open(os.path.join("./data/Makefile.llvm"), "r") as g:
    #             while 1:
    #                 line = g.readline()
    #                 if line == "":
    #                     break
    #                 elif "CCC_OPTS_ADD =" in line:
    #                     line = line.strip("\n") + " {}\n".format(compile_flags)
    #                 elif "CC_OPTS =" in line:
    #                     line = line.strip("\n")  + " \n"
    #                 f.writelines(line)
    #         f.close()

    #         if os.path.exists(os.path.join(path, "*.o")):
    #             os.system("cd {} && make clean".format(path))
    #         os.system("cd {} && make ".format(path))


    #         command = "cd {} && chmod +x a.out && chmod +x ./__run  &&" \
    #             "bash -c '(TIMEFORMAT='%3R'; time ./__run 1  > output.txt) &> time.txt'".format(path) # runtime will be wrote in "time.txt"
    #         os.system(command=command)
    #         # input()
    #         with open(path + "/time.txt","r") as file:
    #             # print(str(file.read()).split('\n'))
    #             run_time = float(file.read().split('\n')[-2])
    #         # print(run_time)
    #         return run_time
