from algorithm.BA import BA
from algorithm.CS import CS
from algorithm.DE import DE
from algorithm.EDA import EDA
from algorithm.FA import FA
from algorithm.FPA import FPA
from algorithm.GWO import GWO
from algorithm.HHO import HHO
from algorithm.JAYA import JAYA
from algorithm.PSO import PSO
from algorithm.SCA import SCA
from algorithm.SSA import SSA
from algorithm.WOA import WOA
from algorithm.GA import GA
from algorithm.EnhancedHybridDE import EnhancedHybridDE
from algorithm.EnhancedHybridBA import EnhancedHybridBA
from algorithm.EnhancedHybridEDA import EnhancedHybridEDA
from algorithm.EnhancedHybridGA import EnhancedHybridGA
from algorithm.EnhancedHybridCS import EnhancedHybridCS
from algorithm.EnhancedHybridFA import EnhancedHybridFA
from algorithm.EnhancedHybridFPA import EnhancedHybridFPA
from algorithm.EnhancedHybridJAYA import EnhancedHybridJAYA
from algorithm.EnhancedHybridPSO import EnhancedHybridPSO
from algorithm.EnhancedHybridSCA import EnhancedHybridSCA



from algorithm.HybridBA import HybridBA
from util import *
from os import system
import argparse
import time
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

help_string = "Usage:"

# polybench时间太长，平时测试先不用
cbench = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian",
]


polybench = ["correlation","covariance","2mm","3mm","atax","bicg","doitgen","mvt","gemm","gemver","gesummv","symm","syr2k","syrk","trmm","cholesky","durbin","gramschmidt","lu","ludcmp","trisolv","deriche","floyd-warshall","adi","fdtd-2d","heat-3d","jacobi-1d","jacobi-2d","seidel-2d"]
testsuite = cbench # + polybench
known_seqs = [['-sroa', '-jump-threading'],['-mem2reg', '-gvn', '-instcombine'],['-mem2reg', '-gvn', '-prune-eh'],['-mem2reg', '-gvn', '-dse'],['-mem2reg', '-loop-sink', '-loop-distribute'],['-early-cse-memssa', '-instcombine'],['-early-cse-memssa', '-dse'],['-lcssa', '-loop-unroll'],['-licm', '-gvn', '-instcombine'],['-licm', '-gvn', '-prune-eh'],['-licm', '-gvn', '-dse'],['-memcpyopt', '-loop-distribute']]  

# 所有已经支持的算法，但部分存在BUG
# algorithm = ["BA", "CS", "DE", "EDA", "FA", "FPA", "GA",
#                 "GWO", "HHO", "JAYA", "PSO", "SCA", "SSA", "WOA"]
algorithm = [ "DE", "EDA", "GA","BA", "CS", "FA", "FPA",
                "JAYA", "PSO", "SCA"]

opts = ["-O0","-O1","-O2","-O3"]

util_instance = Util()
flags = util_instance.gcc_flags


def calculate_fi():
    # 计算testsuite内元素个数
    # flags = util_instance.gcc_flags  # 假设 util_instance 是一个已定义的实例
    n_flags = len(flags)
    n_testsuite = len(testsuite)  # 假设 testsuite 是一个已定义的列表

    # 创建一个n_flags的列表，用于存放每个算法的调和平均值
    fi_times = [0] * n_flags

    # 创建一个n_testsuite*n_flags的二维列表，用于存放每个算法的运行时间倒数
    reciprocal_times = [[0] *  n_flags for _ in range(n_testsuite)]
    speed_times = [[0] *  n_flags for _ in range(n_testsuite)]

    print("O0")
    # 计算每个算法的运行时间倒数，并存储在倒数列表中
    for i in range(n_flags):
        for j, pro in enumerate(testsuite):  # 使用enumerate获取索引
            flag = flags[i]
            speedup = util_instance.run_procedure2(pro, [flag])
            speedup = -speedup  # 假设这只是加速比的一个变换
            speed_times[j][i] = speedup
            # 如果speedup为零，设置倒数为无穷大，避免除零错误
            if speedup == 0:
                reciprocal_times[j][i] = float('inf')  # 或者设置为一个非常大的数，表示无穷大
            else:
                reciprocal_times[j][i] = 1 / speedup  # 计算倒数加速比

    columns = flags  # 每列是不同的编译选项
    #df = pd.DataFrame(reciprocal_times, index=testsuite, columns=columns)

    # 将 DataFrame 导出到 Excel 文件
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
   # df.to_excel("speedup_results_{current_time}.xlsx", sheet_name="Speedup", index_label="Program")

    df_reci = pd.DataFrame(reciprocal_times, index=testsuite, columns=flags)  # 转置
    df_spe = pd.DataFrame(speed_times, index=testsuite, columns=columns)  # 转置
    
    #df_reci = pd.DataFrame(reciprocal_times, index=testsuite[:1], columns=flags[:2])  # 转置
    #df_spe = pd.DataFrame(speed_times, index=testsuite[:1], columns=columns[:2])  # 转置

    # 将 DataFrame 导出到 Excel 文件
    df_reci.to_excel('reciprocal_{}.xlsx'.format(current_time), sheet_name="Speedup", index_label="Program")
    df_spe.to_excel('speed_times_{}.xlsx'.format(current_time), sheet_name="Speedup", index_label="Program")

    # 计算每个算法的调和平均值
    for i in range(n_flags):
        # 计算倒数的平均值，再取倒数得到调和平均值
        # harmonic_mean = n_testsuite / sum(reciprocal_times[k][i] for k in range(n_testsuite))
         harmonic_mean = n_testsuite / sum(reciprocal_times[k][i] for k in range(n_testsuite))
         fi_times[i] = harmonic_mean  # 直接赋值
    print(fi_times)
    df = pd.DataFrame([fi_times], columns=flags)
    df.to_csv('fi_time_{}.csv'.format(current_time), index=False)
    temp_id = []
    selected_flags = []

    # print("aaaaa")
    for i in range(n_flags):
        if fi_times[i] > 1:
            temp_id.append(i)
            flag_tmp = flags[i]
            # print(flag_tmp)
            selected_flags.append(flag_tmp)
    print("len(selected_flags={})".format(len(selected_flags)))
    print("len(tmp_id={})".format(len(temp_id)))
    return selected_flags

def graph_matrix(selected_flags,testsuite1):
    # 初始化一个n*n的矩阵，n为程序的行数
    # df = pd.read_excel('speed_times_20241113_2121.xlsx')
    # df.set_index("Program", inplace=True)
    print("Calculate matrix")
    n = len(selected_flags)
   
    for pro in testsuite1:
        matrix_graph = [[0 for i in range(n)] for j in range(n)]
        file_name = 'matrix_grap_{}.xlsx'.format(pro)
        print("矩阵预计保存为{}".format(file_name))
        # for flag in flags:
        # for flag in selected_flags:
        for i in range(n):
            for j in range(i+1,n):
                print("i={},j={}",i,j)
                flag_temp = [selected_flags[i],selected_flags[j]]
                swapped_flag = [selected_flags[j],selected_flags[i]]
                runtime_i = util_instance.run_procedure_runtime(pro, [selected_flags[i]])
                runtime_j = util_instance.run_procedure_runtime(pro, [selected_flags[j]])
                runtime1 = util_instance.run_procedure_runtime(pro, flag_temp)
                runtime2 = util_instance.run_procedure_runtime(pro, swapped_flag)
        
                if(runtime1 < min(runtime2,min(runtime_i, runtime_j))):
                    matrix_graph[i][j] += 1
                if(runtime2 < min(runtime2,min(runtime_i, runtime_j))):
                    matrix_graph[j][i] += 1
        df = pd.DataFrame(matrix_graph, columns=selected_flags, index=selected_flags)
        df.to_excel(file_name)
        print("执行完程序{}".format(pro))
        print("矩阵已保存为{}".format(file_name))
    return matrix_graph
def save_mat(selected_flags, matrix_graph):
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    df = pd.DataFrame(matrix_graph, columns=selected_flags, index=selected_flags)
    df.to_excel('matrix_grap_{}.xlsx'.format(current_time))
    print("矩阵已保存为 'matrix_grap_{}.xlsx'".format(current_time))

def select_flags(alg_name,floder_name,n_pop=10,n_gen=30):

    parameter = "\"{}\",{},{}".format(floder_name,n_pop,n_gen)
    func = "{}({})".format(alg_name,parameter)

    result_file = "./result/txt/{}_{}.txt".format(alg_name,floder_name)

    if os.path.exists(result_file):
        print(1)
        return
    init_time = time.time()

    model = eval(func)

    [best_flags, min_time]= model.start()
    cost_time = time.time() - init_time
    curve = model.curve.tolist()
    best_flags = best_flags.tolist()

    content = "algorithm:{}\nbenchmark:{}\ngenerate:{}\npopulation:{}\ncurve:{}\nbest_flags:{}\nmin_time:{}\ncost_time:{}\ntimes:{}\n".format(alg_name,floder_name,n_gen,n_pop,curve,best_flags,min_time,cost_time,times)

    with open(result_file,"w") as f:
        f.write(content)


    data_json ={'algorithm':alg_name,'benchmark':floder_name,'generation':n_gen,'population':n_pop,'curve':curve,'best_flags':best_flags,'min_time':min_time,'cost_time':cost_time}
    result_file = "./result/json/{}_{}.json".format(alg_name,floder_name)
    with open(result_file,"w") as f:
        json.dump(data_json,f)
def select_flags_enhance(alg_name, folder_name, n_pop=10, n_gen=30, result_prefix="result_xxx",known_sequences=known_seqs):
    # 构建参数和模型
    # parameter = "\"{}\",{},{}".format(folder_name, n_pop, n_gen, known_seqs)
    parameter = "\"{}\", {}, {}, known_sequences={}".format(
        folder_name, 
        n_pop, 
        n_gen, 
        known_sequences  # 直接传递列表变量
    )

    # func = "{}({})".format(alg_name, parameter)
    func = "{}({})".format(alg_name, parameter)
    
    # 创建结果目录
    txt_dir = os.path.join(result_prefix, "txt")
    json_dir = os.path.join(result_prefix, "json")
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 构建文件路径
    base_name = f"{alg_name}_{folder_name}"
    txt_path = os.path.join(txt_dir, f"{base_name}.txt")
    json_path = os.path.join(json_dir, f"{base_name}.json")

    # 检查文件是否存在
    if os.path.exists(txt_path) and os.path.exists(json_path):
        return

    # 运行算法
    start_time = time.time()
    model = eval(func)
    [best_flags, best_fitness] = model.start()
    cost_time = time.time() - start_time

    # 准备数据
    content = (
        f"algorithm: {alg_name}\n"
        f"benchmark: {folder_name}\n"
        f"generation: {n_gen}\n"
        f"population: {n_pop}\n"
        f"curve: {model.curve.tolist()}\n"
        f"best_flags: {best_flags}\n"
        f"best_fitness: {best_fitness}\n"
        f"cost_time: {cost_time}\n"
    )

    data_json = {
        "algorithm": alg_name,
        "benchmark": folder_name,
        "generation": n_gen,
        "population": n_pop,
        "curve": model.curve.tolist(),
        "best_flags": best_flags,
        "best_fitness": best_fitness,
        "cost_time": cost_time,
    }

    # 保存结果
    with open(txt_path, "w") as f:
        f.write(content)

    with open(json_path, "w") as f:
        json.dump(data_json, f, indent=2)




if __name__ == "__main__":
    
    # calculate_fi()
    # index  = 0
    # for index in range(0,3):
    #     flags = util_instance.gain_flags_cluster(index)
    #     if(index == 0):
    #         programs = ['automotive_qsort1', 'automotive_susan_e', 'automotive_susan_s', 'bzip2d', 'bzip2e', 'consumer_jpeg_c', 'consumer_jpeg_d', 'consumer_tiff2bw', 'consumer_tiff2rgba', 'consumer_tiffdither', 'consumer_tiffmedian', 'office_rsynth', 'office_stringsearch1', 'telecom_adpcm_d']
    #     elif(index == 1):   
    #         programs = ['automotive_bitcount', 'automotive_susan_c', 'network_dijkstra', 'network_patricia', 'security_blowfish_d', 'security_blowfish_e', 'security_sha', 'telecom_adpcm_c', 'telecom_CRC32']
    #     else:
    #         programs = ['security_rijndael_d', 'security_rijndael_e']
        
        
        # print(flags)
        # exit()
        # for pro in programs[0]:
        #     for flag in flags:
        #         speedup = util_instance.run_procedure2(pro, [flag])

      # ]   
    # phase 4
    known_seqs = [['-scoped-noalias-aa', '-lower-expect', '-sroa', '-prune-eh'], ['-loop-unroll', '-early-cse-memssa', '-mem2reg', '-block-freq'], ['-lazy-branch-prob', '-lcssa', '-loop-unroll', '-early-cse-memssa'], ['-instcombine', '-lazy-branch-prob', '-lcssa', '-loop-unroll'], ['-early-cse-memssa', '-mem2reg', '-block-freq', '-adce'], ['-mem2reg', '-gvn', '-prune-eh', '-loops'], ['-licm', '-gvn', '-prune-eh', '-loops'], ['-assumption-cache-tracker', '-lower-expect', '-sroa', '-prune-eh'], ['-early-cse', '-jump-threading', '-lower-expect', '-sroa'], ['-postdomtree', '-loop-unroll', '-early-cse-memssa', '-mem2reg'], ['-gvn', '-lazy-branch-prob', '-lcssa', '-loop-unroll'], ['-cg-profile', '-lazy-branch-prob', '-lcssa', '-loop-unroll'], ['-profile-summary-info', '-instcombine', '-lazy-branch-prob', '-lcssa'], ['-lower-expect', '-sroa', '-prune-eh', '-loops'], ['-jump-threading', '-lower-expect', '-sroa', '-prune-eh'], ['-sroa', '-postdomtree', '-loop-unroll', '-early-cse-memssa'], ['-lcssa', '-loop-unroll', '-early-cse-memssa', '-pgo-memop-opt'], ['-pgo-memop-opt', '-scoped-noalias-aa', '-lower-expect', '-sroa'], ['-annotation-remarks', '-sroa', '-prune-eh', '-loops'], ['-deadargelim', '-early-cse-memssa', '-mem2reg', '-scoped-noalias-aa'], ['-inline', '-early-cse-memssa', '-mem2reg', '-scoped-noalias-aa'], ['-branch-prob', '-deadargelim', '-early-cse-memssa', '-mem2reg'], ['-div-rem-pairs', '-sroa', '-prune-eh', '-loops']]
    

    # enhan_algo = [ "EnhancedHybridBA", "EnhancedHybridEDA", "EnhancedHybridGA","EnhancedHybridBA", "EnhancedHybridCS", "EnhancedHybridFA", "EnhancedHybridFPA",
                # "EnhancedHybridJAYA", "EnhancedHybridPSO", "EnhancedHybridSCA"]
    # for alg in enhan_algo:
    #     # print(alg)
    # print("111")
    new_bench = ['automotive_qsort1', 'automotive_susan_e', 'automotive_susan_s', 'bzip2d', 'bzip2e', 'consumer_jpeg_c', 'consumer_jpeg_d', 'consumer_tiff2bw', 'consumer_tiff2rgba', 'consumer_tiffdither', 'consumer_tiffmedian', 'office_rsynth', 'office_stringsearch1', 'telecom_adpcm_d','automotive_bitcount', 'automotive_susan_c', 'network_dijkstra', 'network_patricia', 'security_blowfish_d', 'security_blowfish_e', 'security_sha', 'telecom_adpcm_c', 'telecom_CRC32','security_rijndael_d', 'security_rijndael_e']
    
    'consumer_jpeg_d'
    'consumer_tiff2bw'
    'consumer_tiff2rgba'
    'consumer_tiffdither'
    'office_rsynth'
    new_bench_sub = ['consumer_jpeg_d','consumer_tiff2bw','consumer_tiff2rgba','consumer_tiffdither','office_rsynth','automotive_qsort1','network_patricia','office_stringsearch1','security_blowfish_d','automotive_susan_c','automotive_susan_e','automotive_susan_s','bzip2d','security_sha','telecom_CRC32','telecom_adpcm_c']
    
    cluster_1_programs = ['automotive_bitcount', 'automotive_susan_c', 'network_dijkstra', 'network_patricia', 'security_blowfish_d', 'security_blowfish_e', 'security_sha', 'telecom_adpcm_c', 'telecom_CRC32']
   
    result_prefix="result_650_phase4_e10_modifi3"
    # pro = "automotive_susan_c"
    # alg_temp = ["EnhancedHybridJAYA","EnhancedHybridPSO","EnhancedHybridSCA"]
    alg_temp = ["EnhancedHybridBA","EnhancedHybridGA","EnhancedHybridEDA","EnhancedHybridDE",]
    # print("main_begin")
    
    # for alg in alg_temp:
    alg = "EnhancedHybridGA"
    for pro in cluster_1_programs:
        tqdm.write("current" + " " + alg + " " + pro)
        select_flags_enhance(alg,pro,10,30,result_prefix,known_seqs)
    print("done")
    print(alg)
   


  
    