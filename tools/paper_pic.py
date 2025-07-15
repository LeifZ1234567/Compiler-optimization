import json
import seaborn as sns
import os
# path = os.getcwd()
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
Green = "#8ECFC9"
Orange = "#FFBE7A"
Red = "#FA7F6F"
Blue = "#82B0D2"
Purpre = "#BEB8DC"
Grey = "#999999"
gcc_flags = []
algorithm = ["BA", "CS", "DE", "EDA", "FA", "FPA", "GA",
             "JAYA", "PSO", "SCA"]
# algorithm = [ "DE", "EDA", "GA","BA", "CS", "FA", "FPA",
#   "JAYA", "PSO", "SCA"]
process = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_rijndael_e", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian",
]
process = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian",
]
full_name = {
    "BA": "Bat-inspired Algorithm",
    "CS": "Cuckoo Search",
    "DE": "Differential Evolution",
    "EDA": "Estimation Distribution Algorithm",+
    "FA": "Firefly Algorithm",
    "FPA": "Flower Pollination Algorithm",
    "GA": "Genetic Algorithm",
    "GWO": "Grey Wolf Optimizer",
    "HHO": "Harris Hawks Optimization",
    "JAYA": "Jaya Algorithm",
    "PSO": "Particle Swarm Optimization",
    "SCA": "Sine Cosine Algorithm",
    "SSA": "Salp Swarm Algorithm",
    "WOA": "Whale Optimization Algorithm"
}

PATH = "../result/"
JSON_PATH = "../result/json/"
IMAGE_PATH = "../result/plot/"
JSON_TYPE = ".json"

files = [x for x in os.listdir(JSON_PATH) if JSON_TYPE in x]


def gain_flags():

    flags = ['-tti', '-tbaa', '-scoped-noalias-aa', '-assumption-cache-tracker', '-targetlibinfo', '-verify', '-lower-expect', '-simplifycfg', '-domtree', '-sroa', '-early-cse', '-profile-summary-info', '-annotation2metadata', '-forceattrs', '-inferattrs', '-callsite-splitting', '-ipsccp', '-called-value-propagation', '-globalopt', '-mem2reg', '-deadargelim', '-basic-aa', '-aa', '-loops', '-lazy-branch-prob', '-lazy-block-freq', '-opt-remark-emitter', '-instcombine', '-basiccg', '-globals-aa', '-prune-eh', '-inline', '-openmp-opt-cgscc', '-function-attrs', '-argpromotion', '-memoryssa', '-early-cse-memssa', '-speculative-execution', '-lazy-value-info', '-jump-threading', '-correlated-propagation', '-aggressive-instcombine', '-libcalls-shrinkwrap', '-postdomtree', '-branch-prob', '-block-freq', '-pgo-memop-opt', '-tailcallelim', '-reassociate', '-loop-simplify', '-lcssa-verification', '-lcssa', '-scalar-evolution', '-licm', '-loop-rotate', '-loop-unswitch', '-loop-idiom', '-indvars', '-loop-deletion', '-loop-unroll', '-mldst-motion', '-phi-values', '-memdep', '-gvn', '-sccp', '-demanded-bits', '-bdce', '-adce', '-memcpyopt', '-dse', '-barrier', '-elim-avail-extern', '-rpo-function-attrs', '-globaldce', '-float2int', '-lower-constant-intrinsics', '-loop-accesses', '-loop-distribute', '-inject-tli-mappings', '-loop-vectorize', '-loop-load-elim', '-slp-vectorizer', '-vector-combine', '-transform-warning', '-alignment-from-assumptions', '-strip-dead-prototypes', '-constmerge', '-cg-profile', '-loop-sink', '-instsimplify', '-div-rem-pairs', '-annotation-remarks']

    return flags


def gain_baselines():
    baselines = {}
    with open(PATH + "baseline.txt", "r") as f:
        while True:
            line = f.readline().rstrip().split()
            if not line:
                break
            baselines[line[0]] = list(map(float, line[1:]))
    return baselines


LLVM_passes = gain_flags()
baselines = gain_baselines()


def hist_pro(mean_col):
    mean = np.mean(mean_col)
    fig, ax = plt.subplots()
    ax.bar([i for i in range(len(process))],
           mean_col-1, color=Blue, label="Speedup")

    plt.xticks([i for i in range(len(process))], process, fontsize=7)
    plt.yticks([i*0.1 for i in range(9)], [1.0, 1.1, 1.2,
               1.3, 1.4, 1.5, 1.6, 1.7, 1.8], fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax.axhline(y=mean-1, c=Red, ls="--", lw=3, label="Average")
    ax.grid(axis='y', color=Grey, linestyle="--")
    ax.axhline(y=0, c="black", ls="-", lw=3, label="Baseline(-O3)")

    # plt.ylim((0.95,1.6))
    # plt.title("Mean of Speed (divided by program)")
    plt.ylabel("Speedup")
    plt.xlabel("Process")
    plt.legend(framealpha=1)
    plt.tight_layout()
    plt.savefig(PATH + "mean_pro.jpg")
    plt.savefig(PATH + "mean_pro.eps", dpi=300)
    plt.cla()


def hist_alg(mean_row):
    mean = np.mean(mean_row)
    fig, ax = plt.subplots()

    plt.grid(axis='y', color=Grey, linestyle="--")
    plt.bar([i for i in range(1, len(algorithm)+1)],
            mean_row, color=Blue, label="Speedup")

    plt.axhline(y=mean, c=Red, ls="--", lw=3, label="Average")
    plt.ylim((1.00, 1.30))
    plt.ylabel("Speedup")
    ax.axhline(y=1, c="black", ls="-", lw=2, label="Baseline(-O3)")
    plt.xlabel("Algorithm")
    # plt.title("Mean of Speed (divided by algorithm)")
    ta = plt.gca()
    plt.xticks([i+1 for i in range(len(algorithm))])
    plt.legend(framealpha=1)
    ta.set_xticklabels(algorithm)
    # plt.yticks([1,1.05,1.10,1.15,1.20,1.25,1.30])

    plt.tight_layout()
    plt.savefig(PATH + "mean_alg.jpg")
    plt.savefig(PATH + "mean_alg.eps", dpi=300)


def generate_curve(alg_name, bench_name, min_time, curve):
    # print(baselines)

    O1_LINE, O2_LINE, O3_LINE = [baselines[bench_name][i] for i in range(3)]
    temp = curve.copy()
    for i, t in enumerate(temp):
        curve[i] = O3_LINE / -t
    # plt.title(full_name[alg_name] + " x " + bench_name ,fontsize = 12)
    # plt.ylim((4.0,4.6))
    plt.ylabel("Time/s", fontsize=12)
    plt.xlabel("Generation", fontsize=12)

    ax = plt.gca()
    ax.xaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.yaxis.grid(True, which='major', linestyle=(0, (8, 4)))
    ax.plot(curve, color='cornflowerblue',
            alpha=0.7, linewidth=3, label=alg_name,marker='s')
    # ax.scatter(np.arange(30),curve, ,marker='o',edgecolors="#FFFFFF")
    # ax.axhline(y=13.538, c="r", ls="-.", lw=2, label='-O0:13.538')
    # ax.axhline(y=O1_LINE, c="steelblue", ls=":", lw=3, label='-O1:{:.3f}s'.format(O1_LINE))
    # ax.axhline(y=O2_LINE, c="steelblue", ls="--", lw=3, label='-O2:{:.3f}s'.format(O2_LINE))
    ax.axhline(y=O3_LINE, c="steelblue", ls="-.", lw=3,
               label='-O3:{:.3f}s'.format(O3_LINE))
    plt.xticks([0, 4, 9, 14, 19, 24, 29], [1, 5, 10, 15, 20, 25, 30])
    
    plt.tight_layout()

    plt.legend(framealpha=1)
    plt.savefig(
        IMAGE_PATH+"{}_{}_curve.eps".format(alg_name, bench_name), dpi=300)
    plt.savefig(IMAGE_PATH+"{}_{}_curve.jpg".format(alg_name, bench_name))
    plt.clf()


def get_com_and_fre(flags):
    # print(flags)
    total = len(algorithm) * len(process) / 100
    len_flags = len(LLVM_passes)
    combination = np.zeros((len_flags, len_flags))
    frequency = np.zeros(len_flags)
    for flag in flags:
        for i in range(len(flag)):  # 程序数量
            for j in range(len_flags):

                frequency[j] += flag[i][j]
                # print(frequency)
                for k in range(len_flags):
                    if j > k:
                        combination[j][k] += (flag[i][j] == flag[i][k])

    # print(len(frequency))

    # frequency /= total
    return combination, frequency


def generate_heapmap(flags):
    combination, frequency = get_com_and_fre(flags)
    print(max(frequency),min(frequency),np.mean(frequency))
    # res = 0
    # for com in combination:
    #     for c in com:
    #         if c:
    #             res +=c
    #     print(res)
    # print(np.max(combination),np.min(combination),np.mean(combination))
    # exit()
    # print(frequency.tolist())
    index = np.argsort(-frequency)
    print(frequency)
    print(index)

    # fre = np.sort(frequency)[::-1]
    # print(index)
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(7)
    ax.grid(axis='y', color=Grey, linestyle="--")

    fre_temp = np.zeros(len(LLVM_passes))
    for idx, alg in enumerate(algorithm):
        # print(len(flags))
        _, frequencys = get_com_and_fre([flags[idx]])

        frequency = [frequencys[i] for i in index]

        # print(len(frequency),len(LLVM_passes))

        ax.bar([i for i in range(len(LLVM_passes))],
               frequency, bottom=fre_temp, label=alg)

        fre_temp += np.array(frequency)

    plt.legend(ncol=5,framealpha=1)

    # plt.xticks([i for i in range(51)],[i+1 for i in index])
    plt.xticks([i for i in range(len(LLVM_passes))], [
               LLVM_passes[i] + "({})".format(i+1) for i in index],fontsize=6)

    # plt.setp(ax.get_xticklabels(), rotation=35, ha="right",
    #  rotation_mode="anchor")
    # plt.title("Frequency of Compilation Flags",fontsize=18)

    plt.xlabel("Compiler flag (No.)", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(PATH + "flag_frequency.jpg")
    plt.savefig(PATH + "flag_frequency.eps", dpi=300)
    plt.cla()
    
    for a,com in enumerate(combination):
        for b,c in enumerate(com):
            if c==163:
                ffflags = gain_flags()
                print(a,ffflags[a],b,ffflags[b])
            # print(c,end= " " )
        # print()
    # print(combination)
    combination /= len(process) * len(algorithm)

    # print(np.argmax(combination))
    # for row in combination:
    #     for i in row:
    #         print(i,end=" ")
    #     print()

    # input()
    # return
    mask = np.zeros_like(combination, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()

    ax = sns.heatmap(combination, cmap="Blues", ax=ax, vmin=0.20,
                     vmax=0.75, mask=mask, linewidths=.5)  # 修改颜色，添加线宽
    ax.set_xlabel('Number of compile flag')  # x轴标题
    ax.set_ylabel('Number of compile flag')
    # plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
    #      rotation_mode="anchor")
    # plt.title("Correlation of Compiling Flags")
    fig.tight_layout()  # 自动调整子图参数,使之填充整个图像区域。
    figure = ax.get_figure()
    plt.savefig(PATH+'heatmap.jpg')  # 保存图片
    plt.savefig(PATH + "heatmap.eps", dpi=300)
    plt.tight_layout()
    ax.cla()
    plt.cla()


best_flags = []
result_speed = []
result_time = []
result_flags = []
bench_flags = []
result_curve = []
# for file in process:
#     print(file,end=" ")
# print()
for alg in algorithm:
    temp_speed = []
    temp_flag = []
    temp_cost = 0
    # print(alg,end=" ")
    for file in process:
        file_path = JSON_PATH + "{}_{}.json".format(alg, file)
        # print(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)

        alg_name = data["algorithm"]
        bench_name = data["benchmark"]
        gen = data["generation"]
        pop = data["population"]

        # print(name,bench,gen ,pop )
        curve = data["curve"]
        temp_speed.append(-min(curve))
        flag = data["best_flags"]

        temp_flag.append(flag)
        min_runtime = -data["min_time"]
        total_runtime = data["cost_time"]
        temp_cost += total_runtime
        # generate_curve(alg_name,bench_name,min_runtime,curve)
    # input()
    result_speed.append(temp_speed)  # 记录所有加速比结果
    result_flags.append(temp_flag)
    # print(alg,temp_cost/23)

# data = np.array(result_speed)
# # print(data)
# mean_col = np.average(data, axis=0)
# mean_row = np.average(data, axis=1)
# # print(mean_col)
# hist_pro(mean_col)
# hist_alg(mean_row)

generate_heapmap(result_flags)
