import os
# path = os.getcwd()
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
import seaborn as sns

Green = "#8ECFC9"
Orange = "#FFBE7A"
Red = "#FA7F6F"
Blue = "#82B0D2"
Purpre = "#BEB8DC"
Grey = "#999999"
gcc_flags = []
algorithm = ["BA", "CS", "DE", "EDA", "FA", "FPA", "GA",
                "JAYA", "PSO", "SCA" ]

# algorithm = ["BA", "CS", "DE", "EDA", "FA", "FPA", "GA",
#                 "GWO", "HHO", "JAYA", "PSO", "SCA", "SSA", "WOA"]


process = [
#     "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_rijndael_e", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian", 
# ]

process = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1",  "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian", 
]

full_name = {
    "BA":"Bat-inspired Algorithm",
    "CS":"Cuckoo Search",
    "DE":"Differential Evolution", 
    "EDA":"Estimation Distribution Algorithm",
    "FA":"Firefly Algorithm", 
    "FPA":"Flower Pollination Algorithm",
    "GA":"Genetic Algorithm",
    "GWO":"Grey Wolf Optimizer",
    "HHO":"Harris Hawks Optimization",
    "JAYA":"Jaya Algorithm",
    "PSO":"Particle Swarm Optimization",
    "SCA":"Sine Cosine Algorithm",
    "SSA":"Salp Swarm Algorithm", 
    "WOA":"Whale Optimization Algorithm"
}

TXT_TYPE = ".txt"

PATH = "../result/"
IMAGE_PATH = "../result/plot/"
TXT_PATH = "../result/txt/"



def gain_gcc_flags():

    command = "sh ../data/diff.sh"

    # res = os.system(command)
    res = os.popen(command)
    flags = res.read().split()
    return flags

def generate_curve(alg_name,bench_name,curve,O1_LINE,O2_LINE,O3_LINE):

    # plt.title(full_name[alg_name] + " x " + bench_name ,fontsize = 12)
    # plt.ylim((4.0,4.6))
    plt.ylabel("Time/s",fontsize = 12)
    plt.xlabel("Generation",fontsize = 12)

    ax = plt.gca()
    ax.xaxis.grid(True, which='major',linestyle = (0,(8,4))) 
    ax.yaxis.grid(True, which='major',linestyle = (0,(8,4))) 
    ax.plot(curve,color = 'cornflowerblue',alpha = 0.7, linewidth=3,label=alg_name)
    # ax.axhline(y=13.538, c="r", ls="-.", lw=2, label='-O0:13.538')
    ax.axhline(y=O1_LINE, c="steelblue", ls=":", lw=3, label='-O1:{:.3f}s'.format(O1_LINE))
    ax.axhline(y=O2_LINE, c="steelblue", ls="--", lw=3, label='-O2:{:.3f}s'.format(O2_LINE))
    ax.axhline(y=O3_LINE, c="steelblue", ls="-.", lw=3, label='-O3:{:.3f}s'.format(O3_LINE))
    plt.xticks([0,4,9,14,19,24,29],[1,5,10,15,20,25,30])
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_PATH+"{}_{}_curve.jpg".format(alg_name,bench_name))
    plt.savefig(IMAGE_PATH+"{}_{}_curve.eps".format(alg_name,bench_name), dpi=300)
    plt.clf()

def get_com_and_fre(flags,total=10*25/100):
    combination = np.zeros((51,51))
    frequency = np.zeros(51)
    for i in range(len(flags)):
        for j in range(51):
            frequency[j] += flags[i][j]
            # print(frequency)
            # input()
            for k in range(51):
                
                if j > k:
                    combination[j][k] += (flags[i][j] == flags[i][k])
                # combination[j][k] += (flags[i][j] == flags[i][k])    
    # print(frequency)
     
    # frequency /= total
    return combination,frequency

def generate_heapmap(flags):
    combination,frequency = get_com_and_fre(flags)
    # print(frequency.tolist())
    index= np.argsort(-frequency)
    
    fre = np.sort(frequency)[::-1]
    # print(index)
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(7)
    ax.grid(axis='y',color=Grey,linestyle ="--")

    fre_temp = np.zeros(len(gcc_flags))
    for idx, alg in enumerate(algorithm):
        _,frequency = get_com_and_fre(flags[idx*25:(idx+1)*25])

        frequency = [frequency[i] for i in index]

        ax.bar([i for i in range(51)],frequency,bottom=fre_temp,label=alg)
          
        fre_temp += np.array(frequency)
          
    plt.legend(ncol=5 )
    # ax.bar([i for i in range(51)],fre,color = Blue)
    
    # sort_index = np.argsort(frequency)
    # for i in range(5):
    #     print(gcc_flags[sort_index[i]],frequency[sort_index[i]],gcc_flags[sort_index[50-i]],frequency[sort_index[50-i]])
    # ax.axhline(y=np.mean(frequency), c=Red, ls="--", lw=3)
    # ax.set_xticklabels(gcc_flags)


    # plt.xticks([i for i in range(51)],[i+1 for i in index])
    plt.xticks([i for i in range(51)],[gcc_flags[i] + "({})".format(i+1) for i in index])


    # plt.setp(ax.get_xticklabels(), rotation=35, ha="right",
        #  rotation_mode="anchor")
    # plt.title("Frequency of Compilation Flags",fontsize=18)
    
    plt.xlabel("Compiler flag (No.)",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
         rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(PATH + "flag_frequency.jpg")
    plt.savefig(PATH + "flag_frequency.eps", dpi=300)
    plt.cla()

    combination /= len(flags)

    

    # print(np.argmax(combination))
    # for row in combination:
    #     for i in row:
    #         print(i,end=" ")
    #     print()
    
    # input()
    # return 
    mask = np.zeros_like(combination, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()

    ax = sns.heatmap(combination, cmap="Blues",ax=ax ,vmin=0.20, vmax=0.75,mask=mask, linewidths=.5)  # 修改颜色，添加线宽
    ax.set_xlabel('Number of compile flag')  # x轴标题
    ax.set_ylabel('Number of compile flag')
    # plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
    #      rotation_mode="anchor")
    # plt.title("Correlation of Compiling Flags")
    fig.tight_layout()  #自动调整子图参数,使之填充整个图像区域。
    figure = ax.get_figure()
    plt.savefig(PATH+'heatmap.jpg')  # 保存图片
    plt.savefig(PATH + "heatmap.eps", dpi=300)
    plt.tight_layout()
    ax.cla()
    plt.cla()
    


def gain_baselines():
    baselines = {}
    with open(PATH + "baseline.txt", "r") as f:
        while True:
            line =  f.readline().rstrip().split()
            if not line:
                break
            baselines[line[0]] = list(map(float,line[1:]))
    return baselines


def gain_curve(string):
    string = string.strip('[').strip(']')
    curve = string.split(',')
    curve = list(map(float,curve))
    return curve

def gain_flag(string):
    string = string.strip('[').strip(']')
    flag = string.split(',')
    flag = list(map(int,flag))
    return flag

def gain_para(string):
    dic = string.split(':')
    para = dic[-1]
    return para

if __name__ == "__main__":
    gcc_flags = gain_gcc_flags()   
    baselines = gain_baselines()
    files = [x for x in os.listdir(TXT_PATH) if TXT_TYPE in x]
    # print(files)
    best_flags = []
    result_speed = []
    result_time = []
    bench_flags = []
    result_curve = []
    for alg in algorithm:
        temp_speed = []
        temp_time = []
        for file in process:
            file_path = TXT_PATH + "{}_{}.txt".format(alg,file)
            with open(file_path,"r") as f:
                data = f.read()

                content = data.split("\n")[:-1]

                parameters = list(map(gain_para,content))
                # print(parameters)

                alg_name = parameters[0]
                bench_name = parameters[1]
                gen = parameters[2]
                pop = parameters[3]

                # print(name,bench,gen ,pop )
                curve =gain_curve(parameters[4])
                result_curve.append( np.argmin(curve))
                flag = gain_flag(parameters[5])
                min_runtime = float(parameters[6])
                total_runtime = float(parameters[7])


                

                # print(curve)
                # print(len(flag))
                # print(min_runtime,total_runtime)
                if bench_name == process[20]:
                    bench_flags.append(flag)

                if bench_name not in baselines.keys():
                    continue
                temp_speed.append((baselines[bench_name][3]/min_runtime-1)+1)
                temp_time.append(total_runtime)
                best_flags.append(flag)
                o1,o2,o3 = [baselines[bench_name][i] for i in range(1,4)]
                # generate_curve(alg_name,bench_name,curve,o1,o2,o3)

        result_speed.append(temp_speed)  
        result_time.append(temp_time) 


    generate_heapmap(best_flags)
    exit()


    result_curve = np.array(result_curve)
    iter_result = [sum(result_curve==i) for i in range(30)]
    # print(iter_result)

    best_flags = np.array(best_flags)
    # print(bench_flags)
    com,fre = get_com_and_fre(bench_flags,10/100)

    plt.bar([i for i in range(51)],fre, color=Blue,label="Speedup")
    plt.xticks([i for i in range(0,51,2)],[i for i in range(1,52,2)],size=7)
    plt.ylabel("Frequency")
    plt.xlabel("Number of compiler flag")
    plt.grid(axis='y',color=Grey,linestyle ="--")
    plt.tight_layout()
    plt.savefig(PATH + "fre_of_sha.jpg")
    plt.savefig(PATH + "fre_of_sha.eps", dpi=300)

    plt.cla()
    # print(fre)


    # print(result)


    # for row in result_time:
    #     for i in row:
    #         print(i,end=" ")
    #     print()
    data = np.array(result_speed)
    best = np.argmax(data,axis=0)
    mean = np.mean(data)

    mean_col = np.average(data, axis=0)
    mean_row = np.average(data, axis=1)

    plt.cla()
    fig ,ax= plt.subplots()
    ax.bar([i for i in range(len(process))], mean_col-1,color=Blue,label="Speedup")

    plt.xticks([i for i in range(len(process))],process,fontsize=7)
    plt.yticks([i*0.1 for i in range(6)],[i*0.1+1 for i in range(6)],fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
         rotation_mode="anchor")
    ax.axhline(y=mean-1, c=Red, ls="--", lw=3,label="Average")
    ax.grid(axis='y',color=Grey,linestyle ="--")
    ax.axhline(y=0, c="black", ls="-", lw=3,label="Baseline(-O3)")

    # plt.ylim((0.95,1.6))
    # plt.title("Mean of Speed (divided by program)")
    plt.ylabel("Speedup")
    plt.xlabel("Process")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH + "mean_pro.jpg")
    plt.savefig(PATH + "mean_pro.eps", dpi=300)
    plt.cla()
    
    fig ,ax= plt.subplots()

    plt.grid(axis='y',color=Grey,linestyle ="--")
    plt.bar([i for i in range(1,len(algorithm)+1)], mean_row,color=Blue,label="Speedup")

    plt.axhline(y=mean, c=Red, ls="--", lw=3,label="Average")
    plt.ylim((1.00,1.13))
    plt.ylabel("Speedup")
    ax.axhline(y=1, c="black", ls="-", lw=2,label="Baseline(-O3)")
    plt.xlabel("Algorithm")
    # plt.title("Mean of Speed (divided by algorithm)")
    ta = plt.gca()
    plt.xticks([i+1 for i in range(len(algorithm))])
    plt.legend()
    ta.set_xticklabels(algorithm)
    plt.yticks([1,1.02,1.04,1.06,1.08,int(mean*100)/100,1.10,1.12])
    
    plt.tight_layout()
    plt.savefig(PATH + "mean_alg.jpg")
    plt.savefig(PATH + "mean_alg.eps", dpi=300)

    # for i in mean_row:
    #     print(i)

    # for i in mean_col:
    #     print(i)

    # generate_heapmap(flags)