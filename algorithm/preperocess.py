# import re
# import os
# from collections import defaultdict

# CBENCH = [
#     "automotive_susan_c", "automotive_susan_e", "automotive_susan_s",
#     "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c",
#     "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e",
#     "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c",
#     "consumer_jpeg_d", "network_patricia", "automotive_qsort1",
#     "security_rijndael_d", "security_sha", "office_stringsearch1",
#     "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither",
#     "consumer_tiffmedian"
# ]

# def parse_optimization_file(input_file, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 构建匹配模式
#     program_pattern = re.compile(r'^(' + '|'.join(re.escape(p) for p in CBENCH) + r')$')
#     opy_pattern = re.compile(r'^\s*opy_level\s*=\s*(.+)$')
#     speedup_pattern = re.compile(r'^\s*Speedup\s*=\s*([\d.]+)$')

#     current_program = None
#     pending_options = None  # 用于跟踪opy_level后的选项
#     program_data = defaultdict(list)

#     with open(input_file, 'r') as f:
#         for line in f:
#             line = line.strip()
            
#             # 匹配程序名
#             if program_pattern.match(line):
#                 current_program = line
#                 pending_options = None  # 新程序开始时重置状态
#                 continue
                
#             # 处理opy_level行
#             if current_program and pending_options is None:
#                 opy_match = opy_pattern.match(line)
#                 if opy_match:
#                     pending_options = opy_match.group(1).strip()
#                     continue  # 等待下一行的Speedup
                    
#             # 处理Speedup行
#             if current_program and pending_options is not None:
#                 speedup_match = speedup_pattern.match(line)
#                 if speedup_match:
#                     speedup = float(speedup_match.group(1))
#                     program_data[current_program].append( (speedup, pending_options) )
#                     pending_options = None  # 重置等待状态

#     # 生成结果文件
#     for program, data in program_data.items():
#         sorted_data = sorted(data, key=lambda x: -x[0])[:5]  # 按加速比降序取前5
        
#         output_file = os.path.join(output_dir, f"{program}_top5.csv")
#         with open(output_file, 'w') as f:
#             f.write("Rank,Options,Speedup\n")
#             for rank, (speedup, options) in enumerate(sorted_data, 1):
#                 f.write(f"{rank},{options},{speedup:.4f}\n")
#         print(f"Generated: {output_file}")

# # 使用示例
# if __name__ == "__main__":
#     parse_optimization_file(
#         input_file="optimization.txt",
#         output_dir="cbench_results"
#     )

import re
import os
import csv
from collections import defaultdict

CBENCH = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s",
    "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c",
    "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e",
    "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c",
    "consumer_jpeg_d", "network_patricia", "automotive_qsort1",
    "security_rijndael_d", "security_sha", "office_stringsearch1",
    "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither",
    "consumer_tiffmedian"
]

def parse_optimization_file(input_file, output_file):
    # 构建匹配模式
    program_pattern = re.compile(r'^(' + '|'.join(re.escape(p) for p in CBENCH) + r')$')
    opy_pattern = re.compile(r'^\s*opy_level\s*=\s*(.+)$')
    speedup_pattern = re.compile(r'^\s*Speedup\s*=\s*([\d.]+)$')

    current_program = None
    pending_options = None  # 用于跟踪opy_level后的选项
    program_data = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 匹配程序名
            if program_pattern.match(line):
                current_program = line
                pending_options = None  # 新程序开始时重置状态
                continue
                
            # 处理opy_level行
            if current_program and pending_options is None:
                opy_match = opy_pattern.match(line)
                if opy_match:
                    pending_options = opy_match.group(1).strip()
                    continue  # 等待下一行的Speedup
                    
            # 处理Speedup行
            if current_program and pending_options is not None:
                speedup_match = speedup_pattern.match(line)
                if speedup_match:
                    speedup = float(speedup_match.group(1))
                    program_data[current_program].append((speedup, pending_options))
                    pending_options = None  # 重置等待状态

    # 保存所有程序的最优加速比和编译选项到一个 CSV 文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Program", "Best Speedup", "Best Optimization"])

        for program, data in program_data.items():
            # 获取最优加速比和对应的优化选项
            best_speedup, best_optimization = max(data, key=lambda x: x[0])
            writer.writerow([program, f"{best_speedup:.4f}", best_optimization])

    print(f"Generated: {output_file}")

# 使用示例
if __name__ == "__main__":
    parse_optimization_file(
        input_file="optimization.txt",  # 输入日志文件
        output_file="cbench_best_results.csv"  # 输出文件
    )
