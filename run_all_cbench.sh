#!/bin/bash

# 定义程序名称数组
# new_bench=(
#     'automotive_qsort1'
#     'automotive_susan_e'
#     'automotive_susan_s'
#     'bzip2d'
#     'bzip2e'
#     'consumer_jpeg_c'
#     'consumer_jpeg_d'
#     'consumer_tiff2bw'
#     'consumer_tiff2rgba'
#     'consumer_tiffdither'
#     'consumer_tiffmedian'
#     'office_rsynth'
#     'office_stringsearch1'
#     'telecom_adpcm_d'
#     'automotive_bitcount'
#     'automotive_susan_c'
#     'network_dijkstra'
#     'network_patricia'
#     'security_blowfish_d'
#     'security_blowfish_e'
#     'security_sha'
#     'telecom_adpcm_c'
#     'telecom_CRC32'
#     'security_rijndael_d'
#     'security_rijndael_e'
# )

new_bench=(
    'consumer_jpeg_d'
    'consumer_tiff2bw'
    'consumer_tiff2rgba'
    'consumer_tiffdither'
    'office_rsynth'
)

# 设置参数
time_budget=2000


# 遍历数组并执行命令
for program in "${new_bench[@]}"; do
    log_file="boca_$(date +%m%d)_${program}.txt"
    echo "Running $program with time_limit=$time_budget  > $log_file"
   
    python boca.py "$program" --iters 1000 --runs 1 --time_limit "$time_budget"   > "$log_file"

done

