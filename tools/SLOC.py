import os


process = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_rijndael_e", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian", 
]

for file_floder in process:

    path = "../cBench_V1.1/" + file_floder + "/src"
    command = "cd {} && find . -name \"*.c\" -or -name \"*.h\"  | xargs wc -l".format(path)
    res = os.popen(command).read().split("\n")[-2].rstrip(" total").lstrip(" ")
    print(file_floder,res)