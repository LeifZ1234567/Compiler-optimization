import json
cbench = [
    "automotive_susan_c", "automotive_susan_e", "automotive_susan_s", "automotive_bitcount", "bzip2d", "office_rsynth", "telecom_adpcm_c", "telecom_adpcm_d", "security_blowfish_d", "security_blowfish_e", "bzip2e", "telecom_CRC32", "network_dijkstra", "consumer_jpeg_c", "consumer_jpeg_d", "network_patricia", "automotive_qsort1", "security_rijndael_d", "security_rijndael_e", "security_sha", "office_stringsearch1", "consumer_tiff2bw", "consumer_tiff2rgba", "consumer_tiffdither", "consumer_tiffmedian", 
]
polybench = ["correlation","covariance","2mm","3mm","atax","bicg","doitgen","mvt","gemm","gemver","gesummv","symm","syr2k","syrk","trmm","cholesky","durbin","gramschmidt","lu","ludcmp","trisolv","deriche","floyd-warshall","adi","fdtd-2d","heat-3d","jacobi-1d","jacobi-2d","seidel-2d"]
testsuite = cbench + polybench

for file in testsuite:
    with open("../result/json/DE_{}.json".format(file)) as f:
        data = f.read()
        res = json.loads(data)
    

        print("{} {} {}".format(res['benchmark'],res['baseline'],res['min_time']))