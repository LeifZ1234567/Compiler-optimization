import pandas as pd

df = pd.read_csv("cbench_best_results.csv",encoding="utf-8")
# print(df)
# print(df['Program'])
program_list = df['Program'].tolist()
print(program_list)