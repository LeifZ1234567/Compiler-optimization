import re
import pandas as pd

def parse_out_file(file_path):
    program_data = {}
    current_program = None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        # 匹配程序名，例如 Programautomotive_susan_c Start → automotive_susan_c
        program_match = re.match(r'^Program(.+?) Start$', line)
        if program_match:
            current_program = program_match.group(1)
            if current_program not in program_data:
                program_data[current_program] = []
            index += 1
            continue
        
        if current_program is not None:
            # 解析优化选项行
            if line.startswith('opy_level = '):
                optimizations = line[len('opy_level = '):].split()
                # 检查下一行是否为 Speedup
                if index + 1 < len(lines):
                    next_line = lines[index + 1].strip()
                    speedup_match = re.search(r'Speedup=([0-9.]+)', next_line)
                    if speedup_match:
                        speedup = float(speedup_match.group(1))
                        program_data[current_program].append({
                            '优化选项': ' '.join(optimizations),
                            '加速比': speedup
                        })
                        index += 2  # 跳过已处理的行
                        continue
            index += 1
        else:
            index += 1

    return program_data

def save_to_excel(data, output_file):
    # 转换为 DataFrame
    rows = []
    for program, entries in data.items():
        for entry in entries:
            rows.append({
                '程序名': program,
                '优化选项': entry['优化选项'],
                '加速比': entry['加速比']
            })
    df = pd.DataFrame(rows)
    # 写入 Excel
    df.to_excel(output_file, index=False)
    print(f"数据已保存到 {output_file}")

# 示例用法
file_path = 'nohup_hybridba20250316.out'  # 替换为你的 .out 文件路径
output_excel = 'optimization_results.xlsx'

program_data = parse_out_file(file_path)
save_to_excel(program_data, output_excel)