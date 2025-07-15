import re
import pandas as pd
from pathlib import Path

def extract_optimization_results(log_path, output_file):
    """
    从日志文件中提取优化结果
    :param log_path: 输入日志文件路径
    :param output_file: 输出Excel文件路径
    """
    result_data = []
    current_program = None
    buffer = []

    # 编译正则表达式模式
    program_pattern = re.compile(r'^[^\s：]+$')  # 匹配不含空格和冒号的程序名
    option_pattern = re.compile(r'最优编译选项：(.+)$')
    speedup_pattern = re.compile(r'最佳加速比：(-?\d+\.\d+)x$')

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测新程序开始
            if program_pattern.match(line) and '：' not in line:
                # 处理前一个程序
                if current_program:
                    process_program_data(current_program, buffer, result_data)
                    buffer = []
                current_program = line
                continue
                
            # 缓存相关数据行
            if current_program:
                buffer.append(line)

        # 处理最后一个程序
        if current_program:
            process_program_data(current_program, buffer, result_data)

    # 保存结果到Excel
    if result_data:
        df = pd.DataFrame(result_data, columns=['程序名', '最优编译选项', '最佳加速比'])
        df.to_excel(output_file, index=False)
        print(f"成功提取 {len(df)} 个程序结果，已保存至：{output_file}")
    else:
        print("未找到有效优化结果")

def process_program_data(program, buffer, result_data):
    """处理单个程序的数据"""
    try:
        # 提取最后两行有效数据
        last_two_lines = [line for line in buffer[-2:] if line]
        
        # 解析最优编译选项
        options_match = re.search(r'最优编译选项：(.+)', last_two_lines[0])
        options = options_match.group(1) if options_match else "N/A"
        
        # 解析加速比（保留原始负值）
        speedup_match = re.search(r'最佳加速比：(-?\d+\.\d+)x', last_two_lines[1])
        speedup = float(speedup_match.group(1)) if speedup_match else 0.0
        
        result_data.append((program, options, speedup))
    except Exception as e:
        print(f"解析程序 {program} 时出错: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_log = Path("./optimization.log")
    output_xlsx = Path("./optimization_results_txt.xlsx")
    
    if not input_log.exists():
        print(f"错误：日志文件 {input_log} 不存在")
    else:
        extract_optimization_results(input_log, output_xlsx)