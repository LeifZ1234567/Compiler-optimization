import os
import json
import csv
from collections import defaultdict

def process_json_files(input_dir, output_dir):
    """
    处理JSON文件并生成分类CSV报告
    :param input_dir: JSON文件所在目录
    :param output_dir: 输出目录（自动创建）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据结构：{算法名: [(程序名, 加速比, 耗时)]}
    algorithm_map = defaultdict(list)
    
    # 遍历JSON文件
    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue
        
        # 解析文件名
        try:
            # 使用partition确保只分割第一个下划线
            algorithm, _, program_part = filename.partition('_')
            program = program_part.rsplit('.', 1)[0]  # 移除.json
        except Exception as e:
            print(f"文件名解析失败: {filename} ({str(e)})")
            continue
        
        # 读取文件内容
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据完整性
            required_fields = ['best_fitness', 'cost_time']
            if not all(field in data for field in required_fields):
                raise KeyError("缺少必要字段")
                
            # 处理加速比（取绝对值）
            speedup = round(abs(data['best_fitness']), 2)
            cost_time = round(data['cost_time'], 2)
            
        except Exception as e:
            print(f"文件处理失败: {filename} ({str(e)})")
            continue
        
        # 存储数据
        algorithm_map[algorithm.upper()].append( (program, speedup, cost_time) )
    
    # 生成CSV文件
    for algorithm, records in algorithm_map.items():
        csv_path = os.path.join(output_dir, f"{algorithm}.csv")
        
        # 按程序名排序
        records.sort(key=lambda x: x[0])
        
        # 写入CSV（使用utf-8-sig编码解决Excel乱码）
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["程序名", "加速比", "消耗时间"])
            
            for program, speedup, cost in records:
                writer.writerow([program, speedup, cost])
                
        print(f"已生成: {os.path.basename(csv_path)}")

if __name__ == "__main__":
    # 配置路径（示例）
    json_dir = "./result_650_phase5/json"
    output_dir = os.path.join(os.path.dirname(json_dir), "process_json")
    
    process_json_files(json_dir, output_dir)