# import os
# import json

# # 指定文件夹路径
# folder_path = '/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/incontext_reflection'

# def count_lines_in_jsonl_folder(folder_path):
#     """统计文件夹中所有 .jsonl 文件的总行数"""
#     total_lines = 0
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.jsonl'):
#             file_path = os.path.join(folder_path, file_name)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 for _ in file:
#                     print(_)
#                 line_count = sum(1 for _ in file)
#                 total_lines += line_count
#     return total_lines

# # # 统计总行数
# # total_lines = count_lines_in_jsonl_folder(folder_path)
# # print(total_lines)


# import os

# def merge_jsonl_files(input_folder, output_file):
#     """将文件夹中的所有 .jsonl 文件合并为一个 .jsonl 文件"""
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         # 遍历文件夹中的所有文件
#         for file_name in os.listdir(input_folder):
#             # 只处理 .jsonl 文件
#             if file_name.endswith('.jsonl'):
#                 file_path = os.path.join(input_folder, file_name)
#                 with open(file_path, 'r', encoding='utf-8') as infile:
#                     # 逐行读取并写入输出文件
#                     for line in infile:
#                         outfile.write(line)
    
#     print(f"合并完成，输出文件: {output_file}")

# # 使用示例
# input_folder = '/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/incontext_reflection'
# output_file = '/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/merged_output.jsonl'
# merge_jsonl_files(input_folder, output_file)

import os
import json

def load_persona_session_pairs(file_path):
    """读取 .jsonl 文件中的 persona_id 和 session_id 组合"""
    pairs = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            persona_id = data.get("persona_id")
            session_id = data.get("session_id")
            if persona_id is not None and session_id is not None:
                pairs.add((persona_id, session_id))
    return pairs

def find_non_existing_pairs(input_file, output_file, result_file):
    """查找 input_file 中不存在于 output_file 的 persona_id 和 session_id 组合"""
    # 加载 output_file 中的所有组合
    output_pairs = load_persona_session_pairs(output_file)
    
    # 打开结果文件
    with open(result_file, 'w', encoding='utf-8') as result:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                persona_id = data.get("persona_id")
                session_id = data.get("session_id")
                
                # 检查组合是否在 output_pairs 中
                if (persona_id, session_id) not in output_pairs:
                    # 如果不存在，则写入结果文件
                    result.write(line)
    
    print(f"查找完成，结果已保存到 {result_file}")

# 使用示例
input_file = '/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/retrieval/bert_3.jsonl'
output_file = '/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/merged_output.jsonl'
result_file = '/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/non_existing_pairs.jsonl'

find_non_existing_pairs(input_file, output_file, result_file)
