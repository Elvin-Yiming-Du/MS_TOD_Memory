import json
import os
import json
import random
import pandas as pd
from collections import defaultdict


def judge_positive(persona_data, session_id):
    session_context = ""
    for session in persona_data["sessions"]:
        if str(session["session_id"]) == session_id:
            if session["exist_confirmation"]:
                for utterance in session["content"]:
                    if utterance["is_confirmation"]:
                        break
                    session_context = session_context + utterance["speaker"] + ":" + utterance["utterance"] + "\n"
                return True, session_context
            else:
                return False, session_context

    return False, session_context


def load_and_generate_pairs(file_path):
    """
    从 JSONL 文件中加载数据，并根据 persona_id 生成训练模型所需的输入数据。
    正样例 (pos_or_neg 为 True) 作为 label 和 context，
    负样例 (pos_or_neg 为 False) 作为 candidate_intents。
    为每个正样例随机匹配 2 到 3 个相同 persona_id 的负样例。
    
    返回:
    - data_pairs: 包含 context, candidate_intents, label 的字典列表
    """
    # 用于存储不同 persona_id 的正负样例
    persona_samples = defaultdict(lambda: {"positive": [], "negative": []})
    
    # 读取 JSONL 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            persona_id = entry['persona_id']
            
            if entry['pos_or_neg']:
                # 正样例
                persona_samples[persona_id]["positive"].append({
                    "context": entry['session'],
                    "label": entry['intent_description']
                })
            else:
                # 负样例
                persona_samples[persona_id]["negative"].append(entry['intent_description'])
    
    # 生成训练数据对
    data_pairs = []
    for persona_id, samples in persona_samples.items():
        positives = samples["positive"]
        negatives = samples["negative"]
        
        for positive in positives:
            # 如果没有负样例则跳过
            if not negatives:
                continue
            
            # 随机选择 2 到 3 个负样例
            num_negatives = random.randint(2, 3)
            selected_negatives = random.sample(negatives, min(num_negatives, len(negatives)))
            
            data_pairs.append({
                "context": positive["context"],
                "candidate_intents": selected_negatives + [positive["label"]],
                "label": positive["label"]
            })
    
    return data_pairs


def get_negative_descriptions(persona_data, persona_memory):
    negative_ids = []
    negative_intent_descriptions = []
    for session in persona_data["sessions"]:
        if not session["exist_confirmation"]:
            negative_ids.append(int(session["session_id"]))  
            
    for persona_mem in persona_memory["sessions"]:
        for k, v in persona_mem.items():
            if int(k) in negative_ids:
                negative_intent_descriptions.append(v["intent_description"])
    return negative_intent_descriptions


if __name__ == "__main__":

    output_file = '/mnt/ailabtemp/duyiming/mmt_tod/pos_neg_test_memory_intent_descriptions.jsonl'
    persona_memory_folder = "/mnt/ailabtemp/duyiming/mmt_tod/qa_summary_memory_bank_train"
    persona_memory_paths = os.listdir(persona_memory_folder)
    persona_data_folder = "/mnt/ailabtemp/duyiming/mmt_tod/personal_dataset_565/"

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for persona_path in persona_memory_paths:
            # 读取 JSON 文件
            persona_memory_file = f'{persona_memory_folder}/{persona_path}'
            with open(persona_memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            persona_file = persona_data_folder + persona_path
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
            # 提取 persona_id
            persona_id = data.get("persona_id", "unknown")

            # 提取每个 session 的 id 和 intent_description，并创建字典

            negative_intent_descriptions = get_negative_descriptions(persona_data, data)

            for session in data.get("sessions", []):
                for session_id, session_data in session.items():
                    pos_or_neg, session_context = judge_positive(persona_data, session_id)
                    intent_description = session_data.get("intent_description", "")
                    if pos_or_neg:
                        # 组织成字典格式
                        output_data = {
                            "persona_id": persona_id,
                            "id": session_id,
                            "intent_description": intent_description,
                            "pos_or_neg":pos_or_neg,
                            "session": session_context
                        }
                        # 写入 JSONL 文件，每行一个 JSON 对象
                        out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    # else:
                        random_intent = random.choice(negative_intent_descriptions)
                        # 组织成字典格式
                        output_data = {
                            "persona_id": persona_id,
                            "id": session_id,
                            "intent_description": random_intent,
                            "pos_or_neg":False,
                            "session": session_context
                        }
                        # 写入 JSONL 文件，每行一个 JSON 对象
                        out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

        print(f"Intent descriptions saved to {output_file}")

    
        # 加载数据并生成训练对
    # # file_path = 'memory_intent_descriptions.jsonl'  # 请将此路径修改为您的 JSONL 文件路径
    # data_pairs = load_and_generate_pairs(output_file)

    # # 将生成的训练数据保存为 CSV 文件
    # df = pd.DataFrame(data_pairs)
    # output_csv_path = '/mnt/ailabtemp/duyiming/mmt_tod/pos_neg_intent_train_data_1111.csv'
    # df.to_csv(output_csv_path, index=False)

    # print(f"生成的训练数据已保存为 {output_csv_path}")