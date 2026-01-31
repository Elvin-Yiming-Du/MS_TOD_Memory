import os
import json
import sys
import re
import numpy as np
import pandas as pd
sys.path.append("/mnt/ailabtemp/duyiming/mmt_tod/")
from utils import cosine_similarity, get_embeddings, qa_total_similarity
from api_llm_utils import get4mini_generate, get4mini_message_generate

# summary the current session intention.
# including intention description
# task slots qa pair generation.
# retrieve the candidate history with the query for the missing task slots.

def evaluate_task_goal_slots(response, task_slots):
    correct_slot = 0
    for slot_value in task_slots:
        if slot_value.lower() in response.lower():
            correct_slot += 1

    slot_accuracy = correct_slot/len(task_slots)
    return slot_accuracy

def extract_current_history(input_message):
    current_session_history = ""
    for message in input_message[:-1]:
        current_session_history = current_session_history + message["role"] + ":" + message["content"] + "\n"

    return current_session_history

# 检索最相关的k个intent的函数
def retrieve_top_k_intent_similarity(intent_description, personal_memory_bank, k=1):
    # 提取所有的intent描述
    descriptions = [session_data.values()[0]["intent_description"] for session_id, session_data in personal_memory_bank["sessions"]]
    
    # 将新intent_description加入描述列表
    descriptions.append(intent_description)
    
    # 获取描述的嵌入
    embeddings = get_embeddings(descriptions)
    
    # 计算新描述与现有描述的余弦相似度
    new_embedding = embeddings[-1]  # 最后一项是新intent的嵌入
    similarities = [cosine_similarity(new_embedding, emb) for emb in embeddings[:-1]]  # 与现有的比较
    
    # 获取相似度最高的k个索引
    top_k_indices = np.argsort(similarities)[-k:][::-1]  # 按相似度从大到小排序，取前k个
    
    # 返回最相关的k个session索引
    return top_k_indices

def extract_triple_quotes_content(text):
    # 使用正则表达式查找 ''' 和 ''' 之间的内容
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)  # re.DOTALL 使 '.' 匹配包括换行符在内的所有字符
    return matches[0]

def summary_current_session_intention(current_session):
    task_goal_attributes = []
    task_goals = current_session["task_goal"]
    for task_goal in task_goals:
        for s_values in task_goal["slot_values"]:
            task_goal_attributes.append(s_values[0])    

    history_context = ""
    for turn in current_session["content"]:
        if turn["is_confirmation"]:
            break
        history_context = history_context + turn["speaker"] + ":" + turn["utterance"] + "\n"

    intent_description_prompt = f'Please generate one detailed sentence to describe the user intent accourding to the dialogue history: {history_context}\n'
    # intent_message = [{"role": "user", "content": intent_description_prompt}]
    intent_description = get4mini_generate(intent_description_prompt, temperature = 0.8)

    task_goal_attributes = ",".join(task_goal_attributes)
    missing_task_goal_query_prompt = """Please help me generate questions, 
    based on the provided conversation history {history}, that correspond to unanswered attributes in the 
    task goal {task_attributes}. 
    1. The questions should start with \'What,\' \'When,\' \'Why,\' \'How,\' or \'Where.\' 
    2. Ensure that the generated questions are in third person.
    fill the following json:
            [
            [Question]:
            ]"""
    missing_task_goal_prompt = missing_task_goal_query_prompt.format(history = history_context, task_attributes = task_goal_attributes)
    # complete_message = [{"role": "user", "content": missing_task_goal_prompt}]
    missing_task_slot_queries = get4mini_generate(missing_task_goal_prompt, temperature = 0.8)

    return intent_description, missing_task_slot_queries

def group_query_similarity(missing_queries, session_candidates):
    scores = {}
    for session_content in session_candidates:
        for session_id, session_candidate in session_content.items():
            total_scores, overall_score = qa_total_similarity(missing_queries, session_candidate)
            scores[session_id] = {"all_scores" : total_scores, "overall_score": overall_score}
    return scores

def search_intent_similar_sessions(top_2_intents_scores, personal_bank):
    results = {}
    for intent, score in top_2_intents_scores:
        for one_session in personal_bank["sessions"]:
            for k, v in one_session.items():
                if intent == v["intent_description"]:
                    v["intent_similar_score"] = score
                    results[k] = v
    return results

def retrieve_qas_by_session_id(personal_bank, session_id):
    qa_summary = []
    for one_session in personal_bank["sessions"]:
        for k, v in one_session.items():
            if k == session_id:
                qa_summary = v["qa_summary"]
                break
    updated_qa_summaries = []
    if type(qa_summary) is list:
        for qa_pairs in qa_summary:
            updated_qa_summaries.append({"Question" : list(qa_pairs.values())[0], "Answer" : qa_pairs["Answer"]})
    return updated_qa_summaries

def retrieve_intent_simiarity(current_session_intent_description, personal_bank, target_session_id, top_k = 2):
    history_descriptions = []
    # Get the embedding for the current session intent
    for one_session in personal_bank["sessions"]:
        for k, v in one_session.items():
            if int(k) > target_session_id - 1:
                break
            history_descriptions.append(v["intent_description"])
    current_embedding = get_embeddings([current_session_intent_description])[0]
    
    # Get embeddings for the historical intents
    history_embeddings = get_embeddings(history_descriptions)
    
    # Compute the cosine similarity between the current intent and each historical intent
    similarities = [cosine_similarity(current_embedding, history_embedding) for history_embedding in history_embeddings]

    # Zip similarities with corresponding history intents
    intent_similarity_pairs = list(zip(history_descriptions, similarities))
    
    # Sort by similarity score in descending order
    sorted_intent_similarity_pairs = sorted(intent_similarity_pairs, key=lambda pair: pair[1], reverse=True)
    
    if top_k < len(sorted_intent_similarity_pairs):
        # Select the top 2 most similar intents
        top_k_intents = sorted_intent_similarity_pairs[:top_k]
    else:
        top_k_intents = sorted_intent_similarity_pairs
    top_k_results = search_intent_similar_sessions(top_k_intents, personal_bank)
    # Return the top 2 history intents along with their similarity scores
    return top_k_results

def top_k_scores(scores, k):
    # 按照 overall_score 进行排序，先将 overall_score 转换为整数用于排序
    sorted_items = sorted(scores.items(), key=lambda item: item[1]['overall_score'], reverse=True)
    
    # 提取前 k 个键
    top_k_keys = [item[0] for item in sorted_items[:k]]
    
    return top_k_keys

def search_ground_truth_session_ids(personal_dialogue, search_session_id, target_id):
    reference_history_session_ids = []
    for session_candidate in personal_dialogue["sessions"]:
        if search_session_id == session_candidate["reference_dialogue_id"] and session_candidate["session_id"] != target_id:
            reference_history_session_ids.append(session_candidate["session_id"])
    return reference_history_session_ids


# confirmation generation
if __name__ == '__main__':

    model_config_path = "/mnt/ailabtemp/duyiming/llm_models/model_path.json"
    with open(model_config_path, 'r', encoding='utf-8') as file:
        model_path = json.load(file)
    non_exist_file = "/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/non_existing_pairs.jsonl"
    non_exist_session_ids = {}
    with open(non_exist_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            persona_id = data.get("persona_id")
            session_id = data.get("session_id")
            if non_exist_session_ids is None or persona_id not in non_exist_session_ids.keys():
                non_exist_session_ids[persona_id] = [session_id]
            else:
                non_exist_session_ids[persona_id].append(session_id)

    personal_file_paths = os.listdir("/mnt/ailabtemp/duyiming/mmt_tod/clean_personal_dataset_0928_131/")
    output_results = []
    MAX_TRY = 5
    found_flag = False
    output_data_file = f'/mnt/ailabtemp/duyiming/mmt_tod/intermediate_results/incontext_reflection/gpt4omini_reflection_others.jsonl'    
    for personal_file in personal_file_paths:
        
        with open(f'/mnt/ailabtemp/duyiming/mmt_tod/clean_personal_dataset_0928_131/{personal_file}', 'r', encoding='utf-8') as file:
            personal_dialogue = json.load(file)
        file.close()
        

        if os.path.exists(output_data_file):
            continue
            
        personal_memory_bank_path = f'/mnt/ailabtemp/duyiming/mmt_tod/qa_summary_memory_bank/personal_{str(personal_dialogue["persona_id"])}.json'
    
        with open(personal_memory_bank_path, 'r') as file:
            personal_memory_bank = json.load(file)
        file.close

        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data["exist_confirmation"]:
                    if personal_dialogue["persona_id"] not in non_exist_session_ids.keys():
                        continue
                    if session_data["session_id"] not in non_exist_session_ids[personal_dialogue["persona_id"]]:
                        continue
                    TRY_NUM = 0
                    while(TRY_NUM < MAX_TRY):
                        try:
                            output_record = {}        
                            intent_description, missing_goal_quries = summary_current_session_intention(session_data)
                            questions_str = extract_triple_quotes_content(missing_goal_quries)
                            questions = json.loads(questions_str)
                            output_record["persona_id"] = personal_dialogue["persona_id"]
                            output_record["session_id"] = session_data["session_id"]
                            output_record["intent_guess"] = intent_description
                            output_record["missing_questions"] = questions
                            output_results.append(output_record)
                            break
                        except Exception as e:
                            print(e)
                            TRY_NUM += 1
            except Exception as e:
                print(e)               
       
    with open(output_data_file, 'w') as new_file:
        for item in output_results:
            json_object = json.dumps(item)
            new_file.write(json_object + '\n')
        new_file.close()
        print("write into file successfully!")









