import os
import json
import sys
import torch
import numpy as np
import pandas as pd
# from utils import cosine_similarity, get_embeddings, qa_total_similarity
# from api_llm_utils import get4mini_generate, get4mini_message_generate
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
sys.path.append("/Users/duyiming/Documents/")
from gpt_proxy_client import openai_proxy 

def gpt4_generate(input_message):
    client = openai_proxy.GptProxy(api_key="74cc20a30beb601722ea5faa69f1dff9")
    rsp = client.generate(
            messages=input_message,
            model="gpt-4o-mini-2024-07-18",  #gpt-4o-2024-08-06-ptu. gpt-4o-mini-2024-07-18
            transaction_id="lsch_test_0004", # 同样transaction_id将被归类到同一个任务，一起统计
            temperature=0.001
        )
    output = rsp.json()["data"]["response_content"]["choices"][0]["message"]["content"]
    return output
# 指定 GPU 设备
# torch.cuda.set_device(3)  # 固定在第三张 GPU（cuda:2）
device = "cuda:3" if torch.cuda.is_available() else "cpu"

response_generation_prompt ="""
You are an dialogue assistant. 
Generate a confirmation response based on the user\'s utterance. Include any relevant task goals [TASK GOALS] identified in the dialogue or related memory [MEMORY]. If [MEMORY] is unavailable, construct your response accurately and comprehensively using the provided conversation details. Ensure your reply acknowledges the user\'s request clearly and incorporates relevant information from both the dialogue and the related memory units [MEMORY].
[TASK GOAL]
{task_goal}

[MEMORY]
{memory}
"""


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


def construct_retrieved_contents(retrieved_id, personal_memory_bank):
    retrieved_context = ""
    for session_content in personal_memory_bank["sessions"]:
        try:
            for k, v in session_content.items():
                if str(retrieved_id) == str(k):
                    retrieved_context += "The following is one memory unit:\n"
                    # retrieved_context += v["intent_description"]
                    summaries = v["qa_summary"]
                    for summary in summaries:
                        retrieved_context += summary["Answer"] + "\n"
        except Exception as e:
            retrieved_context = str(v["qa_summary"])

    return retrieved_context

# confirmation generation
def confirmation_generation(model, tokenizer, retrieved_session_content, session_id, current_session, task_goals):
    system_message = {"role":"system", "content":"You are an dialogue assistant."}
    input_message = [system_message]
    utterance_id = 0
    scores = []
    output_results = []
    for utterance in current_session:
        if utterance["is_confirmation"]:
            if len(input_message) == 0:
                continue

            task_goal = []
            task_slots = []
            
            # if task_goals[0]["utterance_id"] == utterance_id:
            #     current_task_goal = task_goals[0]["slot_values"]
            # else:
            #     continue

            for task_slots_list in task_goals:
                if task_slots_list["utterance_id"] == utterance_id:
                    for ts_item in task_slots_list["slot_values"]:
                        task_goal.append(ts_item[0])
                        task_slots.append(ts_item[1])
            if len(task_goal) == 0:
                continue
            query = input_message[-1]["role"] + ":" + input_message[-1]["content"]
            current_session_dialogue = extract_current_history(input_message)
            # system_message = {"role": "system", "content": f'You are an dialogue assistant. Generate a confirmation response based on the user\'s utterance. Include any relevant task goal slots [{",".join(task_goal)}] identified in the dialogue or retrieved history. If such information is unavailable, construct your response accurately and comprehensively using the provided conversation details. Ensure your reply acknowledges the user\'s request clearly and incorporates relevant information from both the dialogue and retrieved history {retrieved_session_content}.'}
            # system_message = {"role": "system", "content": f'The assistant need to generate one confirmation response within 50 words. \n\n [Example] Assistant: Excellent! Just to confirm, you would like to buy 2 standard tickets for the Nationals Vs Royals game on March 3rd in Washington D.C. Is that correct? [Requirements]Include any relevant task goal slots [{",".join(task_goal)}] identified in the dialogue context or related history: {retrieved_session_content}.'}
            system_message_prompt = response_generation_prompt.replace("{task_goal}", ",".join(task_goal))
            system_message_prompt = system_message_prompt.replace("{memory}", retrieved_session_content)
            input_message[0]["content"] = system_message_prompt

            # input_message[0]["content"] = f'You are an dialogue assistant. Generate a confirmation response based on the user\'s utterance. Include any relevant task goal slots [{",".join(task_goal)}] identified in the dialogue or retrieved history. If such information is unavailable, construct your response accurately and comprehensively using the provided conversation details. Ensure your reply acknowledges the user\'s request clearly and incorporates relevant information from both the dialogue and retrieved history {retrieved_session_content}.'
            target_confirmation_message = input_message.copy()
            # target_confirmation_message.append(system_message)


            response = gpt4_generate(target_confirmation_message)
            # print(response)
            reference = utterance["utterance"]
            slot_accuracy = evaluate_task_goal_slots(response, task_slots)
            scores.append(slot_accuracy)

            output_results.append([personal_dialogue["persona_id"], session_id, retrieved_session_content, current_session_dialogue, query, response, reference, str(task_goals), slot_accuracy])
            break
            input_message.append({"role": "assistant", "content": utterance["utterance"]})
        else:
            # Append user or assistant utterance based on the speaker
            role = "user" if utterance["speaker"].lower() == "user" else "assistant"
            input_message.append({"role": role, "content": utterance["utterance"]})
        utterance_id += 1

    return output_results, scores

def construct_rerank_results_dict(file_path):
    """
    Reads a CSV file and constructs a nested dictionary in the format:
    {
        <persona_id>: {
            <session_id>: {
                "retrieved_sessions": <retrieved_sessions>,
                "reference_ids": <reference_ids>
            }
        }
    }
    """
    # Load the file
    df = pd.read_csv(file_path)
    
    # Initialize the nested dictionary
    nested_dict = {}
    
    # Iterate through each row to populate the dictionary
    for _, row in df.iterrows():
        persona_id = row['persona_id']
        session_id = row['session_id']
        retrieved_sessions = eval(row['retrieved_session_ids'])  # Convert string representation of list to actual list
        reference_ids = eval(row['reference_ids'])  # Convert string representation of list to actual list
        
        if persona_id not in nested_dict:
            nested_dict[persona_id] = {}
        
        nested_dict[persona_id][session_id] = {
            "retrieved_sessions": retrieved_sessions,
            "reference_ids": reference_ids
        }
    
    return nested_dict


    
if __name__ == '__main__':
    # sample_test_persona_ids = [1, 10, 19, 23, 32, 49, 68, 98, 112, 129]
    personal_file_paths = os.listdir("/Users/duyiming/Documents/YimingDU/tod_multi_turn/clean_personal_dataset_0928_131")
    confirmation_slot_accuracy_average = 0
    confirmation_number = 0
    index = 0
    final_output_results = []
    # model_config_path = "/mnt/ailabtemp/duyiming/duyiming/llm_models/model_path_on_server.json"
    # with open(model_config_path, 'r', encoding='utf-8') as file:
    #     model_path = json.load(file)

    # model, tokenizer = load_model_and_tokenizer(model_path["llama3_8B_instruct"])
    model, tokenizer = None, None
    emb_model = "text_embed_10"
    rerank_result_file_path = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/confirmation_generation/updated_text_embedding_small_5_10_rerank_new_results_5_5.csv'
    rerank_results = construct_rerank_results_dict(rerank_result_file_path)

    for personal_file in personal_file_paths:
        with open(f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/clean_personal_dataset_0928_131/{personal_file}', 'r', encoding='utf-8') as file:
            personal_dialogue = json.load(file)
        file.close()
        personal_memory_bank_path = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/qa_summary_memory_bank/personal_{str(personal_dialogue["persona_id"])}.json'
    
        with open(personal_memory_bank_path, 'r') as file:
            personal_memory_bank = json.load(file)
        # if int(personal_dialogue["persona_id"]) not in sample_test_persona_ids:
        #     continue
        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data["exist_confirmation"]:
                    task_goals = session_data["task_goal"]
                    retrieved_session_content = ""
                    retrieved_results = rerank_results[personal_dialogue["persona_id"]][session_data["session_id"]]["retrieved_sessions"]
                    for retreived_score in retrieved_results:
                        retrieved_session_content = retrieved_session_content + construct_retrieved_contents(retreived_score, personal_memory_bank) + "\n"
                
                    eval_output_result, scores = confirmation_generation(model, tokenizer, retrieved_session_content, str(session_data["session_id"]), session_data["content"], task_goals)
                    final_output_results.extend(eval_output_result)
                    for score in scores:
                        confirmation_slot_accuracy_average += score
                    confirmation_number += 1
                    if confirmation_number % 20 == 0:
                        print(confirmation_slot_accuracy_average/confirmation_number)
            except Exception as e:
                print(e)

    print(confirmation_slot_accuracy_average)
    #     # 将数据转换为 DataFrame
    df = pd.DataFrame(final_output_results[1:], columns=final_output_results[0])

    # 写入 CSV 文件
    df.to_csv(f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/confirmation_generation/confirmation_generation_qa_memory_1212_version2_{emb_model}.csv', index=False)

    print("CSV 文件已成功生成。")









