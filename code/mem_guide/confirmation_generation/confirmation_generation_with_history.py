import os
import json
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
# 指定 GPU 设备
# torch.cuda.set_device(1)  # 固定在第三张 GPU（cuda:2）
device = "cuda:1" if torch.cuda.is_available() else "cpu"
sys.path.append("/Users/duyiming/Documents/")
from gpt_proxy_client import openai_proxy 


def gpt4_generate(input_message):
    client = openai_proxy.GptProxy(api_key="74cc20a30beb601722ea5faa69f1dff9")
    rsp = client.generate(
            messages=input_message,
            model="gpt-4o-mini-2024-07-18",  #gpt-4o-2024-08-06-ptu. gpt-4o-mini-2024-07-18
            transaction_id="lsch_test_0004", # 同样transaction_id将被归类到同一个任务，一起统计
            temperature=0.8
        )
    output = rsp.json()["data"]["response_content"]["choices"][0]["message"]["content"]
    return output

def load_model_and_tokenizer(model_name):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},  # 确保模型加载到 cuda:2
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


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
        retrieved_session_ids = eval(row['retrieved_session_ids'])  # Convert string representation of list to actual list
        reference_ids = eval(row['reference_ids'])  # Convert string representation of list to actual list
        
        if persona_id not in nested_dict:
            nested_dict[persona_id] = {}
        
        nested_dict[persona_id][session_id] = {
            "retrieved_session_ids": retrieved_session_ids,
            "reference_ids": reference_ids,
        }
    
    return nested_dict


def llama_generate_response(model, tokenizer, input_messages, max_length=2048, temp = 0.3):

    text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
        temperature = temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
        
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


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


def search_intent_similar_sessions(top_2_intents_scores, personal_bank):
    results = {}
    for intent, score in top_2_intents_scores:
        for one_session in personal_bank["sessions"]:
            for k, v in one_session.items():
                if intent == v["intent_description"]:
                    v["intent_similar_score"] = score
                    results[k] = v
    return results


# confirmation generation
def confirmation_generation(retrieved_session_content, session_id, current_session, task_goals):
    system_message = {"role":"system", "content":"You are an dialogue assistant."}
    input_message = [system_message]
    utterance_id = 0
    scores = []
    output_results = []
    try:
        if current_session and current_session[0]["speaker"].lower() == "assistant":
            input_message.append({"role": "user", "content": "Hello"})  # 一个简单的虚拟User内容
        for utterance in current_session:
            if utterance["is_confirmation"]:
                if len(input_message) == 1:
                    continue

                task_goal = []
                task_slots = []

                for task_slots_list in task_goals:
                    if task_slots_list["utterance_id"] == utterance_id:
                        for ts_item in task_slots_list["slot_values"]:
                            task_goal.append(ts_item[0])
                            task_slots.append(ts_item[1])
                if len(task_goal) == 0:
                    continue
                query = input_message[-1]["role"] + ":" + input_message[-1]["content"]
                current_session_dialogue = extract_current_history(input_message)
                input_message[0]["content"] = f'You are an dialogue assistant. Generate a confirmation response based on the user\'s utterance. Include any relevant task goal slots [{",".join(task_goal)}] identified in the dialogue or retrieved history. If such information is unavailable, construct your response accurately and comprehensively using the provided conversation details. Ensure your reply acknowledges the user\'s request clearly and incorporates relevant information from both the dialogue and retrieved history {retrieved_session_content}.'

                target_confirmation_message = input_message.copy()
                
                
                # target_confirmation_message.append(system_message)


                response = gpt4_generate(target_confirmation_message)
                reference = utterance["utterance"]
                slot_accuracy = evaluate_task_goal_slots(response, task_slots)
                scores.append(slot_accuracy)

                output_results.append([personal_dialogue["persona_id"], session_id, retrieved_session_content, current_session_dialogue, query, response, reference, slot_accuracy])
                break
                # input_message.append({"role": "assistant", "content": utterance["utterance"]})
            else:
                # Append user or assistant utterance based on the speaker
                role = "user" if utterance["speaker"].lower() == "user" else "assistant"
                input_message.append({"role": role, "content": utterance["utterance"]})
            utterance_id += 1
    except Exception as e:
        print(e)

    return output_results, scores


def extract_history_context(session_data):
    history_context = ""
    for turn in session_data["content"]:
        history_context = history_context + turn["speaker"] + ":" + turn["utterance"] + "\n"
    return history_context
    

def load_retrieved_dialogue_memory(personal_dialogue, session_ids):
    dialogue_history = ""
    for session_id in session_ids:
        for session_data in personal_dialogue["sessions"]:
            if int(float(session_data["session_id"])) == int(float(session_id)):
                dialogue_history+= extract_history_context(session_data)
    return dialogue_history

if __name__ == '__main__':
    personal_dataset_folder = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/clean_personal_dataset_0928_131"
    personal_file_paths = os.listdir(personal_dataset_folder)
    # model_config_path = "/mnt/ailabtemp/duyiming/duyiming/llm_models/model_path_on_server.json"
    rerank_result_file_path = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/confirmation_generation/updated_text_embedding_small_5_10_rerank_new_results_5_5.csv'
    output_file = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/final_confirmation_results/gpt4omini/retrieved_dialogue_history/gpt4o_mini_confirmation_generation_with_retrieved_history_text_embed_10_dynamic_rerank.csv'
    confirmation_slot_accuracy_average = 0
    confirmation_number = 0
    index = 0
    final_output_results = []
    emb_model = "text_embed_10"
   
    rerank_results = construct_rerank_results_dict(rerank_result_file_path)
    
    # with open(model_config_path, 'r', encoding='utf-8') as file:
    #     model_path = json.load(file)

    # model, tokenizer = load_model_and_tokenizer(model_path["qwen2_7B_instruct"])
    
    for personal_file in tqdm(personal_file_paths):
        with open(f'{personal_dataset_folder}/{personal_file}', 'r', encoding='utf-8') as file:
            personal_dialogue = json.load(file)
        file.close()

        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data["exist_confirmation"]:
                    task_goals = session_data["task_goal"]
                    retrieved_session_ids =  rerank_results[personal_dialogue["persona_id"]][session_data["session_id"]]["retrieved_session_ids"]
                    retrieved_dialogue_history = load_retrieved_dialogue_memory(personal_dialogue, retrieved_session_ids)
                    eval_output_result, scores = confirmation_generation(retrieved_dialogue_history, str(session_data["session_id"]), session_data["content"], task_goals)
                    final_output_results.extend(eval_output_result)
                    for score in scores:
                        confirmation_slot_accuracy_average += score
                    confirmation_number += 1

            except Exception as e:
                print(e)

    print(confirmation_slot_accuracy_average/confirmation_number)
    #     # 将数据转换为 DataFrame
    df = pd.DataFrame(final_output_results[1:], columns=final_output_results[0])

    # 写入 CSV 文件
    df.to_csv(output_file, index=False)

    print("CSV 文件已成功生成。")







