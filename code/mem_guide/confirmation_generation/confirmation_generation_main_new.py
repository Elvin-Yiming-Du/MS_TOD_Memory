import os
import json
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
sys.path.append("/Users/duyiming/Documents/")
from gpt_proxy_client import openai_proxy 

def gpt4_generate(input_message):
    client = openai_proxy.GptProxy(api_key="74cc20a30beb601722ea5faa69f1dff9")
    rsp = client.generate(
            messages=input_message,
            model="gpt-4o-mini-2024-07-18", 
            transaction_id="lsch_test_0004",
            temperature=0.8
        )
    output = rsp.json()["data"]["response_content"]["choices"][0]["message"]["content"]
    return output

device = "cuda:3" if torch.cuda.is_available() else "cpu"

response_generation_prompt ="""
You are a dialogue assistant. 
Generate a confirmation response based on the user's utterance. Include any relevant task goals [TASK GOALS] identified in the dialogue or related memory [MEMORY]. If [MEMORY] is unavailable, construct your response accurately and comprehensively using the provided conversation details. Ensure your reply acknowledges the user's request clearly and incorporates relevant information from both the dialogue and the related memory units [MEMORY].
[TASK GOAL]
{task_goal}

[MEMORY]
{memory}
"""

def load_model_and_tokenizer(model_name):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device}, 
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def llama_generate_response(model, tokenizer, input_messages, max_length=512, temp = 0.3):
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
    slot_accuracy = correct_slot/len(task_slots) if len(task_slots) > 0 else 0
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

def summarize_memory_unit(memory_unit):
    """
    使用GPT-4对单个memory unit进行总结，提取intent和对应的task attributes。
    您可以根据需要修改提示词，使得总结更准确。
    """
    prompt = [
        {"role": "system", "content": "You are a helpful assistant who summarizes given memory units."},
        {"role": "user", "content": f"Please summarize the following memory unit into a concise intent and corresponding task attributes:\n\n{memory_unit}\n\nFormat your answer as:\nIntent: <summarized intent>\nTask Attributes: <list or brief description>"}
    ]
    summary = gpt4_generate(prompt)
    return summary.strip()

def construct_retrieved_memory_units(retrieved_id, personal_memory_bank):
    """
    返回单独的memory unit列表，而不是直接拼接。
    每个session的qa_summary中有多个Answer，需要分别视为一个memory unit。
    """
    memory_units = []
    for session_content in personal_memory_bank["sessions"]:
        for k, v in session_content.items():
            if str(retrieved_id) == str(k):
                summaries = v.get("qa_summary", [])
                # 将每一个Answer视为一个独立的memory unit
                for summary_item in summaries:
                    memory_units.append(summary_item["Answer"])
    return memory_units

def generate_summarized_memory(retrieved_results, personal_memory_bank):
    """
    对检索到的session进行处理。
    对其中的每个memory unit进行总结，并返回所有总结后的文本。
    """
    all_summaries = []
    for retrieved_id in retrieved_results:
        memory_units = construct_retrieved_memory_units(retrieved_id, personal_memory_bank)
        for unit in memory_units:
            summarized_unit = summarize_memory_unit(unit)
            all_summaries.append(summarized_unit)
    # 将所有memory unit总结拼接起来作为最终的MEMORY内容
    return "\n\n".join(all_summaries)


# confirmation generation
def confirmation_generation(model, tokenizer, retrieved_session_content, session_id, current_session, task_goals):
    system_message = {"role":"system", "content":"You are a dialogue assistant."}
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
            
            for task_slots_list in task_goals:
                if task_slots_list["utterance_id"] == utterance_id:
                    for ts_item in task_slots_list["slot_values"]:
                        task_goal.append(ts_item[0])
                        task_slots.append(ts_item[1])
            if len(task_goal) == 0:
                continue

            query = input_message[-1]["role"] + ":" + input_message[-1]["content"]
            current_session_dialogue = extract_current_history(input_message)

            # 将retrieved_session_content的总结结果作为MEMORY
            system_message_prompt = response_generation_prompt.replace("{task_goal}", ",".join(task_goal))
            system_message_prompt = system_message_prompt.replace("{memory}", retrieved_session_content)
            input_message[0]["content"] = system_message_prompt

            target_confirmation_message = input_message.copy()
            response = gpt4_generate(target_confirmation_message)

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
    df = pd.read_csv(file_path)
    nested_dict = {}
    for _, row in df.iterrows():
        persona_id = row['persona_id']
        session_id = row['session_id']
        retrieved_sessions = eval(row['retrieved_session_ids']) 
        reference_ids = eval(row['reference_ids']) 
        
        if persona_id not in nested_dict:
            nested_dict[persona_id] = {}
        
        nested_dict[persona_id][session_id] = {
            "retrieved_sessions": retrieved_sessions,
            "reference_ids": reference_ids
        }
    return nested_dict


if __name__ == '__main__':
    sample_test_persona_ids = [1, 10, 19, 23, 32, 49, 68, 98, 112, 129]
    personal_file_paths = os.listdir("/Users/duyiming/Documents/YimingDU/tod_multi_turn/clean_personal_dataset_0928_131")
    confirmation_slot_accuracy_average = 0
    confirmation_number = 0
    index = 0
    final_output_results = []

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
        
        if int(personal_dialogue["persona_id"]) not in sample_test_persona_ids:
            continue
        
        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data["exist_confirmation"]:
                    task_goals = session_data["task_goal"]
                    retrieved_results = rerank_results[personal_dialogue["persona_id"]][session_data["session_id"]]["retrieved_sessions"]
                    
                    # 对retrieved session内容先行总结
                    summarized_memory = generate_summarized_memory(retrieved_results, personal_memory_bank)

                    eval_output_result, scores = confirmation_generation(model, tokenizer, summarized_memory, str(session_data["session_id"]), session_data["content"], task_goals)
                    final_output_results.extend(eval_output_result)
                    for score in scores:
                        confirmation_slot_accuracy_average += score
                    confirmation_number += 1
                    if confirmation_number % 2 == 0:
                        print(confirmation_slot_accuracy_average/confirmation_number)
            except Exception as e:
                print(e)

    print(confirmation_slot_accuracy_average)
    # 将数据转换为DataFrame
    if len(final_output_results) > 0:
        df = pd.DataFrame(final_output_results[1:], columns=final_output_results[0])
        df.to_csv(f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/confirmation_generation/confirmation_generation_qa_memory_1212_version1_{emb_model}.csv', index=False)
        print("CSV 文件已成功生成。")
    else:
        print("无结果输出。")
