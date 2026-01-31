import os
import json
import sys
import re
import numpy as np
import pandas as pd
sys.path.append("/Users/duyiming/Documents/YimingDU/tod_multi_turn/")
from api_llm_utils import get4mini_generate, get4mini_message_generate

# summary the current session intention.
# including intention description
# task slots qa pair generation.
# retrieve the candidate history with the query for the missing task slots.


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

# confirmation generation
if __name__ == '__main__':

    # model_config_path = "/mnt/ailabtemp/duyiming/llm_models/model_path.json"
    # with open(model_config_path, 'r', encoding='utf-8') as file:
    #     model_path = json.load(file)
    personal_file_paths = os.listdir("/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_dataset_565")
    
    MAX_TRY = 3
    found_flag = False

    for personal_file in personal_file_paths:
        output_results = []
        with open(f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_dataset_565/{personal_file}', 'r', encoding='utf-8') as file:
            personal_dialogue = json.load(file)
        file.close()
        output_data_file = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/intermediate_results/incontext_reflection_train/train_gpt4omini_reflection_{personal_dialogue["persona_id"]}.jsonl'    

        if os.path.exists(output_data_file):
            continue
            
        personal_memory_bank_path = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/qa_summary_memory_bank_train/personal_{str(personal_dialogue["persona_id"])}.json'
    
        with open(personal_memory_bank_path, 'r') as file:
            personal_memory_bank = json.load(file)
        file.close

        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data["exist_confirmation"]:
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









