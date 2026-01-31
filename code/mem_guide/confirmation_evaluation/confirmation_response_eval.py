import os
import json
import sys
import csv
import time
from openai import OpenAI
import json
os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_BASE_URL"] = "xxx"
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
  base_url=os.environ.get("OPENAI_BASE_URL"),
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
sys.path.append("/Users/duyiming/Documents/")
from gpt_proxy_client import openai_proxy 
from rouge_score import rouge_scorer

def gpt4_generate(input_message,
                   model_name = "gpt-4o-2024-08-06",
                   temperature = 0.1,
                   max_length = 512):
    TRY_NUM = 1
    success = False

    for attempt in range(TRY_NUM):
        try:
             completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_length,
                messages=input_message)
        except Exception as e:
            print(f"{e}")
        else:
            success = True
            break
    if success:
        result = completion.model_dump_json()
    else:
        result = ''
    llm_response = json.loads(result)["choices"][0]["message"]["content"]
    return llm_response

def gpt4_generate_with_retry(input_message, max_retries=3, delay=1):
    """
    调用 gpt4_generate 函数并在失败时重试。
    """
    retries = 0
    while retries < max_retries:
        try:
            output_content = gpt4_generate(input_message)
            rationale, score = parse_gpt_output(output_content)
            return score, rationale
        except Exception as e:
            print(f"Error during gpt4_generate: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Returning None.")
    return None, None

def parse_gpt_output(gpt_output):
    # 1. 清理字符串
    cleaned_output = gpt_output.replace('```json', '').replace('```', '').strip()
    
    # 2. 处理转义字符
    cleaned_output = cleaned_output.encode('utf-8').decode('unicode_escape')
    
    # 3. 尝试将其解析为 JSON
    try:
        parsed_output = json.loads(cleaned_output)
        rationale = parsed_output.get("Rationale", "").strip()
        score = parsed_output.get("Score")
        return rationale, score
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None, None

eval_prompt = """
You are a strict and objective evaluator. Your task is to assess the quality of the final predicted response using the provided conversation context, the user’s target goal attributes, and a reference answer. Your evaluation should be fair, professional, and reflect an expert judgment of the response’s quality.

[Dialogue Context]
{{conversation_history}}

[Task Goal]
{{task_goal}}

[reference_answer]
{{reference_anwser}}

[predict_answer]
{{predict_answer}}

Evaluation Criteria:
Requirement Alignment: Does the final predict_answer meet the user’s task goal?
Content Accuracy: Is the information in the final response correct, clear, and logically organized?
Language Quality: Is the language fluent, coherent, and readable? Are there any obvious grammatical or word choice errors?
Comparison to Reference Answer: Compared to the reference answer, how does the final response differ in terms of completeness, professionalism, and clarity?
Overall Score: Assign a score from 1 to 10 (10 being the best), considering all of the above factors.

The evaluation must be structured in the following JSON format:
```json
{
  "Rationale": "<Explain the rationale of your score.>",
  "Score": <An integer score from 1 to 10.>
}
"""
def compute_bleu(prediction, reference): # reference和prediction应该是字符串，sacrebleu需要列表形式 
    # 创建一个自定义BLEU实例，weights=(1.0, 0, 0, 0) 表示只计算1-gram重合度 (BLEU-1)
    bleu_metric = BLEU(max_ngram_order=1)
    bleu1_score = bleu_metric.corpus_bleu(prediction, [reference])
    return bleu1_score.score

def compute_rouge_l(reference, prediction): 
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 
    scores = scorer.score(reference, prediction) 
    return scores['rougeL'].fmeasure


eval_prompt = """
You are a strict and objective evaluator. Your task is to assess the quality of the final predicted response using the provided conversation context, the user’s target goal attributes, and a reference answer. Your evaluation should be fair, professional, and reflect an expert judgment of the response’s quality.

[Dialogue Context]
{conversation_history}

[Task Goal]
{task_goal}

[Reference Answer]
{reference_answer}

[Predicted Answer]
{predict_answer}

Evaluation Criteria:
Requirement Alignment: Does the final predict_answer meet the user’s task goal?
Content Accuracy: Is the information in the final response correct, clear, and logically organized?
Language Quality: Is the language fluent, coherent, and readable? Are there any obvious grammatical or word choice errors?
Comparison to Reference Answer: Compared to the reference answer, how does the final response differ in terms of completeness, professionalism, and clarity?
Overall Score: Assign a score from 1 to 10 (10 being the best), considering all of the above factors.

The evaluation must be structured in the following JSON format:
```json
{
  "Rationale": "<Explain the rationale of your score.>",
  "Score": <An integer score from 1 to 10.>
}
"""


sample_test_persona_ids = [1, 10, 19, 23, 32, 49, 68, 98, 112, 129]

if __name__ == '__main__':
    task_goals_dict = {}
    personal_file_paths = os.listdir("/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_tod_memory_evaluation_131")
        # 读取CSV文件并逐行遍历
    
    for personal_file in personal_file_paths:
        with open(f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_tod_memory_evaluation_131/{personal_file}', 'r', encoding='utf-8') as file:
            personal_dialogue = json.load(file)
        file.close()
        task_goals_dict[personal_dialogue["persona_id"]] = {}
        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data["exist_confirmation"]:
                    task_goals = session_data["task_goal"]
                    for task_slots_list in task_goals:
                        task_goals_dict[personal_dialogue["persona_id"]][session_data["session_id"]] = task_slots_list["slot_values"]
            except Exception as e:
                print(e)


    output_results = {}
    result_root_folder = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/final_confirmation_results"
    end_flag = False
    for llama_folder in os.listdir(result_root_folder):
        if end_flag:
            print(output_results)
            break
        if llama_folder not in ["gpt4omini"] or llama_folder == ".DS_Store":
            continue
        
        output_results[llama_folder] = {}
        setting_folder = result_root_folder + "/" + llama_folder
        for setting_folder_name in os.listdir(setting_folder):
            if end_flag:
                print(output_results)
                break
            # if setting_folder_name not in ["ground_truth", "qa_memory", "retrieved_dialogue_history", "whole_history"]:
            #     continue
            if setting_folder_name not in ["summary"]:
                continue
            output_results[llama_folder][setting_folder_name] = {}
            setting_folder_path = result_root_folder + "/" + llama_folder + "/" + setting_folder_name
            for file_name in os.listdir(setting_folder_path):

                result_file_path = setting_folder_path + "/" + file_name

                result_file_path = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/confirmation_generation/confirmation_generation_qa_memory_without_pruning_text_embed_10.csv"
                output_file_path = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/confirmation_generation/confirmation_generation_qa_memory_without_pruning_text_score.csv"
                # output_file_path = setting_folder_path + "/" + "score_sample_10_"+ file_name
                all_scores = 0.0
                counter = 0
                references = []
                predictions = []
                bleu_scores = 0.0
                new_data = [["persona_id","session_id","retrieved_memory", "context","query", "predict","reference", "slot accuracy", "gpt4_score", "rationale"]]
                with open(result_file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        try:
                            if int(row[0]) not in sample_test_persona_ids:
                                continue
                            user_task_goal = task_goals_dict[int(row[0])][int(row[1])] 
                            reference_response = row[6]
                            predict_response = row[5]
                            context =  row[3] + "\n" + row[4]

                            new_eval_prompt = eval_prompt.replace("{conversation_history}", context)
                            new_eval_prompt = new_eval_prompt.replace("{task_goal}", str(user_task_goal))
                            new_eval_prompt = new_eval_prompt.replace("{reference_answer}", reference_response)
                            new_eval_prompt = new_eval_prompt.replace("{predict_answer}", predict_response)
                            input_message = [{"role":"user", "content":new_eval_prompt}]
                            score, rationale  = gpt4_generate_with_retry(input_message)
                            print(score)
                            row.append(score)
                            row.append(rationale)
                            new_data.append(row)
                            all_scores += float(score)        
                            counter += 1
                        except Exception as e:
                            print(e)
                print(all_scores/counter)
                print(counter)  
                output_results[llama_folder][setting_folder_name] = {"gpt4_score": all_scores/counter}          
                # 将新数据写入到新CSV文件
                with open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
                    csv_writer = csv.writer(outfile)
                    csv_writer.writerows(new_data)

                print(f"处理完成，结果已保存到 {output_file_path}")
                end_flag = True
                break
