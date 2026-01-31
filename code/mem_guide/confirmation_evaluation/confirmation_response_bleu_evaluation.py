
import os
import csv
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_rouge_l(reference, prediction): 
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 
    scores = scorer.score(reference, prediction) 
    return scores['rougeL'].fmeasure



def extract_reference_response(persona_id, session_id):
    with open(f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_tod_memory_evaluation_131/personal_{persona_id}.json', 'r', encoding='utf-8') as file:
        personal_dialogue = json.load(file)
    file.close()
    for session_data in personal_dialogue["sessions"]:
        if session_data["session_id"] == int(session_id):
            for uttr in session_data["content"]:
                if uttr["is_confirmation"]:
                    return uttr["utterance"]
    return None



if __name__ == '__main__':
    # result_file_path = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/final_confirmation_results/gpt4omini/qa_memory/dynamic_rerank_confirmation_generation_mmt_tod_text_embed_10.csv"
    output_results = {}
    result_root_folder = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/final_confirmation_results"
    for llama_folder in os.listdir(result_root_folder):
        if llama_folder not in ["gpt4omini", "llama", "mistral", "qwen"]:
            continue
        output_results[llama_folder] = {}
        setting_folder = result_root_folder + "/" + llama_folder
        for setting_folder_name in os.listdir(setting_folder):
            if setting_folder_name not in ["ground_truth", "qa_memory", "retrieved_dialogue_history", "whole_history"]:
                continue
            output_results[llama_folder][setting_folder_name] = {}
            setting_folder_path = result_root_folder + "/" + llama_folder + "/" + setting_folder_name
            for file_name in os.listdir(setting_folder_path):
                if "score" in file_name:
                    continue
                if ".DS_Store" in file_name:
                    continue
                result_file_path = setting_folder_path + "/" + file_name
                print(result_file_path)
                counter = 0
                references = []
                predictions = []
                bleu_scores = 0.0
                rougel_scores = 0.0
                slot_accuracy = 0.0
                with open(result_file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        try:
                            reference_response = extract_reference_response(row[0], row[1])
                            if reference_response is None:
                                print("null")
                                reference_response = row[6]
                            predict_response = row[5]
                            # 将参考答案和预测答案分词（句子变成token列表）
                            reference = [reference_response.split()]  # nltk的sentence_bleu需要列表的列表作为参考
                            candidate = predict_response.split()

                            # # 计算BLEU-1，即权重为(1.0, 0, 0, 0)
                            bleu1_score = sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25), 
                                                        smoothing_function=SmoothingFunction().method1)
                            
                            rougel = compute_rouge_l(reference_response, predict_response)
                            rougel_scores += rougel*100
                            # # BLEU值本身为0~1之间，将其乘以100便于查看百分数
                            bleu1_score = bleu1_score * 100
                            bleu_scores += bleu1_score
                            slot_accuracy += float(row[-1])
                            counter += 1
                        except Exception as e:
                            print(e)
                output_results[llama_folder][setting_folder_name] = {"bleu_score":bleu_scores/counter, "rougel":rougel_scores/counter, "slot_accuracy": slot_accuracy/counter}
                print(bleu_scores/counter)
                print(rougel_scores/counter)
    print(output_results)
    







