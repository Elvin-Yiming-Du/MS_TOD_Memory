from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from intent_dataset import IntentDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
from transformers import AdamW
from tqdm import tqdm
import json
import os
import ast
def load_test_data(file_path):
    """加载测试数据"""
    df = pd.read_csv(file_path)
    return df

def extract_current_session(session):
    session_context = ""
    for utterance in session["content"]:
        if utterance["is_confirmation"]:
            break
        session_context = session_context + utterance["speaker"] + ":" + utterance["utterance"] + "\n"
    return session_context


def search_ground_truth_session_ids(personal_dialogue, search_session_id, target_id):
    reference_history_session_ids = []
    for session_candidate in personal_dialogue["sessions"]:
        if search_session_id == session_candidate["reference_dialogue_id"] and session_candidate["session_id"] != target_id:
            reference_history_session_ids.append(session_candidate["session_id"])
    return reference_history_session_ids


# def predict_intent(context, candidate_intents, model, tokenizer, device, top_k=2):
#     model.eval()
    
#     inputs = [
#         tokenizer(
#             f"Context: {context} Intent: {intent}",
#             max_length=512,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         ).to(device)
#         for intent in candidate_intents
#     ]
    
#     intent_scores = []
#     for input_pair in inputs:
#         # 手动创建 decoder_input_ids，通常以 <pad> token 开始
#         decoder_input_ids = torch.full(
#             (1, 1), tokenizer.pad_token_id, device=device
#         )
        
#         with torch.no_grad():
#             outputs = model(
#                 input_ids=input_pair['input_ids'],
#                 attention_mask=input_pair['attention_mask'],
#                 decoder_input_ids=decoder_input_ids
#             )
            
#             logits = outputs.logits
#             probs = torch.softmax(logits, dim=-1)
        
#             # 取概率的平均值作为得分
#             score = probs.mean().item()
#             intent_scores.append(score)
#     # 将得分和意图打包在一起
#     scored_intents = list(zip(candidate_intents, intent_scores))
    
#     # 根据得分从高到低排序
#     scored_intents.sort(key=lambda x: x[1], reverse=True)
    
#     # 返回得分最高的 top_k 个意图
#     top_intents = [intent for intent, score in scored_intents[:top_k]]
#     return top_intents

#     # best_intent_index = torch.tensor(intent_scores).argmax().item()
#     # return candidate_intents[best_intent_index]

# def calculate_similarity_score(model, context, generated_text):
#     # 将 context 和 generated_text 转换为嵌入向量
#     embeddings1 = model.encode(context, convert_to_tensor=True)
#     embeddings2 = model.encode(generated_text, convert_to_tensor=True)
    
#     # 计算余弦相似度
#     similarity = util.cos_sim(embeddings1, embeddings2).item()
#     return similarity

def calculate_similarity_score(context, generated_text):
    """基于 Jaccard 相似度的计算"""
    set1 = set(context.split())
    set2 = set(generated_text.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    return similarity

def retrieve_related_sessions(context, memory_bank, target_session_id, model, tokenizer, device, top_k=3):
    """
    从 memory_bank 中检索与 context 最相关的 session。
    
    参数：
    - context: 当前 session 的上下文文本
    - memory_bank: 包含多个 session 的字典
    - model: 已加载的 T5 模型
    - tokenizer: T5 分词器
    - device: 设备 (CPU/GPU)
    - top_k: 返回的最相关的 session 数量
    
    返回：
    - List[Tuple[session_id, intent_description, score]]
    """
    model.eval()
    candidate_intents = []
    session_ids = []
    # 提取 memory_bank 中所有的 intent_description 和 session_id
    for one_session in memory_bank["sessions"]:
        for k, v in one_session.items():
            if int(k) > target_session_id - 1:
                break
            candidate_intents.append(v["intent_description"])
            session_ids.append(k)
    # candidate_intents = [session["intent_description"] for session in memory_bank["sessions"]]
    # session_ids = [session["session_id"] for session in memory_bank["sessions"]]
    
    # 对每个候选意图进行编码并计算得分
    inputs = [
        tokenizer(
            f"Context: {context} Intent: {intent}",
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        for intent in candidate_intents
    ]
    
    intent_scores = []
    for input_pair in inputs:
        generated_ids = model.generate(
            input_ids=input_pair['input_ids'],
            attention_mask=input_pair['attention_mask'],
            max_length=50,
            num_return_sequences=1,
            num_beams=3  # 可以调整 beam search 参数
        )
        # 将生成的 token 转换为文本
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        similarity = calculate_similarity_score(context, generated_text)
        # score = calculate_similarity_score(model, context, generated_text)
        intent_scores.append(similarity)
        # print(generated_text)
    
    # 将 session_id、intent_description 和 score 打包在一起
    scored_sessions = list(zip(session_ids, candidate_intents, intent_scores))
    
    # 根据得分从高到低排序
    scored_sessions.sort(key=lambda x: x[2], reverse=True)
    
    # 返回得分最高的 top_k 个 session
    top_sessions = scored_sessions[:top_k]
    return top_sessions


def calculate_recall_at_k(test_data, top_k):
    """计算 Recall@top_k 分数"""
    correct_predictions = 0
    total_samples = len(test_data)

    for index, row in test_data.iterrows():
        label = row['label']
        predicted_intents = row['predicted_intent']
        
        # 如果 label 在预测的前 top_k 个意图列表中，则认为是正确的
        if label in predicted_intents:
            correct_predictions += 1

    recall_at_k = correct_predictions / total_samples
    return recall_at_k


def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def retrieve_personal_memory(persona_id):
    """加载个人 memory bank"""
    memory_bank_path = f'/mnt/ailabtemp/duyiming/mmt_tod/qa_summary_memory_bank/personal_{str(persona_id)}.json'
    return load_json(memory_bank_path)

def calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids):
    """计算意图的准确率"""
    correct_matches = sum(1 for session_score_item in intent_des_top_k_results if int(session_score_item[0]) in reference_session_ids)
    return correct_matches / len(reference_session_ids) if reference_session_ids else 0.0
    
def main():
    # 加载 T5 模型和分词器
    model_name = "/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/output_models/best_output_model"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    personal_data_dir = "/mnt/ailabtemp/duyiming/mmt_tod/clean_personal_dataset_0928_131/"
    personal_file_paths = os.listdir(personal_data_dir)
    
    confirmation_number = 0
    all_intent_acc = 0.0

    # 遍历所有个人文件
    for personal_file in tqdm(personal_file_paths, desc="Processing Files"):
        personal_dialogue = load_json(os.path.join(personal_data_dir, personal_file))
        personal_memory_bank = retrieve_personal_memory(personal_dialogue["persona_id"])

        # 遍历每个对话会话
        for session_data in personal_dialogue["sessions"]:
            try:
                if session_data.get("exist_confirmation"):
                    reference_session_ids = search_ground_truth_session_ids(
                        personal_dialogue, session_data["reference_dialogue_id"], session_data["session_id"]
                    )

                    # 提取当前会话文本
                    current_session_txt = extract_current_session(session_data)

                    # 检索与当前会话相关的 sessions
                    intent_des_top_k_results = retrieve_related_sessions(
                        current_session_txt, personal_memory_bank, session_data["session_id"], model, tokenizer, device, top_k=3
                    )

                    # 计算意图准确率
                    intent_acc = calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids)
                    confirmation_number += 1
                    all_intent_acc += intent_acc

                    print(f"Current Intent Accuracy: {intent_acc:.4f}")
                    print(f"Cumulative Accuracy: {all_intent_acc / confirmation_number:.4f}")
                    print("------------------------------")
                    
            except Exception as e:
                print(f"Error processing session {session_data['session_id']}: {e}")

    # 输出总体准确率
    if confirmation_number > 0:
        print(f"Final Average Accuracy: {all_intent_acc / confirmation_number:.4f}")
    else:
        print("No confirmation sessions found.")

if __name__ == "__main__":
    main()


# if __name__ == "__main__":


    # # # 加载 T5 模型和分词器
    # model_name = "/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/output_models/best_output_model"
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # top_k = 3
    # model.to(device)

    # personal_file_paths = os.listdir("/mnt/ailabtemp/duyiming/mmt_tod/clean_personal_dataset_0928_131/")
    # confirmation_number = 0
    # index = 0
    # all_intent_acc = 0.0
    # for personal_file in personal_file_paths:
    #     with open(f'/mnt/ailabtemp/duyiming/mmt_tod/clean_personal_dataset_0928_131/{personal_file}', 'r', encoding='utf-8') as file:
    #         personal_dialogue = json.load(file)
    #     file.close()
    #     personal_memory_bank_path = f'/mnt/ailabtemp/duyiming/mmt_tod/qa_summary_memory_bank/personal_{str(personal_dialogue["persona_id"])}.json'
    
    #     with open(personal_memory_bank_path, 'r') as file:
    #         personal_memory_bank = json.load(file)

    #     for session_data in personal_dialogue["sessions"]:
    #         try:
    #             if session_data["exist_confirmation"]:
    #                 reference_history_session_ids = search_ground_truth_session_ids(personal_dialogue, session_data["reference_dialogue_id"], session_data["session_id"])
    #                 # task_goals = session_data["task_goal"]

    #                 # task_slots = []
    #                 current_session_txt = extract_current_session(session_data)

    #                 ###
    #                 intent_des_top_k_results = retrieve_related_sessions(current_session_txt, personal_memory_bank, session_data["session_id"], model, tokenizer, device, 3)
    #                 # 
    #                 ###
                    
                    
    #                 intent_acc = 0.0
    #                 for session_score_item in intent_des_top_k_results:
    #                     if int(session_score_item[0]) in reference_history_session_ids:
    #                         intent_acc += 1
    #                 confirmation_number += 1
    #                 intent_acc = intent_acc/len(reference_history_session_ids)
    #                 print(intent_acc)
    #                 all_intent_acc+= intent_acc
    #                 print(all_intent_acc/confirmation_number)
    #                 print("------------------------------")
                    
    #         except Exception as e:
    #             print(e)
    # print(all_intent_acc/confirmation_number)
    # # 示例：加载数据并打印一些示例
    # file_path = '/mnt/ailabtemp/duyiming/duyiming/mmt_tod/full_intent_test_data_1110.csv'  # 请确保路径正确
    # # output_model_dir = "/mnt/ailabtemp/duyiming/duyiming/mmt_tod/intent_rag/output_models/"
    # # best_model_dir = output_model_dir + "best_output_model/"

    # # dataset = IntentDataset(file_path, tokenizer)
    # # train_loader, val_loader, test_loader = load_data(file_path)
    # test_data = load_test_data(file_path)
    #     # 进行预测并输出结果
    # predictions = []
    # for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
    #     context = row['context']
    #     candidate_intents = ast.literal_eval(row['candidate_intents'])
    #     # 预测意图
    #     predicted_intents = predict_intent(context, candidate_intents, model, tokenizer, device, top_k)
        
    #     predictions.append(predicted_intents)

    # # 将预测结果添加到 DataFrame
    # test_data['predicted_intent'] = predictions

    # # 计算 Recall@top_k
    # recall_at_k = calculate_recall_at_k(test_data, top_k)
    # print(f"Recall@{top_k}: {recall_at_k:.4f}")



    # 将预测结果添加到 DataFrame 并保存
    # test_data['predicted_intent'] = predictions
    # output_path = '/mnt/ailabtemp/duyiming/duyiming/mmt_tod/intent_rag/test/predicted_intents_output.csv'
    # test_data.to_csv(output_path, index=False)
    # print(f"预测结果已保存到 {output_path}")


        # # 手动创建 decoder_input_ids，通常以 <pad> token 开始
        # decoder_input_ids = torch.full(
        #     (1, 1), tokenizer.pad_token_id, device=device
        # )
        
        # with torch.no_grad():
        #     outputs = model(
        #         input_ids=input_pair['input_ids'],
        #         attention_mask=input_pair['attention_mask'],
        #         decoder_input_ids=decoder_input_ids
        #     )
            
        #     logits = outputs.logits
        #     probs = torch.softmax(logits, dim=-1)
            
        #     # 取概率的平均值作为得分
        #     score = probs.mean().item()
        #     intent_scores.append(score)