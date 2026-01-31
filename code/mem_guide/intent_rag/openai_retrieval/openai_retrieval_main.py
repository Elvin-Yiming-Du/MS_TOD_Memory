import os
import json
import torch
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import os

# Function to get embeddings from OpenAI in batches
def get_embeddings(texts, batch_size=10):

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"  # Use the appropriate model
        )
        embeddings.extend([datum.embedding for datum in response.data])
    return embeddings

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

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def retrieve_personal_memory(persona_id):
    """加载个人 memory bank"""
    memory_bank_path = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_memory_bank_sum/persona_{str(persona_id)}.json'
    return load_json(memory_bank_path)

def calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids):
    """计算意图的准确率"""
    correct_matches = sum(1 for session_score_item in intent_des_top_k_results if int(session_score_item["session_id"]) in reference_session_ids)
    return correct_matches / len(reference_session_ids) if reference_session_ids else 0.0



def batch_retrieve_related_sessions(context, candidate_intents, session_ids, top_k=3):
    # max_length = 32768
    # task_name_to_instruct = {"example": "Given a question, retrieve intent desscription passages that answer the question",}
    # query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    # passage_prefix = ""
    # 获取当前 context 的嵌入，并确保转换为 NumPy 数组
    current_embedding = np.array(get_embeddings([context])[0])
    
    # 如果 current_embedding 是一维数组，则调整为二维
    if len(current_embedding.shape) == 1:
        current_embedding = current_embedding.reshape(1, -1)
    
    # 获取历史意图的嵌入，并将其转换为 NumPy 数组
    history_embeddings = np.array(get_embeddings(candidate_intents))
    
    # 如果 history_embeddings 是一维数组，则调整为二维
    if len(history_embeddings.shape) == 1:
        history_embeddings = history_embeddings.reshape(1, -1)
    
    # 确保每个 history_embedding 都是二维数组
    history_embeddings = [embedding.reshape(1, -1) if len(embedding.shape) == 1 else embedding for embedding in history_embeddings]
    
    # print(current_embedding.shape())
    # Compute the cosine similarity between the current intent and each historical intent
    scores = [cosine_similarity(current_embedding, history_embedding)[0][0] for history_embedding in history_embeddings]
    scores = np.array(scores).flatten()
    top_k_indices = np.argsort(scores)[-top_k:][::-1]  # 按分数降序排序
    
    # 构建输出结果
    top_k_results = [
        {
            "session_id": session_ids[idx],
            "candidate": candidate_intents[idx],
            "score": scores[idx]
        }
        for idx in top_k_indices
    ]
    return top_k_results

def get_t5_embeddings(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def main():
    # 加载 T5 模型和分词器
    # Each query needs to be accompanied by an corresponding instruction describing the task.
    TOP_K = [3,5,10]
    top_k_results = {}
    output_results = {}
    total_recall_results = {}
    for top_k in TOP_K:
        top_k_results[str(top_k)] = 0.0
        output_results[str(top_k)] = []
        total_recall_results[str(top_k)] = 0.0
    # load model with tokenizer
    personal_data_dir = "/Users/duyiming/Documents/YimingDU/tod_multi_turn/personal_tod_memory_evaluation_131"
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
                    candidate_intents = []
                    session_ids = []
                    # 提取 memory_bank 中所有的 intent_description 和 session_id
                    for one_session in personal_memory_bank["sessions"]:
                        if one_session["session_id"] > int(session_data["session_id"]) - 1:
                            break
                        candidate_intents.append(one_session["summary"])
                        session_ids.append(one_session["session_id"])
                    # 提取当前会话文本
                    current_session_txt = extract_current_session(session_data)

                    # 批量检索与当前会话相关的 sessions
                    intent_des_top_k_results = batch_retrieve_related_sessions(
                        current_session_txt, candidate_intents, session_ids, top_k=10
                    )

                    # 计算意图准确率
                    confirmation_number += 1  
                    for top_k in TOP_K:
                        temp_retrieve_results = intent_des_top_k_results[:top_k]
                        intent_acc = calculate_intent_accuracy(temp_retrieve_results, reference_session_ids)
                        top_k_results[str(top_k)] += intent_acc
                        output_record={}
                        output_record["persona_id"] = personal_dialogue["persona_id"]
                        output_record["session_id"] = session_data["session_id"]
                        output_record["retrieval_results"] = temp_retrieve_results
                        output_record["reference_session_ids"] = reference_session_ids
                        output_record[f'recall@{str(TOP_K)}'] = float(intent_acc)
                        output_results[str(top_k)].append(output_record)
                        # print(f"Cumulative Accuracy: {top_k_results[str(top_k)] / confirmation_number:.4f}")
                        # print("------------------------------")   

                    all_intent_acc += intent_acc

                    print(f"Current Intent Accuracy: {intent_acc:.4f}")
                    print(f"Cumulative Accuracy: {all_intent_acc / confirmation_number:.4f}")
                    print("------------------------------")
                    
            except Exception as e:
                print(f"Error processing session {session_data.get('session_id', 'Unknown')}: {e}")

    # 输出总体准确率
    if confirmation_number > 0:
        for top_k in TOP_K:
            print(f"Final Average Accuracy: { top_k_results[str(top_k)] / confirmation_number:.4f}")
    else:
        print("No confirmation sessions found.")

    for top_k in TOP_K:
        output_data_file = f'/Users/duyiming/Documents/YimingDU/tod_multi_turn/intermediate_results/retrieval/sum_text_embedding_3_small_{str(top_k)}.jsonl'    
        with open(output_data_file, 'w') as file:
            for item in output_results[str(top_k)]:
                json_object = json.dumps(item)
                file.write(json_object + '\n')

if __name__ == "__main__":
    main()