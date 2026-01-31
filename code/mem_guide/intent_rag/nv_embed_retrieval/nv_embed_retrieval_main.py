import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

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

def calculate_similarity_score(context, generated_text):
    """基于 Jaccard 相似度的计算"""
    set1 = set(context.split())
    set2 = set(generated_text.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    return similarity

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
    correct_matches = sum(1 for session_score_item in intent_des_top_k_results if int(session_score_item["session_id"]) in reference_session_ids)
    return correct_matches / len(reference_session_ids) if reference_session_ids else 0.0

def find_most_similar_intent(generated_text, candidate_intents):
    """
    根据生成的文本 (generated_text) 找到与候选 intents 中最相关的一个，并返回其分数。
    """
    # 将 generated_text 和所有 candidate_intents 合并到一起
    all_texts = [generated_text] + candidate_intents

    # 使用 TF-IDF 向量化文本
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()

    # 计算 generated_text 与所有 candidate intents 的余弦相似度
    similarity_scores = cosine_similarity([vectors[0]], vectors[1:])[0]

    # 找到相似度最高的 intent 及其分数
    best_match_index = similarity_scores.argmax()
    best_match_score = similarity_scores[best_match_index]
    best_match_intent = candidate_intents[best_match_index]

    return best_match_intent, best_match_score

def batch_retrieve_related_sessions(context, candidate_intents, model, session_ids, device, top_k=3):
    max_length = 32768
    task_name_to_instruct = {"example": "Given a question, retrieve intent desscription passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    passage_prefix = ""
    query_embeddings = model.encode([context], instruction=query_prefix, max_length=max_length).to(device)
    passage_embeddings = model.encode(candidate_intents, instruction=passage_prefix, max_length=max_length).to(device)
    scores = (query_embeddings @ passage_embeddings.T) * 100
    scores = scores.flatten()  # 转换为一维数组
    # 将 scores 从 GPU 移动到 CPU
    scores = scores.cpu().numpy()
    # 获取 top_k 最高分的索引
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


def main():
    # 加载 T5 模型和分词器
    # Each query needs to be accompanied by an corresponding instruction describing the task.


   
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # load model with tokenizer
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).to(device)
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
                    candidate_intents = []
                    session_ids = []
                    # 提取 memory_bank 中所有的 intent_description 和 session_id
                    for one_session in personal_memory_bank["sessions"]:
                        for k, v in one_session.items():
                            if int(k) > int(session_data["session_id"]) - 1:
                                break
                            candidate_intents.append(v["intent_description"])
                            session_ids.append(k)
                    # 提取当前会话文本
                    current_session_txt = extract_current_session(session_data)

                    # 批量检索与当前会话相关的 sessions
                    intent_des_top_k_results = batch_retrieve_related_sessions(
                        current_session_txt, candidate_intents, model, session_ids, device, top_k=3
                    )

                    # 计算意图准确率
                    intent_acc = calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids)
                    confirmation_number += 1
                    all_intent_acc += intent_acc

                    print(f"Current Intent Accuracy: {intent_acc:.4f}")
                    print(f"Cumulative Accuracy: {all_intent_acc / confirmation_number:.4f}")
                    print("------------------------------")
                    
            except Exception as e:
                print(f"Error processing session {session_data.get('session_id', 'Unknown')}: {e}")

    # 输出总体准确率
    if confirmation_number > 0:
        print(f"Final Average Accuracy: {all_intent_acc / confirmation_number:.4f}")
    else:
        print("No confirmation sessions found.")

if __name__ == "__main__":
    main()