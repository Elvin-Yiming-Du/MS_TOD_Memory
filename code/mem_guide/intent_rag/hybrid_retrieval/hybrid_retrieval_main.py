import os
import json
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def retrieve_personal_memory(persona_id):
    """加载个人 memory bank"""
    memory_bank_path = f'/mnt/ailabtemp/duyiming/mmt_tod/qa_summary_memory_bank/personal_{str(persona_id)}.json'
    return load_json(memory_bank_path)

def extract_current_session(session):
    """提取当前会话的上下文"""
    session_context = ""
    for utterance in session["content"]:
        if utterance["is_confirmation"]:
            break
        session_context += f"{utterance['speaker']}:{utterance['utterance']}\n"
    return session_context

def search_ground_truth_session_ids(personal_dialogue, search_session_id, target_id):
    """查找历史对话中相关的 session_id"""
    reference_history_session_ids = []
    for session_candidate in personal_dialogue["sessions"]:
        if search_session_id == session_candidate["reference_dialogue_id"] and session_candidate["session_id"] != target_id:
            reference_history_session_ids.append(session_candidate["session_id"])
    return reference_history_session_ids

def calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids):
    """计算意图的准确率"""
    correct_matches = sum(1 for session_score_item in intent_des_top_k_results if int(session_score_item["session_id"]) in reference_session_ids)
    return correct_matches / len(reference_session_ids) if reference_session_ids else 0.0

def retrieve_related_intentions_bm25(context, candidate_intents, session_ids, top_k=10):
    """使用 BM25 进行初步检索"""
    tokenized_intents = [intent.split() for intent in candidate_intents]
    bm25 = BM25Okapi(tokenized_intents)
    tokenized_context = context.split()
    scores = bm25.get_scores(tokenized_context)
    
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    top_k_results = [
        {
            "session_id": session_ids[idx],
            "candidate": candidate_intents[idx],
            "score": scores[idx]
        }
        for idx in top_k_indices
    ]
    return top_k_results

def rerank_with_bert(context, candidates, model, tokenizer, device):
    """使用 BERT 对 BM25 返回的候选集进行重新排序"""
    inputs = [f"{context} [SEP] {candidate}" for candidate in candidates]
    encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1)[:, 1]  # 获取正类的概率分数

    return scores.cpu().numpy()

def hybrid_retrieve_intentions(context, candidate_intents, session_ids, model, tokenizer, device, top_k_bm25=6, top_k_final=3):
    """混合检索方法：BM25 + BERT 重排序"""
    # Step 1: 使用 BM25 进行初步检索
    bm25_results = retrieve_related_intentions_bm25(context, candidate_intents, session_ids, top_k=top_k_bm25)
    
    # Step 2: 提取 BM25 返回的候选集
    bm25_candidates = [result["candidate"] for result in bm25_results]
    bm25_session_ids = [result["session_id"] for result in bm25_results]
    
    # Step 3: 使用 BERT 对 BM25 返回的候选集进行重排序
    bert_scores = rerank_with_bert(context, bm25_candidates, model, tokenizer, device)
    
    # Step 4: 构建 BERT 重排序后的结果
    final_results = [
        {
            "session_id": bm25_session_ids[idx],
            "candidate": bm25_candidates[idx],
            "score": bert_scores[idx]
        }
        for idx in range(len(bm25_candidates))
    ]
    
    # Step 5: 根据 BERT 分数排序并选取 top_k_final 个结果
    final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k_final]
    return final_results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/output_model")
    model.to(device)
    
    personal_data_dir = "/mnt/ailabtemp/duyiming/mmt_tod/clean_personal_dataset_0928_131/"
    personal_file_paths = os.listdir(personal_data_dir)
    
    confirmation_number = 0
    all_intent_acc = 0.0

    for personal_file in tqdm(personal_file_paths, desc="Processing Files"):
        personal_dialogue = load_json(os.path.join(personal_data_dir, personal_file))
        personal_memory_bank = retrieve_personal_memory(personal_dialogue["persona_id"])

        for session_data in personal_dialogue["sessions"]:
            if session_data.get("exist_confirmation"):
                reference_session_ids = search_ground_truth_session_ids(
                    personal_dialogue, session_data["reference_dialogue_id"], session_data["session_id"]
                )
                candidate_intents = []
                session_ids = []

                for one_session in personal_memory_bank["sessions"]:
                    for k, v in one_session.items():
                        if int(k) > int(session_data["session_id"]) - 1:
                            break
                        candidate_intents.append(v["intent_description"])
                        session_ids.append(k)

                current_session_txt = extract_current_session(session_data)

                intent_des_top_k_results = hybrid_retrieve_intentions(
                    current_session_txt, candidate_intents, session_ids, model, tokenizer, device
                )

                intent_acc = calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids)
                confirmation_number += 1
                all_intent_acc += intent_acc

                print(f"Current Intent Accuracy: {intent_acc:.4f}")
                print(f"Cumulative Accuracy: {all_intent_acc / confirmation_number:.4f}")

    if confirmation_number > 0:
        print(f"Final Average Accuracy: {all_intent_acc / confirmation_number:.4f}")

if __name__ == "__main__":
    main()
