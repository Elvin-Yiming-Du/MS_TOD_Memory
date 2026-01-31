import os
import json
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def retrieve_personal_memory(persona_id):
    """加载个人 memory bank"""
    memory_bank_path = f'D:/Elvin/YimingDU/tod_multi_turn/qa_summary_memory_bank/personal_{str(persona_id)}.json'
    return load_json(memory_bank_path)

def extract_current_session(session):
    """提取当前会话的上下文"""
    confirmation_state = session.get("confirmation_state", False)
    session_context = ""
    for i, utterance in enumerate(session["turns"]):
        if i == confirmation_state["confirmation_utterance_id"]:
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

def retrieve_related_intentions_bm25(context, candidate_intents, session_ids, top_k=3):
    """使用 BM25 进行检索"""
    # 对候选意图进行分词处理
    tokenized_intents = [intent.split() for intent in candidate_intents]
    bm25 = BM25Okapi(tokenized_intents)
    
    # 对输入 context 进行分词
    tokenized_context = context.split()
    
    # 计算 BM25 分数
    scores = bm25.get_scores(tokenized_context)
    
    # # 获取 top_k 最高分的索引
    # top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # 构建输出结果
    top_k_results = [
        {
            "session_id": str(idx+1),
            "candidate": candidate_intents[idx],
            "score": scores[idx]
        }
        for idx in range(len(session_ids))
    ]

    # 按照 score 从高到低排序
    sorted_results = sorted(
        top_k_results, 
        key=lambda x: x["score"], 
        reverse=True
    )
    top_k_results = sorted_results[:top_k]
    return top_k_results


    # # 构建输出结果
    # top_k_results = [
    #     {
    #         "session_id": session_ids[idx],
    #         "candidate": candidate_intents[idx],
    #         "score": scores[idx]
    #     }
    #     for idx in top_k_indices
    # ]
    # return top_k_results

def main():
    personal_data_dir = "D:/Elvin/YimingDU/tod_multi_turn/intent_rag/data"
    personal_file_paths = os.listdir(personal_data_dir)
    
    confirmation_number = 0
    all_intent_acc = 0.0

    session_candidates_statistic = 0

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
                    session_candidates_statistic += len(candidate_intents)
                    # 提取当前会话文本
                    current_session_txt = extract_current_session(session_data)

                    # 使用 BM25 检索相关的意图
                    intent_des_top_k_results = retrieve_related_intentions_bm25(
                        current_session_txt, candidate_intents, session_ids, top_k=10
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
    print(session_candidates_statistic/confirmation_number)

if __name__ == "__main__":
    main()
