import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import Levenshtein

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

# def calculate_similarity_score(context, generated_text):
#     """基于 Levenshtein 编辑距离的相似度"""
#     similarity = Levenshtein.ratio(context, generated_text)
#     return similarity


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
    memory_bank_path = f'/mnt/ailabtemp/duyiming/duyiming/mmt_tod/qa_summary_memory_bank/personal_{str(persona_id)}.json'
    return load_json(memory_bank_path)

def calculate_intent_accuracy(intent_des_top_k_results, reference_session_ids):
    """计算意图的准确率"""
    correct_matches = sum(1 for session_score_item in intent_des_top_k_results if int(session_score_item[0]) in reference_session_ids)
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

def batch_retrieve_related_sessions(context, candidate_intents, model, tokenizer, device, top_k=3):
    # """批量检索与当前会话相关的 sessions"""
    # model.eval()
    
    # # 批量处理输入数据
    # inputs = tokenizer(
    #     [f"Context: {context} Intent: {intent}" for intent in candidate_intents],
    #     max_length=512,
    #     padding='max_length',
    #     truncation=True,
    #     return_tensors='pt'
    # ).to(device)
    
    # with torch.no_grad():
    #     outputs = model.generate(
    #         input_ids=inputs['input_ids'],
    #         attention_mask=inputs['attention_mask'],
    #         max_length=128,
    #         num_return_sequences=1,
    #         num_beams=3,
    #     )
    
    # # 将生成的文本解码
    # generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # # 计算相似度得分（使用余弦相似度）
    # intent_scores = []
    # for generated_text, intent in zip(generated_texts, candidate_intents):
    #     score = calculate_similarity_score(context, generated_text)
    #     intent_scores.append(score)
    
    # # 将 session_id 和得分打包，并根据得分排序
    # scored_sessions = list(zip(range(len(candidate_intents)), candidate_intents, intent_scores))
    # scored_sessions.sort(key=lambda x: x[2], reverse=True)
    
    # return scored_sessions[:top_k]
    # 计算与候选 intents 的相似度，并找到最相关的 intent
    # 调用生成器模型生成文本
    model.eval()
    inputs = tokenizer(
        [f"Context: {context} Intent: {intent}" for intent in candidate_intents],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_return_sequences=1,
            num_beams=3,
        )

    # 将生成的文本解码
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 计算与候选 intents 的相似度，并找到最相关的 intent
    scored_sessions = []
    for generated_text in generated_texts:
        best_intent, best_score = find_most_similar_intent(generated_text, candidate_intents)
        scored_sessions.append((best_intent, best_score))

    # 根据得分排序并返回 top_k 结果
    scored_sessions.sort(key=lambda x: x[1], reverse=True)
    top_k_results = scored_sessions[:top_k]

    print("Top K Results:", top_k_results)
    return top_k_results

def main():
    # 加载 T5 模型和分词器
    model_name = "/mnt/ailabtemp/duyiming/duyiming/mmt_tod/intent_rag/output_models/best_output_model"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    personal_data_dir = "/mnt/ailabtemp/duyiming/duyiming/mmt_tod/clean_personal_dataset_0928_131/"
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
                        current_session_txt, candidate_intents, model, tokenizer, device, top_k=3
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