from transformers import pipeline, BertTokenizer, BertForSequenceClassification

# 加载训练好的模型和分词器
model_path = "/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/output_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

def retrieve_best_intent(session, candidate_intents):
    best_intent = None
    best_score = -float('inf')
    
    for intent in candidate_intents:
        input_text = f"{session} [SEP] {intent}"
        scores = classifier(input_text)
        pos_score = scores[0][1]['score']  # 获取正样本的置信度
        if pos_score > best_score:
            best_score = pos_score
            best_intent = intent
    
    return best_intent

# 示例检索任务
session = "用户：我想查一下最近的订单"
candidate_intents = ["查询订单状态", "取消订单", "修改地址"]
best_match = retrieve_best_intent(session, candidate_intents)
print(f"最佳匹配的意图: {best_match}")