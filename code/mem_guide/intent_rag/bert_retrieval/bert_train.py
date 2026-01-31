import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import json

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# 数据预处理函数
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = examples["label"]
    return tokenized

def prepare_bert_training_data(input_file, output_csv):
    """
    从 JSONL 文件加载数据，并生成 BERT 的训练数据格式。
    输入为 session + intent_description，输出为 pos_or_neg。
    """
    data = []
    
    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            session = entry.get("session", "")
            intent_description = entry.get("intent_description", "")
            label = 1 if entry.get("pos_or_neg", False) else 0
            
            # 构造 BERT 输入格式: session [SEP] intent_description
            input_text = f"{session} [SEP] {intent_description}"
            data.append({"text": input_text, "label": label})
    
    # 转换为 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Training data saved to {output_csv}")


# 创建 Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    # 指定文件路径
    input_file = '/mnt/ailabtemp/duyiming/mmt_tod/pos_neg_test_memory_intent_descriptions.jsonl'
    output_csv = '/mnt/ailabtemp/duyiming/mmt_tod/bert_training_data.csv'

    # 生成 BERT 训练数据
    prepare_bert_training_data(input_file, output_csv)
    
    # 加载数据集
    df = pd.read_csv(output_csv)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # 对训练集和验证集进行分词
    def tokenize_data(df):
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return encodings, torch.tensor(labels)

    # 分词并创建 Dataset 对象
    train_encodings, train_labels = tokenize_data(train_df)
    eval_encodings, eval_labels = tokenize_data(eval_df)


    train_dataset = CustomDataset(train_encodings, train_labels)
    eval_dataset = CustomDataset(eval_encodings, eval_labels)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/bert_train_models",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    # 使用 Trainer API 进行训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练并保存模型
    trainer.train()
    trainer.save_model("/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/output_model")
    tokenizer.save_pretrained("/mnt/ailabtemp/duyiming/mmt_tod/intent_rag/output_model")
    print("Model training complete and saved.")
