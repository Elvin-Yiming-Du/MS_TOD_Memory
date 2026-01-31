# # 加载预训练的BERT模型和分词器
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(candidate_intents))
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

# 定义数据集和数据加载器
def split_dataset(data, train_ratio=0.8, val_ratio=0.2):
    """将数据集拆分为训练集、验证集和测试集"""
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    return random_split(data, [train_size, val_size, test_size])

def load_data(file_path, batch_size=4):
    """
    加载 CSV 文件并创建 T5 数据加载器
    - file_path: CSV 文件路径
    - batch_size: 每批次的数据量
    """
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dataset = IntentDataset(file_path, tokenizer)
    # 拆分数据集
    train_data, val_data, test_data = split_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)


    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

def save_model(model, tokenizer, epoch, output_dir="output_model"):
    """保存模型和分词器"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir} at epoch {epoch}")

def train(model, data_loader, optimizer, device):
    """训练模型"""
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def validate(model, data_loader, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def predict_intent(context, candidate_intents, model, tokenizer, device):
        model.eval()
        
        inputs = [
            tokenizer(
                f"Context: {context} Intent: {intent}",
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            for intent in candidate_intents
        ]
        
        intent_scores = []
        for input_pair in inputs:
            # 手动创建 decoder_input_ids，通常以 <pad> token 开始
            decoder_input_ids = torch.full(
                (1, 1), tokenizer.pad_token_id, device=device
            )
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_pair['input_ids'],
                    attention_mask=input_pair['attention_mask'],
                    decoder_input_ids=decoder_input_ids
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
                # 取概率的平均值作为得分
                score = probs.mean().item()
                intent_scores.append(score)

        best_intent_index = torch.tensor(intent_scores).argmax().item()
        return candidate_intents[best_intent_index]
        

if __name__ == "__main__":
    # 加载 T5 模型和分词器
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 加载数据并创建数据集

    # 示例：加载数据并打印一些示例
    file_path = '/mnt/ailabtemp/duyiming/duyiming/mmt_tod/full_intent_training_data_1110.csv'  # 请确保路径正确
    output_model_dir = "/mnt/ailabtemp/duyiming/duyiming/mmt_tod/intent_rag/output_models/"
    best_model_dir = output_model_dir + "best_output_model/"
    train_loader, val_loader, test_loader = load_data(file_path)

    # dataset = IntentDataset(data, tokenizer)
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 使用AdamW优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 设置参数
    epochs = 20
    save_every_n_epochs = 2
    best_loss = float('inf')
    

    # 开始训练
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")

        # 验证模型
        val_loss = validate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")

        # 每隔 n 个 epoch 保存一次模型
        if (epoch + 1) % save_every_n_epochs == 0:
            save_model(model, tokenizer, epoch + 1, output_dir=f'{output_model_dir}model_epoch_{epoch + 1}')

        # 检查是否为最佳模型，如果是则保存
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, tokenizer, epoch + 1, output_dir=best_model_dir)
            print(f"Best model updated with validation loss {best_loss:.4f}")

    print("Training complete.")
    print(f"Best model saved to {best_model_dir} with validation loss {best_loss:.4f}")

    # # 测试意图预测
    # context = "user:Hi, I have another question about my reservation.\n assistant:Hello! What would you like to know about your reservation?\n user:Can you confirm everything one more time? I just want to make sure."
    # candidate_intents = ['The user intends to reserve a single hotel room for one night, specifically for the following day, and is awaiting confirmation of availability from the assistant.', 'The user intends to confirm the status of their existing reservation at Khana Peena for March 1st at 2 pm for two people and expresses satisfaction upon receiving confirmation, indicating no further assistance is needed at the moment.', 'The user intends to confirm the details of their upcoming appointment at 18/8 Fine Men’s Salons in Palo Alto, and after receiving the appointment specifics from the assistant, they express satisfaction and gratitude for the confirmation.']
    # predicted_intent = predict_intent(context, candidate_intents, model, tokenizer, device)
    # print(f"Predicted intent: {predicted_intent}")




    # def train(model, data_loader, optimizer, device):
    #     model.train()
    #     total_loss = 0
        
    #     for batch in tqdm(data_loader):
    #         inputs, labels = batch
    #         inputs = {k: v.to(device) for k, v in inputs.items()}
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(**inputs, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item()
        
    #     avg_loss = total_loss / len(data_loader)
    #     print(f"Training loss: {avg_loss}")


    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model.to(device)

    # # 开始训练
    # for epoch in range(3):
    #     print(f"Epoch {epoch+1}")
    #     train(model, data_loader, optimizer, device)
