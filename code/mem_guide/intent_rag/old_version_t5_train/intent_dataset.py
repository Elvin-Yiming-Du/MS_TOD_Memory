import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class IntentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        初始化 T5 数据集。
        - file_path: CSV 文件路径
        - tokenizer: T5 分词器
        - max_length: 最大序列长度
        """
        # 加载 CSV 文件
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = row['context']
        candidate_intents = eval(row['candidate_intents'])  # 将字符串列表解析为列表
        label = row['label']
        
        # 构建 T5 输入格式
        input_text = f"Context: {context} Candidates: {' | '.join(candidate_intents)}"
        target_text = label

        # 对输入进行编码
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        labels = self.tokenizer(
            target_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )['input_ids']

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }, labels.squeeze()