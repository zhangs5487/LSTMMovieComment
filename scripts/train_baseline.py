# scripts/train_baseline.py
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import torch
import torch.nn as nn
from src.dataset import DataModule
from src.trainer import Trainer

# 简单的 LSTM（无注意力）
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.lstm = nn.LSTM(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=config['bidirectional']
        )
        # 双向LSTM需要乘以2
        lstm_output_dim = config['hidden_dim'] * (2 if config['bidirectional'] else 1)
        self.fc = nn.Linear(lstm_output_dim, config['num_classes'])
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output

def main():
    # ✅ 修复：指定 utf-8 编码
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 数据准备
    data_module = DataModule(config)
    train_df, test_df = data_module.load_data()
    preprocessor = data_module.build_preprocessor()
    train_dataset, test_dataset = data_module.create_datasets()
    train_loader, test_loader = data_module.create_dataloaders(train_dataset, test_dataset)
    
    # 模型
    vocab_size = len(preprocessor.vocab)
    model = SimpleLSTM(vocab_size, config['model'])
    
    # 训练
    trainer = Trainer(model, config)
    trainer.set_dataset_sizes(len(train_dataset), len(test_dataset))
    trainer.train(train_loader, test_loader)

if __name__ == "__main__":
    main()