# src/model.py
import torch
import torch.nn as nn
from typing import Dict, Any
import logging
import os
logger = logging.getLogger(__name__)

class LSTMWithAttention(nn.Module):
    """生产级LSTM+Attention模型"""
    
    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embedding = nn.Embedding(
            vocab_size, 
            config['embed_dim'], 
            padding_idx=0
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            config['embed_dim'],
            config['hidden_dim'],
            batch_first=True,
            bidirectional=config['bidirectional']
        )
        
        # 注意力层
        hidden_dim_multiplier = 2 if config['bidirectional'] else 1
        self.attention = nn.Linear(
            config['hidden_dim'] * hidden_dim_multiplier, 
            1
        )
        
        # 分类层
        self.fc = nn.Linear(
            config['hidden_dim'] * hidden_dim_multiplier,
            config['num_classes']
        )
        
        # 正则化
        self.dropout = nn.Dropout(config['dropout_rate'])
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"模型初始化完成")
        logger.info(f"  词汇表大小: {vocab_size:,}")
        logger.info(f"  嵌入维度: {config['embed_dim']}")
        logger.info(f"  LSTM隐藏层: {config['hidden_dim']} ({'双向' if config['bidirectional'] else '单向'})")
        logger.info(f"  总参数量: {self.count_parameters():,}")
    
    def _init_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def count_parameters(self) -> int:
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: [batch_size, seq_length]
        Returns:
            logits: [batch_size, num_classes]
        """
        # 嵌入层
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM层
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # 注意力层
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)    # [batch, hidden_dim*2]
        
        # Dropout + 分类层
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)  # [batch, num_classes]
        
        return logits
    
    def save_pretrained(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'vocab_size': self.embedding.num_embeddings
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    @classmethod
    def from_pretrained(cls, filepath: str, device: str = 'cpu'):
        """加载预训练模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            config=checkpoint['config']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"模型已加载: {filepath}")
        logger.info(f"设备: {device}")
        
        return model