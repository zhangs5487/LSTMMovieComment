# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from typing import List, Dict, Optional

from src.data_preprocessing import TextPreprocessor
import logging

logger = logging.getLogger(__name__)

class IMDBDataset(Dataset):
    """生产级IMDB数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 preprocessor: TextPreprocessor, max_length: int):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        logger.info(f"创建数据集: {len(texts)} 条样本")
        logger.info(f"序列长度: {max_length}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 转换为序列
        sequence = self.preprocessor.text_to_sequence(text, self.max_length)
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataModule:
    """数据管理模块"""
    
    def __init__(self, config: dict):
        self.config = config
        self.preprocessor = None
        self.train_df = None
        self.test_df = None
    
    def load_data(self):
        """加载数据"""
        train_path = self.config['data']['train_path']
        test_path = self.config['data']['test_path']
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("数据文件不存在，请先运行 download_imdb.py")
        
        logger.info(f"加载训练数据: {train_path}")
        self.train_df = pd.read_csv(train_path, encoding=self.config['data']['encoding'])
        
        logger.info(f"加载测试数据: {test_path}")
        self.test_df = pd.read_csv(test_path, encoding=self.config['data']['encoding'])
        
        logger.info(f"训练集大小: {len(self.train_df):,} 条")
        logger.info(f"测试集大小: {len(self.test_df):,} 条")
        
        return self.train_df, self.test_df
    
    def build_preprocessor(self):
        """构建预处理器"""
        self.preprocessor = TextPreprocessor(
            max_vocab_size=self.config['data']['max_vocab_size']
        )
        
        # 仅用训练集构建词汇表
        logger.info("使用训练集构建词汇表...")
        vocab = self.preprocessor.build_vocab(self.train_df['text'].tolist())
        
        # 保存词汇表
        vocab_file = self.config['paths']['vocab_file']
        self.preprocessor.save_vocab(vocab_file)
        
        return self.preprocessor
    
    def create_datasets(self):
        """创建数据集"""
        if self.preprocessor is None:
            raise ValueError("需要先调用 build_preprocessor()")
        
        max_length = self.config['data']['max_sequence_length']
        
        logger.info("创建训练数据集...")
        train_dataset = IMDBDataset(
            texts=self.train_df['text'].tolist(),
            labels=self.train_df['label'].tolist(),
            preprocessor=self.preprocessor,
            max_length=max_length
        )
        
        logger.info("创建测试数据集...")
        test_dataset = IMDBDataset(
            texts=self.test_df['text'].tolist(),
            labels=self.test_df['label'].tolist(),
            preprocessor=self.preprocessor,
            max_length=max_length
        )
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, test_dataset: Dataset):
        """创建数据加载器"""
        batch_size = self._get_batch_size()
        
        logger.info(f"创建数据加载器 (batch size: {batch_size})")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows下设置为0避免问题
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"训练批次: {len(train_loader)}")
        logger.info(f"测试批次: {len(test_loader)}")
        
        return train_loader, test_loader
    
    def _get_batch_size(self) -> int:
        """根据GPU内存动态调整batch size"""
        hardware_config = self.config['hardware']
        
        if not torch.cuda.is_available():
            logger.warning("GPU不可用，使用CPU模式")
            return hardware_config['batch_size']['default']
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"检测到GPU内存: {gpu_memory:.1f} GB")
        
        # 根据内存阈值选择batch size
        thresholds = hardware_config['batch_size']['gpu_memory_thresholds']
        for threshold in sorted(thresholds, key=lambda x: x['threshold_gb'], reverse=True):
            if gpu_memory >= threshold['threshold_gb']:
                return threshold['batch_size']
        
        return hardware_config['batch_size']['default']