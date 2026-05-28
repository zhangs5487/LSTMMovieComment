# src/data_preprocessing.py
import re
import json
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """生产级文本预处理器"""
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.vocab: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.word_counter = Counter()
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        生产级文本预处理
        - 转小写
        - 移除特殊字符
        - 保留基本标点
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 转小写
        text = text.lower()
        
        # 移除非字母数字和基本标点
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
        
        # 处理多个空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 分词
        words = text.split()
        
        return words
    
    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        构建生产级词汇表
        - 增量处理大文件
        - 定期日志
        - 保存中间状态
        """
        logger.info(f"构建词汇表 (最大词汇数: {self.max_vocab_size:,})")
        
        for i, text in enumerate(texts):
            if i % 10000 == 0 and i > 0:
                logger.info(f"  已处理 {i:,} 条文本，当前词汇数: {len(self.word_counter):,}")
            
            words = self.preprocess_text(text)
            self.word_counter.update(words)
        
        logger.info(f"词汇统计完成！总词汇数: {len(self.word_counter):,}")
        
        # 构建词汇表
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(self.word_counter.most_common(self.max_vocab_size - 2), start=2):
            vocab[word] = i
        
        self.vocab = vocab
        logger.info(f"词汇表构建完成！最终词汇数: {len(vocab):,}")
        
        return vocab
    
    def text_to_sequence(self, text: str, max_length: int) -> List[int]:
        """将单条文本转换为序列"""
        words = self.preprocess_text(text)
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # 截断或填充
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence += [self.vocab['<PAD>']] * (max_length - len(sequence))
        
        return sequence
    
    def save_vocab(self, filepath: str):
        """保存词汇表到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"词汇表已保存到: {filepath}")
        logger.info(f"词汇表大小: {len(self.vocab):,}")
    
    def load_vocab(self, filepath: str):
        """从文件加载词汇表"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"词汇表文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        logger.info(f"词汇表已加载: {len(self.vocab):,} 个词汇")
        return self.vocab
    
    @classmethod
    def from_existing_vocab(cls, vocab_file: str) -> 'TextPreprocessor':
        """从现有词汇表创建预处理器"""
        processor = cls()
        processor.load_vocab(vocab_file)
        return processor