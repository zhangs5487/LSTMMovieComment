# src/predictor.py
import torch
import json
import logging
from typing import List, Union, Dict
from pathlib import Path

from src.model import LSTMWithAttention
from src.data_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """情感分析预测器"""
    
    def __init__(
        self, 
        model_path: str, 
        vocab_path: str, 
        config: dict,
        use_gpu: bool = True
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            vocab_path: 词汇表JSON文件路径
            config: 模型配置字典（从config.yaml加载）
            use_gpu: 是否使用GPU
        """
        self.config = config
        
        # 设置设备
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("使用CPU")
        
        # 加载预处理器（从保存的词汇表文件）
        self.preprocessor = TextPreprocessor(
            max_vocab_size=config['data']['max_vocab_size']
        )
        self.preprocessor.load_vocab(vocab_path)
        vocab_size = len(self.preprocessor.vocab)
        logger.info(f"词汇表大小: {vocab_size}")
        
        # 获取最大序列长度
        self.max_length = config['data']['max_sequence_length']
        logger.info(f"最大序列长度: {self.max_length}")
        
        # 加载模型
        self.model = self._load_model(model_path, vocab_size)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str, vocab_size: int) -> LSTMWithAttention:
        """加载模型权重"""
        # 准备模型配置字典（与训练时的格式一致）
        model_config = {
            'embed_dim': self.config['model']['embed_dim'],
            'hidden_dim': self.config['model']['hidden_dim'],
            'num_classes': self.config['model']['num_classes'],
            'dropout_rate': self.config['model'].get('dropout_rate', 0.5),
            'bidirectional': self.config['model'].get('bidirectional', True)
        }
        
        # 创建模型实例
        model = LSTMWithAttention(
            vocab_size=vocab_size,
            config=model_config
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 兼容两种保存格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        logger.info(f"模型加载成功: {model_path}")
        
        return model
    
    def predict(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        预测单条或多条文本的情感
        
        Args:
            texts: 单条文本字符串或文本列表
            
        Returns:
            单条: {'text': str, 'sentiment': str, 'confidence': float, 'probabilities': dict}
            多条: List[上述字典]
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        results = []
        for text in texts:
            # 预处理：文本转序列
            sequence = self.preprocessor.text_to_sequence(text, self.max_length)
            input_ids = torch.tensor([sequence], dtype=torch.long).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_ids)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()
            
            # 解析结果
            sentiment = "positive" if pred_class == 1 else "negative"
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'negative': probs[0][0].item(),
                    'positive': probs[0][1].item()
                }
            })
        
        return results[0] if single else results
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[Dict]:
        """批量预测（更高效）"""
        results = []
        total = len(texts)
        
        # 预先编码所有文本
        sequences = [self.preprocessor.text_to_sequence(t, self.max_length) for t in texts]
        
        # 分批处理
        for i in range(0, total, batch_size):
            batch_sequences = sequences[i:i+batch_size]
            input_ids = torch.tensor(batch_sequences, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1)[0]
            
            for j, idx in enumerate(range(i, min(i+batch_size, total))):
                sentiment = "positive" if preds[j].item() == 1 else "negative"
                results.append({
                    'text': texts[idx],
                    'sentiment': sentiment,
                    'confidence': confs[j].item(),
                    'probabilities': {
                        'negative': probs[j][0].item(),
                        'positive': probs[j][1].item()
                    }
                })
            
            if show_progress:
                logger.info(f"批量预测进度: {min(i+batch_size, total)}/{total}")
        
        return results
    
    def evaluate_on_testset(self, test_loader) -> Dict:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试集DataLoader
            
        Returns:
            {'accuracy': float, 'total_samples': int, 'correct': int, 'predictions': list}
        """
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total * 100
        
        logger.info(f"测试集评估结果:")
        logger.info(f"  总样本数: {total}")
        logger.info(f"  正确数: {correct}")
        logger.info(f"  准确率: {accuracy:.2f}%")
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct': correct,
            'predictions': all_preds,
            'labels': all_labels
        }