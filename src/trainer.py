# src/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """完整版训练器：记录训练/验证损失与准确率，支持早停，绘制对比曲线"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = self._get_device()
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录指标
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        
        # 数据集大小（外部传入）
        self.train_size = 0
        self.val_size = 0
    
    def _get_device(self) -> str:
        device_config = self.config['hardware']['device']
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
        logger.info(f"使用设备: {device}")
        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    
    def set_dataset_sizes(self, train_size: int, val_size: int):
        self.train_size = train_size
        self.val_size = val_size
        logger.info(f"数据集大小: 训练={train_size:,}, 验证={val_size:,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='训练', leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config['training']['gradient_clip'])
            self.optimizer.step()
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            avg_loss = total_loss / total_samples
            acc = correct / total * 100
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.1f}%'})
        
        return total_loss / total_samples, correct / total * 100
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / total_samples, correct / total * 100
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[float, float]:
        logger.info("开始训练...")
        total_start = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 保存最佳模型（按准确率）
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_loss = val_loss
                self._save_model('best')
                logger.info(f"  🏆 新最佳模型 | Val Acc: {val_acc:.2f}% | Loss: {val_loss:.4f}")
            
            # 早停检查
            if self._check_early_stopping(val_acc):
                logger.info(f"  ⚠️ 早停触发，停止训练")
                break
            
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1:2d}/{self.config['training']['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )
        
        total_time = time.time() - total_start
        self._save_model('final')
        
        logger.info("=" * 50)
        logger.info(f"训练完成 | 总时间: {total_time/60:.2f} min")
        logger.info(f"最佳验证准确率: {self.best_accuracy:.2f}%")
        logger.info(f"最佳验证损失: {self.best_loss:.4f}")
        logger.info("=" * 50)
        
        return total_time, self.best_accuracy
    
    def _save_model(self, model_type: str):
        if model_type == 'best':
            path = self.config['paths']['best_model']
            meta = {
                'accuracy': self.best_accuracy,
                'loss': self.best_loss,
                'epoch': len(self.train_losses),
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'train_size': self.train_size,
                'val_size': self.val_size
            }
        else:
            path = self.config['paths']['final_model']
            meta = {
                'final_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0.0,
                'final_loss': self.val_losses[-1] if self.val_losses else 0.0,
                'best_accuracy': self.best_accuracy,
                'epoch': len(self.train_losses),
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'train_size': self.train_size,
                'val_size': self.val_size
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metadata': meta
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def _check_early_stopping(self, current_acc: float) -> bool:
        patience = self.config['training']['early_stopping_patience']
        if patience <= 0:
            return False
        if current_acc > self.best_accuracy:
            self.early_stopping_counter = 0
            return False
        self.early_stopping_counter += 1
        return self.early_stopping_counter >= patience
    
    def generate_report(self, total_time: float, vocab_size: int, best_accuracy: float):
        report_path = self.config['paths']['report_file']
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n情感分析训练报告\n" + "=" * 60 + "\n")
            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {self.device}\n")
            f.write(f"训练集: {self.train_size:,} 条\n")
            f.write(f"验证集: {self.val_size:,} 条\n")
            f.write(f"词汇表大小: {vocab_size:,}\n")
            f.write(f"序列长度: {self.config['data']['max_sequence_length']}\n")
            f.write(f"模型: LSTM + Attention\n")
            f.write(f"最终训练准确率: {self.train_accuracies[-1]:.2f}%\n")
            f.write(f"最终验证准确率: {self.val_accuracies[-1]:.2f}%\n")
            f.write(f"最佳验证准确率: {best_accuracy:.2f}%\n")
            f.write(f"总训练时间: {total_time/60:.2f} 分钟\n")
        logger.info(f"报告已保存: {report_path}")
    
    def plot_training_curves(self):
        curve_path = self.config['paths']['curve_file']
        os.makedirs(os.path.dirname(curve_path), exist_ok=True)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(self.train_losses) + 1)   # 整数 1,2,3...
        
        # 左图：损失
        axes[0].plot(epochs, self.train_losses, 'b-o', label='训练损失')
        axes[0].plot(epochs, self.val_losses, 'r-s', label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('损失曲线对比')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 右图：准确率
        axes[1].plot(epochs, self.train_accuracies, 'b-o', label='训练准确率')
        axes[1].plot(epochs, self.val_accuracies, 'r-s', label='验证准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('准确率曲线对比')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"训练曲线已保存: {curve_path}")