# scripts/train.py
import sys
import os
import time
import logging
import yaml
from pathlib import Path

# ✅ 设置项目根目录到Python路径（关键！）
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ 统一绝对导入
from src.data_preprocessing import TextPreprocessor
from src.dataset import DataModule
from src.model import LSTMWithAttention
from src.trainer import Trainer

def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建结果目录
    for dir_path in config['paths'].values():
        if isinstance(dir_path, str):
            if dir_path.endswith(('.pth', '.json', '.png', '.txt')):
                os.makedirs(os.path.dirname(dir_path), exist_ok=True)
            elif os.path.isdir(dir_path) or not os.path.splitext(dir_path)[1]:  # 目录或无扩展名
                os.makedirs(dir_path, exist_ok=True)
    
    return config

def main():
    """主训练函数"""
    # 设置日志
    logger = setup_logging('results/logs')
    
    # 加载配置
    config_path = str(Path(__file__).parent.parent / 'configs' / 'config.yaml')
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        logger.error("请确保在 sentiment_analysis/ 目录下运行此脚本")
        return
    
    config = load_config(config_path)
    
    logger.info("=" * 60)
    logger.info(f"情感分析模型训练 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 1. 数据准备
    logger.info("\n1. 准备数据...")
    data_module = DataModule(config)
    train_df, test_df = data_module.load_data()
    
    # 2. 文本预处理
    logger.info("\n2. 文本预处理...")
    preprocessor = data_module.build_preprocessor()
    
    # 3. 创建数据集
    logger.info("\n3. 创建数据集...")
    train_dataset, test_dataset = data_module.create_datasets()
    train_loader, test_loader = data_module.create_dataloaders(train_dataset, test_dataset)
    
    # 4. 初始化模型
    logger.info("\n4. 初始化模型...")
    vocab_size = len(preprocessor.vocab)
    model = LSTMWithAttention(vocab_size, config['model'])
    
    # 5. 训练
    logger.info("\n5. 开始训练...")
    trainer = Trainer(model, config)
    
    # ✅ 修复：设置数据集大小（解决作用域错误）
    trainer.set_dataset_sizes(len(train_dataset), len(test_dataset))
    
    # ✅ 修复：获取训练结果
    total_time, best_accuracy = trainer.train(train_loader, test_loader)
    
    # 6. 生成报告和可视化
    logger.info("\n6. 生成报告和可视化...")
    # ✅ 修复：传入正确的参数
    trainer.generate_report(total_time, vocab_size, best_accuracy)
    trainer.plot_training_curves()
    
    logger.info("\n✅ 训练完成！所有结果已保存到 results/ 目录")
    logger.info("   - 模型: results/models/")
    logger.info("   - 词汇表: results/vocab.json")
    logger.info("   - 训练曲线: results/figures/")
    logger.info("   - 训练报告: results/training_report.txt")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()