# scripts/predict.py
import sys
import os
import yaml
import logging
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from src.predictor import SentimentPredictor
from src.dataset import DataModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def predict_single_examples(predictor: SentimentPredictor):
    """单条文本预测示例"""
    logger.info("\n" + "=" * 60)
    logger.info("单条文本预测示例")
    logger.info("=" * 60)
    
    test_texts = [
        "This movie is absolutely fantastic! The acting is superb and the plot keeps you on the edge of your seat. Highly recommended!",
        "The story drags on forever, the acting is cringeworthy, and the whole thing is a complete waste of time. Don't bother watching.",
        "It was okay, nothing special. Pretty average overall, didn't leave much of an impression.",
        "The special effects are stunning but the story is painfully weak — classic popcorn flick with no substance.",
        "Visually beautiful with a moving soundtrack, but the ending felt rushed and unsatisfying.",
        "I had no idea what was going on half the time. Two hours of my life I'll never get back.",
        "The cast gives it their all and the performances are solid, but the script is just too weak to save it. A real shame because the talent was there.",
        "One of the best films I've seen all year. Deeply moving, brilliantly written, and the cinematography is breathtaking.",
        "Terrible dialogue, wooden acting, and a plot that makes absolutely no sense. I wanted to walk out of the theater.",
        "Surprisingly good! Went in with low expectations and came out thoroughly entertained. Great pacing and genuine laughs throughout."
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        sentiment_cn = "😊 正面" if result['sentiment'] == 'positive' else "😞 负面"
        
        print(f"\n📝 文本: {text}")
        print(f"🎯 情感: {sentiment_cn}")
        print(f"📊 置信度: {result['confidence']:.2%}")
        print(f"📈 概率分布: 负面={result['probabilities']['negative']:.2%}, 正面={result['probabilities']['positive']:.2%}")


def predict_from_file(predictor: SentimentPredictor, input_file: str, output_file: str = None):
    """从文件读取文本批量预测"""
    logger.info(f"\n从文件读取文本: {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"共读取 {len(texts)} 条文本")
    
    # 批量预测
    results = predictor.predict_batch(texts, batch_size=64, show_progress=True)
    
    # 统计结果
    positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
    negative_count = len(results) - positive_count
    
    logger.info(f"\n批量预测完成:")
    logger.info(f"  正面评论: {positive_count} ({positive_count/len(results)*100:.1f}%)")
    logger.info(f"  负面评论: {negative_count} ({negative_count/len(results)*100:.1f}%)")
    
    # 保存结果
    if output_file:
        output_path = Path(__file__).parent.parent / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("text\tsentiment\tconfidence\tnegative_prob\tpositive_prob\n")
            for r in results:
                f.write(f"{r['text']}\t{r['sentiment']}\t{r['confidence']:.4f}\t"
                       f"{r['probabilities']['negative']:.4f}\t{r['probabilities']['positive']:.4f}\n")
        logger.info(f"结果已保存到: {output_path}")
    
    return results


def evaluate_on_testset(predictor: SentimentPredictor, config: dict):
    """在测试集上评估模型"""
    logger.info("\n" + "=" * 60)
    logger.info("在测试集上评估模型")
    logger.info("=" * 60)
    
    # 加载测试数据
    data_module = DataModule(config)
    train_dataset, test_dataset = data_module.create_datasets()
    _, test_loader = data_module.create_dataloaders(train_dataset, test_dataset)
    
    # 评估
    results = predictor.evaluate_on_testset(test_loader)
    
    logger.info(f"\n✅ 测试集准确率: {results['accuracy']:.2f}%")
    
    return results


def interactive_predict(predictor: SentimentPredictor):
    """交互式预测模式"""
    logger.info("\n" + "=" * 60)
    logger.info("进入交互式预测模式")
    logger.info("输入 'quit' 或 'exit' 退出")
    logger.info("=" * 60)
    
    while True:
        print("\n" + "-" * 40)
        text = input("请输入要分析的评论: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("退出交互模式")
            break
        
        if not text:
            print("请输入有效内容")
            continue
        
        result = predictor.predict(text)
        sentiment_cn = "😊 正面" if result['sentiment'] == 'positive' else "😞 负面"
        
        print(f"\n📊 分析结果:")
        print(f"   情感: {sentiment_cn}")
        print(f"   置信度: {result['confidence']:.2%}")
        print(f"   正面概率: {result['probabilities']['positive']:.2%}")
        print(f"   负面概率: {result['probabilities']['negative']:.2%}")


def main():
    parser = argparse.ArgumentParser(description='情感分析模型预测')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'example', 'file', 'evaluate'],
                       help='预测模式: interactive(交互), example(示例), file(批量文件), evaluate(评估)')
    parser.add_argument('--input', type=str, default=None,
                       help='批量预测的输入文件路径（用于file模式）')
    parser.add_argument('--output', type=str, default='results/predictions.txt',
                       help='批量预测的输出文件路径')
    parser.add_argument('--no-gpu', action='store_true',
                       help='不使用GPU，强制使用CPU')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config()
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    model_path = project_root / config['paths']['best_model']
    vocab_path = project_root / config['paths']['vocab_file']
    
    # 检查文件是否存在
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.error("请先运行训练脚本生成模型")
        return
    
    if not vocab_path.exists():
        logger.error(f"词汇表文件不存在: {vocab_path}")
        logger.error("请先运行训练脚本生成词汇表")
        return
    
    # 初始化预测器
    logger.info("初始化预测器...")
    predictor = SentimentPredictor(
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        config=config,
        use_gpu=not args.no_gpu
    )
    
    # 根据模式执行
    if args.mode == 'example':
        predict_single_examples(predictor)
    
    elif args.mode == 'file':
        if not args.input:
            logger.error("file模式需要指定--input参数")
            return
        predict_from_file(predictor, args.input, args.output)
    
    elif args.mode == 'evaluate':
        evaluate_on_testset(predictor, config)
    
    else:  # interactive
        interactive_predict(predictor)


if __name__ == "__main__":
    main()