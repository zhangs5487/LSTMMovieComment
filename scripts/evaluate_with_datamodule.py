# scripts/evaluate_imdb.py
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import pandas as pd
from src.predictor import SentimentPredictor

def load_config():
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 加载配置
    config = load_config()
    
    # 初始化预测器
    print("加载模型...")
    predictor = SentimentPredictor(
        model_path=str(Path(__file__).parent.parent / 'results' / 'models' / 'best_model.pth'),
        vocab_path=str(Path(__file__).parent.parent / 'results' / 'vocab.json'),
        config=config,
        use_gpu=True
    )
    
    # 加载测试集
    test_csv_path = Path(__file__).parent.parent / 'data' / 'imdb_test.csv'
    
    if not test_csv_path.exists():
        print(f"错误: 测试集文件不存在 {test_csv_path}")
        return
    
    print(f"\n加载测试集: {test_csv_path}")
    df = pd.read_csv(test_csv_path, encoding='utf-8-sig')
    print(f"测试集大小: {len(df)} 条")
    print(f"列名: {df.columns.tolist()}")
    
    # 取前100条进行测试（可以根据需要调整）
    test_size = min(100, len(df))
    df_test = df.head(test_size).copy()
    texts = df_test['text'].tolist()
    true_labels = df_test['label'].tolist()  # 0=负面, 1=正面
    
    print(f"\n开始预测 {test_size} 条评论...")
    
    # 批量预测
    results = predictor.predict_batch(texts, batch_size=32, show_progress=True)
    
    # 计算准确率
    correct = 0
    for i, result in enumerate(results):
        # 转换预测结果: positive->1, negative->0
        pred_label = 1 if result['sentiment'] == 'positive' else 0
        if pred_label == true_labels[i]:
            correct += 1
    
    accuracy = correct / len(results) * 100
    
    print("\n" + "=" * 60)
    print("📊 测试集评估结果")
    print("=" * 60)
    print(f"测试样本数: {len(results)}")
    print(f"正确预测: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"训练时验证准确率: 86.45%")
    
    # 显示样例预测
    print("\n" + "=" * 60)
    print("🔍 预测样例展示（前10条）")
    print("=" * 60)
    
    for i in range(min(10, len(results))):
        sentiment_cn = "正面" if results[i]['sentiment'] == 'positive' else "负面"
        true_cn = "正面" if true_labels[i] == 1 else "负面"
        correct_mark = "✓" if (1 if results[i]['sentiment'] == 'positive' else 0) == true_labels[i] else "✗"
        
        # 截断过长的文本
        text_preview = texts[i][:80] + "..." if len(texts[i]) > 80 else texts[i]
        
        print(f"\n{i+1}. 文本: {text_preview}")
        print(f"   真实标签: {true_cn}")
        print(f"   预测结果: {sentiment_cn} {correct_mark}")
        print(f"   置信度: {results[i]['confidence']:.2%}")
        print(f"   正面概率: {results[i]['probabilities']['positive']:.2%}")
        print(f"   负面概率: {results[i]['probabilities']['negative']:.2%}")
    
    # 统计预测分布
    pred_positive = sum(1 for r in results if r['sentiment'] == 'positive')
    pred_negative = len(results) - pred_positive
    true_positive = sum(true_labels)
    true_negative = len(true_labels) - true_positive
    
    print("\n" + "=" * 60)
    print("📈 预测分布统计")
    print("=" * 60)
    print(f"真实分布: 正面={true_positive}, 负面={true_negative}")
    print(f"预测分布: 正面={pred_positive}, 负面={pred_negative}")
    
    # 保存完整结果
    output_path = Path(__file__).parent.parent / 'results' / 'test_predictions.csv'
    df_test['predicted_sentiment'] = [r['sentiment'] for r in results]
    df_test['predicted_label'] = [1 if r['sentiment'] == 'positive' else 0 for r in results]
    df_test['confidence'] = [r['confidence'] for r in results]
    df_test['positive_prob'] = [r['probabilities']['positive'] for r in results]
    df_test['negative_prob'] = [r['probabilities']['negative'] for r in results]
    df_test['correct'] = [1 if ((1 if r['sentiment'] == 'positive' else 0) == true_labels[i]) else 0 for i, r in enumerate(results)]
    
    df_test.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 完整结果已保存到: {output_path}")

if __name__ == "__main__":
    main()