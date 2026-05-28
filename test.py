# check_baseline_accuracy.py
import torch

# 读取 Baseline 的模型
checkpoint = torch.load('results/models/best_model.pth', map_location='cpu')
metadata = checkpoint.get('metadata', {})

print("=" * 50)
print("Baseline 模型（无注意力）训练结果")
print("=" * 50)
print(f"最佳准确率: {metadata.get('accuracy', 'N/A')}%")
print(f"训练轮数: {metadata.get('epoch', 'N/A')}")
print(f"训练损失: {metadata.get('train_losses', [])}")
print(f"验证准确率: {metadata.get('val_accuracies', [])}")