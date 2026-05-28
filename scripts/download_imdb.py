import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 同时取消所有代理设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

# download_imdb.py
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("=" * 60)
print(f"IMDB数据集下载 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

#start time
start_time = time.time()

# 下载数据集
print("🚀 开始下载IMDB数据集...")
dataset = load_dataset('imdb', cache_dir='./data_cache')
print("✅ 数据集下载完成！")

# 转换为DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# 验证数据
print(f"\n📊 数据集统计:")
print(f"训练集: {len(train_df):,} 条 ({train_df['label'].value_counts().to_dict()})")
print(f"测试集: {len(test_df):,} 条 ({test_df['label'].value_counts().to_dict()})")

# 数据样例
print("\n🔍 训练数据样例:")
for i in range(3):
    label = "正面" if train_df.iloc[i]['label'] == 1 else "负面"
    text = train_df.iloc[i]['text'][:150]
    print(f"{i+1}. [{label}] {text}...")

# 保存数据
train_df.to_csv('data/imdb_train.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('data/imdb_test.csv', index=False, encoding='utf-8-sig')
print("\n💾 数据已保存到CSV文件")
#中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  #  Windows 黑体，或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题
# 生成标签分布图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x='label', data=train_df)
plt.title('训练集标签分布')
plt.xticks([0, 1], ['负面', '正面'])

plt.subplot(1, 2, 2)
sns.countplot(x='label', data=test_df)
plt.title('测试集标签分布')
plt.xticks([0, 1], ['负面', '正面'])




plt.tight_layout()
plt.savefig('results/figures/label_distribution.png', dpi=300, bbox_inches='tight')
print("📊 标签分布图已保存")

print(f"\n✅ 任务完成！耗时: {time.time() - start_time:.2f} 秒")