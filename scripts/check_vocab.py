# check_vocab.py
import json

vocab = json.load(open('results/vocab.json', 'r', encoding='utf-8'))
print("✅ 词汇表加载成功")
print(f"  特殊标记: {list(vocab.keys())[:5]}")
print(f"  词汇数量: {len(vocab):,}")