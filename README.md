# LSTMMovieComment

基于 LSTM + Attention 的英文电影评论情感分析（IMDB 二分类：正面/负面）。

## 模型架构

| 层 | 规格 |
|---|---|
| Embedding | 128 维 |
| LSTM | 256 隐藏单元，双向 |
| Attention | 加性注意力（Linear → Softmax → 加权求和） |
| Classifier | Linear(hidden\*2 → 2)，Dropout 0.5 |

序列长度 200，词汇表 10,000，使用 Adam + 梯度裁剪 + 早停训练。

## 训练结果

| 指标 | 数值 |
|---|---|
| 训练集 | 25,000 |
| 测试集 | 25,000 |
| 最佳验证准确率 | **86.17%** |
| 最终训练准确率 | 89.16% |
| 训练时间 | ~2 分钟 (GPU) |

## 项目结构

```
LSTMmovieComment/
├── configs/config.yaml              # 所有超参数集中管理
├── src/
│   ├── model.py                     # LSTMWithAttention 模型定义
│   ├── data_preprocessing.py        # TextPreprocessor（分词、词汇表构建）
│   ├── dataset.py                   # Dataset + DataModule（DataLoader 封装）
│   ├── trainer.py                   # 训练循环、早停、曲线绘制
│   └── predictor.py                 # 推理封装（单条/批量/交互式）
├── scripts/
│   ├── train.py                     # 训练入口
│   ├── predict.py                   # 预测入口（4 种模式）
│   ├── train_baseline.py            # 消融对比：纯 LSTM 无 Attention
│   ├── download_imdb.py             # 下载 IMDB 数据集
│   ├── np_wordCloud.py              # 词云生成
│   └── evaluate_with_datamodule.py  # 测试集评估
├── results/
│   ├── vocab.json                   # 词汇表
│   └── training_report.txt          # 训练报告
└── data/                            # CSV 数据文件（需运行下载脚本生成）
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA（可选，CPU 也可运行）

### 安装依赖

```bash
pip install torch pandas pyyaml tqdm matplotlib seaborn datasets
```

### 下载数据

```bash
python scripts/download_imdb.py
```

### 训练

```bash
python scripts/train.py
```

训练完成后，模型和词汇表保存在 `results/` 目录下。

### 预测

```bash
# 交互式预测
python scripts/predict.py

# 示例文本预测
python scripts/predict.py --mode example

# 批量文件预测
python scripts/predict.py --mode file --input your_texts.txt --output results/predictions.txt

# 测试集评估
python scripts/predict.py --mode evaluate
```

### 消融对比

```bash
python scripts/train_baseline.py
```

## 核心特性

- **Attention 机制**：逐时间步 Softmax 加权，有效捕捉关键情感词
- **配置与代码分离**：`config.yaml` 管理所有超参数
- **GPU 内存自适应**：根据 GPU 显存自动调整 batch size
- **模型保存完整**：同时保存 `state_dict`、`config`、`vocab_size`，推理可自包含恢复
- **早停 + 梯度裁剪**：防止过拟合和梯度爆炸
- **消融对比**：内置纯 LSTM baseline，可量化 Attention 贡献

## 待改进

- 验证集/测试集当前未分离，应从训练集中划分验证集
- 预处理对 IMDB 特有噪声（HTML 标签如 `<br />`）过滤不够彻底
- 可尝试 Multi-head Attention 或预训练词向量（GloVe）进一步提升效果
