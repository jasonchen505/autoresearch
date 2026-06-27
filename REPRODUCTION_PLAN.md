# AutoResearch 8x4090 复现计划

> 基于实际硬件资源（8卡4090）制定的完整复现方案

---

## 一、硬件资源评估

### 1.1 硬件对比

| 指标 | H100 (原项目) | 4090 (你的资源) | 差异分析 |
|------|--------------|----------------|---------|
| **VRAM** | 80GB | 24GB | **3.3x 减小**，需要调整 batch size |
| **BF16 TFLOPS** | 989.5 | 330 | **3x 减小**，需要调整时间预算 |
| **内存带宽** | 3.35TB/s | 1TB/s | **3.3x 减小**，影响数据加载 |
| **互联** | NVLink | 无 | 多卡通信受限 |
| **Flash Attention** | FA3 (Hopper) | FA2/FA3 (community) | 需要验证兼容性 |

### 1.2 关键约束分析

**内存约束**:
```
原项目: DEVICE_BATCH_SIZE=128, MAX_SEQ_LEN=2048
单 batch 内存: 128 * 2048 * 2 bytes (BF16) ≈ 0.5MB (仅输入)
实际模型内存: ~25GB (参数 + 优化器 + 激活值)
4090 限制: 24GB → 需要减小 batch size 或模型大小
```

**计算约束**:
```
原项目: 5分钟训练，H100 约 989 TFLOPS
4090: 330 TFLOPS → 同样时间只能完成 1/3 的计算
需要: 延长时间 或 减小模型
```

---

## 二、复现策略选择

### 2.1 三种可行方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A: 保持架构，调整超参** | 保持模型架构不变，减小 batch size 和时间预算 | 最接近原版，代码改动最小 | 实验效果可能不如原版 |
| **B: 减小模型规模** | 减小 DEPTH 和 ASPECT_RATIO | 可以在 5 分钟内完成 | 模型能力受限 |
| **C: 多卡并行** | 使用 DDP 或 FSDP 分布训练 | 可以用更大 batch | 实现复杂，调试困难 |

### 2.2 推荐方案：A + B 混合

**核心思路**:
1. 保持模型架构设计不变（保持学习价值）
2. 减小 DEPTH 从 8 到 4-6
3. 减小 DEVICE_BATCH_SIZE 从 128 到 32-64
4. 延长时间预算从 5 分钟到 10-15 分钟（可选）

---

## 三、详细复现计划

### Phase 1: 环境准备（30分钟）

#### 1.1 安装依赖

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
cd /data/home/yizhou
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# 安装依赖
uv sync
```

#### 1.2 验证 GPU 环境

```bash
# 检查 GPU 信息
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 支持
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_capability())"
```

#### 1.3 准备数据

```bash
# 下载数据（只下载部分 shard，节省时间）
uv run prepare.py --num-shards 50

# 如果要完整数据
uv run prepare.py --num-shards -1
```

**注意**: 4090 的内存带宽较低，数据加载可能成为瓶颈。建议先用少量 shard 测试。

---

### Phase 2: 参数调整（1小时）

#### 2.1 计算可行的参数组合

**内存估算公式**:
```
模型参数内存 ≈ num_params * 2 bytes (BF16)
优化器内存 ≈ num_params * 8 bytes (Adam: 2 state + momentum)
激活值内存 ≈ batch_size * seq_len * hidden_dim * num_layers * 2 bytes
```

**4090 可行配置**:

| 配置 | DEPTH | ASPECT_RATIO | DEVICE_BATCH_SIZE | 预计内存 | 预计时间 |
|------|-------|--------------|-------------------|---------|---------|
| **保守** | 4 | 64 | 32 | ~12GB | 5分钟 |
| **中等** | 6 | 64 | 48 | ~18GB | 8分钟 |
| **激进** | 8 | 64 | 64 | ~24GB | 15分钟 |

#### 2.2 修改 train.py

**关键修改点**:

```python
# 1. 修改 H100 峰值性能为 4090 的值
# train.py:463
H100_BF16_PEAK_FLOPS = 330e12  # 4090 的 BF16 性能

# 2. 减小模型深度
# train.py:450
DEPTH = 4  # 从 8 减小到 4

# 3. 减小 batch size
# train.py:451
DEVICE_BATCH_SIZE = 32  # 从 128 减小到 32

# 4. 可选：延长训练时间
# prepare.py:31
TIME_BUDGET = 600  # 从 300 延长到 600 (10分钟)
```

#### 2.3 验证 Flash Attention 兼容性

```python
# 测试 Flash Attention
uv run python -c "
import torch
from kernels import get_kernel
cap = torch.cuda.get_device_capability()
print(f'GPU capability: {cap}')
repo = 'varunneal/flash-attention-3' if cap == (9, 0) else 'kernels-community/flash-attn3'
print(f'Using repo: {repo}')
fa3 = get_kernel(repo).flash_attn_interface
print('Flash Attention loaded successfully')
"
```

---

### Phase 3: 单卡验证（2小时）

#### 3.1 运行基线实验

```bash
# 单卡运行
CUDA_VISIBLE_DEVICES=0 uv run train.py
```

**预期输出**:
```
Vocab size: 8,192
Model config: {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 4, ...}
Parameter counts:
  wte                     : 16,777,216
  value_embeds            : 8,388,608
  lm_head                 : 16,777,216
  transformer_matrices    : 50,331,648
  scalars                 : 8
  total                   : 91,874,696
...
val_bpb:          X.XXXXXX
training_seconds: 300.0
peak_vram_mb:     XXXXX.X
```

#### 3.2 记录基线结果

创建 `results_4090.tsv`:
```
commit	val_bpb	memory_gb	status	description
baseline	0.000000	0.0	keep	DEPTH=4, BS=32, 5min
```

#### 3.3 验证评估指标

```bash
# 检查 val_bpb 计算是否正确
grep "^val_bpb:" run.log
```

---

### Phase 4: 多卡实验（3-4小时）

#### 4.1 方案选择

**方案 A: 数据并行（推荐）**

使用 `torchrun` 进行数据并行：

```bash
# 8卡数据并行
torchrun --nproc_per_node=8 train_ddp.py
```

**需要创建 `train_ddp.py`**:
```python
# 主要修改：
# 1. 初始化分布式环境
import torch.distributed as dist
dist.init_process_group(backend='nccl')

# 2. 包装模型
model = torch.nn.parallel.DistributedDataParallel(model)

# 3. 调整 batch size
DEVICE_BATCH_SIZE = 32  # 每卡 batch size
# 总 batch size = 32 * 8 = 256

# 4. 调整梯度累积
grad_accum_steps = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * world_size)
```

**方案 B: 实验并行（更简单）**

每卡独立运行不同实验：

```bash
# 卡0: 基线
CUDA_VISIBLE_DEVICES=0 uv run train.py &

# 卡1: 改变学习率
CUDA_VISIBLE_DEVICES=1 uv run train.py &

# 卡2: 改变模型大小
CUDA_VISIBLE_DEVICES=2 uv run train.py &

# ... 以此类推
```

**优点**: 简单，无需修改代码
**缺点**: 无法用更大 batch size

#### 4.2 推荐：方案 B（实验并行）

**原因**:
1. 代码改动最小
2. 可以同时探索多个方向
3. 符合 autoresearch 的实验循环理念
4. 4090 没有 NVLink，多卡通信效率低

**脚本示例**:

```bash
#!/bin/bash
# run_parallel.sh

# 创建实验目录
mkdir -p experiments

# 卡0: 基线实验
CUDA_VISIBLE_DEVICES=0 uv run train.py > experiments/gpu0_baseline.log 2>&1 &

# 卡1: 增加学习率
CUDA_VISIBLE_DEVICES=1 MATRIX_LR=0.06 uv run train.py > experiments/gpu1_lr.log 2>&1 &

# 卡2: 减小深度
CUDA_VISIBLE_DEVICES=2 DEPTH=6 uv run train.py > experiments/gpu2_depth.log 2>&1 &

# 卡3: 增加 batch size
CUDA_VISIBLE_DEVICES=3 DEVICE_BATCH_SIZE=48 uv run train.py > experiments/gpu3_bs.log 2>&1 &

# 等待所有实验完成
wait
```

---

### Phase 5: Agent 自动化实验（过夜运行）

#### 5.1 配置 Agent

修改 `program.md` 适配 4090：

```markdown
## 4090 适配说明

由于使用 4090 而非 H100，需要调整以下参数：

1. **DEVICE_BATCH_SIZE**: 从 128 减小到 32-64
2. **DEPTH**: 从 8 减小到 4-6
3. **TIME_BUDGET**: 可以延长到 10-15 分钟
4. **H100_BF16_PEAK_FLOPS**: 修改为 330e12

实验时请优先尝试：
- 调整学习率 (MATRIX_LR, EMBEDDING_LR)
- 调整优化器参数 (WEIGHT_DECAY, ADAM_BETAS)
- 尝试不同的 WINDOW_PATTERN
- 调整 WARMUP_RATIO 和 WARMDOWN_RATIO
```

#### 5.2 过夜实验脚本

```bash
#!/bin/bash
# overnight_experiments.sh

LOG_DIR="overnight_logs_$(date +%Y%m%d)"
mkdir -p $LOG_DIR

# 8卡并行实验，每卡跑不同配置
for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu \
    EXPERIMENT_TAG="gpu${gpu}" \
    uv run train.py > $LOG_DIR/gpu${gpu}.log 2>&1 &
done

echo "Started 8 experiments on GPUs 0-7"
echo "Logs in $LOG_DIR/"
```

#### 5.3 结果收集

```bash
#!/bin/bash
# collect_results.sh

LOG_DIR="overnight_logs_*"
echo "GPU\tval_bpb\tpeak_vram_mb\ttraining_seconds"

for log in $LOG_DIR/gpu*.log; do
    gpu=$(basename $log .log)
    val_bpb=$(grep "^val_bpb:" $log | awk '{print $2}')
    vram=$(grep "^peak_vram_mb:" $log | awk '{print $2}')
    time=$(grep "^training_seconds:" $log | awk '{print $2}')
    echo "$gpu\t$val_bpb\t$vram\t$time"
done
```

---

## 四、关键代码修改清单

### 4.1 必须修改的文件

| 文件 | 修改内容 | 原因 |
|------|---------|------|
| `train.py:463` | `H100_BF16_PEAK_FLOPS = 330e12` | 4090 的 BF16 性能 |
| `train.py:450` | `DEPTH = 4` | 减小模型大小 |
| `train.py:451` | `DEVICE_BATCH_SIZE = 32` | 减小内存占用 |
| `prepare.py:31` | `TIME_BUDGET = 600` | 延长训练时间（可选） |

### 4.2 可选修改的文件

| 文件 | 修改内容 | 原因 |
|------|---------|------|
| `train.py:435` | `WINDOW_PATTERN = "SL"` | 简化注意力模式 |
| `train.py:438` | `TOTAL_BATCH_SIZE = 2**18` | 减小总 batch size |
| `program.md` | 添加 4090 适配说明 | 指导 Agent 实验 |

### 4.3 完整修改示例

```python
# train.py - 关键修改点

# 1. 修改性能指标
H100_BF16_PEAK_FLOPS = 330e12  # 4090 BF16 性能

# 2. 修改模型配置
DEPTH = 4  # 从 8 减小到 4
DEVICE_BATCH_SIZE = 32  # 从 128 减小到 32
WINDOW_PATTERN = "SL"  # 简化注意力模式

# 3. 可选：调整优化参数
TOTAL_BATCH_SIZE = 2**18  # 从 2^19 减小到 2^18
```

---

## 五、预期结果与基准

### 5.1 预期性能指标

| 配置 | 预期 val_bpb | 预期内存 | 预期时间 | 预期 MFU |
|------|-------------|---------|---------|---------|
| **DEPTH=4, BS=32** | ~1.1-1.2 | ~12GB | 5分钟 | ~25-30% |
| **DEPTH=6, BS=48** | ~1.0-1.1 | ~18GB | 8分钟 | ~20-25% |
| **DEPTH=8, BS=64** | ~0.95-1.05 | ~24GB | 15分钟 | ~15-20% |

**注**: 4090 的 MFU 通常低于 H100，因为：
1. 内存带宽较低
2. 没有 NVLink
3. Flash Attention 效率可能较低

### 5.2 成功标准

**Phase 1-2 成功标准**:
- [ ] 环境安装成功
- [ ] 数据准备完成
- [ ] 单卡运行无报错
- [ ] val_bpb 正常输出

**Phase 3 成功标准**:
- [ ] 基线 val_bpb 在预期范围内
- [ ] 内存使用稳定在 24GB 以下
- [ ] 训练时间在 5-15 分钟

**Phase 4 成功标准**:
- [ ] 8 卡并行运行稳定
- [ ] 可以同时进行多个实验
- [ ] 结果可比较

**Phase 5 成功标准**:
- [ ] Agent 可以自动运行实验
- [ ] 过夜实验无崩溃
- [ ] 结果有明显改进

---

## 六、故障排除

### 6.1 常见问题

**问题 1: OOM (Out of Memory)**
```bash
# 解决方案：减小 batch size
DEVICE_BATCH_SIZE = 16  # 或更小
```

**问题 2: Flash Attention 不兼容**
```python
# 解决方案：使用 PyTorch 原生注意力
# 注释掉 Flash Attention 相关代码
# 使用 F.scaled_dot_product_attention 替代
```

**问题 3: 训练速度过慢**
```bash
# 解决方案：
# 1. 延长时间预算
TIME_BUDGET = 900  # 15分钟

# 2. 减小模型
DEPTH = 2
```

**问题 4: 数据加载瓶颈**
```python
# 解决方案：增加 num_workers
# 在 make_dataloader 中添加 num_workers 参数
```

### 6.2 调试命令

```bash
# 检查 GPU 状态
nvidia-smi -l 1

# 检查内存使用
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# 检查进程
ps aux | grep train.py

# 杀死所有训练进程
pkill -f train.py
```

---

## 七、学习路径建议

### 7.1 第一天：环境与基线

**目标**: 跑通单卡实验

1. 安装环境（30分钟）
2. 准备数据（30分钟）
3. 修改参数（30分钟）
4. 运行基线实验（1小时）
5. 分析结果（30分钟）

### 7.2 第二天：多卡实验

**目标**: 跑通多卡并行

1. 测试 2 卡并行（1小时）
2. 扩展到 4 卡（1小时）
3. 扩展到 8 卡（1小时）
4. 分析并行效率（1小时）

### 7.3 第三天：Agent 实验

**目标**: 运行自动化实验

1. 配置 Agent（1小时）
2. 运行 5 轮实验循环（2小时）
3. 分析实验结果（1小时）
4. 优化实验策略（1小时）

### 7.4 第四天：深入优化

**目标**: 探索最佳配置

1. 尝试不同的模型架构（2小时）
2. 尝试不同的优化器配置（2小时）
3. 总结最佳实践（1小时）

---

## 八、代码修改模板

### 8.1 train_4090.py 修改模板

```python
#!/usr/bin/env python3
"""
AutoResearch pretraining script - 4090 适配版
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# 4090 使用 community 版本
repo = "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ... 其他代码保持不变 ...

# ---------------------------------------------------------------------------
# Hyperparameters - 4090 适配
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # 保持不变
HEAD_DIM = 128          # 保持不变
WINDOW_PATTERN = "SL"   # 简化：只用 Short 和 Long

# Optimization
TOTAL_BATCH_SIZE = 2**18  # 从 2^19 减小到 2^18 (~262K tokens)
EMBEDDING_LR = 0.6        # 保持不变
UNEMBEDDING_LR = 0.004    # 保持不变
MATRIX_LR = 0.04          # 保持不变
SCALAR_LR = 0.5           # 保持不变
WEIGHT_DECAY = 0.2        # 保持不变
ADAM_BETAS = (0.8, 0.95)  # 保持不变
WARMUP_RATIO = 0.0        # 保持不变
WARMDOWN_RATIO = 0.5      # 保持不变
FINAL_LR_FRAC = 0.0       # 保持不变

# Model size - 4090 适配
DEPTH = 4                 # 从 8 减小到 4
DEVICE_BATCH_SIZE = 32    # 从 128 减小到 32

# 性能指标 - 4090 适配
H100_BF16_PEAK_FLOPS = 330e12  # 4090 的 BF16 性能

# ... 其他代码保持不变 ...
```

### 8.2 run_all_experiments.sh

```bash
#!/bin/bash
# run_all_experiments.sh - 8卡并行实验脚本

set -e

LOG_DIR="experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "Starting experiments in $LOG_DIR"

# 定义实验配置
declare -A configs
configs[0]="baseline:DEPTH=4:DEVICE_BATCH_SIZE=32"
configs[1]="deeper:DEPTH=6:DEVICE_BATCH_SIZE=24"
configs[2]="wider:DEPTH=4:DEVICE_BATCH_SIZE=48"
configs[3]="lr_high:DEPTH=4:DEVICE_BATCH_SIZE=32:MATRIX_LR=0.06"
configs[4]="lr_low:DEPTH=4:DEVICE_BATCH_SIZE=32:MATRIX_LR=0.02"
configs[5]="sl_window:DEPTH=4:DEVICE_BATCH_SIZE=32:WINDOW_PATTERN=SL"
configs[6]="sssl_window:DEPTH=4:DEVICE_BATCH_SIZE=32:WINDOW_PATTERN=SSSL"
configs[7]="warmdown:DEPTH=4:DEVICE_BATCH_SIZE=32:WARMDOWN_RATIO=0.3"

# 并行运行实验
for gpu in 0 1 2 3 4 5 6 7; do
    IFS=':' read -r name params <<< "${configs[$gpu]}"
    
    # 构建环境变量
    env_vars=""
    for param in $params; do
        env_vars="$env_vars -e $param"
    done
    
    # 运行实验
    echo "Starting experiment $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH=$PWD \
    uv run python -c "
import os
# 设置参数
$(for param in $params; do echo "os.environ['${param%%=*}'] = '${param#*=}'"; done)

# 导入并运行
import train
" > $LOG_DIR/${name}_gpu${gpu}.log 2>&1 &
done

echo "All experiments started. Logs in $LOG_DIR/"
echo "Monitor with: tail -f $LOG_DIR/*.log"
```

---

## 九、总结

### 9.1 核心要点

1. **4090 vs H100**: 内存 3.3x 减小，计算 3x 减慢
2. **关键调整**: DEPTH 4-6, DEVICE_BATCH_SIZE 32-64
3. **推荐方案**: 实验并行（每卡独立实验）
4. **预期结果**: val_bpb ~1.0-1.2，训练时间 5-15 分钟

### 9.2 快速开始

```bash
# 1. 克隆仓库
cd /data/home/yizhou
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# 2. 安装依赖
uv sync

# 3. 准备数据
uv run prepare.py --num-shards 50

# 4. 修改参数（使用 sed 或手动编辑）
sed -i 's/DEPTH = 8/DEPTH = 4/' train.py
sed -i 's/DEVICE_BATCH_SIZE = 128/DEVICE_BATCH_SIZE = 32/' train.py
sed -i 's/H100_BF16_PEAK_FLOPS = 989.5e12/H100_BF16_PEAK_FLOPS = 330e12/' train.py

# 5. 运行实验
CUDA_VISIBLE_DEVICES=0 uv run train.py
```

### 9.3 学习价值

通过这个复现过程，你将学到：
1. **硬件适配**: 如何根据硬件调整模型配置
2. **内存优化**: batch size、模型大小的权衡
3. **分布式训练**: 多卡并行的实现方式
4. **实验管理**: 如何高效地进行多组实验
5. **LLM 训练**: 完整的训练流程和技术细节

---

**下一步**: 按照 Phase 1 开始执行，遇到问题随时调整参数。
