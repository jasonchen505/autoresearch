# 增量学习笔记：8x4090 复现 AutoResearch

> 基于前两轮分析（项目理解 + 面试准备），记录复现过程中新学到的点

---

## 一、硬件适配相关的新知识点

### 1.1 GPU 架构差异深入理解

#### 前两轮认知
- H100 性能更强，内存更大
- 4090 是消费级显卡，性能较弱

#### 本轮新学到
**Compute Capability 的实际影响**:
```python
# train.py:21-23
cap = torch.cuda.get_device_capability()
# H100: (9, 0) → Hopper 架构，支持 FA3 原生版本
# 4090: (8, 9) → Ada Lovelace 架构，需要 community 版本
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
```

**关键理解**:
1. **不只是性能差异**: 不同架构支持的指令集不同，FA3 对 Hopper 有专门优化
2. **Community 版本的限制**: 可能没有针对 Ada Lovelace 的专门优化，效率可能较低
3. **实际影响**: 同样的模型，4090 上 Flash Attention 的效率可能只有 H100 的 60-70%

#### 实践验证方法
```bash
# 检查 GPU 能力
uv run python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'Capability: {torch.cuda.get_device_capability()}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
print(f'BF16 support: {torch.cuda.is_bf16_supported()}')
"
```

---

### 1.2 内存计算的精确理解

#### 前两轮认知
- 模型参数占用内存
- 优化器状态占用额外内存

#### 本轮新学到
**精确的内存计算公式**:

```
总内存 = 模型参数 + 优化器状态 + 激值 + 临时缓冲区

模型参数 (BF16):
- 50M params * 2 bytes = 100MB

优化器状态 (Adam):
- exp_avg: 50M * 4 bytes = 200MB (FP32)
- exp_avg_sq: 50M * 4 bytes = 200MB (FP32)
- 总计: 400MB

激活值 (取决于 batch size 和 sequence length):
- 每层: batch_size * seq_len * hidden_dim * 2 bytes
- 所有层: num_layers * 每层内存
- 估算: 32 * 2048 * 512 * 8 * 2 = 512MB

临时缓冲区:
- 梯度: 100MB (与参数同大小)
- 中间结果: 200-500MB
```

**关键理解**:
1. **激活值是大头**: 通常占总内存的 50-70%
2. **Batch size 的影响**: 线性关系，减半 batch size 减半激活值内存
3. **Sequence length 的影响**: 也是线性关系，但通常固定为 2048
4. **Gradient Checkpointing**: 可以用计算换内存，减少激活值存储

#### 4090 内存预算
```
4090 总内存: 24GB
系统预留: ~2GB
可用: ~22GB

安全配置:
- DEPTH=4, DEVICE_BATCH_SIZE=32: ~12GB (安全)
- DEPTH=6, DEVICE_BATCH_SIZE=48: ~18GB (较安全)
- DEPTH=8, DEVICE_BATCH_SIZE=64: ~24GB (边界，可能 OOM)
```

---

### 1.3 MFU (Model FLOPs Utilization) 的深入理解

#### 前两轮认知
- MFU 是衡量 GPU 利用效率的指标
- 典型值 30-50%

#### 本轮新学到
**MFU 的计算公式**:
```python
# train.py:587
mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
```

**影响 MFU 的因素**:

| 因素 | 影响 | 4090 特点 |
|------|------|----------|
| **内存带宽** | 数据加载速度 | 1TB/s vs H100 3.35TB/s |
| **计算密度** | 矩阵乘法效率 | 较高，但不如 H100 |
| **Flash Attention** | 注意力计算效率 | Community 版本效率较低 |
| **Batch Size** | GPU 利用率 | 需要足够大才能饱和 |

**4090 的 MFU 预期**:
- 单卡: 25-35% (vs H100 35-50%)
- 多卡: 15-25% (无 NVLink，通信开销大)

**提高 MFU 的方法**:
1. **增大 batch size**: 在内存允许范围内尽量大
2. **使用 torch.compile**: 减少 Python overhead
3. **优化数据加载**: 使用 pinned memory, non-blocking transfer
4. **减少 CPU-GPU 同步**: 避免频繁的 .item() 调用

---

## 二、分布式训练的新知识点

### 2.1 数据并行 vs 模型并行

#### 前两轮认知
- 数据并行：每卡完整模型，不同数据
- 模型并行：模型分片到多卡

#### 本轮新学到
**4090 的并行策略选择**:

| 策略 | 适用场景 | 4090 可行性 |
|------|---------|------------|
| **DDP (数据并行)** | 模型能放进单卡 | ✅ 推荐 |
| **FSDP (完全分片)** | 模型太大，单卡放不下 | ⚠️ 可能，但通信开销大 |
| **模型并行 (Tensor Parallel)** | 超大模型 | ❌ 不推荐，4090 无 NVLink |
| **实验并行** | 同时跑多个实验 | ✅ 最简单，推荐 |

**关键理解**:
1. **4090 没有 NVLink**: 多卡通信走 PCIe，带宽只有 32GB/s vs NVLink 900GB/s
2. **DDP 通信开销**: 每步需要 all-reduce 梯度，PCIe 带宽可能成为瓶颈
3. **实验并行更高效**: 每卡独立实验，无通信开销，可以同时探索多个方向

### 2.2 DDP 的实际实现细节

#### 关键代码修改
```python
# 原版 train.py
model = GPT(config)
model.to(device)

# DDP 版本
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 包装模型
model = GPT(config).to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 调整 batch size
# 原来: DEVICE_BATCH_SIZE = 128 (单卡)
# 现在: DEVICE_BATCH_SIZE = 32 (每卡), 总 batch = 32 * 8 = 256
```

#### 踩坑点
1. **随机种子**: 需要每卡设置相同的种子，否则数据增强不一致
2. **日志打印**: 只在 rank 0 打印，避免重复输出
3. **模型保存**: 只在 rank 0 保存，使用 `model.module.state_dict()`
4. **数据采样**: 需要 `DistributedSampler` 确保每卡数据不重叠

---

### 2.3 实验并行的最佳实践

#### 为什么选择实验并行？

**对比分析**:

| 方面 | DDP | 实验并行 |
|------|-----|---------|
| **代码改动** | 大 | 无 |
| **通信开销** | 有 | 无 |
| **调试难度** | 高 | 低 |
| **探索效率** | 一次一个方向 | 一次 8 个方向 |
| **Batch Size** | 可以更大 | 受单卡限制 |

**结论**: 对于 autoresearch 这种探索性实验，实验并行更合适

#### 实验设计
```bash
# 8 卡实验并行，每卡一个实验方向
GPU 0: 基线 (DEPTH=4, LR=0.04)
GPU 1: 更深模型 (DEPTH=6, LR=0.04)
GPU 2: 更大 batch (DEPTH=4, BS=48)
GPU 3: 更高学习率 (DEPTH=4, LR=0.06)
GPU 4: 不同窗口模式 (WINDOW_PATTERN=SL)
GPU 5: 不同 warmdown (WARMDOWN_RATIO=0.3)
GPU 6: 不同权重衰减 (WEIGHT_DECAY=0.1)
GPU 7: 组合优化 (DEPTH=5, BS=40, LR=0.05)
```

---

## 三、训练优化的新知识点

### 3.1 Learning Rate Scaling Law

#### 前两轮认知
- 学习率需要根据模型大小调整
- 通常用 sqrt scaling

#### 本轮新学到
**代码中的 LR Scaling**:
```python
# train.py:248-249
dmodel_lr_scale = (model_dim / 768) ** -0.5
print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
```

**关键理解**:
1. **为什么是 sqrt?**: 理论上，梯度的方差与模型维度成正比，所以 LR 需要 1/sqrt(dim)
2. **768 是基线**: 这是原项目的默认维度，LR 是在这个维度下调好的
3. **实际影响**: 如果 DEPTH=4，model_dim=256，LR 会放大 sqrt(3) ≈ 1.73 倍

**4090 适配建议**:
```python
# 如果减小模型，可能需要调整 LR
# DEPTH=4 → model_dim=256 → LR 放大 1.73x
# 但 4090 计算能力弱，可能需要适当减小 LR
MATRIX_LR = 0.03  # 从 0.04 减小到 0.03
```

---

### 3.2 Gradient Accumulation 的深入理解

#### 前两轮认知
- 用于在小 batch size 下模拟大 batch
- loss 需要除以累积步数

#### 本轮新学到
**为什么需要除以 grad_accum_steps?**

```python
# train.py:550
loss = loss / grad_accum_steps
loss.backward()
```

**数学原理**:
```
假设 TOTAL_BATCH_SIZE = 524K tokens
DEVICE_BATCH_SIZE = 32 * 2048 = 65K tokens
grad_accum_steps = 524K / 65K = 8

每次 backward:
- 计算 65K tokens 的梯度
- 累加到参数的 .grad

如果不除以 8:
- 最终梯度 = 8 * 真实梯度
- 相当于学习率放大了 8 倍

除以 8 后:
- 最终梯度 = 真实梯度
- 等价于用 524K tokens 计算一次梯度
```

**关键理解**:
1. **梯度累加是加法**: backward() 会把梯度加到 .grad 上
2. **loss 缩放是除法**: 为了平均，不是累加
3. **等价性**: 正确实现后，与大 batch 完全等价

**4090 适配**:
```python
# 原配置
TOTAL_BATCH_SIZE = 2**19  # 524K
DEVICE_BATCH_SIZE = 128 * 2048 = 262K
grad_accum_steps = 2

# 4090 配置
TOTAL_BATCH_SIZE = 2**18  # 262K
DEVICE_BATCH_SIZE = 32 * 2048 = 65K
grad_accum_steps = 4

# 注意: grad_accum_steps 越大，训练越慢（更多 Python overhead）
```

---

### 3.3 Warmup 和 Warmdown 的实际效果

#### 前两轮认知
- Warmup 避免初期大梯度
- Warmdown 稳定后期训练

#### 本轮新学到
**代码实现**:
```python
# train.py:518-525
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
```

**原配置分析**:
- WARMUP_RATIO = 0.0: 没有 warmup
- WARMDOWN_RATIO = 0.5: 后 50% 时间做 warmdown
- FINAL_LR_FRAC = 0.0: 最终 LR 降到 0

**为什么这样设计？**
1. **5 分钟训练太短**: Warmup 会浪费宝贵的训练时间
2. **Adam 有自适应性**: 可以自动调整 LR，不需要 warmup
3. **Warmdown 很重要**: 帮助收敛到更好的局部最优

**4090 适配建议**:
```python
# 如果延长训练时间到 10-15 分钟，可以考虑加 warmup
TIME_BUDGET = 900  # 15 分钟
WARMUP_RATIO = 0.1  # 前 10% 做 warmup
WARMDOWN_RATIO = 0.4  # 后 40% 做 warmdown
```

---

## 四、数据处理的新知识点

### 4.1 Best-fit Packing 算法深入

#### 前两轮认知
- 文档打包进固定长度的行
- 100% 利用率，无 padding

#### 本轮新学到
**算法实现细节**:
```python
# prepare.py:314-332
# Find largest doc that fits entirely
best_idx = -1
best_len = 0
for i, doc in enumerate(doc_buffer):
    doc_len = len(doc)
    if doc_len <= remaining and doc_len > best_len:
        best_idx = i
        best_len = doc_len
```

**时间复杂度**:
- 每次查找: O(buffer_size) = O(1000)
- 每行需要多次查找: 取决于文档大小分布
- 总体: O(batch_size * buffer_size * avg_splits_per_row)

**潜在瓶颈**:
1. **CPU 密集**: 查找是纯 CPU 操作
2. **Python 循环**: 效率不高
3. **频繁 pop**: 列表操作有开销

**优化方向**:
```python
# 可以用堆 (heap) 优化查找
import heapq

# 用最大堆存储文档，按长度排序
doc_heap = [(-len(doc), i, doc) for i, doc in enumerate(doc_buffer)]
heapq.heapify(doc_heap)

# 查找最佳文档: O(log n)
while doc_heap:
    neg_len, idx, doc = heapq.heappop(doc_heap)
    if -len(doc) <= remaining:
        return doc
```

---

### 4.2 BOS-aligned 设计的意义

#### 前两轮认知
- 每行以 BOS token 开头
- 帮助模型识别文档边界

#### 本轮新学到
**为什么 BOS-aligned 很重要？**

1. **文档边界识别**: 模型需要知道新文档从哪里开始
2. **因果注意力**: BOS 是"无条件"生成的起点，没有前面的 context
3. **训练-推理一致性**: 推理时通常也从 BOS 开始

**实际影响**:
```python
# prepare.py:286
bos_token = tokenizer.get_bos_token_id()

# 每个文档都 prepend BOS
token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
```

**如果不用 BOS-aligned 会怎样？**
1. **文档混淆**: 模型可能把不同文档的内容混在一起
2. **生成质量下降**: 不知道什么时候该停止生成
3. **评估不准**: val_bpb 计算可能受影响

---

### 4.3 数据加载的性能优化

#### 前两轮认知
- 使用 pinned memory
- non-blocking transfer

#### 本轮新学到
**代码实现**:
```python
# prepare.py:296-303
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
```

**Pinned Memory 的作用**:
1. **不在虚拟内存中**: 直接在物理内存中，不会被 swap
2. **DMA 传输**: 可以直接用 DMA 传输到 GPU，不需要 CPU 参与
3. **异步传输**: `non_blocking=True` 可以让传输和计算重叠

**4090 的注意事项**:
```python
# 4090 的 PCIe 带宽只有 32GB/s
# 如果数据传输成为瓶颈，可以：
# 1. 增加 buffer_size，预加载更多数据
# 2. 使用多个 worker 进程并行加载
# 3. 减小 batch size，减少每步传输量
```

---

## 五、优化器的新知识点

### 5.1 Muon 的实现细节深入

#### 前两轮认知
- Muon 对梯度正交化
- 用 Newton-Schulz 迭代

#### 本轮新学到
**Polar Express 系数**:
```python
# train.py:297-303
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
```

**这些系数是怎么来的？**
- 是通过多项式逼近极分解的最优系数
- 每组 (a, b, c) 对应一次迭代
- 5 次迭代足够收敛到高精度

**数学原理**:
```
极分解: A = UP, 其中 U 是正交矩阵，P 是半正定
目标: 计算 U（正交化的梯度）

Newton-Schulz 迭代:
X_{k+1} = a * X_k + X_k @ (b * X_k^T @ X_k + c * (X_k^T @ X_k)^2)

收敛条件: ||X_0|| < 1
所以需要先归一化: X = X / (||X|| * 1.02 + 1e-6)
```

**4090 的性能影响**:
```python
# Newton-Schulz 迭代需要矩阵乘法
# 4090 的矩阵乘法性能: ~330 TFLOPS
# 每次迭代: O(n^3) FLOPs
# 5 次迭代: 5 * O(n^3)

# 对于 50M 参数的模型，每步额外开销约 5-10%
# 可以接受
```

---

### 5.2 Cautious Weight Decay 的深入理解

#### 前两轮认知
- 只对与梯度方向一致的参数做 weight decay
- 避免对正在更新的参数施加惩罚

#### 本轮新学到
**代码实现**:
```python
# train.py:349-353
# Cautious weight decay + parameter update
lr = lr_t.to(g.dtype)
wd = wd_t.to(g.dtype)
mask = (g * stacked_params) >= 0
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

**数学解释**:
```
标准 weight decay:
θ = θ - lr * (g + wd * θ)

Cautious weight decay:
mask = (g * θ) >= 0  # 梯度和参数同号
θ = θ - lr * (g + wd * θ * mask)

为什么？
- 如果 g * θ >= 0: 梯度和参数同向，更新会让 θ 更远离 0
  → 需要 weight decay 来拉回
- 如果 g * θ < 0: 梯度和参数反向，更新会让 θ 更接近 0
  → 不需要 weight decay，否则会过度惩罚
```

**实际效果**:
1. **更稳定的训练**: 避免 weight decay 破坏正在学习的参数
2. **更好的泛化**: 只对"过拟合"的参数做正则化
3. **需要调整 weight decay**: 因为效果更温和，可能需要更大的 wd 值

---

### 5.3 NorMuon Variance Reduction

#### 前两轮认知
- Muon 有方差缩减机制
- 用 second momentum 记录

#### 本轮新学到
**代码实现**:
```python
# train.py:337-348
# NorMuon variance reduction
beta2 = beta2_t.to(g.dtype)
v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
red_dim_size = g.size(red_dim)
v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
v_norm = v_norm_sq.sqrt()
second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
g = g * final_scale.to(g.dtype)
```

**数学原理**:
```
目标: 对梯度进行归一化，减少方差

步骤:
1. 计算每行/列的方差: v_mean = mean(g^2, dim)
2. 计算总方差: v_norm = sqrt(sum(v_mean) * dim_size)
3. 用 second momentum 平滑: v_smooth = EMA(v_mean)
4. 计算缩放因子: scale = 1 / sqrt(v_smooth)
5. 保持总范数不变: final_scale = scale * (v_norm / new_v_norm)

效果:
- 每行/列的梯度被归一化到相似的尺度
- 减少了不同参数之间的尺度差异
- 类似于 LayerNorm，但是在梯度空间
```

**4090 的性能影响**:
```python
# 需要额外的 reduction 操作
# 在 4090 上，reduction 操作效率较低
# 可能比 H100 慢 10-20%

# 优化方向:
# 1. 减少 red_dim 的计算频率
# 2. 使用 fused kernel
```

---

## 六、评估指标的新知识点

### 6.1 BPB (Bits Per Byte) 的深入理解

#### 前两轮认知
- vocab-size-independent 的评估指标
- 计算 nats 然后转换为 bits

#### 本轮新学到
**代码实现**:
```python
# prepare.py:343-365
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

**关键细节**:
1. **Special tokens 被排除**: `mask = nbytes > 0` 只计算有字节的 token
2. **每个 token 的字节数不同**: 英文 1 字节，中文 3 字节
3. **评估集固定**: 始终用 shard_06542 评估

**为什么 BPB 比 Perplexity 更好？**
```
Perplexity = exp(cross_entropy)
- 依赖于 vocab size
- 不同 tokenizer 的 perplexity 不可比

BPB = cross_entropy / log(2) / bytes_per_token
- 不依赖 vocab size
- 不同 tokenizer 的 BPB 可比
- 直接反映"每个字节需要多少 bit 来编码"
```

**4090 适配**:
```python
# 评估时间可能较长
# 可以减少 EVAL_TOKENS 来加速
EVAL_TOKENS = 20 * 524288  # 从 40 * 524288 减半
```

---

### 6.2 评估的稳定性

#### 前两轮认知
- 评估使用固定的数据集
- 结果应该可重复

#### 本轮新学到
**影响评估稳定性的因素**:

1. **随机性**: 数据加载有随机性（best-fit packing）
2. **Batch size**: 不同 batch size 可能有微小差异
3. **数值精度**: BF16 有随机舍入

**提高稳定性的方法**:
```python
# 1. 固定随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 2. 使用固定的评估集
# prepare.py 已经固定用 shard_06542

# 3. 多次评估取平均
# 当前实现只评估一次，可以改为多次
```

---

## 七、工程实践的新知识点

### 7.1 torch.compile 的实际效果

#### 前两轮认知
- JIT 编译，减少 Python overhead
- 首次运行需要编译

#### 本轮新学到
**代码实现**:
```python
# train.py:508
model = torch.compile(model, dynamic=False)
```

**实际效果**:
1. **首次编译**: 需要 1-3 分钟，取决于模型大小
2. **稳态加速**: 通常 10-30%，取决于模型复杂度
3. **内存开销**: 编译后的代码占用额外内存

**4090 的注意事项**:
```python
# 4090 的 CUDA 核心较少，编译收益可能不如 H100
# 首次编译时间可能更长
# 建议: 先跑一次预热，再开始计时

# 预热代码
for _ in range(3):
    model(torch.randint(0, 1000, (1, 10)))
```

**调试技巧**:
```python
# 如果 torch.compile 出问题，可以回退到 eager mode
model = torch.compile(model, dynamic=False, disable=True)

# 或者用 reduce-overhead 模式
model = torch.compile(model, mode="reduce-overhead")
```

---

### 7.2 GC 管理的实际效果

#### 前两轮认知
- Python GC 会导致 ~500ms 停顿
- 在 step 0 冻结 GC

#### 本轮新学到
**代码实现**:
```python
# train.py:593-598
if step == 0:
    gc.collect()
    gc.freeze()
    gc.disable()
elif (step + 1) % 5000 == 0:
    gc.collect()
```

**gc.freeze() 的作用**:
1. **冻结已分配对象**: 不再追踪这些对象的引用
2. **减少 GC 开销**: GC 不需要扫描这些对象
3. **风险**: 如有循环引用，不会被回收

**为什么每 5000 步 collect？**
- 防止内存泄漏
- 清理临时对象
- 5000 步大约对应 30-60 秒

**4090 的注意事项**:
```python
# 4090 内存较小，可能需要更频繁地 GC
# 如果观察到内存持续增长，改为每 1000 步
elif (step + 1) % 1000 == 0:
    gc.collect()
```

---

### 7.3 混合精度训练的细节

#### 前两轮认知
- 用 BF16 减少内存和计算
- BF16 比 FP16 更稳定

#### 本轮新学到
**代码实现**:
```python
# train.py:462
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# 使用
with autocast_ctx:
    loss = model(x, y)
```

**BF16 vs FP16 对比**:

| 方面 | BF16 | FP16 |
|------|------|------|
| **动态范围** | 大（与 FP32 相同） | 小 |
| **精度** | 低（7 位尾数） | 高（10 位尾数） |
| **需要 loss scaling** | 否 | 是 |
| **溢出风险** | 低 | 高 |
| **4090 支持** | 是 | 是 |

**为什么 BF16 更适合训练？**
1. **不需要 loss scaling**: 省去额外的复杂度
2. **梯度更稳定**: 不会因为溢出变成 NaN
3. **与 FP32 一致的动态范围**: 不会因为范围不够而丢失信息

**4090 的 BF16 性能**:
```python
# 4090 的 BF16 性能: ~330 TFLOPS
# 4090 的 FP16 性能: ~660 TFLOPS (with sparsity)
# 理论上 FP16 更快，但需要 loss scaling，实际差距不大

# 建议: 保持用 BF16，简单且稳定
```

---

## 八、学习过程中的意外发现

### 8.1 WINDOW_PATTERN 的实际影响

#### 发现
原项目默认用 "SSSL"，但 README 建议小模型用 "L"

#### 分析
```python
# "SSSL" 模式:
# - 3 层 Short window (只看一半)
# - 1 层 Long window (看全部)
# - 最后一层必须是 Long

# "L" 模式:
# - 所有层都是 Long window

# 对于小模型 (DEPTH=4):
# - "SSSL" → 3 层 Short + 1 层 Long
# - "SL" → 1 层 Short + 1 层 Long (循环)
# - "L" → 所有层 Long
```

**实际影响**:
1. **Short window 省计算**: 注意力复杂度从 O(n²) 降到 O(n * window)
2. **但可能丢信息**: 长距离依赖在 Short 层被截断
3. **小模型更敏感**: 参数少，信息流更重要

**建议**:
```python
# 4090 配置，先用 "L" 保证不丢信息
WINDOW_PATTERN = "L"

# 如果想省计算，用 "SL"
WINDOW_PATTERN = "SL"
```

---

### 8.2 ASPECT_RATIO 的意义

#### 发现
`model_dim = depth * ASPECT_RATIO`

#### 分析
```python
# train.py:469-472
def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
```

**设计意图**:
1. **深度和宽度的权衡**: ASPECT_RATIO 固定，DEPTH 控制深度
2. **Head 数量自动计算**: model_dim / HEAD_DIM = num_heads
3. **对齐到 HEAD_DIM**: 确保 head_dim 是整数

**4090 配置**:
```python
# DEPTH=4, ASPECT_RATIO=64
# base_dim = 4 * 64 = 256
# model_dim = 256 (已经是 128 的倍数)
# num_heads = 256 / 128 = 2

# 注意: 只有 2 个 head，注意力可能不够
# 可以考虑减小 HEAD_DIM
HEAD_DIM = 64  # 从 128 减小到 64
# num_heads = 256 / 64 = 4
```

---

### 8.3 Value Embeddings 的层选择

#### 发现
```python
def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
```

#### 分析
**对于 DEPTH=8**:
- layer 0: 0 % 2 == 7 % 2 → False
- layer 1: 1 % 2 == 7 % 2 → True
- layer 2: 2 % 2 == 7 % 2 → False
- ...
- layer 7: 7 % 2 == 7 % 2 → True

**结果**: 只有奇数层有 Value Embeddings

**对于 DEPTH=4**:
- layer 0: 0 % 2 == 3 % 2 → False
- layer 1: 1 % 2 == 3 % 2 → True
- layer 2: 2 % 2 == 3 % 2 → False
- layer 3: 3 % 2 == 3 % 2 → True

**结果**: 只有 layer 1 和 3 有 Value Embeddings

**实际影响**:
1. **减少参数**: Value Embeddings 参数量约 50% of word embeddings
2. **信息流**: 只在部分层注入额外信息
3. **可能不够**: 对于小模型，每层都加 VE 可能更好

**建议**:
```python
# 如果想每层都有 Value Embeddings，修改 has_ve
def has_ve(layer_idx, n_layer):
    return True  # 每层都有
```

---

## 九、总结与下一步

### 9.1 核心新学到的点

| 类别 | 新学到的点 |
|------|-----------|
| **硬件适配** | Compute Capability 影响 FA 版本选择 |
| **内存计算** | 激活值是内存大头，batch size 线性影响 |
| **分布式训练** | 实验并行比 DDP 更适合探索性实验 |
| **训练优化** | LR scaling law, cautious weight decay 的数学原理 |
| **数据处理** | Best-fit packing 的时间复杂度，BOS-aligned 的意义 |
| **优化器** | Polar Express 系数的来源，NorMuon variance reduction |
| **评估指标** | BPB vs Perplexity 的数学区别 |
| **工程实践** | torch.compile 的首次编译开销，GC freeze 的风险 |

### 9.2 下一步行动

1. **立即执行**: 按照 REPRODUCTION_PLAN.md 开始 Phase 1
2. **重点验证**: Flash Attention 在 4090 上的兼容性
3. **性能测试**: 测量实际的 MFU 和内存使用
4. **迭代优化**: 根据测试结果调整参数

### 9.3 待解决的问题

1. **Flash Attention 效率**: Community 版本在 4090 上的效率如何？
2. **最佳 DEPTH**: 4、6、8 哪个性价比最高？
3. **多卡通信**: PCIe 带宽是否成为瓶颈？
4. **Agent 适配**: 如何让 Agent 适应 4090 的配置？

---

**下一步**: 开始执行 Phase 1，记录实际遇到的问题和解决方案。
