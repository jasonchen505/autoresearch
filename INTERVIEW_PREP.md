# AutoResearch 面试准备指南

> 基于 Karpathy 的 autoresearch 框架，针对 LLM 算法实习岗位的深度准备

---

## 一、项目概述与核心理念

### 1.1 项目背景

AutoResearch 是 Andrej Karpathy 在 2026 年提出的**自动化 AI 研究框架**。核心思想是让 AI Agent 自主进行 LLM 训练实验，实现"睡一觉起来就有更好的模型"。

**关键引用**（来自 README）:
> "One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun... That era is long gone."

### 1.2 设计哲学

| 设计选择 | 原因 |
|---------|------|
| **单文件修改** (`train.py`) | 保持范围可控，diff 可审查 |
| **固定时间预算** (5分钟) | 实验可直接比较，不受硬件影响 |
| **自包含** | 无外部依赖，单 GPU、单文件、单指标 |
| **Markdown 作为指令** (`program.md`) | 轻量级"技能"定义，人类编写，Agent 执行 |

---

## 二、技术架构深度解析

### 2.1 项目结构

```
autoresearch/
├── prepare.py      # 数据准备 + 运行时工具（只读）
├── train.py        # 模型 + 优化器 + 训练循环（Agent 修改）
├── program.md      # Agent 指令（人类修改）
└── pyproject.toml  # 依赖
```

### 2.2 核心指标：val_bpb (Bits Per Byte)

**为什么用 BPB 而不是 perplexity？**

```python
# prepare.py:344-365
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte.
    """
    token_bytes = get_token_bytes(device="cuda")
    # ...
    total_nats += (loss_flat * mask).sum().item()
    total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

**面试深挖点**:
- BPB vs Perplexity: BPB 是 vocab-size-independent 的，不同 tokenizer 配置可以公平比较
- 计算公式: `BPB = total_nats / (ln(2) * total_bytes)`
- 为什么排除 special tokens (byte length = 0)?

### 2.3 模型架构 (train.py:32-292)

#### GPTConfig 数据类

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

#### 关键架构组件

**1. RoPE (Rotary Position Embeddings)**

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

**面试考察点**:
- RoPE 的数学原理：通过旋转矩阵编码位置信息
- 为什么 RoPE 比 learned position embeddings 更好？
- RoPE 的长度外推问题

**2. Value Embeddings (Value Residual / ResFormer)**

```python
# train.py:83-87
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

**面试深挖**:
- 这是 ResFormer 论文的技术
- 通过 input-dependent gate 混合 value embedding
- 交替层使用（`has_ve` 函数）

**3. 滑动窗口注意力**

```python
def _compute_window_sizes(self, config):
    pattern = config.window_pattern.upper()  # "SSSL"
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
```

**面试考察**:
- SSSL 模式：Short-Short-Short-Long
- 为什么最后一层必须是 Long window？
- 滑动窗口 vs 全注意力的 trade-off

**4. RMSNorm**

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

**5. ReLU² 激活**

```python
def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()  # ReLU squared
    x = self.c_proj(x)
    return x
```

**面试问题**: ReLU² vs GELU vs SwiGLU 的比较

**6. Logit Soft-capping**

```python
softcap = 15
logits = self.lm_head(x)
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)
```

**面试深挖**: 为什么要 softcap？防止 logits 爆炸，稳定训练

**7. 残差连接优化**

```python
# train.py:276-277
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

**面试考察**: 这是 DeepNet 风格的残差缩放，`x0` 是初始 embedding

### 2.4 优化器：MuonAdamW

这是项目最独特的部分！

#### Muon 优化器原理

```python
# train.py:316-353
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, ...):
    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    # Newton-Schulz iteration for orthogonalization
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B

    # NorMuon variance reduction
    # ...
```

**面试深挖点**:

1. **Muon 的核心思想**: 对梯度进行正交化处理，类似于 Natural Gradient 的近似
2. **Polar Express**: 使用 Newton-Schulz 迭代计算矩阵的极分解
3. **为什么只对 2D matrix params 使用 Muon?**
   - Embedding 和 lm_head 是 1D 或非矩阵结构
   - 只有 transformer block 中的线性层是 2D 矩阵

4. **Cautious Weight Decay**

```python
mask = (g * stacked_params) >= 0
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

**面试问题**: 什么是 cautious weight decay? 为什么需要?

#### 参数组策略

```python
param_groups = [
    dict(kind='adamw', params=lm_head_params, lr=0.004, ...),
    dict(kind='adamw', params=embedding_params, lr=0.2, ...),
    dict(kind='muon', params=matrix_params, lr=0.04, ...),
]
```

**面试考察**: 不同参数组使用不同优化器和学习率的原因

### 2.5 学习率调度

```python
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
```

**面试考察**: Warmup + Warmdown 策略，为什么没有 cosine schedule?

### 2.6 数据加载与打包

```python
def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit.
    100% utilization (no padding).
    """
```

**面试深挖**:
- Best-fit packing 算法
- 为什么每行都要以 BOS 开头?
- 如何处理文档边界?

---

## 三、Agent 自动化研究流程

### 3.1 实验循环 (program.md)

```
LOOP FOREVER:
1. Look at the git state
2. Tune train.py with an experimental idea
3. git commit
4. Run: uv run train.py > run.log 2>&1
5. Extract results: grep "^val_bpb:" run.log
6. If improved → keep commit
7. If worse → git reset
```

**面试考察点**:
- 这是一个经典的 **explore-exploit** 问题
- 使用 git 作为版本控制和回滚机制
- 结果记录在 `results.tsv` (不提交到 git)

### 3.2 Agent 的约束与自由度

**可以做的**:
- 修改模型架构
- 修改优化器
- 修改超参数
- 修改 batch size

**不能做的**:
- 修改 `prepare.py`
- 安装新包
- 修改评估函数

**面试深挖**: 为什么这样设计？确保实验的公平性和可比性

---

## 四、面试考察点与深挖问题

### 4.1 LLM 基础知识

#### Q1: 解释 Transformer 的自注意力机制

**要点**:
- QKV 矩阵
- Scaled dot-product attention
- Multi-head attention
- 复杂度 O(n²d)

#### Q2: RoPE 的工作原理

**深挖**:
- 旋转矩阵的数学推导
- 为什么 RoPE 能编码相对位置?
- RoPE 的长度外推问题及解决方案 (NTK-aware, YaRN, etc.)

#### Q3: 解释不同的位置编码方案

- Learned absolute position
- Relative position (T5)
- RoPE (LLaMA, Mistral)
- ALiBi

#### Q4: 为什么使用 RMSNorm 而不是 LayerNorm?

**要点**:
- 计算效率更高
- 没有 re-centering，只有 re-scaling
- 实践中效果相当

#### Q5: 比较不同的激活函数

| 激活函数 | 特点 |
|---------|------|
| ReLU | 简单，但有 dead neuron 问题 |
| GELU | 平滑，常用 |
| SwiGLU | LLaMA 使用，需要更多参数 |
| ReLU² | 本项目使用，稀疏性更好 |

### 4.2 优化器相关

#### Q6: 解释 Adam 优化器

**要点**:
- 一阶矩估计 (momentum)
- 二阶矩估计 (adaptive learning rate)
- Bias correction
- AdamW 的 weight decay 实现

#### Q7: 什么是 Muon 优化器? 为什么有效?

**深挖**:
- Muon = Momentum + Orthogonalization
- 使用 Newton-Schulz 迭代进行极分解
- 类似于 Natural Gradient 的低秩近似
- 为什么只对矩阵参数有效?

#### Q8: 解释 gradient accumulation

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()
```

**面试考察**: 为什么需要? 什么时候用? 如何正确实现?

### 4.3 训练技巧

#### Q9: 为什么需要 learning rate warmup?

**要点**:
- 避免训练初期的大梯度破坏参数
- Adam 的 bias correction 在初期不够
- 与 batch size 的关系

#### Q10: 解释 mixed precision training

```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
```

**深挖**:
- FP16 vs BF16 的区别
- 为什么 BF16 更适合训练?
- Loss scaling 的必要性

#### Q11: 什么是 MFU (Model FLOPs Utilization)?

```python
mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
```

**面试考察**: 如何计算? 影响因素? 典型值是多少?

### 4.4 Agent 与自动化研究

#### Q12: 如何设计一个自动化的研究框架?

**要点**:
- 明确的目标函数 (val_bpb)
- 可控的实验环境
- 自动化的实验循环
- 版本控制和回滚机制

#### Q13: Agent 如何做出实验决策?

**深挖**:
- 探索 vs 利用的平衡
- 如何避免局部最优?
- 如何利用历史实验结果?

#### Q14: 这个框架的局限性是什么?

- 只能修改单个文件
- 固定时间预算限制了大规模实验
- 没有多 GPU 支持
- 评估指标单一 (只有 val_bpb)

### 4.5 代码细节深挖

#### Q15: 解释 `torch.compile` 的作用

```python
model = torch.compile(model, dynamic=False)
```

**要点**:
- JIT 编译
- 图优化
- 减少 Python overhead
- `dynamic=False` 假设输入形状固定

#### Q16: 解释 pinned memory 和 non-blocking transfer

```python
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
gpu_buffer.copy_(cpu_buffer, non_blocking=True)
```

**面试考察**: 为什么? 如何优化数据加载?

#### Q17: 解释 GC 管理策略

```python
if step == 0:
    gc.collect()
    gc.freeze()
    gc.disable()
elif (step + 1) % 5000 == 0:
    gc.collect()
```

**深挖**: Python GC 会造成什么问题? 为什么 freeze?

---

## 五、项目介绍建议 (面试用)

### 5.1 一分钟版本

> "AutoResearch 是 Karpathy 提出的自动化 LLM 研究框架。核心思想是让 AI Agent 自主修改训练代码、运行实验、评估结果，实现持续优化。我在研究中深入分析了它的架构设计，包括使用 RoPE、Value Residual、Muon 优化器等现代技术，以及如何通过固定时间预算和单一指标实现实验的公平比较。这个项目让我对 LLM 训练的全栈有了深入理解。"

### 5.2 三分钟版本

**背景**:
- 项目目的：自动化 LLM 研究
- 核心约束：5分钟训练，单 GPU，单文件修改

**技术亮点**:

1. **模型架构**: 现代 GPT 变体，包含 RoPE、Value Embeddings、滑动窗口注意力
2. **优化器**: MuonAdamW，结合了 Muon 的正交化和 Adam 的自适应性
3. **数据处理**: Best-fit packing，100% 利用率，BOS-aligned
4. **评估**: BPB 指标，vocab-size-independent

**Agent 设计**:
- 使用 git 进行版本控制
- 自动化的实验循环
- 基于结果的保留/丢弃决策

**我的收获**:
- 理解了 LLM 训练的全栈
- 学会了如何设计可重复的实验
- 掌握了现代优化器的原理

### 5.3 五分钟版本 (添加代码细节)

在三分钟版本基础上，添加：
- 具体的代码片段解释
- 与其他实现的比较 (如 nanoGPT, LLaMA)
- 自己的改进想法

---

## 六、延伸学习建议

### 6.1 相关论文

1. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
2. **Value Residual**: "ResFormer: Scaling ViTs with Multi-Resolution Training"
3. **Muon**: "Muon: An Orthogonal Optimizer for Large-Scale Deep Learning"
4. **RMSNorm**: "Root Mean Square Layer Normalization"
5. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention"

### 6.2 相关项目

- **nanoGPT**: Karpathy 的简单 GPT 实现
- **nanochat**: 本项目的父项目
- **LLaMA**: Meta 的开源 LLM
- **Mistral**: 滑动窗口注意力

### 6.3 实践建议

1. **跑一遍代码**: 理解整个流程
2. **尝试修改**: 改变超参数，观察效果
3. **阅读源码**: PyTorch 的 attention 实现
4. **复现论文**: 尝试实现一个简单的优化器改进

---

## 七、常见面试陷阱

### 陷阱1: "这个项目很简单，就是调参"

**正确回应**: 强调架构设计、优化器创新、Agent 设计的深度

### 陷阱2: "为什么不直接用现成的框架?"

**正确回应**: 解释简化设计的好处：可控性、可理解性、实验公平性

### 陷阱3: "5分钟能训练出什么好模型?"

**正确回应**: 这是研究方法论，不是追求 SOTA；重点是实验效率和自动化

### 陷阱4: "这个项目有什么实际应用?"

**正确回应**: 
- 快速原型验证
- 超参数搜索
- 架构探索
- 自动化机器学习研究

---

## 八、关键代码片段速查

### 8.1 模型初始化

```python
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
```

### 8.2 训练循环核心

```python
while True:
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    # Update LR, momentum, weight decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
```

### 8.3 评估函数

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    # 计算 bits per byte
    return total_nats / (math.log(2) * total_bytes)
```

---

## 九、面试模拟问答

### Q: 请介绍一下你的项目

A: "我研究了 Karpathy 的 AutoResearch 框架，这是一个让 AI Agent 自主进行 LLM 研究的系统。核心设计是让 Agent 修改训练代码，运行5分钟实验，根据结果决定保留或丢弃。技术上，它使用了 RoPE、Value Residual、Muon 优化器等现代技术。这个项目让我深入理解了 LLM 训练的全栈。"

### Q: 你从这个项目学到了什么?

A: "三个方面：1) LLM 训练的技术细节，如优化器设计、位置编码；2) 实验方法论，如何设计公平可比的实验；3) Agent 设计，如何让 AI 自主进行研究。"

### Q: 如果让你改进这个项目，你会怎么做?

A: "几个方向：1) 支持多 GPU 分布式训练；2) 添加更多评估指标，如生成质量；3) 使用更先进的 Agent 策略，如 MCTS；4) 支持更多模型架构。"

### Q: 解释 Muon 优化器为什么有效

A: "Muon 的核心是对梯度进行正交化处理。它使用 Newton-Schulz 迭代计算梯度矩阵的极分解，这相当于近似自然梯度。自然梯度考虑了参数空间的几何结构，因此收敛更快。Muon 只对2D矩阵参数有效，因为正交化需要矩阵结构。"

---

## 十、总结

这个项目虽然代码量不大，但涵盖了 LLM 训练的几乎所有核心概念：
- 模型架构 (Transformer, RoPE, Value Residual)
- 优化器 (Adam, Muon, 学习率调度)
- 训练技巧 (Mixed Precision, Gradient Accumulation)
- 评估方法 (BPB)
- Agent 设计 (自动化实验)

掌握这个项目，就掌握了 LLM 训练的核心知识，足以应对大多数面试场景。
