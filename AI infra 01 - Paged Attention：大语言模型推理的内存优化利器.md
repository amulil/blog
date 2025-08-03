# AI infra 01 - Paged Attention：大语言模型推理的内存优化利器

> *"Creativity is just connecting things. When you ask creative people how they did something, they feel a little guilty because they didn't really do it, they just saw something."*
> —— **Steve Jobs**

## 引言

在大语言模型（LLM）推理过程中，**内存管理**是一个至关重要的挑战。随着模型规模和序列长度的增长，传统的注意力机制会消耗大量内存，尤其是 KV Cache（Key-Value Cache）的存储。

**Paged Attention** 正是"站在巨人肩膀上"的创新典范——它巧妙地将**操作系统中成熟的虚拟内存分页管理思想**迁移到深度学习领域，通过"分页"的方式彻底重新定义了注意力机制的内存管理。这种跨领域的技术借鉴，完美诠释了乔布斯所说的"创造力就是连接事物"。

本文将基于一个完整的代码实现，深入浅出地解释 Paged Attention 如何将操作系统的智慧运用到 AI 推理中，具体代码可以参考 [cleanvllm](https://github.com/amulil/cleanvllm/blob/main/qwen3_0_6B.py)。

## 一、传统 Attention 的内存挑战

### 1.1 KV Cache 的作用

在自回归语言模型中，每生成一个新 token，都需要计算该 token 与之前所有 token 的注意力。为了避免重复计算，我们将之前计算过的 Key 和 Value 缓存起来，这就是 **KV Cache**。

```python
# 传统方式：每次都重新计算所有位置的 Key 和 Value
for i in range(seq_len):
    q_i = query[i]
    k_all = key[:i+1]  # 从第0个到第i个token的key
    v_all = value[:i+1]  # 从第0个到第i个token的value
    attention_output[i] = attention(q_i, k_all, v_all)

# 使用KV Cache：复用之前计算的结果
for i in range(seq_len):
    if i == 0:
        kv_cache = [key[0], value[0]]
    else:
        kv_cache.append([key[i], value[i]])  # 只计算新token
    attention_output[i] = attention(query[i], kv_cache)
```

### 1.2 内存碎片化问题

传统的 KV Cache 管理存在以下问题：

1. **内存浪费**：每个序列的KV Cache通常按最大长度预分配，导致短序列浪费内存
2. **内存碎片**：不同长度的序列结束后，留下大小不一的内存碎片
3. **难以复用**：相同前缀的序列无法共享KV Cache

## 二、Paged Attention 的核心思想

### 2.1 操作系统的启发

Paged Attention 借鉴了操作系统中**虚拟内存分页管理**的思想：

- 将连续的逻辑地址空间映射到不连续的物理内存页面
- 通过页表管理逻辑地址到物理地址的映射
- 实现内存的灵活分配和高效利用

### 2.2 KV Cache 的分块管理

在 Paged Attention 中：

```python
class Block:
    """内存块类 - 对应操作系统中的内存页"""
    def __init__(self, block_id):
        self.block_id = block_id        # 块ID（物理地址）
        self.ref_count = 0              # 引用计数
        self.hash = -1                  # 内容哈希（用于去重）
        self.token_ids = []             # 存储的token序列
```

每个Block存储固定数量的 token 的 KV 值：

```python
# 配置
block_size = 256  # 每个块存储256个token的KV值

# 一个长度为1000的序列需要的块数
seq_len = 1000
num_blocks = (seq_len + block_size - 1) // block_size  # = 4个块
```

### 2.3 块表（Block Table）映射

每个序列维护一个**块表**，记录逻辑位置到物理块的映射：

```python
class Sequence:
    def __init__(self, token_ids, sampling_params):
        self.token_ids = token_ids
        self.block_table = []           # 块表：[物理块ID1, 物理块ID2, ...]
    
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    def block(self, i):
        """获取第i个逻辑块的内容"""
        start = i * self.block_size
        end = (i + 1) * self.block_size
        return self.token_ids[start:end]
```

## 三、核心组件实现解析

### 3.1 BlockManager：内存池管理器

`BlockManager` 是 Paged Attention 的核心，负责管理所有内存块：

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}      # 哈希 -> 块ID映射（用于去重）
        self.free_block_ids = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids = set()     # 已使用块集合
```

#### 3.1.1 块分配算法

```python
def allocate(self, seq: Sequence):
    """为序列分配内存块"""
    h = -1
    cache_miss = False
  
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)  # 获取第i个逻辑块的token
    
        # 计算块的哈希值（用于去重）
        if len(token_ids) == self.block_size:
            h = compute_hash(token_ids, h)  # 累积哈希
        else:
            h = -1  # 不完整的块不参与去重
        
        # 检查是否已存在相同内容的块
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        
        if cache_miss:
            # 分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 复用现有块
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                # 增加引用计数
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                # 重新激活块
                block = self._allocate_block(block_id)
            
        # 更新块内容和映射
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        
        seq.block_table.append(block_id)
```

#### 3.1.2 前缀缓存优化

前缀缓存是 Paged Attention 的重要特性：

```python
def compute_hash(token_ids: list[int], prefix: int = -1):
    """计算累积哈希，支持前缀复用"""
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 包含前缀哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**示例场景**：

```python
# 两个序列有共同前缀
seq1: "请介绍一下人工智能的发展历史"
seq2: "请介绍一下人工智能的应用领域"
#     |-------- 共同前缀 -------|

# 前缀块可以被两个序列共享
# seq1.block_table = [共享块1, 共享块2, 独有块3]
# seq2.block_table = [共享块1, 共享块2, 独有块4]
```

### 3.2 Context：推理上下文管理

`Context` 类管理推理时的各种元数据：

```python
@dataclass
class Context:
    is_prefill: bool = False            # 是否为预填充阶段
    cu_seqlens_q: torch.Tensor = None   # 查询序列累积长度
    cu_seqlens_k: torch.Tensor = None   # 键值序列累积长度
    max_seqlen_q: int = 0               # 最大查询序列长度
    max_seqlen_k: int = 0               # 最大键值序列长度
    slot_mapping: torch.Tensor = None   # 槽位映射
    context_lens: torch.Tensor = None   # 上下文长度
    block_tables: torch.Tensor = None   # 块表
```

#### 3.2.1 槽位映射（Slot Mapping）

槽位映射将逻辑位置转换为物理存储位置：

```python
def prepare_prefill(self, seqs: list[Sequence]):
    slot_mapping = []
    for seq in seqs:
        for i in range(start_pos, len(seq)):
            block_idx = i // self.block_size        # 逻辑块索引
            block_offset = i % self.block_size      # 块内偏移
            physical_block_id = seq.block_table[block_idx]  # 物理块ID
            slot = physical_block_id * self.block_size + block_offset
            slot_mapping.append(slot)
    return torch.tensor(slot_mapping)
```

### 3.3 Attention 计算实现

#### 3.3.1 KV Cache 存储

使用 Triton 内核高效存储 KV 值：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, value_ptr, k_cache_ptr, v_cache_ptr, 
    slot_mapping_ptr, D: tl.constexpr):
    """Triton内核：将KV值存储到对应槽位"""
    idx = tl.program_id(0)
  
    # 加载Key和Value
    key = tl.load(key_ptr + idx * D + tl.arange(0, D))
    value = tl.load(value_ptr + idx * D + tl.arange(0, D))
  
    # 获取目标槽位
    slot = tl.load(slot_mapping_ptr + idx)
  
    # 存储到KV Cache
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

#### 3.3.2 注意力计算

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    context = get_context()
  
    # 存储当前KV到Cache
    store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
  
    if HAS_FLASH_ATTN:
        if context.is_prefill:
            # 预填充阶段：使用Flash Attention处理变长序列
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True
            )
        else:
            # 解码阶段：使用KV Cache
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), self.k_cache, self.v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
    else:
        # 备用：PyTorch原生实现
        o = self._pytorch_attention(q, k, v, context, self.k_cache, self.v_cache)
  
    return o
```

## 四、内存分配策略

### 4.1 KV Cache 内存估算

```python
def allocate_kv_cache(self, gpu_memory_utilization: float):
    """分配KV Cache内存"""
    total_memory, used_memory, free_memory = get_gpu_memory()
  
    # 计算每个块的内存大小
    head_dim = self.config.hf_config.hidden_size // self.config.hf_config.num_attention_heads
    num_kv_heads = self.config.hf_config.num_key_value_heads
    num_layers = len(self.model.model.layers)
  
    # 内存大小 = 块大小 × 头维度 × KV头数 × 2(K和V) × 2(float16) × 层数
    bytes_per_block = (self.block_size * head_dim * num_kv_heads * 
                      2 * 2 * num_layers)
  
    # 基于可用内存计算块数量
    available_memory = free_memory * gpu_memory_utilization
    num_blocks = max(1, int(available_memory // bytes_per_block))
  
    print(f"KV Cache: {num_blocks} blocks, "
          f"{bytes_per_block//1024//1024:.1f}MB per block")
```

### 4.2 动态内存管理

```python
def may_append(self, seq: Sequence):
    """序列增长时的动态内存管理"""
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
  
    if len(seq) % self.block_size == 1:
        # 需要新的块
        assert last_block.hash != -1  # 上一块应该已经完成
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)
    
    elif len(seq) % self.block_size == 0:
        # 当前块已满，计算哈希
        token_ids = seq.block(seq.num_blocks-1)
        prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = compute_hash(token_ids, prefix_hash)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

## 五、调度器集成

### 5.1 序列调度

```python
class Scheduler:
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
    
        # 预填充阶段：调度等待队列中的序列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
                not self.block_manager.can_allocate(seq)):
                break
            
            self.block_manager.allocate(seq)  # 分配内存块
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # 预填充阶段
        
        # 解码阶段：处理运行中的序列
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            if not self.block_manager.can_append(seq):
                self.preempt(seq)  # 抢占处理
                break
            else:
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
            
        return scheduled_seqs, False  # 解码阶段
```

### 5.2 内存抢占机制

```python
def preempt(self, seq: Sequence):
    """抢占序列，释放其内存块"""
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # 释放内存
    self.waiting.appendleft(seq)        # 重新排队
```

## 六、Paged Attention 的优势

### 6.1 内存效率

1. **消除内存碎片**：固定大小的块避免了碎片化
2. **按需分配**：只为实际使用的token分配内存
3. **前缀共享**：相同前缀的序列共享内存块

### 6.2 前缀缓存效果

```python
# 示例：批量问答场景
prompts = [
    "根据以下文档回答问题：[长文档内容...] 问题1：...",
    "根据以下文档回答问题：[长文档内容...] 问题2：...",
    "根据以下文档回答问题：[长文档内容...] 问题3：...",
]

# 传统方式：每个序列都存储完整的文档tokens
# Paged Attention：文档部分只存储一次，被多个序列共享
# 内存节省率 = (N-1) * 前缀长度 / (N * 总长度)
```

### 6.3 灵活的批处理

```python
# 不同长度的序列可以高效地批处理
batch = [
    "短问题",                    # 需要1个块
    "中等长度的问题描述...",      # 需要2个块
    "很长的问题描述..." * 100,   # 需要10个块
]
# 传统方式：按最长序列预分配，浪费大量内存
# Paged Attention：每个序列按需分配，内存利用率接近100%
```

## 七、性能优化技巧

TODO：这里还需要深入思考 + 完善

### 7.1 块大小选择

```python
# 块大小的权衡
block_size = 256  # 推荐值

# 过小的块：
# - 优点：内存浪费少，灵活性高
# - 缺点：管理开销大，前缀共享效果差

# 过大的块：
# - 优点：管理开销小，前缀共享效果好
# - 缺点：内存浪费多，灵活性差
```

### 7.2 CUDA Graph 优化

```python
def capture_cudagraph(self):
    """捕获CUDA图优化解码阶段"""
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # 预分配变量
            input_ids = torch.zeros(bs, dtype=torch.int64, device="cuda")
            # ... 其他变量
            outputs = self.model(input_ids, positions)
        self.graphs[bs] = graph
```

### 7.3 异步预填充

```python
# 预填充和解码可以并行执行
# 当解码阶段进行时，新的序列可以在后台进行预填充
```

## 八、总结

Paged Attention 通过将连续的逻辑序列映射到不连续的物理内存块，实现了：

1. **高效的内存利用**：消除碎片，按需分配
2. **智能的前缀缓存**：自动识别和共享相同前缀
3. **灵活的批处理**：支持变长序列的高效批处理
4. **简单的扩展性**：通过增加块数量轻松扩展容量

这种设计使得大语言模型能够在有限的内存资源下处理更多的并发请求，显著提升了推理服务的吞吐量和效率。

通过理解 Paged Attention 的原理和实现，我们可以更好地优化大语言模型的推理性能，为实际应用提供更强大的技术支撑。

---

**参考实现**：本文基于完整的 Qwen3-0.6B 模型实现，包含了 Paged Attention 的所有核心组件，可直接运行和学习。
