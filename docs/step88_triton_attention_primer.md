# 从零理解 Triton unified attention

> 面向“只写过最基本 Triton kernel”的读者。
> 目标：从 GPU 线程/内存讲起，解释 K/V layout 为什么会影响性能、Astra 发现了什么、`cp.async` 到底是什么。
> 代码对象：`minivllm/attention/ops/triton_unified_attention.py`。
> 当前版本包含 Astra 的 stride 特化；basic 版对应 commit `f3bd003`。

## 0. 先记住三句话

```text
1. GPU kernel 的性能，很多时候不是“算得慢”，而是“数据搬得慢”。
2. 同一个逻辑 tensor，可以有不同的“线程持有方式”，这就是 layout。
3. cp.async 不是少搬数据，而是让“搬下一块”和“算当前块”同时进行。
```

---

## 1. GPU 是怎么执行 kernel 的

### 1.1 thread / warp / block / SM

```text
thread   一个线程
warp     32 个线程，一起执行同一条指令
block    一组 warp，也叫 CTA；一个 block 在一个 SM 上
SM       显卡里的计算单元，一个 SM 可以同时跑多个 block
```

Triton 里的“一个 program”通常对应一个 **block/CTA**。

```python
pid = tl.program_id(0)
```

就是“我是第几个 block”。

### 1.2 内存层级

从快到慢：

```text
寄存器 register       最快，每个线程私有
共享内存 shared       一个 block 内共享，很快，但容量小
L1/L2 cache          较近
全局显存 global       很大，但延迟高
local memory         寄存器不够时的“溢出区”，实际在显存里，很慢
```

一个 block 的 shared 容量有限。  
本机 RTX 5090 上，一个 CTA 的 shared 上限约 101 KB；申请太多会编译失败。

### 1.3 为什么“搬数据”是主角

Attention 的每一步都要读 K/V：

```text
Q @ K -> 读 K
P @ V -> 读 V
```

如果 K/V 在全局显存里，每次读都有几百个 cycle 的延迟。  
GPU 靠“很多 warp 同时跑”来掩盖延迟；但如果：

- 每个线程只读 1 个元素；
- 数据布局不连续；
- 寄存器不够，变量被 spill 到 local memory；

就会变成“大家都在等内存”。

---

## 2. Triton 在做什么

### 2.1 逻辑 tensor 和 block

Triton 里写：

```python
offs = tl.arange(0, 128)
x = tl.load(ptr + offs)
```

你看到的是一个长度为 128 的逻辑 tensor。  
但 GPU 上，它是**分配给很多线程**的：

```text
thread0 拿哪些元素？
thread1 拿哪些元素？
每个线程拿几个连续元素？
```

这就是 layout。

### 2.2 `tl.load` 不只是“读内存”

```python
x = tl.load(ptr + offs)
```

它同时决定了：

- 每个线程读哪些地址；
- 一次读几个连续元素（标量/向量）；
- 读出来的 tensor 在寄存器里怎么排；
- 后面 `tl.dot` 还需不需要重排。

### 2.3 `tl.dot` 需要特定摆放

`tl.dot(A, B)` 在 NVIDIA GPU 上通常走 Tensor Core。  
Tensor Core 对操作数的摆放有要求：

```text
A 的线程分布
B 的线程分布
shared 里的排布
ldmatrix 怎么读
```

如果 `tl.load` 读出来的摆放和 `tl.dot` 要的不一致，编译器就要插入：

```text
convert_layout
```

通常意味着：

```text
寄存器 -> 写 shared -> barrier -> 再读回寄存器
```

这一步很贵。

---

## 3. 最基本的 attention kernel

### 3.1 数学

对每个 query：

```text
S = Q @ K^T
P = softmax(S)
O = P @ V
```

在 kernel 里不是一次算完整序列，而是按 KV tile 循环：

```text
for 每个 KV tile:
    load K/V tile
    S_tile = Q @ K_tile
    mask
    online softmax 更新 M/L/P
    acc += P_tile @ V_tile
out = acc / L
```

### 3.2 一个 program 做什么

2D 路径：

```text
grid = (query_blocks, num_kv_heads)
```

一个 program 负责：

```text
一个 query block
× 一个 KV head
```

它串行扫这个请求的 KV tile。

### 3.3 四个容易混的名字

| 名字 | 含义 |
|---|---|
| `TILE_SIZE` | 一次循环处理多少个连续 KV 位置 |
| `BLOCK_SIZE` | 物理 KV cache 一个 block 有多少 slot |
| `BLOCK_M` | 一个 program 一次处理多少 query/GQA 行 |
| `BLOCK_Q` | 一个 program 覆盖多少个 query token |

例子：

```text
TILE_SIZE=32
BLOCK_SIZE=16
```

一个 KV tile 会跨 2 个物理 cache block。

```text
BLOCK_M=8
GROUP=8
BLOCK_Q=1
```

一个 program 处理 1 个 query 位置 × 8 个 GQA head。

---

## 4. 性能问题从哪来

### 4.1 spill：寄存器不够

每个线程的寄存器有上限。  
变量太多、live range 太长，编译器就把一些变量放到 local memory。

```text
255 regs / 58 spills
```

意思是：

```text
寄存器用到上限
还有 58 个 spill
每次读写 spill 都去显存
```

basic 版一开始就是这个状态。

### 4.2 layout conversion：摆放不对

如果 load 的摆放和 dot 的摆放不一致：

```text
数据在寄存器里
-> 写 shared
-> barrier
-> 按另一种 layout 读回来
```

这会消耗：

- shared 容量；
- shared 带宽；
- barrier 时间；
- 寄存器 live range。

### 4.3 tile/并行度：活分得不好

- tile 太小：循环多、load 多；
- tile 太大：shared/寄存器爆；
- program 太少：SM 闲着；
- program 太多：每个都重复扫 KV。

### 4.4 同步 load：等内存

普通 `tl.load`：

```text
发 load
等数据回来
再算
```

如果 K/V 在全局显存，等待很长。  
`cp.async` 就是为这个设计的。

---

## 5. K/V layout 的故事

### 5.1 K/V 在显存里怎么放

```text
K cache: [num_blocks, block_size, num_kv_heads, head_size]
V cache: [num_blocks, block_size, num_kv_heads, head_size]
```

固定一个物理块、一个 KV head 后：

```text
slot 0: h0 h1 h2 ... h255
slot 1: h0 h1 h2 ... h255
```

一个 slot 的 head 向量是连续的。

### 5.2 basic 版的 K/V

basic 版：

```text
K: [H, T]  → 直接给 Q @ K
V: [T, H]  → 直接给 P @ V
```

为什么 K 是 `[H,T]`？

```text
Q [M,H] @ K [H,T] -> S [M,T]
```

为什么 V 是 `[T,H]`？

```text
P [M,T] @ V [T,H] -> O [M,H]
```

两个都符合 dot，但它们的 **load layout 不一样**。

### 5.3 两套 layout 的代价

可以把它想成：

```text
K 的数据按“列优先”发给线程
V 的数据按“行优先”发给线程
```

到了 `tl.dot` 前，编译器需要：

```text
K 用一套 shared 摆法
V 用另一套 shared 摆法
```

于是：

- 两套 shared layout；
- 更多 `convert_layout`；
- 更多 shared/寄存器；
- 每个 tile 都要搬。

实测 basic/中期版本：

```text
198 regs / 0 spills
shared 40,960 B
q3/ctx1024 约 0.151 ms
```

### 5.4 改成共用一种 load layout

改成：

```text
K: [T,H] load
K = tl.trans(K) 再 dot
V: [T,H] load
```

K 和 V 的 load layout 变成同一类；K 只多一次小块 transpose。

实测：

```text
140 regs / 0 spills
shared 25,600 B
q3/ctx1024 约 0.068 ms
```

为什么快这么多？

```text
不是 FLOPs 变了
是 K/V 的 shared/layout conversion 少了
寄存器压力也降了
```

### 5.5 为什么不直接让 K 永远 [H,T]

因为那样 K/V 又分叉了。

两种写法都“数学正确”，但：

```text
K [H,T] + V [T,H]
   → 两套 layout
   → 省了 transpose，但多了转换

K [T,H] + V [T,H] + K transpose
   → 一套 load layout
   → 多一次 transpose，但少很多转换
```

后者在目标 shape 上更快。

---

## 6. Astra 发现了什么

### 6.1 一个被忽略的参数

Kernel 签名里有：

```python
stride_k_cache_3: tl.int64
stride_v_cache_3: tl.int64
```

它们是 K/V **最后一维的 stride**。

K cache 是连续的，所以：

```text
stride_k_cache_3 = 1
```

但它是用 `tl.int64` 传进来的。

### 6.2 runtime stride 为什么坏

地址计算是：

```text
addr = base + d * stride
```

如果 `stride` 是运行时参数：

```text
compiler 只知道“有一个整数 stride”
不知道它一定是 1
```

于是它不能证明：

```text
d=0,1,2,3... 的地址是连续的
```

Triton 的 AxisInfo 会把这段的 contiguity 降到 1。

### 6.3 Coalesce 选成 scalar

Triton 的 Coalesce pass 根据连续性选 load layout：

```text
连续性未知
→ sizePerThread=[1,1]
→ 每个线程只读 1 个元素
```

BF16 一个元素 2 字节。

### 6.4 async 的 32-bit 门槛

Triton 的 pipeline 在判断一个 load 能不能转成 async 时，要求每个线程至少搬 32 bit。

```text
1 个 BF16 = 16 bit < 32 bit
→ 不够资格
→ 不生成 cp.async
```

所以 basic 版的 K/V 一直没有 async。

### 6.5 改成 constexpr

Astra 的改动：

```diff
- stride_k_cache_3: tl.int64,
+ stride_k_cache_3: tl.constexpr,
- stride_v_cache_3: tl.int64,
+ stride_v_cache_3: tl.constexpr,
```

调用点仍然传真实 stride：

```python
stride_k_cache_3=k.stride(3),
```

不是硬编码 1。

原理：

```text
把 stride 变成编译期常量
→ addr = base + d * 1
→ 动态乘法消失
→ AxisInfo 能证明连续
→ Coalesce 选 [1,8]
→ 每个线程搬多个 BF16
→ ≥32 bit
→ async 资格成立
→ cp.async 出现
```

### 6.6 async 出现后还要重新选 tile

只改 constexpr：

```text
async 出现了
但 q3/ctx1024 仍约 0.050 ms
```

因为旧 tile 是为“scalar load”设计的。

Astra 最终：

```text
full BF16 head<=256：TILE32 + BLOCK_M16
head512：BLOCK_M8
SWA/FP32：保留旧 TILE64/BM8
```

结果：

```text
q3/ctx1024: 0.0411 ms
ctx2048:    0.0790 ms
hd512:      0.0580 ms
```

---

## 7. cp.async 到底是什么

### 7.1 同步 load

普通路径：

```text
global -> register
register -> shared
shared -> tensor core
```

问题：

```text
global load 很慢
线程要等数据回来
```

### 7.2 异步 copy

`cp.async` 允许：

```text
global -> shared
```

直接在后台进行，不经过寄存器。

于是可以：

```text
先发下一块 K/V 的 cp.async
同时算当前块的 dot
算完再等下一块
```

这叫软件流水线。

时间线对比：

```text
同步：
  load K0 -> 等 -> dot0 -> load K1 -> 等 -> dot1

异步：
  load K0
  load K1 (async)  ┐
  dot0             ├ 重叠
  wait K1          ┘
  dot1
```

### 7.3 它不是“少搬数据”

数据量没变。  
它只是把“搬数据”和“计算”重叠。

如果：

- shared 不够；
- layout 不对；
- tile 太大；
- 计算本身太短；

async 也不一定快。

### 7.4 它不是手写汇编

Triton 的 pipeline pass 自动生成：

```text
ttg.async_copy_global_to_local
ttg.async_commit_group
ttg.async_wait
cp.async
```

vLLM 的 normal NVIDIA 路径也没有手写这些；  
它只是写了普通 `tl.load`，让 compiler 选。

### 7.5 为什么 32 bit 是门槛

async copy 是“按块搬”的，太小没意义。  
每线程 1 个 BF16 = 2 字节，粒度太细；  
至少多个元素凑到 4 字节，才值得做成 async。

所以：

```text
stride 未知 -> scalar 16 bit -> 没资格
stride=1  -> vector 多元素 -> 有资格
```

---

## 8. 完整时间线

| 阶段 | 改了什么 | 解决什么 | q3/ctx1024 |
|---|---|---|---:|
| basic | TILE32、K `[H,T]`、V `[T,H]`、v_keep | 无 | 0.1903 ms |
| 1 | 去 2D v_keep、尾块 clamp、8 warps | spill | 0.1522 ms |
| 2 | K/V 统一 `[T,H]`、K transpose、BM8、4 warps | layout conversion | 0.068 ms |
| 3 | TILE64 load + 2×32 exact subtile | 循环/load 次数 | 0.0522 ms |
| 4 | stride constexpr | async 资格 | async 出现 |
| 5 | full BF16 TILE32/BM16 | async 下重新平衡 | 0.0411 ms |

---

## 9. 为什么 SWA/FP32 不跟着改

### 9.1 SWA 的数值边界

window=512，query 绝对位置 574，第一个可见 key 是 63。

```text
旧 TILE64 从 0 开始：
  key 0..31 全 mask
  M 先被钳到 0

直接 TILE32 从 32 开始：
  key 32..63 中只有 63 有效
  直接按它的 score 更新 M
```

如果第一个 score 是负的：

```text
旧 M=0
新 M=-2.56
```

`P` 会先转 BF16；`exp(-2.56)` 和 `1` 的舍入不同。  
后面再缩放也无法保证逐位一致。

所以 SWA 保留旧 tile 边界。

### 9.2 FP32

FP32 需要 IEEE dot；  
TILE32/BM16 在 FP32 上出现过 `1e-7` 级差异，  
不满足逐位 0 差，所以保留旧路径。

---

## 10. 不要盲调的参数

| 尝试 | 结果 |
|---|---|
| `tl.range(num_stages=2..5)` | 只把 block-table 流水化，K/V 仍 scalar，变慢 |
| `TILE64` 用在 hd512 | 304 spills |
| K 改回 `[H,T]` | 252 spills |
| `.cg` / `evict_first` / `evict_last` | 2048 更慢，不稳定 |
| BLOCK_M4 | 长上下文无收益 |
| 更多 warps | 早期有用，layout 改后 4 更好 |

这说明：

```text
不要从“加 stages / 加 warps”开始
先看 regs/spills/shared/IR
```

---

## 11. 现在应该怎么读这个 kernel

建议顺序：

1. `unified_attention` wrapper：看 grid、TILE、BLOCK_M、2D/3D 选择；
2. `kernel_unified_attention`：看 `_resolve_seq`、`_tile_loop_bounds`；
3. 主循环：load K/V、S、mask、softmax、acc；
4. `v_keep` / tail / SWA 边界；
5. 3D 的 `segm_*` / `reduce_segments`；
6. 最后再看 TTGIR/PTX，不要一开始就钻。

每读一个优化，问三个问题：

```text
1. 它减少的是哪种数据移动？
2. 它增加了什么成本（shared/寄存器/循环）？
3. 输出还是 0 差吗？
```

---

## 12. 术语表

| 术语 | 一句话 |
|---|---|
| thread / warp / CTA | 线程 / 32 线程 / 一个 block |
| register | 线程私有最快存储 |
| spill | 寄存器不够，变量掉到 local memory |
| shared memory | block 内共享的高速内存 |
| global memory | 显存，大但慢 |
| layout | 逻辑 tensor 到线程/shared 的物理分配 |
| convert_layout | 两种 layout 之间搬一次 |
| coalesced | 相邻线程读相邻地址，适合向量化 |
| `cp.async` | global→shared 的异步拷贝 |
| pipeline | 搬下一块和算当前块重叠 |
| `tl.constexpr` | 编译期常量，参与特化 |
| AxisInfo | Triton 的连续性/对齐分析 |
| Coalesce | 根据连续性选 load layout 的 pass |
| online softmax | 分 tile 更新 M/L/acc，不需要一次看完整序列 |

---

## 13. 补充：把 cp.async 讲透

这一节专门回答四个问题：

```text
1. 同步 load 为什么是 global -> register？三个箭头在干什么？
2. Coalesce 是什么？
3. “每个线程搬多个 BF16”在 program 的所有线程里怎么体现？
4. cp.async 怎么和 tl.dot 重叠？不是还得先搬完才能算吗？
```

### 13.1 同步 load 的三个箭头

先把一次 `tl.dot` 前的数据流画出来：

```text
global memory
    │  ld.global
    ▼
register
    │  st.shared
    ▼
shared memory
    │  ldmatrix
    ▼
tensor core / MMA
```

对应到 Triton：

```python
K = tl.load(k_ptrs)   # 逻辑上：global -> register
S = tl.dot(Q, K)      # 编译器负责 register -> shared -> MMA
```

#### 箭头 1：global -> register

GPU 的普通 load 指令是：

```text
ld.global
```

它的目标位置是**寄存器**。

所以“global 直接到 register”不是说它绕过 cache，而是说：

```text
数据从全局显存出发
经过 L2/L1 cache
最终写进线程的寄存器
```

寄存器是线程私有的，CPU 风格的“变量”就住在这里。

#### 箭头 2：register -> shared

Tensor Core 的矩阵乘法通常不直接从寄存器读普通布局的数据。  
编译器需要把操作数按 MMA 需要的 layout 放进 shared memory：

```text
st.shared
```

这一步会消耗：

- shared 容量；
- shared 带宽；
- barrier；
- 寄存器 live range。

#### 箭头 3：shared -> tensor core

真正给 MMA 用的读指令通常是：

```text
ldmatrix
```

它把 shared 里的数据按矩阵块读进 Tensor Core 需要的寄存器布局。

所以完整路径是：

```text
global -> register -> shared -> tensor core
```

#### cp.async 改的是哪一段

`cp.async` 把前两段合并成：

```text
global -> shared
```

不经过寄存器：

```text
global memory
    │  cp.async
    ▼
shared memory
    │  ldmatrix
    ▼
tensor core / MMA
```

它不是“绕过所有内存层级”，而是：

```text
global -> L2/L1 -> shared
```

只是**目标不是寄存器，而是 shared**。

### 13.2 Coalesce 是什么

`Coalesce` 在 Triton 里有两层含义，别混。

#### 1）Triton 编译器的 Coalesce pass

它根据指针的：

- contiguity（连续元素数）；
- divisibility（对齐）；
- shape；
- 线程数；

选择一个 load layout，让一个 warp 的线程尽量读连续地址。

TTGIR 里的 `#blocked` 参数就是它选出来的：

```text
sizePerThread = [1, 8]
threadsPerWarp = [1, 32]
warpsPerCTA = [1, 4]
order = [1, 0]
```

含义：

```text
每个线程在第二维拿 8 个连续元素
一个 warp 的 32 个线程沿着第二维排
每个 CTA 有 4 个 warp
第二维是最快变化的
```

#### 2）GPU 硬件的内存 coalescer

一个 warp 的 32 个线程同时执行一条 load 指令。  
硬件会把 32 个地址合并成尽量少的 32B/128B memory transaction。

如果 32 个线程读连续地址：

```text
thread0 -> addr0
thread1 -> addr1
...
thread31 -> addr31
```

硬件可以合并成少数几个 sector。

如果地址乱跳：

```text
thread0 -> addr0
thread1 -> addr1000
thread2 -> addr37
...
```

就会变成很多次 transaction。

#### 两者关系

```text
Triton Coalesce pass：编译期选“哪个线程拿哪些元素”
GPU coalescer：运行期把 warp 的地址合并成 transaction
```

layout 选得好，硬件 coalescer 才有东西可合并。

### 13.3 一个 program 的线程怎么搬 K tile

用一个具体例子。

target：

```text
TILE_SIZE = 32
HEAD_SIZE = 256
dtype = BF16
num_warps = 4
```

一个 K tile：

```text
32 × 256 = 8192 个 BF16
8192 × 2 B = 16 KiB
```

一个 CTA：

```text
4 warps × 32 threads = 128 threads
```

#### scalar layout：sizePerThread=[1,1]

每个线程一次拿 1 个 BF16：

```text
每线程：1 × 2 B = 2 B
每 warp：32 × 2 B = 64 B
每 CTA 一轮：128 × 2 B = 256 B
搬完 16 KiB 需要：16 KiB / 256 B = 64 轮
```

每个 warp 要发 64 次 load 指令。

而且：

```text
每线程只有 2 B
< cp.async 的 4 B 门槛
```

所以没有 async 资格。

#### vector layout：sizePerThread=[1,8]

每个线程一次拿 8 个连续 BF16：

```text
每线程：8 × 2 B = 16 B
每 warp：32 × 16 B = 512 B
每 CTA 一轮：128 × 16 B = 2 KiB
搬完 16 KiB 需要：16 KiB / 2 KiB = 8 轮
```

每个 warp 只需要 8 次 load/copy 指令。

而且：

```text
每线程 16 B
>= cp.async 的 4 B 门槛
```

所以有 async 资格。

#### 所有线程同时搬吗

不是“128 个线程在同一个 cycle 一起搬”。

实际是：

```text
一个 warp 的 32 个线程锁步执行同一条指令
4 个 warp 由 SM 调度器调度
可能并行，也可能交错
```

但每个 warp 内部：

```text
32 个线程各自拿到自己的地址
硬件 coalescer 把这 32 个地址合并成 transaction
```

所以“每个线程搬 8 个 BF16”指的是：

```text
每条向量 load/copy 指令
每个线程负责 16 B
一个 warp 负责 512 B
```

不是“一个线程串行搬 8 次”。

#### 一轮一轮怎么覆盖 tile

```text
第 1 轮：4 个 warp 各搬 512 B → 2 KiB
第 2 轮：再搬 2 KiB
...
第 8 轮：搬完 16 KiB
```

这就是 `[1,8]` 在“一个 program 全部线程”场景下的具体体现。

### 13.4 cp.async 怎么和 tl.dot 重叠

你的疑问非常合理：

> 我也得搬完数据才能算矩阵乘法吧？

对，**当前 tile 必须先搬完才能算当前 dot**。

cp.async 重叠的不是“当前 tile 的搬和当前 tile 的算”，而是：

```text
搬下一块 K/V
和
算当前块
```

重叠。

#### 同步版本

```text
load K0
wait K0
dot0
load K1
wait K1
dot1
```

时间线：

```text
load0 100
dot0  100
load1 100
dot1  100
总计 400
```

#### async 版本

```text
load K0
wait K0

load K1 async   ┐
dot0            ├ 同时进行
wait K1         ┘

dot1
```

时间线：

```text
load0 100
load1 100  ┐
dot0  100  ┘ 重叠
dot1  100
总计 300
```

如果 load 和 dot 都是 100：

```text
同步：2 个 tile 需要 400
async：2 个 tile 需要 300
```

tile 越多，越接近：

```text
max(load, dot) + 一次 warmup
```

#### 为什么需要两个 buffer

当前 dot 正在用 buffer A 里的 K/V。  
如果 cp.async 直接覆盖 A，dot 就读到一半新、一半旧的数据。

所以需要：

```text
buffer A：当前 dot 用
buffer B：cp.async 正在写下一块
```

dot 完成后交换。

这就是 double buffering / multi-stage pipeline。

#### 谁负责 wait

`cp.async` 是异步的，不能假设它已经完成。  
需要在消费数据前：

```text
async_wait
barrier
```

Triton 的 pipeline pass 会插入：

```text
ttg.async_commit_group
ttg.async_wait
```

`tl.dot` 本身不知道数据是同步 load 还是 async copy；  
是编译器在它前面安排 wait。

#### 为什么能重叠

因为 GPU 有不同管线：

```text
内存管线：cp.async 搬数据
计算管线：Tensor Core 做 MMA
```

它们可以同时工作。  
普通 `ld.global` 的目标是寄存器，线程要等结果才能继续；  
`cp.async` 把结果写到 shared，线程可以继续发指令。

#### 为什么不是永远更快

如果：

```text
dot 只需要 10 cycle
load 需要 100 cycle
```

那么 overlap 也只能隐藏 10 cycle，剩下 90 cycle 还是在等。

如果：

```text
shared 不够，放不下两个 buffer
```

pipeline 就起不来。

如果：

```text
layout 不对，每线程只搬 2 B
```

连 async 资格都没有。

所以 cp.async 的收益取决于：

```text
load 时间
compute 时间
shared 容量
layout/向量宽度
tile 大小
```

### 13.5 一句话总结

```text
同步：
  global -> register -> shared -> MMA
  线程等 load，算和搬串行

cp.async：
  global -> shared -> MMA
  当前 tile 仍然要等；
  但下一 tile 的搬可以和当前 tile 的算重叠
```

Coalesce 决定：

```text
一个 warp 的 32 个线程怎么读、每线程读几个
```

Astra 的 `constexpr` 决定：

```text
编译器能不能证明“这些地址是连续的”
→ 能不能选 [1,8]
→ 每线程能不能搬够 4 B
→ 能不能进 cp.async pipeline
```

### 13.6 “32 个线程锁步执行同一条指令”，为什么地址还能不同？

这是 SIMT（Single Instruction, Multiple Threads）最容易被误解的地方。

“锁步执行同一条指令”指的是：

```text
同一个 warp 的 32 个线程
共享同一个 program counter / 同一条指令流
同一时刻发射同一条指令
```

不是：

```text
32 个线程的寄存器值一样
32 个线程的地址一样
32 个线程的数据一样
```

#### 每个线程有自己的寄存器

GPU 的寄存器是**per-thread** 的。

假设指令是：

```text
LDG R0, [R1]
```

它的含义是：

```text
从 R1 里的地址 load
结果写到 R0
```

但对 warp 里 32 个线程：

```text
lane0 的 R1 = 0x1000
lane1 的 R1 = 0x1010
lane2 的 R1 = 0x1020
...
```

同一条 `LDG` 指令，每个 lane 用自己的 R1，  
所以发出去的是 32 个不同地址。

#### 地址是怎么算出来的

Triton 里写：

```python
offs = tl.arange(0, 256)
addr = base + offs * 2
x = tl.load(addr)
```

编译后，`offs` 不是“一个数组变量”，  
而是被 layout 分配到各线程的寄存器。

例如 `sizePerThread=[1,8]` 时，概念上：

```text
lane0 负责元素 0..7
lane1 负责元素 8..15
lane2 负责元素 16..23
...
```

每个 lane 计算自己的起始地址：

```text
lane0: base + 0*2
lane1: base + 8*2
lane2: base + 16*2
...
```

然后一条向量 load 指令：

```text
每个 lane 从自己的地址读 16B
```

#### 一个 warp 的地址长什么样

以 BF16、每线程 8 个元素为例：

```text
lane0  -> 0x1000 .. 0x100F
lane1  -> 0x1010 .. 0x101F
lane2  -> 0x1020 .. 0x102F
...
lane31 -> 0x11F0 .. 0x11FF
```

32 个 lane 合起来：

```text
0x1000 .. 0x11FF
= 512B 连续区域
```

硬件 memory coalescer 看到 32 个地址，  
把它们合并成尽量少的 32B/128B transaction。

#### 如果地址不连续

```text
lane0  -> 0x1000
lane1  -> 0x5000
lane2  -> 0x2000
...
```

同一条指令仍然只发一次，  
但 32 个地址落在很多不同的 sector，  
coalescer 只能拆成很多 transaction。

这就是“coalesced”和“non-coalesced”的区别。

#### “锁步”不等于“不能分支”

如果 warp 内线程走不同分支：

```text
if (lane_id < 16) { A } else { B }
```

warp 会先执行 A（部分 lane 活跃），再执行 B（另一部分活跃）。  
这叫 warp divergence。

但对一条直线 load：

```text
address = base + lane_offset
x = load(address)
```

32 个 lane 都执行同一条指令，只是各自地址不同。

#### 一句话

```text
锁步 = 同一个 warp 共享指令流 / PC
每线程 = 自己的寄存器、自己的 lane 地址、自己的数据
coalescer = 把 32 个 lane 地址合并成 transaction
```

所以“同一条指令”和“32 个不同地址”并不矛盾。

### 13.7 这和 CUDA GEMM tutorial 里的 double buffer 是一回事吗？

**是同一类东西，但要区分“模式”和“工具”。**

```text
double buffer / multi-stage pipeline：一种流水线模式
cp.async：实现这种模式的一种硬件指令
```

#### CUDA GEMM tutorial 的经典 double buffer

概念上：

```text
shared buffer A
shared buffer B

load A0/B0 -> buffer A
wait

for k:
    async load A(k+1)/B(k+1) -> buffer B
    compute C += A(k) @ B(k)   // 用 buffer A
    wait
    swap A/B
```

也就是：

```text
搬下一块
和
算当前块
重叠
```

#### attention 的 cp.async

```text
load K0/V0 -> buffer A
wait

for tile:
    async load K(t+1)/V(t+1) -> buffer B
    compute S = Q @ K(t), softmax, acc += P @ V(t)
    wait
    swap A/B
```

所以结构完全一样：

```text
GEMM：A/B tile
attention：K/V tile
```

区别只是 attention 多了一个 online softmax 状态（M/L/acc）。

#### 但 double buffer 不一定等于 cp.async

double buffer 可以这样实现：

```text
ld.global -> register
st.shared -> buffer
```

这是同步的，只是用了两个 buffer。  
它不一定能重叠，因为线程要等 `ld.global` 回寄存器。

`cp.async` 的价值是：

```text
global -> shared
不经过寄存器
线程可以继续发指令
```

所以：

```text
double buffer 是“有两个 buffer”
cp.async 是“让写 buffer 这件事异步”
```

两者经常一起用，但不是同一个概念。

#### Triton 里谁做 double buffer

Triton 的 pipeline pass 会根据：

```text
num_stages
shared 预算
load 的 async 资格
```

自动分配 buffer、插入：

```text
async_copy
async_commit_group
async_wait
barrier
```

所以 Triton 里你通常不手写 buffer A/B。  
你写普通 `tl.load`，编译器决定是否流水化。

#### 为什么参数 scan 没触发

如果 K/V load 没有 async 资格：

```text
每线程 1 个 BF16 = 2B
< 4B 门槛
```

那么：

```text
num_stages 再大
→ 只能流水化 block-table
→ K/V 还是同步 load
→ double buffer 对 K/V 没用
```

所以 Astra 先修的是：

```text
stride constexpr
→ 连续性证明
→ [1,8] 向量化
→ 每线程 16B
→ cp.async 资格
```

然后 tile/stage 才有意义。

#### 一句话

```text
CUDA GEMM tutorial 的 double buffer
和 attention 的 cp.async pipeline
是同一个软件流水线思想。

区别：
  GEMM 教程常手写 buffer/commit/wait；
  Triton 把 buffer 和 cp.async 交给 compiler 自动做。
```

### 13.8 源码里看不到软件流水线：因为它在编译后的 TTGIR 里

你说得对：

```python
for j in range(loop_lo, loop_hi):
    K = tl.load(k_ptrs)
    V = tl.load(v_ptrs)
    S = tl.dot(Q, K)
    ...
```

从源码看，就是：

```text
load 当前 tile
dot 当前 tile
```

看不到 buffer A/B，也看不到“下一块”。

原因是：

```text
Triton 源码是前端 IR
软件流水线是后面的 compiler pass 改写的
```

不是源码里写了一个显式的 pipeline。

#### 编译后的循环

当前 target 编译后，TTGIR 里可以看到：

1. loop 前先做 prologue：

```text
%physical_block_85 = ttg.local_alloc : () -> 1x32xi32 shared
%K_86 = ttg.local_alloc : () -> 1x32x256xbf16 shared
%V   = ttg.local_alloc : () -> 1x32x256xbf16 shared

%physical_block_93 = ttg.async_copy_global_to_local ...
%K_123 = ttg.async_copy_global_to_local ... bf16
%V_126 = ttg.async_copy_global_to_local ... bf16
```

2. 然后进入 `scf.for`，循环携带 K/V 的 memdesc：

```text
%acc_136 = scf.for ... iter_args(..., %K_154 = %K_124, %V_155 = %V_127, ...)
```

3. 每次循环开始，先等当前 K/V：

```text
%K_164 = ttg.async_wait %K_154, %V_155, %physical_block_157 {num = 0}
```

4. 从 shared 读当前 K，做 QK：

```text
%S_165 = ttg.local_load %K_120 token %K_164
%K_166 = tt.trans %S_165
%S_168 = tt.dot %Q_167, %K_166
```

5. 从 shared 读当前 V，做 PV：

```text
%V_191 = ttg.local_load %V_125 token %K_164
%acc_192 = tt.dot %acc_190, %V_191, %acc_187
```

6. 算完当前 tile 后，才发未来的 async copy：

```text
%K_219 = ttg.async_copy_global_to_local %k_ptrs_206 ... bf16
%V_221 = ttg.async_copy_global_to_local %v_ptrs_213 ... bf16
```

注意第 6 步在循环体末尾，但它的目标是**后面的 tile**。  
下一轮循环开始时：

```text
wait 当前
compute 当前
issue 下一块
```

就形成了：

```text
第 t 轮：
  wait Kt
  compute Kt
  issue K(t+2)

第 t+1 轮：
  wait K(t+1)
  compute K(t+1)
  issue K(t+3)
```

`K(t+2)` 的 copy 在第 t 轮末尾发出，  
在第 t+2 轮开始前才需要完成。  
所以它有整整一轮多的时间去搬，和计算重叠。

#### 为什么当前 tile 仍然要等

因为：

```text
Kt 必须完整在 shared 里
才能做 Q @ Kt
```

TTGIR 里的 `async_wait` 就是在等当前 tile。  
pipeline 不是“当前 tile 没搬完就 dot”，而是：

```text
当前 tile：上一轮已经预取好，本轮 wait 后直接用
下一块：本轮计算时在后台搬
```

#### 源码里的 tl.load 去哪了

前端 IR 里是：

```text
tt.load
```

pipeline pass 之后，K/V 的 load 被改写成：

```text
ttg.async_copy_global_to_local
```

同时分配 shared buffer：

```text
ttg.local_alloc
```

dot 前再从 shared 读：

```text
ttg.local_load
```

所以你在源码里只看到 `tl.load`，  
但在 TTGIR 里已经不是普通 load 了。

#### 什么时候不会发生

如果 K/V load 没有 async 资格：

```text
每线程 1 个 BF16 = 2B
< 4B 门槛
```

pipeline pass 不会把 K/V 改成 async copy。  
这时 TTGIR 里：

```text
K/V 是普通 tt.load
没有 K/V 的 async_copy
只有 block-table 的 async_copy
```

这就是早期版本的情况。

Astra 的 `stride constexpr` 让 K/V：

```text
contiguity=8
#blocked1 [1,8]
每线程 16B
```

pipeline pass 才有资格把它们放进 async pipeline。

#### 一句话

```text
源码：load -> dot，顺序语义
TTGIR：prologue async copy
       -> loop: wait 当前, compute 当前, async copy 未来
       -> epilogue
```

软件流水线是 compiler 在 TTGIR 里改出来的，  
不是源码里手写的 buffer A/B。
