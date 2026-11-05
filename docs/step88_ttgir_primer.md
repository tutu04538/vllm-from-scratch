# TTGIR 从零教程：以 unified attention 的软件流水线为例

> 面向完全没看过 MLIR/TTGIR 的读者。
> 目标：能看懂 `minivllm/attention/ops/triton_unified_attention.py` 编译出来的 TTGIR，并能指出软件流水线在哪。
> 配合教程：`docs/step88_triton_attention_primer.md`。

## 0. 先建立一张地图

Triton 的编译流程大致是：

```text
Python kernel
  ↓  Triton 前端
TTIR        （Triton IR，逻辑算子）
  ↓  layout 分配 / Coalesce / Pipeline 等 pass
TTGIR       （Triton GPU IR，带线程/shared layout）
  ↓
LLIR        （LLVM IR）
  ↓
PTX
  ↓
SASS / cubin
```

你写的是 Python，看到的是 TTGIR。

```text
TTIR 关心：算什么
TTGIR 关心：每个线程拿哪些元素、放寄存器还是 shared、怎么搬
```

软件流水线就是在 TTGIR 这一层改出来的。

---

## 1. MLIR 最小语法

TTGIR 是 MLIR 的一种文本形式。先认识 6 个东西。

### 1.1 SSA 值：`%name`

```mlir
%K_166 = tt.trans %S_165 {order = array<i32: 1, 0>}
```

含义：

```text
%S_165 是一个已经定义的值
tt.trans 是操作
%K_166 是结果
```

SSA 规则：

```text
每个 %name 只定义一次
后面可以多次使用
```

名字里的编号：

```text
%K_166
```

是编译器生成的，不稳定。  
不要硬记编号；要看“谁定义、谁使用”。

### 1.2 类型：`: type`

```mlir
%S_168 : tensor<16x32xf32, #mma>
```

拆开：

```text
tensor<16x32xf32>   逻辑 shape [16,32]，元素 f32
#mma                物理 layout
```

标量：

```text
i32 / i64 / f32 / i1
```

指针：

```text
!tt.ptr<bf16>
```

shared 描述：

```text
!ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
```

### 1.3 操作：op

```mlir
%x = arith.addi %a, %b : i32
```

`arith.addi` 是整数加法。  
冒号后面是结果类型。

常见 op：

```text
arith.addi / muli / divsi / cmpi
math.exp
tt.load / tt.store
tt.dot
tt.trans
scf.for
```

### 1.4 常量

```mlir
%c0_i32 = arith.constant 0 : i32
%cst = arith.constant 1.000000e+00 : f32
```

名字里常带 `c0`、`cst`。

### 1.5 区域和 block

`scf.for` 有循环体：

```mlir
%result = scf.for %i = %c0 to %n step %c1
    iter_args(%acc = %init) -> (f32) {
  ...
  scf.yield %new_acc : f32
}
```

含义：

```text
i 从 0 到 n
acc 是循环携带值
每轮产生 new_acc，下一轮变成 acc
```

TTGIR 的 pipeline 会把 K/V 的 shared buffer 放进 `iter_args`。

### 1.6 location

```mlir
loc(#loc245)
```

这是源码位置/调试信息。  
读 IR 时可以忽略它；它不影响语义。

---

## 2. TTGIR 里的 dialect

你会看到很多前缀：

| 前缀 | 含义 |
|---|---|
| `tt.` | Triton 逻辑算子：load/dot/trans |
| `ttg.` | Triton GPU：layout、shared、async copy |
| `arith.` | 整数/浮点算术 |
| `math.` | exp 等数学函数 |
| `scf.` | 循环/条件 |

例如：

```mlir
%S = tt.dot %Q, %K : ...
%K2 = tt.trans %S
%buf = ttg.local_alloc : () -> ...
%tok = ttg.async_copy_global_to_local ...
```

---

## 3. layout 是什么，在 TTGIR 里长什么样

### 3.1 逻辑 shape 和物理 layout

```mlir
tensor<16x32xf32, #mma>
```

前面是逻辑 shape，后面是物理 layout。

同一个 shape 可以有不同的 `#...`。

### 3.2 `#blocked`

```mlir
#blocked1 = #ttg.blocked<{
    sizePerThread = [1, 8],
    threadsPerWarp = [1, 32],
    warpsPerCTA = [4, 1],
    order = [1, 0]
}>
```

含义：

```text
sizePerThread：每线程每维拿几个元素
threadsPerWarp：warp 的线程怎么排
warpsPerCTA：CTA 的 warp 怎么排
order：哪一维最快变化
```

`sizePerThread=[1,8]`：

```text
第二维每个线程拿 8 个连续元素
```

`sizePerThread=[1,1]`：

```text
每个线程只拿 1 个
```

### 3.3 `#shared`

```mlir
#shared = #ttg.swizzled_shared<{
    vec = 8,
    perPhase = 1,
    maxPhase = 8,
    order = [1, 0]
}>
```

这是 shared memory 的排布：

```text
vec       向量宽度
perPhase  swizzle 参数
maxPhase  swizzle 参数
order     维度顺序
```

swizzle 是为了减少 shared bank conflict。

### 3.4 `#mma` 和 `#dot_op`

```mlir
#mma = #ttg.nvidia_mma<{
    versionMajor = 2,
    versionMinor = 0,
    warpsPerCTA = [1, 4],
    instrShape = [16, 8]
}>
```

`#mma` 是 Tensor Core 输出/累加器的 layout。

```mlir
#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
```

`#dot_op` 是 dot 操作数的 layout：

```text
opIdx=0：左操作数
opIdx=1：右操作数
```

### 3.5 `#linear`

```mlir
#linear = #ttg.linear<{
    register = [[0, 1], [0, 8], [0, 16], ...],
    lane = [[0, 2], [0, 4], [1, 0], ...],
    warp = [[8, 0], [16, 0]],
    block = []
}>
```

`linear` 是更一般的线性布局：

```text
register：寄存器内怎么展开
lane：线程内怎么展开
warp：warp 内怎么展开
```

### 3.6 `#slice`

```mlir
#ttg.slice<{dim = 1, parent = #blocked1}>
```

它表示从某个 layout 里切掉一维后的 view。

例如 `[32,256]` 的 layout，`dim=1` 切掉后用于 `[32]` 的 mask。

### 3.7 `convert_layout`

```mlir
%x2 = ttg.convert_layout %x : tensor<... #blocked> -> tensor<... #shared>
```

它表示“换一种物理摆放”。  
通常意味着：

```text
寄存器 -> shared -> barrier -> 再读
```

这是性能敏感点。

---

## 4. TTGIR 里的 shared 和 async copy

### 4.1 `ttg.local_alloc`

```mlir
%K_buf = ttg.local_alloc : () ->
    !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
```

含义：

```text
在 shared memory 里分配一块 K buffer
```

它不是普通 tensor，而是 `memdesc`：

```text
shared memory 的描述符
```

### 4.2 `ttg.local_load`

```mlir
%K = ttg.local_load %K_buf token %tok :
    !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
    -> tensor<32x256xbf16, #linear>
```

从 shared 读进寄存器/目标 layout。

`token` 是同步令牌：  
它确保读之前对应的 async copy 已经完成。

### 4.3 `tt.load` 和 `ttg.async_copy`

普通 load：

```mlir
%K = tt.load %k_ptrs : tensor<32x256x!tt.ptr<bf16>, #blocked1>
```

async copy：

```mlir
%tok = ttg.async_copy_global_to_local %k_ptrs, %buf
    mask %mask other %zero
    : tensor<32x256x!tt.ptr<bf16>, #blocked1>
    -> !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
```

区别：

```text
tt.load：global -> register
async_copy：global -> shared，异步
```

### 4.4 `async_commit_group`

```mlir
%tok2 = ttg.async_commit_group tokens %tok
```

把之前发出的 async copy 组成一个 group，便于等待。

### 4.5 `async_wait`

```mlir
%ready = ttg.async_wait %tok2 {num = 0 : i32}
```

等待 group 完成。

`num=0` 表示等待到没有未完成的 group（具体语义见 Triton 文档；这里理解为“等这批 copy 完成”即可）。

---

## 5. 当前 kernel 的 TTGIR 怎么读

以 q3/ctx1024/target 当前编译结果为例：

```text
regs 155 / spills 0
shared 42,112 B
num_warps 4
num_stages 3

async_copy 7
async_commit 7
async_wait 3
local_alloc 5
local_load 6
scf.for 1
```

注意：

```text
这些是 TTGIR 里的静态操作数
不是运行时执行次数
loop 里的一个 async_copy 会执行很多次
```

### 5.1 prologue：循环前先搬第一批

TTGIR 里先出现：

```mlir
%physical_block_85 = ttg.local_alloc : () ->
    !ttg.memdesc<1x32xi32, #shared1, #smem, mutable>

%K_86 = ttg.local_alloc : () ->
    !ttg.memdesc<1x32x256xbf16, #shared, #smem, mutable>

%V = ttg.local_alloc : () ->
    !ttg.memdesc<1x32x256xbf16, #shared, #smem, mutable>
```

然后：

```mlir
%physical_block_93 = ttg.async_copy_global_to_local ...
%K_123 = ttg.async_copy_global_to_local ... bf16
%V_126 = ttg.async_copy_global_to_local ... bf16
```

这三条是：

```text
block table
K tile
V tile
```

先异步搬进 shared。

### 5.2 进入循环：K/V 是 loop-carried

```mlir
%acc_136:9 = scf.for ... iter_args(
    ...,
    %K_154 = %K_124,
    %V_155 = %V_127,
    ...
)
```

`iter_args` 里的 `%K_154` / `%V_155` 是：

```text
这一轮要用的 K/V shared buffer / token
```

循环每轮把它们传给下一轮。

### 5.3 每轮开始：等当前 tile

```mlir
%K_164 = ttg.async_wait %K_154, %V_155, %physical_block_157 {
    num = 0 : i32
}
```

含义：

```text
等当前 K/V/block-table 的 async copy 完成
```

然后才允许读 shared。

### 5.4 读当前 K，做 QK

```mlir
%S_165 = ttg.local_load %K_120 token %K_164 :
    !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
    -> tensor<32x256xbf16, #linear>

%K_166 = tt.trans %S_165 {order = array<i32: 1, 0>} :
    tensor<32x256xbf16, #linear>
    -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

%Q_167 = ttg.local_load %Q_66 :
    !ttg.memdesc<16x256xbf16, #shared, #smem>
    -> tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

%S_168 = tt.dot %Q_167, %K_166, %cst_5, inputPrecision = tf32 :
    tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    * tensor<256x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    -> tensor<16x32xf32, #mma>
```

读法：

```text
local_load K：shared -> 寄存器（#linear）
tt.trans K：[32,256] -> [256,32]，变成 dot 右操作数
local_load Q：shared -> dot 左操作数
tt.dot：Q @ K -> S [16,32]
```

### 5.5 softmax / mask

中间是一堆：

```text
arith.cmpi / arith.select / math.exp / tt.reduce
```

对应源码：

```python
keep = row_mask[:, None] & seq_mask
S = tl.where(keep, S, -inf)
M, L, P, alpha = _softmax_step(S, M, L)
acc = acc * alpha
```

TTGIR 里读起来很啰嗦，但语义就是 online softmax。

### 5.6 读当前 V，做 PV

```mlir
%V_191 = ttg.local_load %V_125 token %K_164 :
    !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
    -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

%acc_192 = tt.dot %acc_190, %V_191, %acc_187 :
    tensor<16x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    -> tensor<16x256xf32, #mma>
```

含义：

```text
P @ V -> 累加到 acc
```

### 5.7 算完当前，发未来 tile 的 async copy

循环体末尾：

```mlir
%K_219 = ttg.async_copy_global_to_local %k_ptrs_206, %K_120
    mask %acc_218 other %cst_10 {contiguity = 8 : i32}
    : tensor<32x256x!tt.ptr<bf16>, #blocked1>
    -> !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>

%V_221 = ttg.async_copy_global_to_local %v_ptrs_213, %V_125
    mask %acc_218 other %cst_10 {contiguity = 8 : i32}
    : tensor<32x256x!tt.ptr<bf16>, #blocked1>
    -> !ttg.memdesc<32x256xbf16, #shared, #smem, mutable>
```

注意：

```text
这两条在 compute 后面
但目标地址是后面的 tile
不是当前 tile
```

下一轮循环开始时：

```text
wait 当前（下一轮）
compute 当前（下一轮）
issue 再下一块
```

### 5.8 循环结束还有 wait

循环外：

```mlir
%acc_137 = ttg.async_wait {num = 0 : i32}
```

处理最后几个 stage 的收尾。

---

## 6. 软件流水线在 TTGIR 里的形状

把关键操作按执行顺序排：

```text
prologue:
  local_alloc block/K/V
  async_copy block/K/V
  async_commit

loop:
  async_wait 当前 K/V
  local_load K
  trans K
  dot QK
  softmax/mask
  local_load V
  dot PV
  async_copy 未来 K/V
  async_commit
  yield 下一轮的 K/V token

epilogue:
  async_wait
```

这不是源码里的顺序：

```text
源码：
  load K
  dot QK
  load V
  dot PV
```

而是 compiler 把 **load 提前/延后**，把 K/V 变成 shared buffer 和 async token。

### 6.1 为什么能提前

因为 K/V 的地址只依赖：

```text
j（循环变量）
block_table
seq_offset
```

compiler 可以计算未来 `j` 的地址，  
在算当前 tile 时发出未来的 copy。

### 6.2 当前 tile 仍然要等

`async_wait` 在每轮开始：

```text
Kt 没准备好，不能 dot
```

pipeline 只是把：

```text
K(t+1), K(t+2)... 的搬运
```

提前到当前计算期间。

### 6.3 为什么需要 shared buffer

`async_copy` 的目标是：

```text
shared memory
```

不是寄存器。

dot 读的是 shared 里的 K/V。

所以 TTGIR 里有：

```text
local_alloc
local_load
```

### 6.4 为什么 async 数是 7，不是“7 个 tile”

TTGIR 是静态 IR：

```text
loop 内的一条 async_copy
运行时每轮执行一次
```

7 是静态操作数，不是运行时次数。

---

## 7. 怎么自己读一份 TTGIR

### 7.1 先找 layout 定义

文件开头：

```mlir
#blocked = ...
#blocked1 = ...
#shared = ...
#mma = ...
```

先看 `sizePerThread` / `order`，  
判断 load 是 scalar 还是 vectorized。

### 7.2 再找 loop

```text
scf.for
```

看：

- `iter_args` 里有什么；
- 循环体里谁先 wait；
- dot 在哪；
- async_copy 在哪。

### 7.3 找 async

搜索：

```text
async_copy_global_to_local
async_commit_group
async_wait
local_alloc
local_load
```

如果 K/V 的 `async_copy` 是 `!tt.ptr<bf16>`，  
那才是真正的 K/V 数据 copy；  
`!tt.ptr<i32>` 是 block-table。

### 7.4 找 convert_layout

```text
ttg.convert_layout
```

数量多通常意味着 layout 不匹配。

### 7.5 忽略什么

- `loc(...)`
- SSA 编号
- `arith` 的细节（除非在查数值）
- softmax 的 reduce 细节（除非在查数值）

---

## 8. 这份 TTGIR 和源码的对应

| 源码 | TTGIR |
|---|---|
| `K = tl.load(k_ptrs)` | `ttg.async_copy_global_to_local` + `local_load` |
| `V = tl.load(v_ptrs)` | 同上 |
| `S = tl.dot(Q, K)` | `tt.dot`，K 先 `tt.trans` |
| `keep = ...` | `arith.cmpi/andi/select` |
| `_softmax_step` | `tt.reduce` + `math.exp` + `arith` |
| `acc += tl.dot(P, V)` | `tt.dot` 带 `acc` 参数 |
| `for j in range` | `scf.for` + `iter_args` |
| `num_stages=3` | prologue/steady-state/epilogue 的 stage 数 |

---

## 9. 常见误解

### 误解 1：`%K_164` 是 K tensor

它可能是 async token / memdesc / 中间值。  
看类型：

```text
tensor<...>      普通 tensor
!ttg.memdesc<...> shared 描述
!ttg.async_token  async token
```

### 误解 2：`#blocked` 和 `#shared` 只是名字

它们是不同的物理布局。  
`convert_layout` 就是在两者之间搬。

### 误解 3：`num_stages=3` 就是三份 K/V

不一定。  
shared 预算、pass 选择、哪些 load 被 pipeline，都会影响实际 buffer 数。  
以 TTGIR 的 `local_alloc` / `async_copy` 为准。

### 误解 4：源码里没有 pipeline，所以没有 pipeline

pipeline 是 compiler pass 改出来的。  
源码顺序语义不变，TTGIR 里才看得到 prologue/wait/copy。

### 误解 5：async 数就是运行时次数

TTGIR 是静态 IR。  
loop 里的一条 async_copy 会在运行时执行很多次。

---

## 10. 一句话总结

```text
TTGIR = 带 layout 的 Triton IR。
%name = SSA 值；
tensor<...> = 逻辑 shape；
#blocked/#shared/#mma = 物理摆放；
local_alloc/local_load = shared 读写；
async_copy/commit/wait = 异步搬运；
scf.for + iter_args = 循环携带的 buffer/token。
```

这份 unified attention 的 TTGIR 里：

```text
prologue: async copy 第一批 K/V
loop:
  async_wait 当前
  local_load K -> trans -> dot QK
  softmax
  local_load V -> dot PV
  async_copy 未来 K/V
epilogue: async_wait
```

所以软件流水线不是源码里的 buffer A/B，  
而是 compiler 在 TTGIR 里把：

```text
当前 tile 的计算
和
未来 tile 的 global -> shared copy
```

排成了重叠的 stage。
