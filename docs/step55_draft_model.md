# step55：draft model、双 KV 与一般拒绝采样

需求：[185（本关需求）](../../vllm-omni/learning_notes/14_vllm_from_scratch/185_第五十五关_draft_model双KV与一般拒绝采样.md)、
[187（真实模型方案）](../../vllm-omni/learning_notes/14_vllm_from_scratch/187_第五十五关改用真实Qwen3模型.md)。
代码：`step55/`（从 `step54/` 复制，旧包不改）。包指纹 `b4a44645272de161`（19 个 .py / 4774 行）。

## 0. 需求大概

前四关的投机是 **n-gram 提议**：草稿来自重复历史，`q(d)=1`，验证只要 `p[d]`。这一关换成
**真正的第二个模型**：

- 独立的小模型提议（Qwen3-0.6B 提，Qwen3-1.7B 验），不再依赖重复历史；
- **两套 KV**：target 与 draft 各按自己的层数/头数/精度建池子，物理块绝不互换；
- **一般 q 分布**的 `min(1, p/q)` 接受与 `max(p-q, 0)` 纠正；
- 双目录加载（1.7B 是**分片** safetensors）；
- 多请求批量提议、双池容量不足时缩 K 或退回普通 target、抢占后两套 KV 都能恢复。

不承诺加速：本关只把机制做完整、做对。

下面 §1 先讲**算法本身的原理**（为什么要这样接受、这样纠正，为什么它是无偏的），
§2 起讲本关怎么把它落到两个模型、两套 KV 上。只需要看实现的话可以从 §2 开始。

## 1. 算法原理

### 1.1 它想省的是什么

自回归解码每步一次 forward 只出一个 token，瓶颈是**显存带宽**（权重与 KV 每步都要重读一遍）
而不是算力——batch 小的时候算力大量闲置。投机解码的出发点是**让一次 target forward 多验几个位置**：

1. 用便宜的小模型（draft）自回归地猜 K 枚草稿 `d_0..d_{K-1}`；
2. target **一次** forward 就能并行算出这 K+1 个位置各自的分布——因果 attention 保证
   第 i 个位置不依赖它后面的 token，所以草稿可以直接当输入喂进去；
3. 验证之后一般能接受好几枚，于是一次 target forward 产出多个 token。

代价是 draft 的 K 次前向（同轮多请求可以合成一个 batch，见 §4）。所以能不能赚，取决于
**接受率**与**两个模型的成本比**——本关只做机制，不做这个矩阵。

### 1.2 难点：草稿是「另一个分布」抽出来的

draft 从它自己的分布 q 抽草稿，而我们要的是 target 的分布 p。直接接受草稿 = 输出分布变成 q，
质量掉到小模型的水平；直接拒绝 = 退化成普通解码，白算。所以要一套**无偏**（unbiased）的
接受/拒绝规则：**输出分布必须严格等于 p**。

### 1.3 先约定记号（概率论小白版）

后面所有的推导只用得上下表这几样东西。

| 记号 | 读法 | 含义 |
|---|---|---|
| $p(y)$ | 「p 在 y 上的值」 | target 模型给 token $y$ 的概率。所有 token 的概率加起来正好是 1：$\sum_y p(y)=1$ |
| $q(y)$ | 同上 | draft 模型给 token $y$ 的概率，同样 $\sum_y q(y)=1$ |
| $d$ | 草稿 | 从 $q$ 里**抽**出来的那一枚 token。所以「抽到的正好是 $d$」这件事的概率就是 $q(d)$ |
| $y,\ x,\ z$ | — | 词表里任意一个 token（数学里叫「哑变量」，就是「随便哪一个」的意思） |
| $\sum_y$ | 求和 | 把词表里每个 token $y$ 对应的项都加起来（词表有 15 万，就加 15 万项） |
| $\min(a,b)$ / $\max(a,b)$ | 取小 / 取大 | 两个数里较小 / 较大的那个。$\max(a,0)$ 就是「负数一律当 0」 |
| $\mathrm{normalize}(v)$ | 归一化 | 把一串非负数按比例放大缩小，让它们的和变成 1：$\mathrm{normalize}(v)(y)=\dfrac{v(y)}{\sum_z v(z)}$ |
| $P(\cdot)$ | 概率 | $P(\text{某件事})$ 就是这件事发生的概率；$P(A \mid B)$ 是「已知 $B$ 发生的条件下 $A$ 的概率」 |

两个需要先接受的说法：

1. **抽样**：给定一个分布（每个 token 一个非负概率、总和为 1），按概率大小随机挑一个——
   概率大的更容易被挑中。代码里就是 `torch.multinomial`。「草稿 $d$ 服从 $q$」= $d$ 是这么挑出来的。
2. **$\max(p-q,0)$ 是逐 token 算的**：对每个 token $y$ 各算一个 $\max(p(y)-q(y),\,0)$，
   得到一串非负数，再用 $\mathrm{normalize}$ 把它的和拉回 1，才是一个合法分布。
   代码里就是 `(probs - q).clamp(min=0)` 之后除以它的和。

### 1.4 单枚草稿：规则与无偏性证明

先看最简单的情况：这一轮只猜了 **1 枚**草稿 $d$（$K=1$）。规则是

$$
\text{以概率}\ \min\!\left(1,\ \frac{p(d)}{q(d)}\right)\ \text{接受这枚草稿；}
\quad\text{否则从}\ r=\mathrm{normalize}\big(\max(p-q,\,0)\big)\ \text{里再抽一枚。}
$$

后半句里的 $r$ 叫**纠正分布**（residual）。现在证明：**这套规则吐出来的 token，分布严格等于 $p$**。

#### 第一步：把「输出是 y」拆成两条互斥的来路

输出的 token 记作 $Y$。它要成为 $y$，只可能是这两种情况之一，不可能同时发生：

- **事件 $A$**：草稿本身就是 $y$（即 $d=y$），并且这枚草稿被接受了；
- **事件 $B$**：草稿被拒绝了，随后从纠正分布 $r$ 里抽到了 $y$。

所以（这叫全概率公式，说人话就是「把所有可能的来路加起来」）

$$
P(Y=y)=\underbrace{P(A)}_{\text{来路一}}+\underbrace{P(B)}_{\text{来路二}}
$$

#### 第二步：来路一

$$
P(A)=P(d=y)\cdot P(\text{接受}\mid d=y)
=q(y)\cdot\min\!\left(1,\frac{p(y)}{q(y)}\right)
=\min\big(q(y),\,p(y)\big)
$$

逐项解释：$P(d=y)$ 就是 $q(y)$（草稿服从 $q$）；给定草稿是 $y$，接受概率是
$\min(1,p(y)/q(y))$；最后一步是恒等式 $q\cdot\min(1,p/q)=\min(q,p)$——把 $\min$ 里的 $q$ 乘进去即可
（$q(y)=0$ 时草稿不可能是 $y$，两边都是 0，恒等式照样成立）。

#### 第三步：来路二上半段——「草稿被拒」的概率是多少

草稿可能是词表里**任意**一个 token $x$（概率 $q(x)$），每种情况下被拒的概率是
$1-\min(1,p(x)/q(x))$。仍用「把所有来路加起来」：

$$
P(\text{拒绝})=\sum_x q(x)\left(1-\min\!\left(1,\frac{p(x)}{q(x)}\right)\right)
=\sum_x\Big(q(x)-\min\big(q(x),p(x)\big)\Big)
=1-\sum_x\min\big(q(x),p(x)\big)
$$

最后一步用了 $\sum_x q(x)=1$（$q$ 是分布，总和为 1）。

#### 第四步：来路二下半段——纠正分布的分母

纠正分布是「把 $\max(p-q,0)$ 归一化」，它的分母是 $\sum_z\max(p(z)-q(z),0)$。这里有一个
**逐 token 的恒等式**：

$$
\max\big(p(z)-q(z),\,0\big)=p(z)-\min\big(p(z),q(z)\big)
$$

验一下两种情形就信了：$p(z)>q(z)$ 时，左边 $=p(z)-q(z)$、右边 $=p(z)-q(z)$；
$p(z)\le q(z)$ 时，左边 $=0$、右边 $=p(z)-p(z)=0$。

两边对 $z$ 求和：

$$
\sum_z\max\big(p(z)-q(z),0\big)=\sum_z p(z)-\sum_z\min\big(p(z),q(z)\big)=1-\sum_z\min\big(p(z),q(z)\big)
$$

**这个分母与第三步里「拒绝的概率」一模一样。** 记住这一点，下一步就只剩约分了。

#### 第五步：来路二

$$
P(B)=P(\text{拒绝})\cdot r(y)
=\Big(1-\sum_z\min(p,q)\Big)\cdot\frac{\max\big(p(y)-q(y),0\big)}{1-\sum_z\min(p,q)}
=\max\big(p(y)-q(y),\,0\big)
$$

被拒的质量（第三步）刚好等于纠正分布的分母（第四步），一约就没了——所以「拒绝带来的那部分
概率」不多不少，正好是 $\max(p-q,0)$。**在保持接受概率 $\min(1,p/q)$ 的前提下**，这就是
纠正分布不能换成别的原因：换掉后这个约分不成立，$P(B)$ 不再是 $\max(p-q,0)$，合计也就不再是 $p$。

（强调「保持接受概率」是因为**接受概率与纠正分布是一对**：两边同时换成另一套（接受概率
$=p(d)$、纠正分布「挖掉 $d$」）仍然无偏，只是接受率低得多——见 §1.5，那里有实测对照。）

#### 第六步：合计

$$
P(Y=y)=\min\big(q(y),p(y)\big)+\max\big(p(y)-q(y),\,0\big)=p(y)
$$

最后一步是逐点恒等式 $\min(a,b)+\max(a-b,0)=a$，同样验两种情形：$a>b$ 时
$\min=b$、$\max=a-b$，和为 $a$；$a\le b$ 时 $\min=a$、$\max=0$，和为 $a$。

**证毕：输出 token 的分布就是 $p$。** 用一句话复述这套规则在干什么：

- 两个模型都看好的 token：$\min(q,p)$ 那一项就够了，草稿本来就是对的；
- **target 比 draft 更看好**的 token（$p>q$）：draft 抽得不够多，差的那部分
  $\max(p-q,0)$ 完全由「拒绝 + 纠正」补上；
- **draft 比 target 更看好**的 token（$q>p$）：被拒掉的质量 $q-\min(q,p)$ 恰好等于别人需要
  补的质量（两边都等于 $1-\sum\min$），一分不多一分不少。

### 1.5 一个常见的疑问：能不能把草稿「当确定性提议」？

有人会想：草稿不也就是**抽出来的一枚 id** 吗？那干脆把它当成确定性提议（像 n-gram 那样），
用「接受概率 $=p(d)$、纠正分布 $=$ 挖掉 $d$」这一套不行吗？」

**行，它确实也是无偏的**——但**接受率白白砍掉一截**，而且在 $p=q$（draft 完全靠谱）时亏得最狠。
先把三种写法摆在一起（$q$ 指**实际抽出这枚草稿的那个分布**）：

| 接受概率 | 纠正分布 | 单 token 分布 | 实测接受率（$p=[.6,.3,.1]$、$q=[.2,.5,.3]$） |
|---|---|---|---|
| $\min\!\left(1,\dfrac{p(d)}{q(d)}\right)$ | $\mathrm{normalize}\big(\max(p-q,0)\big)$ | $p$ ✓ | $0.60=\sum_y\min(p(y),q(y))$ |
| $p(d)$（当确定性） | 挖掉 $d$ | $p$ ✓ | $0.30=\sum_y p(y)q(y)$ |
| $\min\!\left(1,\dfrac{p(d)}{q(d)}\right)$ | 挖掉 $d$ | $\ne p$ ✗ | — |

第二行就是「把草稿当确定性提议」：**它不是错的，是浪费的**。为什么它仍然无偏？照 §1.4 的
六步重算一遍，只把接受概率换成 $p(d)$、纠正分布换成「挖掉 $d$」：

$$
P(Y=y)=q(y)\,p(y)+\sum_{x\ne y}q(x)\big(1-p(x)\big)\cdot\frac{p(y)}{1-p(x)}
=q(y)p(y)+p(y)\!\!\sum_{x\ne y}\!q(x)
=q(y)p(y)+p(y)\big(1-q(y)\big)=p(y)
$$

两个 $1-p(x)$ 一约，剩下的 $p(y)\sum_{x\ne y}q(x)$ 正好把 $q$ 那部分补齐——这就是它对的原因。

**代价在接受率上。** 把两种规则的接受率写出来：

$$
\text{标准}：\sum_y q(y)\min\!\left(1,\frac{p(y)}{q(y)}\right)=\sum_y\min\big(p(y),q(y)\big)
\qquad\text{变体}：\sum_y q(y)\,p(y)=\langle p,q\rangle
$$

而 $\min(p,q)\ge p\cdot q$ 对每个 token 都成立（概率都 $\le 1$），所以**标准规则恒不差于变体**，
只在两端相等（$p$、$q$ 是同一枚 one-hot）时打平。差多少，取决于两个分布有多像：

- **$p=q$（draft 与 target 一样好）**：标准规则下 $\min(1,p/q)=1$ 恒成立 → **100% 全接受**；
  变体只接受 $\sum_y p(y)^2$——本关那个例子是 $0.46$，等于把「draft 完全猜对」这件事
  白白浪费掉一半；
- **$q$ 比较平**（draft 很没把握，例如接近均匀）：变体的接受率 $\langle p,q\rangle\approx 1/V$
  （$V$ 是词表大小，实际是 15 万），基本等于 0；标准规则仍能拿到 $p$ 的大部分质量。

直觉：**该看的比例是 $p/q$，不是 $p$ 本身**。$\min(1,p/q)=1$ 的意思是「draft 没有比 target
更看好这枚 token，那就白送」；变体却要求 $p(d)$ 绝对地大才接受，等于把 draft 的**信息**
（它对这个 token 有多自信）整段丢掉——draft 越自信，越不该被怀疑。

n-gram 之所以能用「$p(d)$ + 挖掉 $d$」，是因为**它的 $q$ 真的是一枚 one-hot**：这时
$p(d)=\min(1,p(d)/1)$ 成立，$\max(p-q,0)$ 也正好退化成「挖掉 $d$」，两行是同一个式子。
换句话说 n-gram 不是「另一套规则」，而是标准规则在 $q$ 为 one-hot 时的样子；把 $q$ 不再是
one-hot 的场合套用那套**公式**，就落到上表第二行——不偏，但接受率被砍。

第三行才是真的错：接受仍用 $\min(1,p/q)$、纠正却挖掉 $d$，实测首 token 分布
$[0.5038,0.3672,0.129]$ 明显偏离 $p$。§1.4 最后那步约分不成立，指的就是它。

这三行都有常驻用例盯着（`check_step55_rejection.py` 的「变体对照」一节）。**所以本关的选择
不是「对不对」的问题，而是「同样的无偏性，要哪一档接受率」**——投机解码的收益全在
接受率上，没有理由选差的那一档。

### 1.6 多枚草稿为什么仍然无偏

把「普通解码」想成一台机器：每一步都按 $p(\cdot\mid\text{当前历史})$ 抽一个 token、接在历史后面。
投机解码要做的是**用同样的条件分布、同样的顺序去抽，只是换了个抽法**。于是可以归纳：

- **第 1 步**：由 §1.4，提交的第一个 token 服从 $p(\cdot\mid\text{原历史})$；
- **归纳步**：假设前 $i$ 步的联合分布与普通解码一致。分两种情形：
  - 第 $i$ 枚草稿**被接受**：它成为新历史，第 $i+1$ 枚的判定用的是
    $p_{i}(\cdot\mid\text{历史}+d_i)$ 与 $q_{i}(\cdot\mid\text{历史}+d_i)$——正是「历史延长之后」
    的同一套论证（这就是**逐行临时惩罚历史**必须对的原因：每行的 $p_i,q_i$ 都得是**以该行看到的
    历史为条件**的分布，见 §4 与 §3）；
  - 第 $i$ 枚草稿**被拒**：本轮以纠正 token 结束（首次拒绝即停），而它的分布由 §1.4 恰好补回 $p$。

bonus 是「全部接受之后，从最后一行的 $p$ 里抽一枚」，也是这台机器的一步。链式法则把每一步的
条件分布乘起来：

$$
P(Y_1=y_1,\ \dots,\ Y_m=y_m)=P(Y_1=y_1)\prod_{i=2}^{m}P\big(Y_i=y_i\mid Y_1{=}y_1,\dots,Y_{i-1}{=}y_{i-1}\big)
$$

右边每个因子都等于普通解码的对应因子，所以**整条输出序列与普通解码同分布**——这就是
「投机只改变怎么算，不改变算什么」的确切含义。

K=2 时把它写开，就是验收里对着经验值比的那两个式子（`check_step55_rejection.py`）：

$$
\begin{aligned}
P\big(\text{提交 }(y,z)\big)&=q_y\cdot\min\!\left(1,\tfrac{p_y}{q_y}\right)\cdot p_{\text{row1}}[z]
&&\text{（草稿 }y\text{ 被接受，}z\text{ 是 bonus，来自第二行）}\\[2pt]
P\big(\text{提交 }(c)\big)&=\sum_y\Big[q_y-q_y\cdot\min\!\left(1,\tfrac{p_y}{q_y}\right)\Big]\cdot r_y(c)
&&\text{（草稿 }y\text{ 被拒，}c\text{ 是纠正 token）}
\end{aligned}
$$

注意 $r_y$ 与拒绝质量都是**逐枚草稿**的：草稿不同，$r$ 不同，接受概率也不同。

### 1.7 四个特例（代码里都能找到对应的分支）

| 情形 | 提议分布 $q$ | 接受概率 | 纠正分布 | 代码 |
|---|---|---|---|---|
| draft 与 target 同结构同权重（$p=q$） | $p$ | 恒为 $1$ → **整轮全接受**，一次随机数都不抽 | 用不到 | `accept_prob >= 1` |
| n-gram 确定性提议 | $d$ 上的 one-hot | $p(d)$ | 挖掉 $d$ 再归一化（$\max(p-q,0)$ 的退化） | `draft_probs=None` |
| 贪心 target + 贪心 draft | 两边都是 one-hot | $p(d)\in\{0,1\}$，即「argmax 是否相同」 | target 的 argmax | `_is_greedy_without_penalty` 的整批 argmax 快路径 |
| 草稿被 top-k/top-p 过滤掉（$p(d)=0$） | — | $0$ → **必拒且不抽随机数** | $r=p$（挖掉的本就是零质量） | `accept_prob <= 0` |

第二行说明了为什么第 52~54 关的代码在第五十五关一行没改还能用：n-gram 只是 $q$ 的一个特例。

### 1.8 为什么「提议必须真的从 q 抽」

证明里的 $q(y)$ 是**草稿等于 $y$ 的概率**。如果提议用 argmax、验证却拿 softmax 出来的 $q$，
那么实际的提议分布是一个 one-hot（记作 $q_{\text{eff}}$），而验证用的是另一个分布——接受概率
$\min(1,p/q)$ 算的不是真值，输出就有偏了。同理 $q_i(d_i)=0$ 直接报错：从 $q$ 里根本抽不出
一个质量为零的 token，出现这种输入一定是调用方算错了。

### 1.9 为什么会「差一个位置」（两套 KV 的形状从哪来）

要提 $K$ 枚草稿，draft 需要喂 $K$ 个 token：$x, d_0, \dots, d_{K-2}$（第 $j$ 枚草稿是「喂了前 $j$ 个
token」之后预测出来的），所以它只写了 $K$ 个位置，**最后一枚 $d_{K-1}$ 还没进过 draft**；
而 target 一次吃 $[x, d_0, \dots, d_{K-1}]$，写 $K+1$ 个位置。全接受时 draft 天然落后 1，
下一轮提议前必须先补算那一枚（§2 的轨迹）。

反方向也要对齐：被拒的草稿在两边都得回滚。target 上它属于「本轮算过、但不属于真实前缀」
的部分；draft 上它更糟——那是**用错的 token 算出来的 KV**。KV 是因果的，只有真实前缀的 KV
才有意义，错 token 的 KV 会把后面所有位置带偏，所以既不能留、也不能拿去接着提议。

### 1.10 加速比长什么样（以及为什么本关不承诺加速）

先把「一轮平均能白赚几个 token」算出来。第 $i$ 枚草稿能进入输出的前提是**前 $i$ 枚都被接受**
（首次拒绝即停），所以定义「第 $i$ 枚被接受」这件事的概率为

$$
\beta_i=P(\text{第 }1,\dots,i\text{ 枚都被接受})=\prod_{j=1}^{i}\alpha_j
\qquad(\alpha_j=P(\text{第 }j\text{ 枚被接受}\mid\text{前 }j-1\text{ 枚都被接受}))
$$

这是条件概率的乘法：先要第 1 枚过，再在第 1 枚过了的条件下第 2 枚过……一步步乘起来。

**期望**（$E[X]$，读作「X 的期望」）就是「长期平均下来 X 是多少」。用一个小技巧：令
$I_i=1$ 表示第 $i$ 枚被接受、否则 $0$，那么「被接受的枚数」$=\sum_{i=1}^{K}I_i$，
而期望可以逐项相加（期望的线性性，它不要求各枚相互独立）：

$$
E[\text{接受枚数}]=\sum_{i=1}^{K}E[I_i]=\sum_{i=1}^{K}P(\text{第 }i\text{ 枚被接受})
=\sum_{i=1}^{K}\prod_{j=1}^{i}\alpha_j
$$

- $K=1$ 时就是 $\alpha_1$；
- $K=2$、每枚接受概率都是 $0.5$ 时是 $0.5+0.25=0.75$ 枚。

一轮的产出 ≈ $1+E[\text{接受枚数}]$（含 bonus），成本 ≈ 1 次 target forward $+\ K$ 次 draft forward。
若 draft 单次前向比 target 便宜 $c$ 倍：

$$
\text{加速比}\approx\frac{1+\sum_{i=1}^{K}\prod_{j\le i}\alpha_j}{1+c\cdot K}
$$

$\alpha$ 高、$c$ 小才赚；小 batch、短输出、draft 与 target 差距不大时完全可能亏。接受率低
**不是**实现错误（随机初始化的玩具模型接受率必然低），本关的判据始终是「输出分布对不对、
两套 KV 对不对」，不是快不快。

### 1.11 一遍数值例子

需求 §4 的例子：$p=[0.6,0.3,0.1]$、$q=[0.2,0.5,0.3]$，草稿抽到了 $d=1$。

**先算规则里的两个量。**

$$
\text{接受概率}=\frac{p(1)}{q(1)}=\frac{0.3}{0.5}=0.6
\qquad
\max(p-q,0)=\big[\max(0.6{-}0.2,0),\ \max(0.3{-}0.5,0),\ \max(0.1{-}0.3,0)\big]=[0.4,\ 0,\ 0]
$$

$$
r=\mathrm{normalize}\big([0.4,0,0]\big)=\frac{[0.4,0,0]}{0.4+0+0}=[1,\ 0,\ 0]
$$

所以：有 60% 的概率直接接受草稿 1；否则（40%）一定吐出 token 0。

**再验算无偏性**（三枚 token 的输出概率都要回到 $p$）。拒绝概率
$=1-\big(\min(0.2,0.6)+\min(0.5,0.3)+\min(0.3,0.1)\big)=1-(0.2+0.3+0.1)=0.4$：

| token $y$ | 来路一：$q(y)\cdot\min\!\big(1,\tfrac{p(y)}{q(y)}\big)$ | 来路二：$0.4\times r(y)$ | 合计 | 目标 $p(y)$ |
|---|---:|---:|---:|---:|
| 0 | $0.2\times1=0.2$ | $0.4\times1=0.4$ | $\mathbf{0.6}$ | 0.6 ✓ |
| 1 | $0.5\times0.6=0.3$ | $0.4\times0=0$ | $\mathbf{0.3}$ | 0.3 ✓ |
| 2 | $0.3\times\frac{1}{3}=0.1$ | $0.4\times0=0$ | $\mathbf{0.1}$ | 0.1 ✓ |
| 合计 | 0.6 | 0.4 | 1.0 | 1.0 ✓ |

读法：draft 偏爱 token 1（$q=0.5$）而 target 没那么喜欢（$p=0.3$），于是 token 1 的最终概率被
「削」到 0.3；多出来的 0.4 通过「拒绝 + 纠正」一分不少地还给了 token 0（target 更偏爱的那个）。
**注意 token 2 一分钱没拿到**：它的 $p<q$，不是「差的那一方」。

### 1.12 一轮的伪代码

```python
# 提议（draft 模型，同一轮多请求在同一个 batch 里逐位置进行）
for j in range(K):
    token = x if j == 0 else drafts[j - 1]          # 第 0 枚喂 pending x
    q_j = distribution(draft_model(token), params, 历史=真实生成 + 前 j 枚草稿)
    drafts.append(sample(q_j, draft_generator))     # 必须从 q 抽

# 验证（target 一次 forward 出 K+1 行）
for j in range(K):
    ratio = p_j[drafts[j]] / q_j[drafts[j]]         # q 缺省时就是 p_j[drafts[j]]
    if ratio >= 1 or (ratio > 0 and uniform() < ratio):
        接受，继续
    else:
        提交 已接受的草稿 + sample(normalize(max(p_j - q_j, 0)))，本轮结束
if 全部接受:
    提交 所有草稿 + sample(p_K)                      # bonus 来自最后一行
两套 KV 各自回滚到真实前缀                            # target: 保留长度; draft: 夹到同一条边界
```

## 2. 一轮到底发生了什么

一轮的顺序（实现在 `engine.py:step()` 与 `draft.py:run_round()`）：

```text
Scheduler.schedule()      真实 token 预算 + 草稿名额预留（num_reserved_drafts）+ 两池只读容量检查
  → DraftModelProposer.run_round()
        补算：把已提交历史补进 draft 的 KV（按 chunk，受 draft 自己的预算限制）
        提议：第 j 步把「还想要第 j 枚」的请求合成一个 batch 跑一次 draft
  → 组装 target 输入（此时才知道最终草稿数）→ plan_sample_rows
  → target 一次 forward
  → 采样层：一般拒绝采样 + 两套 KV 的回滚/对齐（唯一提交入口）
  → post_step（发布前缀、判停、回收）
```

`run_round` 必须在**组装输入之前**：它改的正是本轮的输入行与计数。这一条踩过坑——
先组装再提议，target 会拿旧的输入配新的计数，直接 `IndexError`。

### 两套 KV 的具体轨迹（脚本模型，K=2，prompt 长 6，两个模型都「整体 +1」）

```text
进轮前：target.length = 6，draft.length = 6，真实历史 [1,2,3,1,2,3,4]，x = 4（位置 6）

提议：draft 喂 x=4 → d0=5（draft.length 6→7）
      draft 喂 d0=5 → d1=6（draft.length 7→8）
      ⚠ 此时 d1 还没进过 draft 模型
target：一次输入 [4,5,6]，写 KV 到 9
验证：  全接受 → 提交 [5,6,7]（d0、d1、bonus=7）
        target 保留 1+K = 3 个输入位置 → target.length = 9
        draft 只看得到 x、d0 → draft.length = 8，**比 target 少 1**
下一轮：补算那 1 枚（d1=6）→ draft.length = 9 追平 → 再从 x=7 提议
```

三种验证结果对应的对齐（需求 §3 的表，这里给实测数字）：

| 目标验证结果 | 提交 | target 保留 | draft 处理 |
|---|---|---:|---|
| 首拒绝（第 1 枚被拒） | `[d0, c]` | `6+2 = 8` | 提议后 draft 正好 8，**齐平** |
| 部分接受 / 首枚即拒（跨块例） | `[x]` | `8+1 = 9` | draft 从 12 夹到 9（丢掉 3 个位置） |
| 全接受 | `[d0,d1,b]` | `6+3 = 9` | draft 8 → 下一轮补算 1 枚 |

**对齐规则只有两条**（`draft.py:align()`）：draft 比 target 长就夹到 `seq.cache.length`
（多出来的一定是被拒或未验证的草稿），比 target 短就什么都不做、下一轮提议前统一补算。
时机与 target 的回滚一样，必须在 `post_step()` 之前。

### 连续四轮的真实轨迹（脚本模型，K=2，prompt `[1,2,3,1,2,3]`）

目标脚本「整体 +1」，但在**位置 11** 故意分叉成 3——于是第二轮会真的被拒一次：

| 步 | 本轮输入 | 草稿 | 结果 | target KV | draft KV | 累计输出 |
|---|---|---|---|---|---|---|
| 1 | `[1,2,3,1,2,3]` | — | prefill，采样出 4 | 6 | 0 | `[4]` |
| 2 | `[4, 5, 6]` | `[5,6]` | **全接受** → 提交 `[5,6,7]` | 9 | **8**（差 1） | `[4,5,6,7]` |
| 3 | `[7, 8, 9]` | `[8,9]` | 接受 d0、**拒绝 d1** → 提交 `[8, 3]` | 11 | 11（齐平） | `[4,5,6,7,8,3]` |
| 4 | `[3, 4, 5]` | `[4,5]` | **全接受** → 提交 `[4,5,6]` | 14 | **13**（差 1） | `[4,5,6,7,8,3,4,5,6]` |

读法：步 2 结束后 draft 停在 8 而 target 到了 9，步 3 一开始就**先补算那一枚**（草稿 6
的 KV）再提议；步 3 被拒的是第 2 枚，所以两边齐平、不需要夹；步 4 又是全接受，draft 再次
落后 1——**「落后 → 补算 → 提议」就是稳态**。每一步提交时刻都断言
`draft.length <= target.length`（draft 绝不领先于真实历史）。

## 3. 一般 p/q 拒绝采样

原理见 §1.4——这一节只说代码里的形状（第 $i$ 枚草稿用第 $i$ 行的两个分布 $p_i,q_i$ 判定）：

$$
\text{接受概率}_i=\min\!\left(1,\ \frac{p_i(d_i)}{q_i(d_i)}\right),
\qquad
\text{纠正分布}_i=\mathrm{normalize}\big(\max(p_i-q_i,\ 0)\big)
$$

**在接受概率仍用 $\min(1,p/q)$ 的前提下，纠正分布不能换成别的**——§1.4 最后那步约分
就不成立了：第 52~54 关的 n-gram 是确定性提议（$q$ 是 token 上的一枚 one-hot），那时
$\max(p-q,0)$ 退化成「把该 token 挖掉再归一化」；对一般 $q$ 继续只挖掉一个 token 得到的
就不是 $p$：需求 §4 的例子（$p=[0.6,0.3,0.1]$、$q=[0.2,0.5,0.3]$、$d=1$）正确值是 $[1,0,0]$，
只挖 token 1 会得到 $[6/7,0,1/7]$，实测首 token 分布 $[0.5038,0.3672,0.129]$。

（**接受概率与纠正分布是一对**，不能只看一个。把「挖掉 $d$」配上「接受概率 $=p(d)$」也是
无偏的——那是另一个有效的采样器，只是接受率低得多，见 §1.5。）

`draft_probs=None` 表示确定性提议，两条路径共用同一个函数、同一份 EOS 与「保留多少 KV」
规则（`_finish_candidates`）。边界约定（判的都是**接受概率** $\min(1,p_i(d_i)/q_i(d_i))$）：

- 接受概率 $\ge 1$（即 $p_i(d_i)\ge q_i(d_i)$，含 $p(d)=1$）：必接受，不抽 uniform；
- 接受概率 $\le 0$（即 $p_i(d_i)=0$）：必拒，不抽 uniform；
- $p=q$ 时每一枚都落在必接受上：整轮全接受、一次接受随机数都不抽，也不会走到 residual；
- $q_i(d_i)=0$ 是**非法输入**（从 $q$ 里抽不出质量为零的 token），明确报错、不除零；
- 接受的草稿若是终止 token：**在接受的当下就停**（第五十四关复验的结论，见
  `docs/step54_random_speculative.md` §8.10），后面的草稿不验证、bonus 不抽。

**为什么不用「更简单」的那套**（把草稿当确定性提议：接受概率 $p(d)$、纠正分布挖掉 $d$）：
它也**无偏**，但**接受率被砍**——这一关的收益全在接受率上。需求 §4 的例子实测
$0.30$ vs 标准规则的 $0.60$；$p=q$（draft 完全靠谱）时更狠：标准规则 $\min(1,p/q)=1$
恒成立（100% 全接受），那套只接受 $\sum_y p(y)^2=0.46$。推导与三行对照表在 §1.5，
常驻用例在 `check_step55_rejection.py` 的「变体对照」一节。**本关选的是接受率更高的那条，
不是「更简单」的那条。**

## 4. 提议层（`draft.py`）

`DraftModelProposer` 持有 draft 模型、**第二个** KV 池、采样后端与 draft 自己的 token 预算。
它不认识 `Engine`，也不认识 `Scheduler`。五个入口：

| 方法 | 做什么 |
|---|---|
| `backlog(seq)` | draft 还差多少个 token 才追上 target 的已计算前缀 |
| `fits(seq, k)` | 只读地问：draft 池容不容得下「补算缺口 + k 枚草稿」（给计划阶段用） |
| `catch_up(seq, budget)` | 把已提交历史 `[draft.length, target.length)` 补进 draft 的 KV。
  **一轮只做一个 chunk**（取「还差多少」与「本轮剩余预算」里小的那个）：补不满就返回
  False、这一轮不提议，缺口下一轮继续——`backlog()` 每轮重算，所以「补到哪儿了」只有
  `draft_cache.length` 一处状态 |
| `run_round(items)` | 补算 + 逐位置批量提议，把草稿写回本轮计划（含 q） |
| `align(seq)` / `release(seq)` | 验证后夹回边界 / 抢占、完成、失败时释放 |

**补算只有一条路径**：draft 落后只会发生在（a）刚准入或命中前缀，（b）被抢占后重算，
（c）上一轮没排上、草稿池不够或走了 fallback。三种情形都走 `catch_up()`——不维护第二份
可分叉的真实 token 列表，draft 也**从不读 target 的 KV**（测试用「draft 的 KV 必须等于它
自己对已提交前缀重算的结果」钉死这一点）。

**提议是批量的**：第 j 步把所有还想要第 j 枚的请求合成一个 batch（每请求一行），
不套「请求循环 × K 次单请求前向」。真实模型上实测：提议前向 13 次提出 19 枚（平均
1.46 枚/次），补算另计 13 次 / 54 个 token。请求的 K 可以不同（终止 token、池子不够、
预算见底），batch 因此是 ragged 的。

**终止 token 之后不再提议**：它要么被接受（验证在那里就结束）、要么被拒（首个拒绝就是它），
后面的草稿永远读不到。

**预算分开计数**：`draft_max_num_batched_tokens` 是 draft 自己的预算（默认与 target 相同），
补算的 chunk 与提议的位置都从**同一个**预算里扣，报告里 `num_catchup_forwards` /
`num_proposal_forwards` 分开统计——小模型的前向不能偷偷记成 0。

两者共享预算是刻意的、也有代价：预算小到「补算就吃光」时，这条请求**一直**走普通 target
路径（实测 `draft_max_num_batched_tokens=2`、prompt 长 6 时提议 0 枚），而 backlog 每轮还
随 target 前进长 1。这条降级路径有测试盯着：输出必须与不开投机**逐 token 相同**——
草稿是可选加速，降级只能影响速度，不能影响结果。

## 5. 双池容量与失败原子性

- 计划阶段（`Scheduler._plan_tokens`）只发**草稿名额**（`num_reserved_drafts`），草稿要跑过
  draft 才知道，写在实际草稿 `draft_ids` 里；名额与实际枚数可以不一致（实际更少），
  缩草稿缩的是名额；
- `_reserve_blocks()` 缩草稿时**两个池子都问**：`can_grow(seq, num_scheduled_tokens)`
  与 `draft.fits(seq, k)`，任一不满足就砍一枚，砍到 0 就是普通的 1-token 路径；
- 所有池操作都是**先计划后提交**（`ensure_blocks_for` 失败时一个字节都不改），
  不会出现「target 成功分配、draft 失败」的泄漏；
- draft 池不够时**只缩草稿**，绝不为可选草稿去抢占 target 的其他请求；
- 实际草稿比预留少时，多预留的 **target** 块由验证后的回滚（`truncate` 到保留长度）
  还回，draft 那边由 `align()` 夹回边界——两个池子各有一条归还路径，不重复还；
- **一枚草稿都没提出来时（fallback）也要还**：这条最初漏了——「多预留的块由验证后的回滚
  自然归还」只对**实际草稿非空**成立（普通采样路径根本不 truncate）。于是「批准 K=2、
  实际 K=0」时 target 会一直挂着 1 个用不到的整块（不是永久泄漏——请求结束时引用照样
  归零——但运行中白占，会挤占别的请求的容量；只查「最后池子归零」发现不了）。
  现在 `SampleRuntime.run()` 结尾有一个**统一的收尾**：对本轮每个 item 检查
  「块表长度 > `ceil(cache.length / block_size)`」就把尾巴还给池子。它对有草稿的路径是
  空操作（已经 truncate 过），对普通路径也通常是空操作——但正是它**执行**了三条不变量中的
  「持块数 == 已算到的整块数」，而不只是写在文档里。第五十五关复验就是这么抓出来的
  （`benchmarks/review_step55_fallback_blocks.py`）。
- 请求完成 / 失败 / 被抢占：两套活动 KV 一起释放（`_preempt` / `_finish_completed_requests`
  / `_fail` 三处），真实历史、优先级、两条随机流全部保留。

测试构造了两种「draft 资源不够」：**计划阶段**就放不下历史（K 直接缩到 0，全程走普通
target 路径）与**运行时**才不够（两条请求的计划都通过了只读检查，第一条补算+提议吃掉
大部分池子，第二条当场补不动）。两种情况下 target 的输出都与「不开投机」逐 token 相同。

## 6. RNG 与 seed 派生

- **两条独立的流**：draft 抽提议用 `seq.draft_generator`，target 抽接受/纠正/bonus 用
  `seq.sampling_state.generator`。
- draft 的种子由请求的 `seed` **稳定派生**（`derive_draft_seed()`：一次线性同余混合，
  常数写在代码里）。**不用 Python `hash()`**——字符串的 hash 被 PYTHONHASHSEED 打乱，
  拿它派生会让「同 seed 同工作序列可复现」失效。
- `seed=None`（调用方没要求复现）时从**全局随机源**取一个：就是 PyTorch 的默认生成器
  `torch.default_generator`（**CPU 上的那一个**，由 `torch.manual_seed()` 播种；不传
  `generator=` 的 `torch.randint` 用的正是它，与 CUDA 的默认生成器是两个）。它是
  **进程级共享**的，所以只在**请求创建时**碰一次、绝不在 `step()` 里抽——这样别的请求
  的推进不会扰动已存在请求的随机流。「同 seed 同工作序列可复现」里的**同工作序列**
  就是这个意思：全局源推进到哪儿取决于之前建过多少条请求。
- **本包从不调用 `torch.manual_seed()`**（只有请求自己的 `torch.Generator.manual_seed()`
  与测试脚本里的调用）：库去播种进程级全局源会踩掉调用方自己的随机流，也会让引擎行为
  取决于 import / 调用顺序。所以「可复现」的前提分两种——请求写了 `seed=` 就**无条件**
  可复现（种子由它稳定派生，与全局状态无关）；没写 `seed=` 则要调用方先播种全局源
  （测试里就是 `build()` 开头的 `torch.manual_seed(seed)`），再加上「同工作序列」。
  一个没播种的新进程，全局源在进程启动时就是随机的（实测三个进程三个不同的值）。
- **为什么 PyTorch 不像 C 的 `rand()` 那样「不播种就每次同一个序列」**：现代库的取向是
  「可复现必须显式 opt-in」——不播种就真的不可预测（Python 的 `random`、NumPy 的新
  `Generator` API 都一样，C 的「等价于 `srand(1)`」被普遍当成历史包袱）。反过来，
  **新建的 `torch.Generator()` 反而是定值**（实测跨进程都是 `67280421310721`），
  这正是 `SamplingState` / `make_draft_generator()` 必须显式 `manual_seed()` 的原因：
  不播的话所有没写 seed 的请求会共读同一条流。
- 上界写 `2**63 - 1` 而不是 `2**63`：后者超出 int64 的范围，`torch.randint` 会直接抛
  `Overflow when unpacking long long`。这条路径（**随机采样 + 不写 seed**）一度没有被
  任何用例覆盖——draft_model 的测试要么贪心（`make_draft_generator()` 提前返回）、
  要么显式给了 seed——现在 §9 有专门的用例盯着。
- 抢占、重算、缩草稿、fallback 都**不重置也不消耗**任何一条流。
- 草稿被拒后**不回退** draft 的随机流：KV 回滚与 RNG 回滚是两件事。
- 同 seed 同工作序列可复现；不要求与「普通随机解码」的文本相同（两条流的抽样次数不同）。

## 7. 加载：分片 safetensors 与双目录

- `formats.read_raw_weights()` 现在同时支持单文件与分片目录：按
  `model.safetensors.index.json` 的 `weight_map` 读，每个唯一分片只 `load_file` 一次，
  合并成一份字典（本关不做流式低峰值加载）。
- 四类坏索引明确报错：分片缺失、索引声明的参数在分片里不存在、分片里有索引未声明的
  参数、参数被声明到别的分片（重复）。分片名必须是**目录内的文件名**，挡 `../` 穿越。
- `Engine.from_model_dir(target_dir, draft_model_dir=...)`：两个模型各读自己的目录，
  结构可以完全不同（本关的 0.6B 是 28 层 hidden 1024、1.7B 是 28 层 hidden 2048），
  draft 池按 draft 自己的层数/头数/精度建。
- `check_draft_model()` 校验：同设备、**同词表**、**同 EOS 集合**、draft 的 `max_seq_len`
  不短于 target、输入缓冲不小于 draft 预算、必须显式给 `draft_num_kv_blocks`。
  词表大小相同不等于 tokenizer 相同——本关只接受调用方提供的同词表模型，不做转换。

## 8. 模块划分

```text
speculative.py  算法层（纯函数）：n-gram 提议 + 一般拒绝采样（p/q）+ 纠正分布
      ↓
sampling.py     采样原语：参数/状态/惩罚/分布(TorchSampler.distribution)/取 token(draw)
      ↓
draft.py        draft 提议层：补算 → 批量提议 → 对齐 → 释放（第二套 KV，不认 Engine/Scheduler）
      ↓
sample_runtime.py  采样执行层：行映射 → 三路径 → 验证与两套 KV 回滚 → 唯一提交入口
      ↓
scheduler.py    本轮跑谁、跑几个 token、两个池子的容量取舍
      ↓
engine.py       装配（两个模型、两个池子、提议层）与一轮的编排
```

两处低层接口是本关新加的：`KVCachePool` 的 `ensure_blocks_for / can_grow_for /
truncate_cache / release_cache`（按显式 `CacheConfig` 操作，target 池与 draft 池共用
同一份块管理，不复制第二份），`TorchSampler.draw()`（「分布 → token」的唯一实现，
提议与采样共用）。

## 9. 验证

### 9.1 脚本清单（全部通过，共 326 项）

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step55_speculative.py` | 57 | 第五十二关的纯函数 + 本关的 draft_model 配置校验 |
| `check_step55_batch.py` | 35 | 第五十三关的行映射 / 混批 / 预算 / 容量 / 恢复 |
| `check_step55_rejection.py` | 54 | 拒绝采样验证层（含一般 p/q、统计检验、与「当确定性提议」变体的接受率对照） |
| `check_step55_random.py` | 22 | 第五十四关的引擎状态（临时计数、RNG、重算、混批） |
| `check_step55_engine.py` | 55 | 单/多请求等价性、目录加载入口 |
| `check_step55_combinations.py` | 17 | priority / 前缀缓存与投机的组合 |
| `check_step55_draft_kv.py` | 64 | **本关的双 KV**（轨迹、对齐、边界、回退、抢占、priority 动态到达、前缀命中、惩罚一致性、
  补算 chunk 与预算共享、draft 随机流、CUDA 冒烟） |
| `check_step55_loading.py` | 12 | 分片权重 + 双目录加载 |
| `check_step55_real_qwen3.py` | 10 | 真实 1.7B + 0.6B 端到端（CUDA BF16） |
| `diff_step54_step55.py` | 88 | `speculative_mode=None` 下与 step54 **逐步逐字节一致** |

### 9.2 有判别力的几条

- **一般 q 的统计检验**：草稿每次从 q 抽，最终首 token 的经验分布回到 p
  （`[0.5998, 0.2998, 0.1004]`，20 万次，5σ 内）；条件两 token 的联合分布与「接受部分
  `q_y·min(1,p_y/q_y)·p_row1[z]` + 拒绝部分 `(q_y - accept)·residual_y`」的理论值一致。
  **反证**：把接受概率里的 q 去掉，确定性用例立刻 FAIL；把纠正分布退回「只挖掉草稿」，
  4 条 FAIL，统计用例实测 `[0.5051, 0.3659, 0.129]`——正落在预期的错误分布上。
- **两套 KV 与单独重算一致**（本关最重要）：跑完之后取每套 KV 的**有效部分**，与该模型
  对已提交前缀单独重算一遍的结果逐个张量比对。同一个模型在**不同 batch 形状**下前向会有
  FP32 级舍入差（实测 ~3e-7），所以按 `atol=1e-5` 比；**并配了「故意改坏一个有效位置」
  的对照**（改坏后差 5e-1，立刻 FAIL），证明这条比对真的在看东西。
- **对齐契约**：全接受时 draft 恰好比 target 少 1（最后一枚被接受的草稿还没进 draft 的
  KV），下一轮补算后追平；第二轮从**上一轮最后一个提交的 token**（bonus）继续提议，
  不重复喂 x；部分接受时两边齐平；跨块回滚真的把多占的整块还回池子（4 块 → 3 块）。
- **真实模型 greedy 逐 token 相同**：1.7B target + 0.6B draft，两条请求各 16 个 token，
  与不开投机的贪心**逐个 ID 相等**。贪心时目标分布是 one-hot，投机只能改「怎么算」。
- **target 每轮最多一次 forward**（16 次 / 16 步）、提议是批量的（1.46 枚/次）。
- **每步收尾后两套池子的三条不变量**（`check_step55_draft_kv.py` 的
  `check_round_invariants()`）：持块数 == `ceil(cache.length / block_size)`、
  引用计数 == 真实持有数、可分配链 == 真正空闲的块。第三条以前没有，验收方用
  「零草稿回退」抓出了漏网的路径；**反证**：去掉收尾这一步，它立刻报
  `cache 持块 3 != ceil(length 7 / 4) = 2`。
- **带三种惩罚项的 greedy 与普通路径逐 token 相同**：惩罚让每一行的历史都不同
  （真实生成 + 前 i 枚草稿），两个模型的提议与验证都按这条规则喂历史。
- **CUDA FP32 / BF16 冒烟**：draft_model / ngram / 关 三种模式各跑一遍，两条请求
  都完成、两套池子归零。

### 9.3 反证清单

| 断言 | 怎么反证的 |
|---|---|
| `max(p-q,0)` 纠正分布 | 退回「只挖掉草稿」→ 4 条 FAIL，统计值落在错误分布上 |
| `min(1,p/q)` 接受概率 | 去掉 q → 确定性用例 FAIL |
| 两套 KV 与重算一致 | 故意改坏一个位置 → 差 5e-1 立刻 FAIL |
| 双 KV 对齐（draft ≤ target） | 每一步的提交时刻都断言 `draft.length <= target.length` |
| 分片加载的错误处理 | 五种坏索引逐个造出来，确认每一种都报错 |
| 零草稿回退的块归还 | 去掉收尾 → `check_step55_draft_kv.py` 的每步不变量立刻失败
  （`cache 持块 3 != ceil(7/4) = 2`），验收方的 `review_step55_fallback_blocks.py`
  也从 FAIL 变 PASS |
| 「草稿被接受过」这条断言 | 旧写法只看「提议数 > 0 且生成满 16 枚」，草稿全被拒也成立
  ✗；现在看采样层记的**实际接受枚数**（真实模型：贪心 10/19、随机 10/27） |
| 「当确定性提议」这一变体 | 实测两件事：它的首 token 分布**也**回到 $p$（无偏），但接受率
  $0.30$ vs 标准的 $0.60$；$p=q$ 时更是 $1.00$ vs $0.46$——证明标准规则是「同样的无偏性、
  更高的接受率」的选择（§1.5） |

## 10. 接口变化与遗留

### 10.1 接口变化

**新增**：`speculative_mode="draft_model"`；`Engine(..., draft_model=, draft_num_kv_blocks=,
draft_max_num_batched_tokens=)`（全部 keyword-only，旧位置参数一个都没挪）；
`Engine.from_model_dir(..., draft_model_dir=...)`；`draft.DraftModelProposer` /
`DraftProposal` / `derive_draft_seed()` / `make_draft_generator()`；
`speculative.verify_drafts_random(..., draft_probs=)`；`residual_probs(p, q, token_id)`；
`sampling.TorchSampler.draw()`；`SampleRuntime.num_accepted_drafts`（累计接受的草稿枚数，
与提议侧的 `DraftModelProposer.num_proposed_tokens` 配套）；`KVCachePool` 的四个 `*_for` / `*_cache` 方法；
`SequenceConfig.draft_cache` / `draft_generator` / `draft_seed`。

**不变**：`speculative_mode=None` 与 `"ngram"` 的行为（`diff_step54_step55.py` 88 项
逐步逐字节一致）；旧包的 import 路径；`TorchSampler.select()` 的语义（仍收**原始**行）。

### 10.2 遗留

1. **不做 GPU rejection kernel**：验证仍是逐请求的 Torch 循环，允许必要的设备同步
   （与第五十四关同一条约束）。
2. **不做异步调度下的采样**：`draft` 与 `target` 的前向都在本轮内同步完成。
3. **draft 池不做前缀共享**：恢复、命中、fallback 全部靠 `catch_up()` 从已提交历史补算。
   好处是两套缓存的交互只有一条路径，代价是重复计算。
4. **不做流式低峰值加载**：分片先合并成一份字典（1.7B 峰值约 4 GB），不做逐片装入。
5. **不做自适应 K、不重发缩 K 后的闲置预算**（需求明确不要求）。
6. **不承诺加速**：本关没有任何吞吐结论。真实模型上小 K、短输出的场景里草稿前向本身
   也是成本，收益取决于接受率与 batch 形状——那是后续专题的事。
7. **同词表假设**：只校验 `vocab_size` 与 EOS 集合，不做 tokenizer 转换（需求允许）。
8. **`speculative_mode="ngram"` 与 `"draft_model"` 不能同时开**：本关只做二选一。
9. **被否决的方案：把草稿当确定性提议**（接受概率 $p(d)$ + 纠正分布「挖掉 $d$」）。
   它同样无偏（§1.5 有推导与实测），但接受率恒不优于标准规则（逐 token
   $\min(p,q)\ge p\cdot q$）：需求 §4 的例子 $0.30$ vs $0.60$，$p=q$ 时 $0.46$ vs $1.00$。
   被否决的原因是**代价落在本关唯一在意的地方**（接受率），不是因为它错。
   验一下这个取舍随时可以复跑：`benchmarks/check_step55_rejection.py` 的「变体对照」一节
   把它和标准规则、以及真正有偏的混用写法（$\min(1,p/q)$ + 挖掉 $d$，实测分布
   $[0.5038,0.3672,0.129]$）放在同一组用例里。
