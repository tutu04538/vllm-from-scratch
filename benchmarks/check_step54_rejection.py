"""第 54 关：拒绝采样验证层（需求 §5①②③）。

分三块，都不启动引擎：
  1. 确定性分支：注入 draw_uniform / draw_token，构造全接受、首拒绝、部分接受、
     bonus、草稿 EOS、纠正 EOS，并数清**随机数调用次数**；
  2. 分布正确性：固定 p=[0.6,0.3,0.1]、草稿 1，重复采样的经验分布要收敛到 p，
     而不是「拒绝后仍从原 p 重抽」的错误分布 [0.42,0.51,0.07]；两 token 的条件
     联合分布同样要对上 p(a)·p(b|a)；
  3. 惩罚与过滤：温度 / top-k / top-p 把草稿过滤掉时 p[d]=0；重复草稿让第二行的
     frequency 惩罚与第一行不同；逐行概率与「普通单步采样」的参考实现一致。
"""

import math
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step54.sampling import SamplingParams, SamplingState, TorchSampler
from step54.speculative import DraftVerification, residual_probs, verify_drafts_random

FAIL = []


def _raises(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


EOS = {63}


class Draws:
    """可注入的两个回调 + 调用计数（需求 §5① 要求数清消耗了几个随机数）。"""

    def __init__(self, uniforms=(), tokens=()):
        self.uniforms = list(uniforms)
        self.tokens = list(tokens)
        self.n_uniform = 0
        self.n_token = 0

    def uniform(self):
        self.n_uniform += 1
        return self.uniforms.pop(0)

    def token(self, probs):
        self.n_token += 1
        if self.tokens:
            return self.tokens.pop(0)
        return int(torch.argmax(probs))


def probs(*values):
    return torch.tensor(values, dtype=torch.float32)


def verify(draft_ids, row_probs, draws, remaining=8, eos=EOS):
    return verify_drafts_random(draft_ids, row_probs, eos, remaining, draws.uniform, draws.token)


# ------------------------------------------------ 1. 确定性分支

# 全接受：两枚草稿的 p[d] 都是 1 -> 不抽 uniform，只抽一次 bonus
d = Draws(tokens=[9])
r = verify([1, 2], [probs(0, 1, 0, 0), probs(0, 0, 1, 0), probs(0, 0, 0, 1)], d)
check("全接受：提交两枚草稿 + 一枚 bonus，保留 1+K 个输入",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (2, [1, 2, 9], 3))
check("全接受：p[d]=1 时不消耗 uniform，只抽一次 bonus",
      (d.n_uniform, d.n_token) == (0, 1), f"uniform={d.n_uniform} token={d.n_token}")

# 首拒绝：p[d0] = 0 -> 必拒，且纠正分布就是原 p（挖掉的本来就是 0）
d = Draws(tokens=[2])
r = verify([1, 2], [probs(0.6, 0.0, 0.4, 0.0), probs(0, 1, 0, 0), probs(0, 0, 1, 0)], d)
check("首枚草稿被 top-k/top-p 过滤（p[d]=0）：必拒，纠正分布就是原 p",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (0, [2], 1))
check("p[d]=0 不消耗 uniform（抽了也是白抽），纠正只抽一次 token",
      (d.n_uniform, d.n_token) == (0, 1), f"uniform={d.n_uniform} token={d.n_token}")

# 部分接受：第一枚 p=1 接受，第二枚 p=0.5 且 u=0.9 >= 0.5 -> 拒，抽纠正
d = Draws(uniforms=[0.9], tokens=[3])
r = verify([1, 2], [probs(0, 1, 0, 0), probs(0.2, 0.5, 0.3, 0), probs(0, 0, 0, 1)], d)
check("部分接受：提交 [d0, 纠正]，保留 1+1 个输入",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (1, [1, 3], 2))
check("部分接受：消耗 1 个 uniform + 1 次纠正抽样", (d.n_uniform, d.n_token) == (1, 1))

# 同一位置 u < p -> 接受
d = Draws(uniforms=[0.1], tokens=[9])
r = verify([2], [probs(0.2, 0.5, 0.3, 0), probs(0, 0, 0, 1)], d)
check("u < p[d] 时接受这一枚", (r.num_accepted, r.committed_ids) == (1, [2, 9]))
check("接受这一枚后没有发生纠正抽样（token 只抽了 bonus）", (d.n_uniform, d.n_token) == (1, 1))

# 草稿本身是 EOS（走**必接受**分支看到它）。小词表测试用 token 3 当终止 token。
# 后面那枚草稿故意给 p=0.4：「先验证完整个前缀、再回头截断」的旧写法会为它抽一次
# uniform，而正确的写法在接受的当下就停——这里正好把两者分开（验收方复验第五十四关
# 时用这个形状抓出过漏子：文本对，随机流错）。
d = Draws(uniforms=[0.1], tokens=[9])
r = verify([3, 2], [probs(0, 0, 0, 1), probs(0.1, 0.2, 0.4, 0.3), probs(0, 0, 0, 1)], d, eos={3})
check("被接受的草稿本身是 EOS：只提交到它，num_accepted 不含它后面的草稿",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (1, [3], 1))
check("接受 EOS 后当场结束：不抽 bonus，也不为后面的草稿抽 uniform",
      (d.n_uniform, d.n_token) == (0, 0), f"uniform={d.n_uniform} token={d.n_token}")

# 第二枚草稿是 EOS：第一枚的 KV 要留
d = Draws(tokens=[9])
r = verify([1, 3], [probs(0, 1, 0, 0), probs(0, 0, 0, 1), probs(0, 0, 0, 1)], d, eos={3})
check("第二枚草稿是 EOS：留第一枚的 KV，后面的不抽不提交",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (2, [1, 3], 2) and d.n_token == 0)

# 全部接受、**bonus 是 EOS**：与「草稿是 EOS」不同，bonus 那次抽样照常发生；
# 它从没作为输入行进过模型，所以草稿的 KV 一个都不用退（1+K）
d = Draws(tokens=[3])
r = verify([1, 2], [probs(0, 1, 0, 0), probs(0, 0, 1, 0), probs(0, 0, 0, 1)], d, eos={3})
check("全部接受且 bonus 是 EOS：提交到 bonus 为止，草稿的 KV 都留（1+K）",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (2, [1, 2, 3], 3))
check("bonus 是 EOS 与草稿是 EOS 的区别：bonus 那次抽样照常发生",
      (d.n_uniform, d.n_token) == (0, 1), f"uniform={d.n_uniform} token={d.n_token}")

# 同一件事的另一条接受分支：中间那枚草稿靠 uniform **接受**上 EOS
d = Draws(uniforms=[0.1, 0.1], tokens=[9])
r = verify([1, 3, 2],
           [probs(0, 1, 0, 0), probs(0.2, 0.3, 0.0, 0.5), probs(0.1, 0.2, 0.4, 0.3),
            probs(0, 0, 0, 1)], d, eos={3})
check("uniform 接受分支碰上 EOS：同样当场结束，第三枚草稿不再验证",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (2, [1, 3], 2))
check("uniform 接受 EOS 后只花了那一次 uniform（第三枚那一次没有发生）",
      (d.n_uniform, d.n_token) == (1, 0), f"uniform={d.n_uniform} token={d.n_token}")

# 纠正 token 是 EOS：照常停下
d = Draws(uniforms=[0.9], tokens=[3])
r = verify([1], [probs(0.5, 0.5, 0, 0), probs(0, 0, 0, 1)], d, eos={3})
check("纠正 token 是 EOS：照常停下，保留 1 个输入",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (0, [3], 1))

# K=0：不抽接受随机数，只抽一次
d = Draws(tokens=[2])
r = verify([], [probs(0.2, 0.5, 0.3, 0)], d)
check("K=0：不抽不存在的接受随机数，只从那一行抽一枚",
      (r.num_accepted, r.committed_ids, r.kept_inputs) == (0, [2], 1)
      and (d.n_uniform, d.n_token) == (0, 1))

check("前置条件：K+1 > R 报错（禁止超预算后静默截断）",
      _raises(lambda: verify([1, 2], [probs(1, 0)] * 3, Draws(), remaining=1)) is not None)
check("行数与草稿数不匹配报错",
      _raises(lambda: verify([1, 2], [probs(1, 0)] * 2, Draws())) is not None)


def _raises(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)

# residual：挖掉草稿再归一化；无剩余质量时明确报错
res = residual_probs(probs(0.6, 0.3, 0.1), 1)
check("纠正分布：挖掉 d 再归一化（需求 §2 的例子 [6/7, 0, 1/7]）",
      res[1].item() == 0.0 and abs(res[0].item() - 6 / 7) < 1e-6
      and abs(res[2].item() - 1 / 7) < 1e-6, str([round(v, 4) for v in res.tolist()]))
check("纠正分布没有剩余质量时明确报错，不偷偷退回 argmax",
      _raises(lambda: residual_probs(probs(0.0, 1.0, 0.0), 1)) is not None)

# ------------------------------------------------ 2. 分布正确性

def empirical(draft_id, target, n, seed):
    """重复采一轮，统计**第一个提交的 token** 的经验分布。

    K=1 时要给两行分布（第 0 行验证草稿、第 1 行抽 bonus）；bonus 那行的取值不影响
    第一个 token —— 草稿被接受时第一个 token 就是草稿本身，被拒时是纠正 token。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    counts = [0] * len(target)
    for _ in range(n):
        r = verify_drafts_random([draft_id], [target, target], set(), 8,
                                 lambda: float(torch.rand((), generator=gen)),
                                 lambda pr: torch.multinomial(pr, 1, generator=gen))
        counts[r.committed_ids[0]] += 1
    return [c / n for c in counts]


N = 200000
target = probs(0.6, 0.3, 0.1)
got = empirical(1, target, N, seed=20240925)
sigma = [math.sqrt(p * (1 - p) / N) for p in (0.6, 0.3, 0.1)]
check("固定 p=[0.6,0.3,0.1]、草稿 1：经验分布收敛到 p（5σ 容差）",
      all(abs(got[i] - [0.6, 0.3, 0.1][i]) < 5 * sigma[i] for i in range(3)),
      str([round(v, 4) for v in got]))
wrong = [0.42, 0.51, 0.07]
check("经验分布明显偏离「拒绝后仍从原 p 重抽」的错误分布",
      all(abs(got[i] - wrong[i]) > 10 * sigma[i] for i in range(3)),
      f"实测 {[round(v, 4) for v in got]}  vs 错的 {wrong}")

# 两 token 的条件联合分布：第一枚草稿被接受时，第二枚来自**第二行**的分布；
# 第一枚被拒时本轮就结束了（首次拒绝即停），不会再有第二个 token。
# 这条同时抓两类问题：第一 token 的边缘分布不对（接受概率算错），
# 以及后续行用了错的历史（第二 token 的分布不是 row1）。
row0 = probs(0.5, 0.5, 0.0)
row1 = probs(0.25, 0.75, 0.0)
row2 = probs(0.2, 0.3, 0.5)
n2, seed2 = 120000, 7
gen = torch.Generator(device="cpu").manual_seed(seed2)
joint = {}
for _ in range(n2):
    r = verify_drafts_random([0], [row0, row1], set(), 8,
                             lambda: float(torch.rand((), generator=gen)),
                             lambda pr: torch.multinomial(pr, 1, generator=gen))
    joint[tuple(r.committed_ids)] = joint.get(tuple(r.committed_ids), 0) + 1
theory = {(1,): 0.5,                       # 首枚被拒（1-0.5）-> 纠正分布 [0,1,0] -> token 1
          (0, 0): 0.5 * 0.25,              # 接受 0 号草稿后，bonus 来自 row1
          (0, 1): 0.5 * 0.75}
worst = max(abs(joint.get(k, 0) / n2 - v) for k, v in theory.items())
tolerance = max(5 * math.sqrt(v * (1 - v) / n2) for v in theory.values())
check("K=1 的条件联合分布：接受则「草稿 + 第二行的 bonus」，被拒则本轮只有纠正 token",
      worst < tolerance,
      f"实测 { {k: round(v / n2, 4) for k, v in sorted(joint.items())} }，"
      f"理论 { {k: round(v, 4) for k, v in sorted(theory.items())} }")
check("被拒的轮次只提交一个 token（不会偷偷补第二个）",
      all(len(k) == 1 or len(k) == 2 for k in joint) and set(joint) == set(theory))

# 全部接受时第二个 token 来自**第二行**，而不是第一行
gen = torch.Generator(device="cpu").manual_seed(11)
n3 = 60000
second = {0: 0, 1: 0, 2: 0}
for _ in range(n3):
    r = verify_drafts_random([0], [probs(1, 0, 0), row1], set(), 8,
                             lambda: float(torch.rand((), generator=gen)),
                             lambda pr: torch.multinomial(pr, 1, generator=gen))
    second[r.committed_ids[1]] += 1
got2 = [second[i] / n3 for i in range(3)]
check("全部接受后 bonus 来自**第二行**的分布（不是第一行）",
      abs(got2[0] - 0.25) < 5 * math.sqrt(0.25 * 0.75 / n3)
      and abs(got2[1] - 0.75) < 5 * math.sqrt(0.25 * 0.75 / n3),
      str([round(v, 4) for v in got2]))

# ------------------------------------------------ 3. 惩罚与过滤

def state_for(**kw):
    params = SamplingParams(vocab_size=4, **kw)
    return SamplingState(params, [1, 2], torch.device("cpu")), params


def hist(state, counts):
    """一份只读的惩罚历史：`apply_penalties` / `distribution()` 只用到这两个属性。

    引擎里也是这么给每一行造临时历史的（临时计数不跟真实状态共享存储）。
    """
    return SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                           generated_counts=counts)


sampler = TorchSampler()

# top-k 把草稿过滤掉：分布里那个 token 的概率是 0 -> 必拒
st, params = state_for(temperature=1.0, top_k=2, seed=1)
row = probs(0.0, 0.5, 3.0, 1.0)          # 前 2 名是 token2、token3
dist = sampler.distribution(row, params, st)
check("top-k 把草稿过滤掉：该 token 的目标概率为 0 -> 必拒",
      dist[1].item() == 0.0 and abs(dist.sum().item() - 1.0) < 1e-6,
      str([round(v, 4) for v in dist.tolist()]))

# top-p 同理
st, params = state_for(temperature=1.0, top_p=0.5, seed=1)
row = probs(0.0, 0.0, 3.0, 0.1)
dist = sampler.distribution(row, params, st)
check("top-p 把草稿过滤掉：目标概率为 0", dist[1].item() == 0.0)

# 温度：低温度把分布推向 one-hot
st, params = state_for(temperature=0.01, seed=1)
row = probs(0.0, 1.0, 3.0, 2.0)
dist = sampler.distribution(row, params, st)
check("温度参与分布构造（0.01 时几乎 one-hot 在最大 logit 上）",
      dist[2].item() > 0.99, str([round(v, 4) for v in dist.tolist()]))

# 逐行历史：草稿重复时第二行的 frequency 惩罚与第一行不同
st, params = state_for(temperature=0.0, frequency_penalty=1.0)
draft = [2, 2]
temp = hist(st, dict(st.generated_counts))
rows = []
for j in range(len(draft) + 1):
    rows.append(sampler.distribution(probs(0.0, 0.0, 2.0, 2.0), params, temp))
    if j < len(draft):
        temp.generated_counts[draft[j]] = temp.generated_counts.get(draft[j], 0) + 1
# 参考：第一行没有被惩罚过（token2 与 token3 都是 2.0）-> 并列取小下标 = 2
#       第二行 token2 被 frequency 惩罚过（计数 1）-> 2.0-1.0 = 1.0 < 2.0 -> 取 3
#       第三行 token2 计数 2 -> 0.0，还是 3
check("逐行历史：重复草稿让第二行的 frequency 惩罚与第一行不同",
      [int(torch.argmax(r)) for r in rows] == [2, 3, 3],
      str([int(torch.argmax(r)) for r in rows]))
check("逐行惩罚与「普通单步采样」的参考实现逐行一致",
      all(torch.equal(rows[j], sampler.distribution(probs(0.0, 0.0, 2.0, 2.0), params,
                                                hist(st, counts)))
          for j, counts in enumerate([{}, {2: 1}, {2: 2}])))


# 贪心路径**一次 uniform 都不该抽**——引擎里那句 `draw_uniform` 就是写成
# 「被调用就抛 AssertionError」的。它成立的前提是：贪心分布是**精确**的 one-hot。
# 这里把这条前提与后果一起钉住（两边用的是同一份实现，交叉验证）。
greedy_params = SamplingParams(vocab_size=4, repetition_penalty=1.4, presence_penalty=0.5,
                               frequency_penalty=0.3)
greedy_state = SamplingState(greedy_params, [1, 2], torch.device("cpu"))
greedy_state.generated_counts = {2: 1}
not_onehot = []
for trial in range(2000):
    row = torch.randn(4) * (10 ** (trial % 4))
    dist = sampler.distribution(row, greedy_params,
                            hist(greedy_state, dict(greedy_state.generated_counts)))
    if not set(dist.tolist()) <= {0.0, 1.0}:
        not_onehot.append(dist.tolist())
check("贪心分布是**精确**的 one-hot（每一项只能是 0.0 或 1.0，2000 组随机行）",
      not not_onehot, str(not_onehot[:2]))


def exploding_uniform():
    raise AssertionError("贪心路径不该抽接受随机数")


# 这一行的 argmax：token2 被三惩罚压到 -0.8，token3 是 3.0 -> 草稿取 3（p[d]=1 -> 必接受）
greedy_rows = [sampler.distribution(probs(0.0, 0.0, 4.0, 3.0), greedy_params,
                                hist(greedy_state, {2: 1})) for _ in range(2)]
r = verify_drafts_random([3], greedy_rows, EOS, 8, exploding_uniform,
                         lambda pr: int(torch.argmax(pr)))
check("贪心分布喂进验证函数：一次都不会去抽 uniform（会抛的那个没被调用）",
      r.num_accepted == 1 and r.committed_ids == [3, 3],
      f"num_accepted={r.num_accepted} committed={r.committed_ids}")

# 反面对照：随机分布（非 one-hot）**必须**抽 uniform，否则接受判定就是假的
random_params = SamplingParams(vocab_size=4, temperature=0.8, seed=1)
random_state = SamplingState(random_params, [1, 2], torch.device("cpu"))
random_rows = [sampler.distribution(probs(0.0, 1.0, 3.0, 3.0), random_params, random_state)
               for _ in range(2)]
d = Draws(uniforms=[0.5], tokens=[2])
r = verify_drafts_random([1], random_rows, EOS, 8, d.uniform, d.token)
check("随机分布（非 one-hot）确实会抽 uniform——两个回调的选择是配套的",
      d.n_uniform == 1 and r.num_accepted + 1 == len(r.committed_ids))

# 贪心 + 惩罚：分布是 one-hot，且落在「施加惩罚之后」的 argmax 上
st, params = state_for(repetition_penalty=4.0)
temp = hist(st, {2: 1})
row = probs(0.0, 0.0, 4.0, 3.0)
dist = sampler.distribution(row, params, temp)
check("贪心 + 惩罚：one-hot 落在惩罚之后的 argmax 上（4.0/4 = 1.0 < 3.0）",
      dist[3].item() == 1.0 and dist[2].item() == 0.0,
      str([round(v, 4) for v in dist.tolist()]))

# 贪心分布与「普通路径」的参考一致：TorchSampler 的贪心就是惩罚后的 argmax。
# 两边喂**同一份**历史（token2 被压到 4/4=1.0 < 3.0，argmax 落在 token3）
st, params = state_for(repetition_penalty=4.0)
temp = hist(st, {2: 1})
row = probs(0.0, 0.0, 4.0, 3.0)
check("贪心分布与 TorchSampler 的贪心取到同一个 token（同一个 argmax）",
      int(torch.argmax(sampler.distribution(row, params, temp)))
      == int(sampler.select(row, params, temp)))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
