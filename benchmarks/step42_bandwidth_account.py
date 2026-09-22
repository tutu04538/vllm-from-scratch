"""第四十二关的带宽账：decode 一步的投影 GEMM 到底跑了多少带宽。

    python benchmarks/step42_bandwidth_account.py

两件事必须做对，否则账是假的：

1. **每个形状用互不相同的权重缓冲。** 本机 L2 有 64 MB，而单个权重只有 4~12 MB；
   重复重放同一个 GEMM 会让权重留在 L2 里，测到的是**含 L2 命中的有效带宽**。
   注意：有效带宽本来就可以高于外部显存带宽，两者是不同口径，不是「物理上不可能」。
2. **用 CUDA Graph 计时**，否则测到的是 CPU 发射开销（每个 GEMM 约 40 µs，
   而纯 GPU 时间只有 12~21 µs）。

参考带宽由 measure_copy_bandwidth() 实测（拷贝张量，计读+写）。它代表一种特定访问
模式，不是所有 kernel 的统一上限；本文件报出的一律是**有效带宽**，不是外部显存流量。

本文件只覆盖 28 层的四种投影，不含 lm_head；attention / norm / RoPE 没有带宽账。
"""

import torch, statistics
d, hq, hkv, hd, inter, L, M = 1024, 16, 8, 128, 3072, 28, 8
shapes = [("qkv", (hq+2*hkv)*hd, d), ("o", d, hq*hd), ("gate_up", 2*inter, d), ("down", d, inter)]
# 28 层各一套互不相同的权重，共 840 MB —— 一次遍历远超 64 MB L2
Ws = [[torch.empty(of, inf, dtype=torch.bfloat16, device="cuda").normal_()
       for _, of, inf in shapes] for _ in range(L)]
xs = [torch.empty(M, inf, dtype=torch.bfloat16, device="cuda").normal_() for _, _, inf in shapes]
total = sum(w.numel()*2 for layer in Ws for w in layer)
print(f"一次遍历的权重总量: {total/2**20:.0f} MB（L2 只有 64 MB）")

def one_pass():
    for layer in Ws:
        for (name, of, inf), w, x in zip(shapes, layer, xs):
            x @ w.T

for _ in range(3): one_pass()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): one_pass()
for _ in range(3): g.replay()
torch.cuda.synchronize()
ts = []
for _ in range(15):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    ts.append(a.elapsed_time(b))
med = statistics.median(ts)
print(f"一次完整遍历（= decode_c8 一步的投影）: 中位 {med:.2f} ms  "
      f"区间 [{min(ts):.2f}, {max(ts):.2f}]")
print(f"  -> 有效带宽（含 L2 命中可能，非外部显存流量） {total/(med/1000)/1e9:.0f} GB/s")
print(f"  参考：本机 copy_ 256/2048 MiB 实测约 524 / 741 GB/s（见 measure_copy_bandwidth）")
print(f"  -> 相当于 copy_ 2048 MiB 参考值（741 GB/s）的 {total/(med/1000)/1e9/741*100:.0f}%")
# 逐形状：同样用互不相同的缓冲，单独测每一类（也远超 L2）
print("\n逐形状（每类 28 个不同权重缓冲，合计 112~336 MB，同样超 L2）:")
for si, (name, of, inf) in enumerate(shapes):
    ws = [layer[si] for layer in Ws]
    x = xs[si]
    def one_shape():
        for w in ws: x @ w.T
    for _ in range(3): one_shape()
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2): one_shape()
    for _ in range(3): g2.replay()
    torch.cuda.synchronize()
    t2 = []
    for _ in range(15):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g2.replay(); b.record(); torch.cuda.synchronize()
        t2.append(a.elapsed_time(b))
    m2 = statistics.median(t2); nb = ws[0].numel()*2*L
    print(f"    {name:<8} {nb/2**20:6.0f} MB  {m2:6.2f} ms  {nb/(m2/1000)/1e9:6.0f} GB/s")
