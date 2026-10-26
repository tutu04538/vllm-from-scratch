"""只下载 Step3.5 MTP 子集（约 5.9 GB），不拉 344 GB 全量权重。

用法：
    HF_ENDPOINT=https://hf-mirror.com python benchmarks/download_step3p5_mtp.py \
        --output models/Step-3.5-Flash-MTP

只取：
  - model.embed_tokens.weight（Step3p5MTP 自己的词嵌入）
  - 3 个 MTP 层（model.layers.45/46/47）的 enorm/hnorm/eh_proj/shared_head/block

不取 target 的 45 层、lm_head、vision/audio；文件用 HTTP Range 按 tensor 读取，
HF CDN 返回 206 才继续，服务器不支持 Range 就明确失败，不会静默退化成全量下载。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import requests

REPO = "stepfun-ai/Step-3.5-Flash"
REVISION = "ab446a3de5e171ea341227e24bb1f090e1b771f7"
INDEX = "model.safetensors.index.json"
CONFIG = "config.json"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "minivllm-step3p5-mtp-downloader"})
    return session


def _get(session, url, *, range_header=None, expected=None):
    headers = {} if range_header is None else {"Range": range_header}
    response = session.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    if range_header is not None and response.status_code != 206:
        raise RuntimeError(
            f"服务器不支持 Range（{url} 返回 {response.status_code}）；"
            f"拒绝退化成整文件下载")
    if expected is not None and int(response.headers.get("Content-Length", -1)) != expected:
        raise RuntimeError(
            f"{url} 的 Content-Length 不是 {expected}（Range 被忽略？）")
    return response


def _read_header(session, url):
    with _get(session, url, range_header="bytes=0-7", expected=8) as response:
        prefix = response.content
    header_len = struct.unpack("<Q", prefix)[0]
    with _get(session, url, range_header=f"bytes=0-{7 + header_len}",
              expected=8 + header_len) as response:
        raw = response.content
    return json.loads(raw[8:]), 8 + header_len


def _copy_range(session, url, start, end, out):
    length = end - start
    with _get(session, url, range_header=f"bytes={start}-{end - 1}",
              expected=length) as response:
        written = 0
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                out.write(chunk)
                written += len(chunk)
    if written != length:
        raise RuntimeError(f"tensor 字节数不符：期望 {length}，写入 {written}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/Step-3.5-Flash-MTP")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    endpoint = args.endpoint.rstrip("/")
    base = f"{endpoint}/{REPO}/resolve/{args.revision}"
    session = _session()
    with _get(session, f"{base}/{INDEX}") as response:
        index = response.json()
    weight_map = index["weight_map"]
    with _get(session, f"{base}/{CONFIG}") as response:
        config = response.json()
    start_layer = int(config["num_hidden_layers"])
    end_layer = start_layer + int(config["num_nextn_predict_layers"] or 1)

    selected = {}
    for name, shard in weight_map.items():
        spec = None
        prefix = f"model.layers.{start_layer}."
        for layer in range(start_layer, end_layer):
            if name.startswith(f"model.layers.{layer}."):
                spec = layer
                break
        if spec is not None:
            selected[name] = shard
        elif name == "model.embed_tokens.weight":
            selected[name] = shard
    if not selected:
        raise RuntimeError("没有选中任何 MTP 权重，检查 config/index")

    shards = {}
    for name, shard in selected.items():
        shards.setdefault(shard, []).append(name)

    total = 0
    headers = {}
    for shard in sorted(shards):
        header, data_start = _read_header(session, f"{base}/{shard}")
        headers[shard] = (header, data_start)
        for name in shards[shard]:
            if name not in header:
                raise KeyError(f"{shard} 的 safetensors header 里没有 {name}")
            begin, end = header[name]["data_offsets"]
            total += end - begin
    if args.dry_run:
        print(f"选中 {len(selected)} 个 tensor，{len(shards)} 个 shard，"
              f"{total / 1e9:.2f} GB（未下载）")
        for shard, names in sorted(shards.items()):
            print(f"  {shard}: {len(names)} tensors")
        return 0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG).write_text(json.dumps(config, indent=2) + "\n")

    entries = []
    offset = 0
    for shard in sorted(shards):
        header, _ = headers[shard]
        for name in sorted(shards[shard]):
            begin, end = header[name]["data_offsets"]
            size = end - begin
            entries.append((name, shard, begin, end, offset, size))
            offset += size
    output_header = {"__metadata__": {"format": "pt"}}
    for name, _shard, _begin, _end, out_offset, size in entries:
        source = headers[_shard][0][name]
        output_header[name] = {
            "dtype": source["dtype"],
            "shape": source["shape"],
            "data_offsets": [out_offset, out_offset + size],
        }
    header_bytes = json.dumps(output_header, separators=(",", ":")).encode()
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    output_file = out_dir / "model.safetensors"
    print(f"写入 {output_file}：{len(entries)} tensors，{offset / 1e9:.2f} GB")
    with output_file.open("wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for number, (name, shard, begin, end, _out_offset, _size) in enumerate(entries, 1):
            data_start = headers[shard][1]
            print(f"[{number}/{len(entries)}] {name} ({(end - begin) / 1e6:.1f} MB)")
            _copy_range(session, f"{base}/{shard}", data_start + begin,
                        data_start + end, out)
    print(f"完成：{output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
