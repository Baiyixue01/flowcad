#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 Step-level RL JSONL 中随机抽取固定数量样本。

示例：
python dataset/sample_step_rl_jsonl.py \


import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def _read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"JSONL 解析失败: line={i}, err={e}") from e
    return rows


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sample_random(rows, k: int, rng: random.Random):
    if k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def _sample_stratified_op(rows, k: int, rng: random.Random):
    """
    按 op 分层抽样，尽量保持原始 op 分布。
    不足部分按全局余量补齐，最终返回固定 k 条（k <= len(rows)）。
    """
    by_op = defaultdict(list)
    for r in rows:
        op = str(r.get("op", "")).strip().lower()
        by_op[op].append(r)

    total = len(rows)
    selected = []
    remainders = []  # (fraction, op)

    # 先按比例分配 floor 配额
    for op, group in by_op.items():
        exact = k * (len(group) / total)
        base = int(exact)
        remainders.append((exact - base, op))
        if base > 0:
            selected.extend(rng.sample(group, min(base, len(group))))

    # 若不足 k，按余数从大到小补齐
    need = k - len(selected)
    if need > 0:
        chosen_ids = {id(x) for x in selected}
        remainders.sort(reverse=True, key=lambda x: x[0])

        for _, op in remainders:
            if need <= 0:
                break
            candidates = [x for x in by_op[op] if id(x) not in chosen_ids]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            selected.append(pick)
            chosen_ids.add(id(pick))
            need -= 1

    # 如果仍不足（极端分布），全局补齐
    if len(selected) < k:
        chosen_ids = {id(x) for x in selected}
        leftovers = [x for x in rows if id(x) not in chosen_ids]
        if leftovers:
            selected.extend(rng.sample(leftovers, k - len(selected)))

    return selected[:k]


def main():
    parser = argparse.ArgumentParser(description="Sample fixed-size subset from Step-level RL JSONL")
    parser.add_argument("--input", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--num", type=int, required=True, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--strategy",
        choices=["random", "stratified_op"],
        default="random",
        help="抽样策略：random=纯随机；stratified_op=按 op 分层",
    )
    args = parser.parse_args()

    if args.num <= 0:
        raise ValueError("--num 必须 > 0")

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = _read_jsonl(in_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"输入为空: {in_path}")

    k = min(args.num, n)
    if args.num > n:
        print(f"[WARN] 请求抽样数 {args.num} > 数据量 {n}，将输出全部样本。")

    rng = random.Random(args.seed)
    if args.strategy == "random":
        sampled = _sample_random(rows, k, rng)
    else:
        sampled = _sample_stratified_op(rows, k, rng)

    _write_jsonl(out_path, sampled)
    print(f"[OK] input={in_path} total={n}")
    print(f"[OK] sampled={len(sampled)} strategy={args.strategy} seed={args.seed}")
    print(f"[OK] output={out_path}")


if __name__ == "__main__":
    main()

