#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 split_result.json + prompt.csv + {pre_code, gt_code} 组合为 Step-level RL 训练数据 JSONL。
"""

import os
import json
import csv
import argparse
from textwrap import dedent
from pathlib import Path

import pandas as pd

# ========== 默认路径（可用命令行覆盖） ==========
SPLIT_JSON = "/home/baiyixue/project/op-cad/data/split_result.json"
PROMPT_CSV = "/home/baiyixue/project/op-cad/data/prompt.csv"
PRE_CODE_DIRS = [
    "/home/baiyixue/project/op-cad/data/pre_code"
]
GT_CODE_DIR  = "/home/baiyixue/project/op-cad/data/gt_code"
OUT_DIR      = "/home/baiyixue/project/op-cad/others/train_data/sft_cop_full"

# ========== 工具函数 ==========
def build_incremental_cq_prompt(
    previous_code: str,
    operation_instruction: str,
    op_kind: str = None,
) -> str:
    """
    构建“增量式 CadQuery 代码生成”Prompt。
    - link_mode=None       ：不强制如何命名结果变量，仅提供温和建议
    - link_mode="inplace"  ：建议基于最近变量继续，并覆盖同名变量（非强制）
    - link_mode="append_new": 建议写入 next_var_name（非强制；未给则自动 current+1）
    """
    # ---------------- 识别操作类型 ----------------
    opk = (op_kind or "").lower()
    is_modify = any(k in opk for k in ["fillet", "chamfer"])  # chamfer / fillet / chamfer_fillet
    # ------------------------------------------------

    # 4) 硬性要求（不含变量命名强制）
    hard_reqs = [
"Output **only** the new CadQuery code snippet (no extra text).",
"If previous code exists: do not recreate the base model or redefine previous variables unnecessarily.",
"The code must be directly executable when appended to the context.",
"Do not include comments or explanations.",
]

    hard_reqs_block = "\n".join([f"{i+1}. {r}" for i, r in enumerate(hard_reqs)])
    
    shape_bool_rules = dedent(f"""
    ### Shape-then-Bool Rules (ENFORCED)
    - You MUST output exactly two sections in this exact order: #shape then #bool. Under #shape, include only shape construction code (including plane/workplane definition, sketch creation, and 3D feature generation). Under #bool, include only the final boolean operation statement (exactly one of union or cut) that combines the current solid with shape.
    - In **#shape**: Build the required feature(s) as **independent solid(s)** without referencing **previous results**; if multiple bodies are created, **union** them into a single solid and assign to **shape**.
    - In **#bool**: Apply **only** one of **union** or **cut** between the current solid and **shape**.
    """).strip()

    plane_usage_rules = dedent("""
    ### Workplane and Face Selection Rules (MANDATORY)
    - **Before modeling operation, you must explicitly define a new workplane to ensure geometric consistency.**
    - **NEVER** use `.faces()` or `.face()` to select faces or workplanes.
    - Always construct workplanes **explicitly** using `Plane` and `Vector`.
    Example:
    ```python
    wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    ```
    """).strip()


    modify_rules = dedent("""
    ### Edge-Selection and Application Rules (MANDATORY for fillet/chamfer)
    - Before applying any fillet or chamfer, you MUST reset the workplane coordinate system exactly as follows (no modifications allowed):
        wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    - You **MUST** split the operation into two distinct steps using sequential variable names (e.g., `edges_1`, `edges_2`...).
        - **Step 1: Select Edges.**
        Select edges from the *current* result (e.g., `result_0`) and assign the **selection workplane** to a new variable (e.g., `edges_1`).
        - **Step 2: Apply Operation.**
        Call `.fillet()` or `.chamfer()` on the **selected edges** (e.g., `edges_1`) and assign the **modified shape** to another new variable (e.g., `shape_1`).
    - Finally, if there are multiple modified shapes, combine all modified shapes using intersection to form the final result, if there is only a single modified shape, assign it directly to `result`.
    Example:
    ```python
    wp = cq.Workplane(inPlane=Plane(origin=(0, 0, 0), normal=Vector(0, 0, 1), xDir=Vector(1, 0, 0)))
    edges_1 = result_0.edges(cq.NearestToPointSelector((x,y,z)))
    shape_1 = edges_1.fillet(fillet_radius)
    edges_2 = result_1.edges(cq.NearestToPointSelector((x,y,z)))
    shape_2 = edges_2.chamfer(chamfer_distance_1, chamfer_distance_2)
    result = reduce(lambda a, b: a.intersect(b), [shape_1, shape_2])
    ```
    """)


# ---- 组装 Prompt ----
    sections = []
    sections.append("### Role\nYou are an expert CAD modeling assistant specialized in CadQuery.\nGenerate ONLY the incremental CadQuery code needed to perform the requested operation, as a continuation of the provided previous code context.")
    sections.append(f"""### Context (already executed Python code)
```python
{previous_code if previous_code.strip() else '# No previous code — this is the first modeling step.'}
```""")
    sections.append(f"""### Instruction
Perform the following operation **as a continuation** of the existing model:
> {operation_instruction}
""")
    if not is_modify:
        sections.append(plane_usage_rules)  
    sections.append("### Hard Requirements\n" + hard_reqs_block)

    if is_modify:
        sections.append(modify_rules)
        sections.append(dedent(f"""
### Output Format (STRICT)
```python
edge_1 = result_m.edges{{edge_selection}}
shape_1 = edge_1.{{operation}}
edge_2 = result_m.edges{{edge_selection}}
shape_2 = edge_2.{{operation}}
...
result = reduce(lambda a, b: a.intersect(b), [shape_1, shape_2, ...])(if multiple shapes)
```
""").strip())
    else:
        sections.append(shape_bool_rules)
        sections.append(dedent(f"""
### Output Format (STRICT)
```python
#shape
{{generated_shape_code}}
#bool
{{generated_bool_code}}
```
""").strip())
    return dedent("\n\n".join(sections)).strip()

def id_to_filename(sample_id: str) -> str:
    """
    "00002_index_1/step1" -> "00002_index_1_step1.py"
    """
    return sample_id.replace("/", "_") + ".py"

def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def wrap_code_py(s: str) -> str:
    s = s.strip()
    # 去掉已有围栏，避免重复
    if s.startswith("```"):
        return s
    return f"```python\n{s}\n```"

def load_prompt_map(csv_path: str):
    """
    读取 prompt.csv，期望至少有:
      - group_index（形如 00002_index_1/step1）
      - prompt_text 或 instruction
      - op（用于 op_kind；可选但强烈建议）
    """
    df = pd.read_csv(csv_path)

    # key
    key_col = None
    for cand in ["group_index", "id", "sample_id"]:
        if cand in df.columns:
            key_col = cand
            break
    if key_col is None:
        raise ValueError("prompt.csv 中未找到标识列（group_index / id / sample_id）")

    # prompt text
    val_col = None
    for cand in ["prompt_text", "instruction", "prompt"]:
        if cand in df.columns:
            val_col = cand
            break
    if val_col is None:
        raise ValueError("prompt.csv 中未找到指令列（prompt_text / instruction / prompt）")

    prompt_map = dict(zip(df[key_col].astype(str), df[val_col].astype(str)))

    # op_kind
    op_col = None
    for cand in ["op_kind", "op", "operation"]:
        if cand in df.columns:
            op_col = cand
            break
    op_map = dict(zip(df[key_col].astype(str), df[op_col].astype(str))) if op_col else {}

    return prompt_map, op_map, key_col, val_col, op_col


from typing import Optional

def find_pre_code_path(sample_id: str) -> Optional[Path]:
    """
    依次在 PRE_CODE_DIRS 里查找
    """
    fname = id_to_filename(sample_id)
    for d in PRE_CODE_DIRS:
        p = Path(d) / fname
        if p.exists():
            return p
    return None

def build_split(items, prompt_map, op_map, out_path: Path, missing_rows: list):
    def is_step0(sid: str) -> bool:
        return sid.endswith("/step0") or sid.endswith("_step0")

    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for sid in items:
            op_instr = prompt_map.get(sid)
            if not op_instr:
                missing_rows.append([sid, "missing_prompt", "", ""])
                continue

            op_kind = op_map.get(sid)  # 可能为空

            pre_code_path = find_pre_code_path(sid)
            gt_code_path = Path(GT_CODE_DIR) / id_to_filename(sid)
            if not gt_code_path.exists():
                missing_rows.append([sid, "missing_gt_code", str(pre_code_path) if pre_code_path else "", ""])
                continue

            try:
                gt_code = read_text(gt_code_path)
            except Exception as e:
                missing_rows.append([sid, f"read_error_gt:{e}", str(pre_code_path) if pre_code_path else "", str(gt_code_path)])
                continue

            # step0 允许无 pre_code；其他 step 必须有
            if pre_code_path is None:
                if is_step0(sid):
                    pre_code = ""  # builder 会写 "No previous code..."
                else:
                    missing_rows.append([sid, "missing_pre_code", "", str(gt_code_path)])
                    continue
            else:
                try:
                    pre_code = read_text(pre_code_path)
                except Exception as e:
                    missing_rows.append([sid, f"read_error_pre:{e}", str(pre_code_path), str(gt_code_path)])
                    continue

            # === 用你刚写的 prompt builder 生成完整 prompt ===
            full_prompt = build_incremental_cq_prompt(
                previous_code=pre_code,
                operation_instruction=op_instr,
                op_kind=op_kind,
            )

            # Step-level RL/GRPO 样本：
            # - prompt: 供模型生成 completion
            # - group_index/op: 供 reward_fn 通过 kwargs 读取
            rec = {
                "prompt": full_prompt,
                "group_index": sid,
                "op": str(op_kind).strip().lower() if op_kind is not None else "",
                "prompt_text": op_instr,
                "previous_code": pre_code,
                "gt_code": gt_code.strip(),  # 可选保留，便于离线分析/回放
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    return n_ok



# ========== 主流程 ==========
def main():

    global PRE_CODE_DIRS, GT_CODE_DIR
    parser = argparse.ArgumentParser(description="Build Step-level RL JSONL from split + prompts + codes")
    parser.add_argument("--split_json", default=SPLIT_JSON)
    parser.add_argument("--prompt_csv", default=PROMPT_CSV)
    parser.add_argument("--pre_code_dirs", nargs="*", default=PRE_CODE_DIRS)
    parser.add_argument("--gt_code_dir", default=GT_CODE_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()


    PRE_CODE_DIRS = args.pre_code_dirs
    GT_CODE_DIR   = args.gt_code_dir

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 split
    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_ids = [str(x) for x in split.get("train", [])]
    val_ids   = [str(x) for x in split.get("val", [])]

    # 读取 prompt
    prompt_map, op_map, key_col, val_col, op_col = load_prompt_map(args.prompt_csv)

    # 生成
    missing_rows = []  # [sample_id, reason, pre_path, gt_path]

    train_path = out_dir / "step_rl_train.jsonl"
    val_path = out_dir / "step_rl_val.jsonl"

    n_train = build_split(train_ids, prompt_map, op_map, train_path, missing_rows)
    n_val   = build_split(val_ids,   prompt_map, op_map, val_path,   missing_rows)

    # 写缺失报表
    rep_path = out_dir / "missing_report.csv"
    with open(rep_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "reason", "pre_code_path", "gt_code_path"])
        w.writerows(missing_rows)

    print(f"[OK] Train written: {n_train} -> {train_path}")
    print(f"[OK] Val written:   {n_val} -> {val_path}")
    print(f"[INFO] Missing/Errors: {len(missing_rows)} -> {rep_path}")

if __name__ == "__main__":
    main()
