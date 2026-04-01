#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-1 Step-level RL 训练脚本（基于 TRL GRPOTrainer）。

数据输入（JSONL）建议字段：
- prompt
- group_index
- op

可选字段：
- prompt_text
- previous_code
- gt_code

示例：
python train_stage1_rl.py \
  --train-jsonl /path/to/step_rl_train.jsonl \
  --eval-jsonl /path/to/step_rl_val.jsonl \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir /path/to/outputs/stage1_rl \
  --gt-single-step-dir /path/to/gt_single_step \
  --gt-single-pc-dir /path/to/gt_single_pc \
  --op-orient-dir /path/to/op_orientated_step \
  --gt-full-pc-dir /path/to/gt_full_pc \
  --gt-edges-dir /path/to/gt_edges_json \
  --pre-code-dir /path/to/pre_code \
  --tmp-dir /path/to/tmp_reward
"""

import reward.pipeline as pl
import reward.reward_fun as rf

import argparse
import os
from typing import Any

from datasets import load_dataset
from transformers import set_seed
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer
import requests


GT_SINGLE_PATH_KEYS = (
    "gt_single_path",
    "gt_single_pc_path",
    "gt_single_step_path",
)
GT_FULL_PATH_KEYS = (
    "gt_full_path",
    "gt_full_pc_path",
    "gt_full_step_path",
)


def _normalize_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "nan":
            return None
        return value
    return None


def _pick_first_path(example: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        path = _normalize_optional_path(example.get(key))
        if path:
            return path
    return None


def _infer_file_kind(path: str | None) -> str:
    if not path:
        return "missing"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return "pointcloud"
    if ext in {".step", ".stp"}:
        return "step"
    return ext.lstrip(".") or "unknown"


def enrich_dataset_with_gt_paths(dataset, dataset_name: str):
    """
    为 reward 显式补齐 GT 文件路径。
    优先复用 JSONL 中已有路径列；若缺失则根据 group_index + 全局目录解析。
    """

    stats = {
        "single_pointcloud": 0,
        "single_step": 0,
        "single_missing": 0,
        "full_pointcloud": 0,
        "full_step": 0,
        "full_missing": 0,
    }

    def _add_gt_paths(example: dict[str, Any]) -> dict[str, Any]:
        gt_single_path = _pick_first_path(example, GT_SINGLE_PATH_KEYS)
        gt_full_path = _pick_first_path(example, GT_FULL_PATH_KEYS)

        if (not gt_single_path or not gt_full_path) and example.get("group_index"):
            resolved_single, resolved_full = pl.resolve_gt_paths(str(example["group_index"]).strip(), pl.GT_SINGLE_STEP_DIR)
            gt_single_path = gt_single_path or resolved_single
            gt_full_path = gt_full_path or resolved_full

        single_kind = _infer_file_kind(gt_single_path)
        full_kind = _infer_file_kind(gt_full_path)

        stats[f"single_{single_kind}" if f"single_{single_kind}" in stats else "single_missing"] += 1
        stats[f"full_{full_kind}" if f"full_{full_kind}" in stats else "full_missing"] += 1

        return {
            "gt_single_path": gt_single_path or "",
            "gt_full_path": gt_full_path or "",
            "gt_single_kind": single_kind,
            "gt_full_kind": full_kind,
        }

    dataset = dataset.map(_add_gt_paths, desc=f"resolve_gt_paths[{dataset_name}]")
    print(
        f"[{dataset_name}] GT sources: "
        f"single(pointcloud={stats['single_pointcloud']}, step={stats['single_step']}, missing={stats['single_missing']}), "
        f"full(pointcloud={stats['full_pointcloud']}, step={stats['full_step']}, missing={stats['full_missing']})"
    )
    return dataset




def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-1 Step-level RL model with GRPO.")

    # 数据/模型
    parser.add_argument("--train-jsonl", required=True, help="step_rl_train.jsonl 路径")
    parser.add_argument("--eval-jsonl", default=None, help="step_rl_val.jsonl 路径（可选）")
    parser.add_argument("--model-name", required=True, help="基础模型或SFT模型路径")
    parser.add_argument("--output-dir", required=True, help="训练输出目录")

    # reward 相关路径
    parser.add_argument("--pre-code-dir", required=True, help="前序代码目录（std 模式）")
    parser.add_argument("--cop-pre-code-dir", default="", help="前序代码目录（cop 模式，可选）")
    parser.add_argument("--gt-single-step-dir", required=True, help="GT 单步 STEP 目录")
    parser.add_argument("--gt-single-pc-dir", default=None, help="可选：GT 单步 NPY 点云目录")
    parser.add_argument("--op-orient-dir", required=True, help="GT 累计形状（full）STEP 目录")
    parser.add_argument("--gt-full-pc-dir", default=None, help="可选：GT 累计形状（full）NPY 点云目录")
    parser.add_argument("--gt-edges-dir", required=True, help="GT 边标签目录")
    parser.add_argument("--dedup-csv", required=True, help="去重映射 CSV（必填）")
    parser.add_argument("--tmp-dir", required=True, help="reward 临时目录")
    parser.add_argument("--mode", choices=["std", "cop"], default="std", help="与 reward 读取 pre_code 的模式")
    parser.add_argument(
        "--reward-server-url",
        default="",
        help="可选：远程 reward 服务地址（例如 http://127.0.0.1:8005）。设置后训练会通过 HTTP 获取 reward。",
    )
    parser.add_argument(
        "--reward-timeout",
        type=float,
        default=120.0,
        help="远程 reward 请求超时（秒）。",
    )
    parser.add_argument(
        "--reward-max-retries",
        type=int,
        default=2,
        help="远程 reward 请求最大重试次数（失败后返回惩罚分）。",
    )

    # 训练超参
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed-config",
        default="",
        help="DeepSpeed 配置文件路径（JSON，可选）。设置后将传给 GRPOConfig.deepspeed。",
    )

    return parser.parse_args()


def configure_reward_env(args):
    """
    将路径配置注入 pipeline/reward_fun，确保 reward_fn 可用。
    """
    cop_pre = args.cop_pre_code_dir if args.cop_pre_code_dir else args.pre_code_dir
    cop_flag = args.mode == "cop"

    # pipeline 内函数（如 resolve_gt_paths）依赖这些全局变量
    pl.PRE_CODE_DIR = args.pre_code_dir
    pl.COP_PRE_CODE_DIR = cop_pre
    pl.GT_SINGLE_STEP_DIR = args.gt_single_step_dir
    pl.GT_SINGLE_PC_DIR = args.gt_single_pc_dir
    pl.GT_FULL_PC_DIR = args.gt_full_pc_dir
    pl.OP_ORIENT_DIR = args.op_orient_dir
    pl.GT_EDGES_DIR = args.gt_edges_dir
    pl.DEDUP_CSV = args.dedup_csv
    pl.TMP_DIR = args.tmp_dir
    pl.COP = cop_flag

    # reward_fun 内直接读取这些同名全局变量
    rf.PRE_CODE_DIR = args.pre_code_dir
    rf.COP_PRE_CODE_DIR = cop_pre
    rf.GT_SINGLE_STEP_DIR = args.gt_single_step_dir
    rf.GT_SINGLE_PC_DIR = args.gt_single_pc_dir
    rf.GT_FULL_PC_DIR = args.gt_full_pc_dir
    rf.GT_EDGES_DIR = args.gt_edges_dir
    rf.TMP_DIR = args.tmp_dir
    rf.COP = cop_flag

    os.makedirs(args.tmp_dir, exist_ok=True)


def build_remote_reward_fn(base_url: str, timeout: float, max_retries: int):
    """
    返回一个与 TRL/GRPO 接口兼容的 reward 函数：
    - 输入: prompts/completions/kwargs(来自 dataset 列)
    - 输出: list[float]
    """

    url = f"{base_url.rstrip('/')}/reward"
    session = requests.Session()

    def _remote_reward_fn(prompts, completions, **kwargs):
        batch_size = len(completions)
        metas: list[dict[str, Any]] = []

        for i in range(batch_size):
            meta_i = {}
            for key, val in kwargs.items():
                if isinstance(val, (list, tuple)):
                    if i < len(val):
                        meta_i[key] = val[i]
                else:
                    meta_i[key] = val
            metas.append(meta_i)

        payload = {
            "prompts": list(prompts),
            "completions": list(completions),
            "metas": metas,
        }

        last_err = None
        for _ in range(max(1, max_retries + 1)):
            try:
                resp = session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                rewards = data.get("rewards", [])
                ok = data.get("ok", True)
                if ok and isinstance(rewards, list) and len(rewards) == batch_size:
                    return [float(r) for r in rewards]
                last_err = RuntimeError(f"Invalid reward response: {data}")
            except Exception as e:
                last_err = e

        print(f"[reward-server] request failed, fallback to -2.0: {last_err}")
        return [-2.0] * batch_size

    return _remote_reward_fn


def main():
    args = parse_args()
    set_seed(args.seed)
    configure_reward_env(args)

    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    train_ds = enrich_dataset_with_gt_paths(train_ds, "train")
    eval_ds = None
    if args.eval_jsonl:
        eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")
        eval_ds = enrich_dataset_with_gt_paths(eval_ds, "eval")

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     trust_remote_code=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name,
    #     trust_remote_code=True,
    # )

    # 🔥 LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # # 🔥 挂 LoRA
    # model = get_peft_model(model, lora_config)

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # model.config.pad_token_id = tokenizer.pad_token_id
    # if hasattr(model, "generation_config") and model.generation_config is not None:
    #     model.generation_config.pad_token_id = tokenizer.pad_token_id
    #     model.generation_config.eos_token_id = tokenizer.eos_token_id
    #     model.generation_config.bos_token_id = tokenizer.bos_token_id

    deepspeed_config = args.deepspeed_config if args.deepspeed_config else None

    # 配置 GRPO 训练参数
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=deepspeed_config,
        report_to="none",
        # use_vllm=True,
        # vllm_mode="server",
        # vllm_server_base_url="http://127.0.0.1:8001",
    )


    # 配置 reward 函数
    reward_func = rf.reward_fn
    if args.reward_server_url:
        reward_func = build_remote_reward_fn(
            base_url=args.reward_server_url,
            timeout=args.reward_timeout,
            max_retries=args.reward_max_retries,
        )
    # 初始化 GRPOTrainer
    trainer = GRPOTrainer(
        model=args.model_name,
        args=grpo_args,
        reward_funcs=reward_func,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 开始训练
    trainable = 0
    total = 0
    for n, p in trainer.model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"trainable params: {trainable}")
    print(f"total params: {total}")
    print(f"trainable ratio: {100 * trainable / total:.6f}%")
    trainer.train()
    trainer.save_model(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
