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
  --gt-image-dir /path/to/gt_image \
  --gt-single-step-dir /path/to/gt_single_step \
  --op-orient-dir /path/to/op_orientated_step \
  --gt-edges-dir /path/to/gt_edges_json \
  --pre-code-dir /path/to/pre_code \
  --tmp-dir /path/to/tmp_reward
"""

import argparse
import os
from typing import Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
import requests

import evaluation.pipeline as pl
import evaluation.reward_fun as rf


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
    parser.add_argument("--gt-image-dir", required=True, help="GT 图像目录")
    parser.add_argument("--gt-single-step-dir", required=True, help="GT 单步 STEP 目录")
    parser.add_argument("--op-orient-dir", required=True, help="GT 累计形状（full）STEP 目录")
    parser.add_argument("--gt-edges-dir", required=True, help="GT 边标签目录")
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
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)

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
    pl.GT_IMAGE_DIR = args.gt_image_dir
    pl.GT_SINGLE_STEP_DIR = args.gt_single_step_dir
    pl.OP_ORIENT_DIR = args.op_orient_dir
    pl.GT_EDGES_DIR = args.gt_edges_dir
    pl.TMP_DIR = args.tmp_dir
    pl.COP = cop_flag

    # reward_fun 内直接读取这些同名全局变量
    rf.PRE_CODE_DIR = args.pre_code_dir
    rf.COP_PRE_CODE_DIR = cop_pre
    rf.GT_IMAGE_DIR = args.gt_image_dir
    rf.GT_SINGLE_STEP_DIR = args.gt_single_step_dir
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
    eval_ds = None
    if args.eval_jsonl:
        eval_ds = load_dataset("json", data_files=args.eval_jsonl, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        report_to="none",
    )

    reward_func = rf.reward_fn
    if args.reward_server_url:
        reward_func = build_remote_reward_fn(
            base_url=args.reward_server_url,
            timeout=args.reward_timeout,
            max_retries=args.reward_max_retries,
        )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_func,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
