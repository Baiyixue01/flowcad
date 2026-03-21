PYTHON := /data/baiyixue/miniforge3/envs/trl/bin/python
SCRIPT := /home/baiyixue/project/flowcad/train_stage1_rl.py
DEEPSPEED := deepspeed

TRAIN_JSONL := dataset/RL_stageI_step_level/step_rl_train_sampled.jsonl
EVAL_JSONL := dataset/RL_stageI_step_level/step_rl_val_sampled.jsonl

MODEL := /data/baiyixue/inference_model/Qwen2.5-Coder-0.5B-Instruct
OUTPUT_DIR := /data/baiyixue/inference_model/stage1_rl

PRE_CODE_DIR := data/pre_code
GT_SINGLE_STEP_DIR := /data/baiyixue/CAD/step_files
OP_ORIENT_DIR := /data/baiyixue/CAD/op_oriented_step
GT_EDGES_DIR := data/gt_edges_json
TMP_DIR := /home/baiyixue/project/flowcad/tmp_reward
DEDUP_CSV := data/dedup.csv
DS_CONFIG := configs/deepspeed/stage1_zero2.json


MAX_INPUT = 16384
MAX_OUPUT = 1024

# 训练参数
LR := 1e-6
BATCH := 1
GRAD_ACC := 2
GEN := 2

# =========================
# 默认目标（本地 reward）
# =========================
train:
	NCCL_P2P_DISABLE=1 \
	NCCL_SHM_DISABLE=1 \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	torchrun --standalone --nproc_per_node=8 \
	$(SCRIPT) \
	--train-jsonl $(TRAIN_JSONL) \
	--eval-jsonl $(EVAL_JSONL) \
	--model-name $(MODEL) \
	--output-dir $(OUTPUT_DIR) \
	--pre-code-dir $(PRE_CODE_DIR) \
	--gt-single-step-dir $(GT_SINGLE_STEP_DIR) \
	--op-orient-dir $(OP_ORIENT_DIR) \
	--gt-edges-dir $(GT_EDGES_DIR) \
	--dedup-csv $(DEDUP_CSV) \
	--tmp-dir $(TMP_DIR) \
	--learning-rate $(LR) \
	--per-device-train-batch-size $(BATCH) \
	--gradient-accumulation-steps $(GRAD_ACC) \
	--max-prompt-length $(MAX_INPUT) \
	--max-completion-length $(MAX_OUPUT) \
	--num-generations $(GEN) \
	--gradient-checkpointing \
	--fp16

# =========================
# 使用远程 reward server
# =========================
train-remote:
	$(PYTHON) $(SCRIPT) \
	--train-jsonl $(TRAIN_JSONL) \
	--eval-jsonl $(EVAL_JSONL) \
	--model-name $(MODEL) \
	--output-dir $(OUTPUT_DIR) \
	--pre-code-dir $(PRE_CODE_DIR) \
	--gt-single-step-dir $(GT_SINGLE_STEP_DIR) \
	--op-orient-dir $(OP_ORIENT_DIR) \
	--gt-edges-dir $(GT_EDGES_DIR) \
	--dedup-csv $(DEDUP_CSV) \
	--tmp-dir $(TMP_DIR) \
	--reward-server-url http://127.0.0.1:8005 \
	--learning-rate $(LR) \
	--per-device-train-batch-size $(BATCH) \
	--gradient-accumulation-steps $(GRAD_ACC) \
	--num-generations $(GEN) \
	--bf16

# =========================
# 使用 DeepSpeed 训练
# =========================
train-deepspeed:
	NCCL_P2P_DISABLE=1 \
	NCCL_SHM_DISABLE=1 \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	accelerate launch --config_file accelerate_ds.yaml \
	$(SCRIPT) \
	--train-jsonl $(TRAIN_JSONL) \
	--eval-jsonl $(EVAL_JSONL) \
	--model-name $(MODEL) \
	--output-dir $(OUTPUT_DIR)_ds \
	--pre-code-dir $(PRE_CODE_DIR) \
	--gt-single-step-dir $(GT_SINGLE_STEP_DIR) \
	--op-orient-dir $(OP_ORIENT_DIR) \
	--gt-edges-dir $(GT_EDGES_DIR) \
	--dedup-csv $(DEDUP_CSV) \
	--tmp-dir $(TMP_DIR) \
	--learning-rate $(LR) \
	--per-device-train-batch-size $(BATCH) \
	--gradient-accumulation-steps $(GRAD_ACC) \
	--max-prompt-length $(MAX_INPUT) \
	--max-completion-length $(MAX_OUPUT) \
	--num-generations $(GEN) \
	--gradient-checkpointing \
	--bf16 \
	--deepspeed-config $(DS_CONFIG)

vllm-serve:
		NCCL_P2P_DISABLE=1 \
		NCCL_SHM_DISABLE=1 \
		CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
		--model $(MODEL) \
		--tensor-parallel-size 2 \
		--port 8001

# =========================
# Debug（小步跑）
# =========================
debug:
	$(PYTHON) $(SCRIPT) \
	--train-jsonl $(TRAIN_JSONL) \
	--model-name $(MODEL) \
	--output-dir $(OUTPUT_DIR)_debug \
	--pre-code-dir $(PRE_CODE_DIR) \
	--gt-single-step-dir $(GT_SINGLE_STEP_DIR) \
	--op-orient-dir $(OP_ORIENT_DIR) \
	--gt-edges-dir $(GT_EDGES_DIR) \
	--dedup-csv $(DEDUP_CSV) \
	--tmp-dir $(TMP_DIR) \
	--max-steps 10 \
	--num-generations 2 \
	--logging-steps 1

# =========================
# 清理
# =========================
clean:
	rm -rf $(OUTPUT_DIR)
