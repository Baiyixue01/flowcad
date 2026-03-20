PYTHON := /data/baiyixue/miniforge3/envs/trl/bin/python
SCRIPT := /home/baiyixue/project/flowcad/train_stage1_rl.py

# 用脚本默认参数
run:
	$(PYTHON) $(SCRIPT)

TRAIN_JSONL := dataset/RL_stageI_step_level/step_rl_train_sampled.jsonl
EVAL_JSONL := dataset/RL_stageI_step_level/step_rl_val_sampled.jsonl

MODEL := /data/baiyixue/inference_model/Llama-3.1-8B_ar_coop_sketch_sft_full
OUTPUT_DIR := /data/baiyixue/inference_model/stage1_rl

PRE_CODE_DIR := /home/baiyixue/project/op-cad/data/pre_code_scale_short
GT_SINGLE_STEP_DIR := /data/baiyixue/CAD/step_files
OP_ORIENT_DIR := /data/baiyixue/CAD/op_oriented_step
GT_EDGES_DIR := /home/baiyixue/project/op-cad/data/gt_edges_json_scale
TMP_DIR := /home/baiyixue/project/flowcad/tmp_reward

# 训练参数
LR := 1e-6
BATCH := 1
GRAD_ACC := 8
GEN := 4

# =========================
# 默认目标（本地 reward）
# =========================
train:
	$(PYTHON) $(SCRIPT) \
	--train-jsonl $(TRAIN_JSONL) \
	--eval-jsonl $(EVAL_JSONL) \
	--model-name $(MODEL) \
	--output-dir $(OUTPUT_DIR) \
	--pre-code-dir $(PRE_CODE_DIR) \
	--gt-single-step-dir $(GT_SINGLE_STEP_DIR) \
	--op-orient-dir $(OP_ORIENT_DIR) \
	--gt-edges-dir $(GT_EDGES_DIR) \
	--tmp-dir $(TMP_DIR) \
	--learning-rate $(LR) \
	--per-device-train-batch-size $(BATCH) \
	--gradient-accumulation-steps $(GRAD_ACC) \
	--num-generations $(GEN) \
	--bf16

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
	--tmp-dir $(TMP_DIR) \
	--reward-server-url http://127.0.0.1:8005 \
	--learning-rate $(LR) \
	--per-device-train-batch-size $(BATCH) \
	--gradient-accumulation-steps $(GRAD_ACC) \
	--num-generations $(GEN) \
	--bf16

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
	--tmp-dir $(TMP_DIR) \
	--max-steps 10 \
	--num-generations 2 \
	--logging-steps 1

# =========================
# 清理
# =========================
clean:
	rm -rf $(OUTPUT_DIR)