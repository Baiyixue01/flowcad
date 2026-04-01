import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import signal

# 复用你现有函数（来自 compute_3D.py）
from reward.utils.compute_3D import sample_from_step  # :contentReference[oaicite:1]{index=1}

# ======================
# 配置
# ======================
NUM_POINTS = 2048   # ⚠️ 和你原来完全一致
NPROC = 24          # 根据CPU调整
TIMEOUT_SECONDS = 180


class SampleTimeoutError(TimeoutError):
    pass


def _handle_timeout(signum, frame):
    raise SampleTimeoutError(f"sampling exceeded {TIMEOUT_SECONDS}s")

# ======================
# 单个文件处理
# ======================
def process_one(args):
    step_path, out_path = args

    try:
        print(f"Processing: {step_path}")
        start_time = time.time()

        # 给单个样本的抽点过程设置硬超时，超过 3 分钟直接跳过。
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(TIMEOUT_SECONDS)

        # STEP → 点云（不做 normalize）
        pts = sample_from_step(step_path, num_points=NUM_POINTS)
        elapsed = time.time() - start_time
        signal.alarm(0)

        # 保存 float32（节省一半空间）
        np.save(out_path, pts.astype(np.float32))

        return "success", f"{step_path} | {elapsed:.2f}s"

    except SampleTimeoutError:
        signal.alarm(0)
        return "timeout", step_path

    except Exception as e:
        signal.alarm(0)
        return "error", f"{step_path} | {e}"


# ======================
# 主函数
# ======================
def build_gt_pointclouds(
    step_root_dir: str,
    out_root_dir: str,
    suffix=".step"
):
    """
    step_root_dir: 你的 GT step 根目录
    out_root_dir:  输出点云目录（建议新建一个，如 gt_points）
    """

    tasks = []

    # -------- 遍历所有 STEP --------
    for root, _, files in os.walk(step_root_dir):
        for f in files:
            if not f.lower().endswith(suffix):
                continue

            step_path = os.path.join(root, f)

            # 构建输出路径（保持结构）
            rel = os.path.relpath(step_path, step_root_dir)
            out_path = os.path.join(out_root_dir, rel.replace(suffix, ".npy"))

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 跳过已存在
            if os.path.exists(out_path):
                continue

            tasks.append((step_path, out_path))

    print(f"[INFO] Total STEP files: {len(tasks)}")

    timeout_log_path = os.path.join(out_root_dir, "skipped_timeout_samples.txt")
    error_log_path = os.path.join(out_root_dir, "failed_samples.txt")

    # -------- 多进程处理 --------
    success = 0
    fail = 0
    timeout = 0

    timeout_records = []
    error_records = []

    with mp.Pool(NPROC) as pool:
        for status, msg in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            if status == "success":
                success += 1
            elif status == "timeout":
                timeout += 1
                timeout_records.append(msg)
                print("[TIMEOUT]", msg)
            else:
                fail += 1
                error_records.append(msg)
                print("[ERROR]", msg)

    if timeout_records:
        with open(timeout_log_path, "w", encoding="utf-8") as f:
            for item in timeout_records:
                f.write(item + "\n")

    if error_records:
        with open(error_log_path, "w", encoding="utf-8") as f:
            for item in error_records:
                f.write(item + "\n")

    print(f"\nDone: {success} success, {timeout} timeout-skipped, {fail} failed")
    print(f"[INFO] Timeout log: {timeout_log_path}")
    print(f"[INFO] Error log: {error_log_path}")


# ======================
# 运行入口
# ======================
if __name__ == "__main__":
    GT_STEP_DIR = "/data/baiyixue/CAD/op_oriented_step_sketch"
    OUT_DIR = "/data/baiyixue/CAD/op_oriented_step_pc_2048"

    build_gt_pointclouds(GT_STEP_DIR, OUT_DIR)
