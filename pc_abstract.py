import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import traceback

# 复用你现有函数（来自 compute_3D.py）
from reward.utils.compute_3D import sample_from_step  # :contentReference[oaicite:1]{index=1}

# ======================
# 配置
# ======================
NUM_POINTS = 8192   # ⚠️ 和你原来完全一致
NPROC = 24          # 根据CPU调整

# ======================
# 单个文件处理
# ======================
def process_one(args):
    step_path, out_path = args

    try:
        print(f"Processing: {step_path}")
        # STEP → 点云（不做 normalize）
        pts = sample_from_step(step_path, num_points=NUM_POINTS)

        # 保存 float32（节省一半空间）
        np.save(out_path, pts.astype(np.float32))

        return True, step_path

    except Exception as e:
        return False, f"{step_path} | {e}"


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

    # -------- 多进程处理 --------
    success = 0
    fail = 0

    with mp.Pool(NPROC) as pool:
        for ok, msg in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            if ok:
                success += 1
            else:
                fail += 1
                print("[ERROR]", msg)

    print(f"\nDone: {success} success, {fail} failed")


# ======================
# 运行入口
# ======================
if __name__ == "__main__":
    GT_STEP_DIR = "/data/baiyixue/CAD/step_files_sketch"
    OUT_DIR = "/data/baiyixue/CAD/step_files_pc"

    build_gt_pointclouds(GT_STEP_DIR, OUT_DIR)