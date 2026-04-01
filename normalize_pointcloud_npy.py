import os
import multiprocessing as mp

import numpy as np
from tqdm import tqdm


NUM_POINTS = 2048
NPROC = 24


def normalize_pointcloud(points: np.ndarray, num_points: int = 2048) -> np.ndarray:
    """对现有点云做重采样 + 中心化 + 单位球归一化。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise RuntimeError(f"Invalid point cloud shape: {pts.shape}")

    if len(pts) != num_points:
        replace = len(pts) < num_points
        indices = np.random.choice(len(pts), size=num_points, replace=replace)
        pts = pts[indices]

    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def process_one(args):
    in_path, out_path = args
    try:
        pts = np.load(in_path)
        norm_pts = normalize_pointcloud(pts, num_points=NUM_POINTS)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, norm_pts.astype(np.float32))
        return True, in_path
    except Exception as e:
        return False, f"{in_path} | {e}"


def build_normalized_npy_tree(in_root_dir: str, out_root_dir: str):
    """对已存在的 .npy 点云目录整树归一化，保持相对路径不变。"""
    tasks = []
    for root, _, files in os.walk(in_root_dir):
        for f in files:
            if not f.lower().endswith(".npy"):
                continue
            in_path = os.path.join(root, f)
            rel = os.path.relpath(in_path, in_root_dir)
            out_path = os.path.join(out_root_dir, rel)
            if os.path.exists(out_path):
                continue
            tasks.append((in_path, out_path))

    print(f"[INFO] Total NPY files to process: {len(tasks)}")

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


if __name__ == "__main__":
    IN_ROOT_DIR = "/data/baiyixue/CAD/step_files_pc_2048"
    OUT_ROOT_DIR = "/data/baiyixue/CAD/step_files_pc_2048_normalized"

    build_normalized_npy_tree(IN_ROOT_DIR, OUT_ROOT_DIR)
