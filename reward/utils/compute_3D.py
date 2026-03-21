#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import cadquery as cq
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ---------------------- 基础工具 ----------------------
def step_to_stl(step_path: str, stl_path: str) -> None:
    """把 STEP 用 CadQuery 导出为 STL（三角网）"""
    shape = cq.importers.importStep(step_path)  # 可能是 Compound
    cq.exporters.export(shape, stl_path)


def load_mesh_as_points(mesh_path: str, num_points: int = 8192) -> np.ndarray:
    """读取三角网，并进行表面均匀采样为点云坐标（N, 3）"""
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"Empty mesh: {mesh_path}")

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(points, dtype=np.float64)


def sample_and_normalize_from_step(step_path: str, num_points: int = 8192) -> np.ndarray:
    """STEP → STL → 点云，并做中心化+单位球缩放（独立归一化）"""
    with tempfile.TemporaryDirectory() as td:
        stl_path = os.path.join(td, "tmp.stl")
        step_to_stl(step_path, stl_path)
        pts = load_mesh_as_points(stl_path, num_points=num_points)

    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def _rotation_matrix_xyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """按 XYZ 欧拉角（单位：度）生成旋转矩阵，等价于 Rx @ Ry @ Rz。"""
    xr, yr, zr = np.radians([x_deg, y_deg, z_deg])

    cx, sx = np.cos(xr), np.sin(xr)
    cy, sy = np.cos(yr), np.sin(yr)
    cz, sz = np.cos(zr), np.sin(zr)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return rx @ ry @ rz


def _rotate_points(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return points @ rot.T


def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    t1 = cKDTree(points1)
    t2 = cKDTree(points2)
    d1, _ = t1.query(points2)
    d2, _ = t2.query(points1)
    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


def hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    t1 = cKDTree(points1)
    t2 = cKDTree(points2)
    d1, _ = t1.query(points2)
    d2, _ = t2.query(points1)
    return float(max(np.max(d1), np.max(d2)))


# ---------------------- 评估（旋转枚举） ----------------------
def compare_step_chamfer_with_rotation_only(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 8192,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis",
):
    """只做欧拉角枚举（X/Y/Z 轴）、不做 ICP。"""
    if save_vis:
        # 统一保留参数以兼容旧调用，但不再执行可视化。
        _ = vis_prefix

    src = sample_and_normalize_from_step(step_path_1, num_points=num_points)
    tgt = sample_and_normalize_from_step(step_path_2, num_points=num_points)

    best_cd = float("inf")
    best_score = float("inf")
    best_align = None
    best_angles = (0, 0, 0)

    for x in angles:
        for y in angles:
            for z in angles:
                rot = _rotation_matrix_xyz(x, y, z)
                rotated = _rotate_points(src, rot)

                cd = chamfer_distance(rotated, tgt)
                hd = hausdorff_distance(rotated, tgt)
                score = cd + hd

                if score < best_score:
                    best_score = score
                    best_cd = cd
                    best_align = rotated
                    best_angles = (x, y, z)

    best_hd = hausdorff_distance(best_align, tgt)
    return best_cd, best_hd, best_angles


def compare_step_chamfer_with_icp_rotation(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 8192,
    angles: list[int] = (0, 90, 180, 270),
    max_corr: float = 0.01,
    save_vis: bool = False,
    vis_prefix: str = "vis",
):
    """兼容旧接口：当前实现已去除 open3d/ICP，退化为纯旋转枚举。"""
    _ = max_corr, save_vis, vis_prefix
    return compare_step_chamfer_with_rotation_only(
        step_path_1=step_path_1,
        step_path_2=step_path_2,
        num_points=num_points,
        angles=angles,
        save_vis=False,
    )


@dataclass
class MetricsResult:
    cd: Optional[float]
    hd: Optional[float]
    best_euler_angle: Optional[tuple[int, int, int]]
    ok: bool
    reason: str = ""


def get_cd_hd(
    pred_step_path: str,
    gt_step_path: str,
    num_points: int = 8192,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis",
):
    """计算 CD/HD，去除 open3d 依赖，默认禁用可视化。"""
    cd, hd, best_angles = compare_step_chamfer_with_rotation_only(
        step_path_1=pred_step_path,
        step_path_2=gt_step_path,
        num_points=num_points,
        angles=angles,
        save_vis=save_vis,
        vis_prefix=vis_prefix,
    )
    return MetricsResult(cd=cd, hd=hd, best_euler_angle=best_angles, ok=True)
