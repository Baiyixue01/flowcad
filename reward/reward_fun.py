import os, re, math, tempfile, json
import numpy as np
from reward.pipeline import resolve_gt_paths, _load_prev_code_from_dir, safe_exec_from_path, _safe_get_cd_hd, _eval_pred_edges_from_blocks, _load_gt_edges_for_pid, _compute_cf_iou_metrics, geometry_valid
from reward.pipeline import *
from reward.utils.post_code_process import build_iso_code, build_integrated_code
# 你已有的函数（来自现有脚本/模块）
# - resolve_gt_paths(pid, GT_SINGLE_STEP_DIR) -> ( gt_single_step, gt_full_step)
# - _load_prev_code_from_dir(pid, PRE_CODE_DIR or COP_PRE_CODE_DIR) -> prev_code str
# - build_iso_code(prev_code, gen_code, iso_export_path, first_step) -> (single_code, info_shape)
# - build_integrated_code(prev_code, gen_code, full_export_path, first_step) -> (integrated_code, final_lhs)
# - safe_exec_from_path(py_path) -> (ok, loc, err)
# - _safe_get_cd_hd(pred_step_path, gt_step_path, angles=None) -> MetricsResult(cd, hd, best_euler_angle, ok, reason)
# - _eval_pred_edges_from_blocks(prev_code, gen_code) -> (pred_f, pred_c, parse_err)
# - _load_gt_edges_for_pid(GT_EDGES_DIR, pid) -> (gt_f, gt_c)
# - _compute_cf_iou_metrics(pred_f, pred_c, gt_f, gt_c) -> dict with cf_iou etc.
# - geometry_valid(shape_obj) -> (ok, info)

_PID_RE = re.compile(r"^\s*#\s*PID\s*:\s*(.+?)\s*$", re.IGNORECASE | re.M)
_OP_RE  = re.compile(r"^\s*#\s*OP\s*:\s*(.+?)\s*$",  re.IGNORECASE | re.M)

def _extract_pid_op(prompt: str):
    m1 = _PID_RE.search(prompt or "")
    m2 = _OP_RE.search(prompt or "")
    pid = (m1.group(1).strip() if m1 else None)
    op  = (m2.group(1).strip().lower() if m2 else "")
    return pid, op

def _to_list(x):
    """把 kwargs 里的 batch 字段尽量转成 list；标量返回单元素 list。"""
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        try:
            y = x.tolist()
            return y if isinstance(y, list) else [y]
        except Exception:
            pass
    return [x]

def _get_by_idx(batch_field, idx):
    if batch_field is None:
        return None
    if idx < len(batch_field):
        return batch_field[idx]
    return None

def _extract_code_block(text: str) -> str:
    """允许 ```python ...``` 或者纯代码。"""
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return (m.group(1) or "").strip()
    return text.strip()


def _safe_path_component(text: str) -> str:
    """将任意字符串转换成较安全的路径片段。"""
    s = (text or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] if s else "unknown"


def _build_unique_workdir(tmp_dir: str, pid: str, sample_idx: int) -> str:
    """
    为每个 completion 创建唯一目录，避免多 rank / 多 completion 并发覆盖。
    目录结构: TMP_DIR/grpo_reward/<pid>/<rank>_<sample>_<random>/
    """
    pid_part = _safe_path_component(pid)
    rank = os.environ.get("RANK", "0")
    base = os.path.join(tmp_dir, "grpo_reward", pid_part)
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"r{rank}_i{sample_idx:03d}_", dir=base)

def _format_ok(gen_code: str, op_kind: str = "") -> bool:
    """
    对齐 build_incremental_cq_prompt 的输出格式要求：
    - 非 chamfer/fillet: 必须有且仅有顺序正确的 #shape -> #bool，两段均非空，且 #bool 内有 result 赋值
    - chamfer/fillet: 必须包含 wp 重置、edge 选择、fillet/chamfer 调用、result 赋值
    """
    if not gen_code or not gen_code.strip():
        return False

    code = gen_code.strip()
    op = (op_kind or "").strip().lower()

    # 尽量避免“解释文字”混入；允许纯代码或从 ```python``` 提取后的代码
    # 这里不做非常激进的自然语言过滤，避免误杀合法注释/变量名。

    if op == "chamfer_fillet":
        # 输出格式要求中的固定 wp 模板（全局坐标系）
        wp_pat = re.compile(
            r"wp\s*=\s*cq\.Workplane\(\s*inPlane\s*=\s*Plane\(\s*origin\s*=\s*\(0\s*,\s*0\s*,\s*0\)\s*,\s*"
            r"normal\s*=\s*Vector\(\s*0\s*,\s*0\s*,\s*1\s*\)\s*,\s*xDir\s*=\s*Vector\(\s*1\s*,\s*0\s*,\s*0\s*\)\s*"
            r"\)\s*\)"
        )
        if not wp_pat.search(code):
            return False

        # 必须先选边，再在“同一个 edges 变量”上做 fillet/chamfer
        edge_sel_pat = re.compile(
            r"(?m)^\s*(edges(?:_\d+)?)\s*=\s*result(?:_\d+)?\s*\.\s*edges\([^)]*\)\s*$"
        )
        selections = list(edge_sel_pat.finditer(code))
        if not selections:
            return False

        op_calls = list(
            re.finditer(
                r"(?m)^\s*(?:shape(?:_\d+)?\s*=\s*)?(edges(?:_\d+)?)\s*\.\s*(fillet|chamfer)\s*\(",
                code,
            )
        )
        if not op_calls:
            return False

        selected_names = {m.group(1): m.start() for m in selections}
        valid_chain = False
        for m in op_calls:
            name = m.group(1)
            if name in selected_names and selected_names[name] < m.start():
                valid_chain = True
                break
        if not valid_chain:
            return False

        # 要求存在 result 最终赋值
        if not re.search(r"(?m)^\s*result\s*=\s*.+$", code):
            return False
        return True

    # 默认：shape-then-bool
    shape_tag = re.search(r"(?mi)^\s*#\s*shape\s*$", code)
    bool_tag = re.search(r"(?mi)^\s*#\s*bool\s*$", code)
    if not shape_tag or not bool_tag:
        return False
    if shape_tag.start() >= bool_tag.start():
        return False

    shape_block = code[shape_tag.end():bool_tag.start()].strip()
    bool_block = code[bool_tag.end():].strip()
    if not shape_block or not bool_block:
        return False

    # #bool 段必须有 result 赋值（与 prompt 的严格格式一致）
    if not re.search(r"(?m)^\s*result\s*=\s*.+$", bool_block):
        return False

    return True

def _tanh_improve(prev_val, now_val, sigma: float):
    """改变量奖励：越改善越正。"""
    if prev_val is None or now_val is None:
        return 0.0
    return float(math.tanh((prev_val - now_val) / max(sigma, 1e-9)))

def reward_fn(prompts, completions, **kwargs):
    """
    TRL/GRPO 会传入等长 prompts & completions（已展平）。
    你返回等长 reward list[float]。
    """
    rewards = []
    # print(f"Completions: {completions}")
    # 你可以根据你数据统计给一个尺度；先给保守默认
    SIGMA_CD = 0.02
    SIGMA_HD = 0.02

    group_indices = _to_list(kwargs.get("group_index"))
    ops = _to_list(kwargs.get("op"))

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        # 优先用数据列，避免从 prompt 正则解析
        pid = _get_by_idx(group_indices, i)
        op_kind = _get_by_idx(ops, i)

        if pid is not None:
            pid = str(pid).strip()
        if op_kind is not None:
            op_kind = str(op_kind).strip().lower()

        gen_code = _extract_code_block(completion)

        # ---------- 门控 ：格式 ----------
        if not _format_ok(gen_code, op_kind=op_kind):
            rewards.append(-2.0)
            continue

        # step0 判断
        m = re.search(r"step(\d+)", pid)
        step_num = int(m.group(1)) if m else -1
        first_step = (step_num == 0)

        # GT 路径（你现有逻辑：single + full）
        gt_single_step, gt_full_step = resolve_gt_paths(pid, GT_SINGLE_STEP_DIR)

        # 前序代码（训练里通常你会把 prev_code 嵌进 prompt，但这里直接按你的目录读也行）
        prev_code = _load_prev_code_from_dir(pid, COP_PRE_CODE_DIR if COP else PRE_CODE_DIR)

        # 为本条样本创建临时工作目录（避免不同样本互相覆盖）
        workdir = _build_unique_workdir(TMP_DIR, pid, i)
        single_step_path = os.path.join(workdir, "pred_single.step")
        full_step_path   = os.path.join(workdir, "pred_full.step")
        single_py        = os.path.join(workdir, "pred_single.py")
        full_py          = os.path.join(workdir, "pred_full.py")

        # ---------- 分支：Chamfer/Fillet ----------
        if op_kind == "chamfer_fillet":
            # 只做 full（与你评测一致）
            try:
                integrated_code, _ = build_integrated_code(prev_code, gen_code, full_step_path, first_step=first_step)
                with open(full_py, "w", encoding="utf-8") as f:
                    f.write(integrated_code)
            except Exception:
                rewards.append(-2.0)
                continue

            ok_full, _, err_full = safe_exec_from_path(full_py)
            if not ok_full or not os.path.exists(full_step_path):
                rewards.append(-2.0)  # 执行失败强罚
                continue

            # 边匹配 IoU（你已有）
            pred_f, pred_c, parse_err = _eval_pred_edges_from_blocks(prev_code, gen_code)
            gt_f, gt_c = _load_gt_edges_for_pid(GT_EDGES_DIR, pid)

            cfm = _compute_cf_iou_metrics(pred_f, pred_c, gt_f, gt_c)
            cf_iou = float(cfm.get("cf_iou", 0.0))

            # 组合奖励：可执行+IoU
            # 你也可以加“预测边数量”约束，防止全选刷分
            r = 1.0 + 2.0 * cf_iou
            rewards.append(r)
            continue

        # ---------- 普通几何操作：single + full ----------
        # 1) 生成 single/full 代码并落盘
        try:
            single_code, info_shape = build_iso_code(prev_code, gen_code, single_step_path, first_step=first_step)
            integrated_code, _ = build_integrated_code(prev_code, gen_code, full_step_path, first_step=first_step)
            with open(single_py, "w", encoding="utf-8") as f:
                f.write(single_code)
            with open(full_py, "w", encoding="utf-8") as f:
                f.write(integrated_code)
        except Exception:
            rewards.append(-2.0)
            continue

        # 2) 执行
        ok_single, _, err_single = safe_exec_from_path(single_py)
        ok_full,   _, err_full   = safe_exec_from_path(full_py)

        pred_single_exists = ok_single and os.path.exists(single_step_path)
        pred_full_exists   = ok_full   and os.path.exists(full_step_path)

        # ---------- 门控：至少 full 或 single 有一个成功 ----------
        if not (pred_single_exists or pred_full_exists):
            rewards.append(-2.0)
            continue

        # 3) 指标：single 对 gt_single；full 对 gt_full（与你评测一致）
        # single
        if pred_single_exists and gt_single_step:
            res_s = _safe_get_cd_hd(pred_step_path=single_step_path, gt_step_path=gt_single_step)
        else:
            res_s = None

        # full：step0 复用 single；step>=1 固定角度 [0]
        if pred_full_exists and gt_full_step:
            if first_step and res_s is not None and getattr(res_s, "ok", False):
                res_f = res_s
            else:
                res_f = _safe_get_cd_hd(pred_step_path=full_step_path, gt_step_path=gt_full_step, angles=[0])
        else:
            res_f = None

        # ---------- 奖励组装（门控 + 连续） ----------
        r = 0.0

        # a) 存活奖励：能执行就加分
        r += 0.1 * float(pred_single_exists) + 0.2 * float(pred_full_exists)

        # b) 几何有效性（可选，但很建议：防空 mesh / 退化）
        # 你目前 safe_exec_from_path 没返回 shape 对象，严格做几何 valid 需要你在 build_* 里把 shape 存成变量并读取；
        # 先用 “存在 step 文件” 作为弱几何门控即可。

        # c) CD/HD 奖励（用绝对值也能跑；更推荐“改变量”，但改变量需要 baseline）
        # 这里先做绝对：越小越好，用 exp/tanh 规约
        if res_s is not None and getattr(res_s, "ok", False) and res_s.cd is not None:
            r += 0.3 * float(math.exp(-5.0 * float(res_s.cd)))
        if res_f is not None and getattr(res_f, "ok", False) and res_f.cd is not None:
            # full 更重要
            r += 1.5 * float(math.exp(-5.0 * float(res_f.cd)))

        # d) 失败轻惩罚：能执行但没算出指标（比如空 mesh）
        if pred_full_exists and (res_f is None or not getattr(res_f, "ok", False)):
            r -= 0.5
        if pred_single_exists and (res_s is None or not getattr(res_s, "ok", False)):
            r -= 0.2

        # 兜底：避免 reward 全为正导致区分度不足
        rewards.append(float(r))
    print(f"Rewards: {rewards}")
    return rewards


if __name__ == "__main__":

    # 正确 shape-bool
    case1 = """
# shape
shape = cq.Workplane("XY").box(1,1,1)

# bool
result = shape
"""

    # 缺少 bool
    case2 = """
# shape
shape = cq.Workplane("XY").box(1,1,1)
"""

    # 顺序错误
    case3 = """
# bool
result = shape

# shape
shape = cq.Workplane("XY").box(1,1,1)
"""

    # 正确 chamfer/fillet
    case4 = """
wp = cq.Workplane(inPlane = Plane(origin = (0,0,0), normal = Vector(0,0,1), xDir = Vector(1,0,0)))

edges = result.edges("|Z")
shape = edges.fillet(0.5)

result = shape
"""

    # chamfer 缺少 edge 选择
    case5 = """
wp = cq.Workplane(inPlane = Plane(origin = (0,0,0), normal = Vector(0,0,1), xDir = Vector(1,0,0)))

shape = edges.fillet(0.5)

result = shape
"""

    tests = [
        ("valid shape-bool", case1, "", True),
        ("missing bool", case2, "", False),
        ("wrong order", case3, "", False),
        ("valid chamfer", case4, "chamfer_fillet", True),
        ("invalid chamfer", case5, "chamfer_fillet", False),
    ]

    for name, code, op, expected in tests:
        res = _format_ok(code, op)
        print(f"{name}: {res} (expected {expected})")
