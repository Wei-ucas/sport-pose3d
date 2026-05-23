#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_image2ground.py  — 验证 image2ground 函数的正确性

验证方案（往返一致性测试）：
  1. 在地面平面（Z=0）铺一张均匀网格，作为"真值"世界坐标
  2. 用 camera_project 正向投影到各视角图像坐标
  3. 用 image2ground  反投影回世界坐标
  4. 计算误差 ||world_gt − world_recovered||（单位：米）
  5. 在视频第一帧上绘制网格，并用颜色编码误差大小
  6. 输出每个视角的误差统计，保存结果图

Usage:
  /data/wangwei/conda-envs/basket/bin/python scripts/verify_image2ground.py \
      --workdir "work-dirs/严九九(YJJ),女,11.23测" \
      --game_id seg-001 \
      --view_list 00 01 02 03 04 05 06 07 \
      --court work-dirs/court.json \
      --grid_step 0.5
"""

import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.camera import load_camera_params, camera_project, image2ground
from src.utils.constant import load_court, CourtConfig, VolleyballCourt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_ground_grid(court: CourtConfig, step: float = 0.5) -> np.ndarray:
    """生成地面平面（Z=0）的均匀网格点，坐标范围覆盖 court（含 margin）。

    Returns:
        grid: (N, 3), float64, Z 全为 0
    """
    m = court.court_margin
    xs = np.arange(-m, court.court_width + m + 1e-6, step)
    ys = np.arange(-m, court.court_height + m + 1e-6, step)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)
    grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float64)
    return grid


def error_to_color(err: float, max_err: float = 0.05) -> tuple:
    """将误差值映射为 BGR 颜色：蓝(0) → 绿(mid) → 红(max)。"""
    t = min(err / max_err, 1.0)
    if t < 0.5:
        s = t * 2
        return (int((1 - s) * 255), int(s * 255), 0)  # 蓝→绿
    else:
        s = (t - 0.5) * 2
        return (0, int((1 - s) * 255), int(s * 255))  # 绿→红


def _safe_pt(xy: np.ndarray) -> tuple:
    return (int(round(float(xy[0]))), int(round(float(xy[1]))))


def process_view(
    view_id: str,
    cam_idx: int,
    camera_params: dict,
    video_dir: str,
    video_type: str,
    court: CourtConfig,
    grid_step: float,
    output_dir: str,
    error_scale: float,
) -> None:
    """处理单个视角：测试往返误差，绘图并保存。"""

    # ---- 视频第一帧 ----
    frame = None
    for ext in [video_type, video_type.lower(), video_type.upper()]:
        vpath = os.path.join(video_dir, f"{view_id}{ext}")
        if os.path.exists(vpath):
            cap = cv2.VideoCapture(vpath)
            ret, frame = cap.read()
            cap.release()
            if ret:
                break
    if frame is None:
        print(f"  [{view_id}] 找不到视频，跳过。")
        return

    H_img, W_img = frame.shape[:2]

    # ---- 单视角相机字典 ----
    cam = {
        "K": camera_params["K"][cam_idx],
        "dist": camera_params["dist"][cam_idx],
        "R": camera_params["R"][cam_idx],
        "T": camera_params["T"][cam_idx],
        "RT": camera_params["RT"][cam_idx],
        "invK": camera_params["invK"][cam_idx],
    }

    # ---- 地面网格 ----
    grid_world = build_ground_grid(court, step=grid_step)  # (N, 3)

    # ---- 正向投影：世界 → 图像 ----
    pts_img = camera_project(grid_world, cam)  # (N, 2)

    # 过滤掉投影到图像外的点（防止干扰误差统计）
    in_frame = (
        (pts_img[:, 0] >= 0)
        & (pts_img[:, 0] < W_img)
        & (pts_img[:, 1] >= 0)
        & (pts_img[:, 1] < H_img)
    )
    grid_world_vis = grid_world[in_frame]
    pts_img_vis = pts_img[in_frame]

    # ---- 反向投影：图像 → 世界（Z=0 约束）----
    pts_img_2d = pts_img_vis[:, :2].astype(np.float32)
    recovered = image2ground(pts_img_2d, cam)  # (N, 2)

    # ---- 误差（米）----
    errors = np.linalg.norm(grid_world_vis[:, :2] - recovered, axis=1)

    max_e = errors.max()
    mean_e = errors.mean()
    p90_e = np.percentile(errors, 90)

    print(
        f"  [{view_id}]  点数={len(errors):4d}  "
        f"mean={mean_e*100:.2f}cm  max={max_e*100:.2f}cm  P90={p90_e*100:.2f}cm"
    )

    # ---- 绘图 ----
    vis = frame.copy()

    for i in range(len(pts_img_vis)):
        pt = _safe_pt(pts_img_vis[i])
        err = errors[i]
        color = error_to_color(err, max_err=error_scale)

        cv2.circle(vis, pt, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, pt, 4, color, -1, cv2.LINE_AA)

        # 误差文本（仅当误差较大时显示，避免杂乱）
        if err > error_scale * 0.3:
            cv2.putText(
                vis,
                f"{err*100:.1f}cm",
                (pt[0] + 5, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"{err*100:.1f}cm",
                (pt[0] + 5, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    # 图例色条
    _draw_colorbar(vis, error_scale)

    # 统计信息
    info = (
        f"View {view_id} | mean={mean_e*100:.2f}cm  "
        f"max={max_e*100:.2f}cm  P90={p90_e*100:.2f}cm  "
        f"(grid step={grid_step}m)"
    )
    cv2.putText(
        vis, info, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        vis,
        info,
        (16, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    save_path = os.path.join(output_dir, f"{view_id}_verify.jpg")
    cv2.imwrite(save_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"         → {save_path}")

    # ---- 附加：数值结果保存为 txt ----
    txt_path = os.path.join(output_dir, f"{view_id}_errors.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("world_x  world_y  proj_u  proj_v  rec_x   rec_y   err_m\n")
        for i in range(len(pts_img_vis)):
            f.write(
                f"{grid_world_vis[i,0]:.3f}  {grid_world_vis[i,1]:.3f}  "
                f"{pts_img_vis[i,0]:.1f}  {pts_img_vis[i,1]:.1f}  "
                f"{recovered[i,0]:.4f}  {recovered[i,1]:.4f}  "
                f"{errors[i]:.6f}\n"
            )


def _draw_colorbar(img: np.ndarray, max_err: float) -> None:
    """在图像右下角绘制简单色条。"""
    h, w = img.shape[:2]
    bar_h, bar_w = 120, 16
    margin = 10
    x0 = w - bar_w - margin - 40
    y0 = h - bar_h - margin - 20

    for i in range(bar_h):
        t = 1.0 - i / bar_h
        color = error_to_color(t * max_err, max_err)
        cv2.rectangle(img, (x0, y0 + i), (x0 + bar_w, y0 + i + 1), color, -1)

    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    fw = 1

    def _lbl(txt, y):
        cv2.putText(
            img, txt, (x0 + bar_w + 3, y), font, fs, (0, 0, 0), fw + 1, cv2.LINE_AA
        )
        cv2.putText(
            img, txt, (x0 + bar_w + 3, y), font, fs, (255, 255, 255), fw, cv2.LINE_AA
        )

    _lbl(f"{max_err*100:.0f}cm", y0 + 8)
    _lbl(f"{max_err*50:.0f}cm", y0 + bar_h // 2 + 4)
    _lbl("0cm", y0 + bar_h + 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="验证 image2ground 往返一致性：地面网格点 → 投影 → 反投影，测量误差。"
    )
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--game_id", type=str, required=True)
    parser.add_argument("--view_list", nargs="+", required=True)
    parser.add_argument("--video_type", type=str, default=".MP4")
    parser.add_argument("--court", type=str, default="work-dirs/court.json")
    parser.add_argument(
        "--grid_step", type=float, default=0.5, help="地面网格间距（米），默认 0.5"
    )
    parser.add_argument(
        "--error_scale",
        type=float,
        default=0.05,
        help="色条最大误差（米），默认 0.05 (5cm)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    court = load_court(args.court)
    print(
        f"球场: {court.court_width}m × {court.court_height}m, "
        f"顶高={court.court_top_height}m, margin={court.court_margin}m"
    )
    print(f"网格间距: {args.grid_step}m, 色条上限: {args.error_scale*100:.0f}cm\n")

    camera_dir = os.path.join(args.workdir, "prepare", "camera", args.game_id)
    intri_path = os.path.join(camera_dir, "intri.yml")
    extri_path = os.path.join(camera_dir, "extri.yml")
    for p in (intri_path, extri_path):
        if not os.path.exists(p):
            sys.exit(f"[ERROR] 找不到: {p}")

    camera_params = load_camera_params(intri_path, extri_path, args.view_list)

    video_dir = os.path.join(args.workdir, "videos", args.game_id)
    output_dir = args.output_dir or os.path.join(
        args.workdir, "verify_image2ground", args.game_id
    )
    os.makedirs(output_dir, exist_ok=True)

    for i, view_id in enumerate(args.view_list):
        process_view(
            view_id=view_id,
            cam_idx=i,
            camera_params=camera_params,
            video_dir=video_dir,
            video_type=args.video_type,
            court=court,
            grid_step=args.grid_step,
            output_dir=output_dir,
            error_scale=args.error_scale,
        )

    print(f"\n完成。结果目录: {output_dir}")


if __name__ == "__main__":
    main()
