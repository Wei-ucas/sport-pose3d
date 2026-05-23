#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_camera.py - 相机参数可视化检查工具

在各视角视频第一帧上绘制：
  1. 世界坐标系原点及三坐标轴  (X=红, Y=绿, Z=蓝)
  2. 以 court 参数为边界的立方体线框  (青色)

Usage example:
  python scripts/check_camera.py \
      --workdir "work-dirs/严九九(YJJ),女,11.23测" \
      --game_id seg-001 \
      --view_list 00 01 02 03 04 05 06 07 \
      --court work-dirs/court.json
"""

import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.camera import load_camera_params, camera_project
from src.utils.constant import load_court, CourtConfig, VolleyballCourt

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _safe_pt(pt_2d: np.ndarray) -> tuple:
    """将 float 2-D 点转为 int tuple，防止 cv2 崩溃。"""
    return (int(round(float(pt_2d[0]))), int(round(float(pt_2d[1]))))


def draw_axis(img: np.ndarray, cam: dict, axis_len: float = 1.0, thickness: int = 3):
    """在图像上绘制世界坐标系三轴 (原点 = 世界坐标系原点)。

    X 轴 = 红色, Y 轴 = 绿色, Z 轴 = 蓝色 (BGR 约定)。
    """
    origin = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )
    pts_2d = camera_project(origin, cam)

    o = _safe_pt(pts_2d[0])
    px = _safe_pt(pts_2d[1])
    py = _safe_pt(pts_2d[2])
    pz = _safe_pt(pts_2d[3])

    tip_len = 0.25
    cv2.arrowedLine(img, o, px, (0, 0, 255), thickness, cv2.LINE_AA, tipLength=tip_len)
    cv2.arrowedLine(img, o, py, (0, 255, 0), thickness, cv2.LINE_AA, tipLength=tip_len)
    cv2.arrowedLine(img, o, pz, (255, 0, 0), thickness, cv2.LINE_AA, tipLength=tip_len)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7
    fw = 2
    for pt, label, color in [
        (px, "X", (0, 0, 255)),
        (py, "Y", (0, 255, 0)),
        (pz, "Z", (255, 0, 0)),
    ]:
        cv2.putText(
            img, label, (pt[0] + 6, pt[1] + 6), font, fs, (0, 0, 0), fw + 2, cv2.LINE_AA
        )
        cv2.putText(
            img, label, (pt[0] + 6, pt[1] + 6), font, fs, color, fw, cv2.LINE_AA
        )

    # 原点圆
    cv2.circle(img, o, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, o, 7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_court_box(
    img: np.ndarray,
    cam: dict,
    court: CourtConfig,
    color=(0, 255, 255),
    thickness: int = 2,
):
    """绘制以 court 参数为边界的立方体线框。

    下底面使用带 margin 的 court_ground_corners；
    上底面使用相同的 x/y 范围，z = court_top_height。
    """
    m = court.court_margin
    w = court.court_width
    h = court.court_height
    ht = court.court_top_height

    # 底面四角 (含 margin, z=0)
    bot = np.array(
        [
            [-m, -m, 0.0],
            [-m, h + m, 0.0],
            [w + m, h + m, 0.0],
            [w + m, -m, 0.0],
        ],
        dtype=np.float64,
    )

    # 顶面四角 (含 margin, z=court_top_height)
    top = bot.copy()
    top[:, 2] = ht

    all_pts = np.vstack([bot, top])  # (8, 3)
    pts_2d = camera_project(all_pts, cam)  # (8, 2)

    bot_2d = [_safe_pt(pts_2d[i]) for i in range(4)]
    top_2d = [_safe_pt(pts_2d[i + 4]) for i in range(4)]

    # 底面
    for i in range(4):
        cv2.line(img, bot_2d[i], bot_2d[(i + 1) % 4], color, thickness, cv2.LINE_AA)
    # 顶面
    for i in range(4):
        cv2.line(img, top_2d[i], top_2d[(i + 1) % 4], color, thickness, cv2.LINE_AA)
    # 垂直棱
    for i in range(4):
        cv2.line(img, bot_2d[i], top_2d[i], color, thickness, cv2.LINE_AA)

    # 标注顶部高度
    label = f"top={ht:.1f}m"
    cx = int(np.mean([p[0] for p in top_2d]))
    cy = int(np.mean([p[1] for p in top_2d]))
    cv2.putText(
        img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="相机参数可视化检查：在各视角视频第一帧上绘制坐标轴和球场立方体线框。"
    )
    parser.add_argument("--workdir", type=str, required=True, help="工作目录")
    parser.add_argument("--game_id", type=str, required=True, help="比赛 ID (子目录名)")
    parser.add_argument("--view_list", nargs="+", required=True, help="视角 ID 列表")
    parser.add_argument(
        "--video_type", type=str, default=".MP4", help="视频文件后缀，默认 .MP4"
    )
    parser.add_argument(
        "--court",
        type=str,
        default="work-dirs/court.json",
        help="球场配置：预设名 (volleyball/badminton) 或 court.json 路径",
    )
    parser.add_argument(
        "--axis_len", type=float, default=1.0, help="坐标轴显示长度 (米)，默认 1.0"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果目录，默认 {workdir}/check_camera/{game_id}/",
    )
    args = parser.parse_args()

    # ---- 球场 ----
    court = load_court(args.court)
    print(
        f"球场: {court.court_width}m × {court.court_height}m, "
        f"顶高={court.court_top_height}m, margin={court.court_margin}m"
    )

    # ---- 相机参数 ----
    camera_dir = os.path.join(args.workdir, "prepare", "camera", args.game_id)
    intri_path = os.path.join(camera_dir, "intri.yml")
    extri_path = os.path.join(camera_dir, "extri.yml")
    for p in (intri_path, extri_path):
        if not os.path.exists(p):
            sys.exit(f"[ERROR] 找不到相机参数文件: {p}")

    camera_params = load_camera_params(intri_path, extri_path, args.view_list)
    print(f"已加载 {len(args.view_list)} 个视角的相机参数。")

    # ---- 输出目录 ----
    output_dir = args.output_dir or os.path.join(
        args.workdir, "check_camera", args.game_id
    )
    os.makedirs(output_dir, exist_ok=True)

    # ---- 逐视角处理 ----
    for i, view_id in enumerate(args.view_list):
        # 尝试多种后缀（大小写）
        for ext in [args.video_type, args.video_type.lower(), args.video_type.upper()]:
            video_path = os.path.join(
                args.workdir, "videos", args.game_id, f"{view_id}{ext}"
            )
            if os.path.exists(video_path):
                break
        else:
            print(
                f"[WARN] 视角 {view_id}: 找不到视频文件，跳过。"
                f" (尝试: {video_path})"
            )
            continue

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"[WARN] 视角 {view_id}: 无法读取第一帧，跳过。")
            continue

        # 单视角相机字典
        cam = {
            "K": camera_params["K"][i],
            "dist": camera_params["dist"][i],
            "R": camera_params["R"][i],
            "T": camera_params["T"][i],
        }

        # 绘制
        draw_court_box(frame, cam, court, color=(0, 255, 255), thickness=2)
        draw_axis(frame, cam, axis_len=args.axis_len, thickness=3)

        # 视角标签
        label = f"View: {view_id}"
        cv2.putText(
            frame,
            label,
            (18, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (18, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        save_path = os.path.join(output_dir, f"{view_id}_check.jpg")
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  [{view_id}] 保存 → {save_path}")

    print(f"\n完成。结果目录: {output_dir}")


if __name__ == "__main__":
    main()
