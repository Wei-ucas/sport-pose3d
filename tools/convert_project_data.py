"""
将 mocap-project JSON 数据格式转换为 sport-cap 项目所需的目录结构和相机标定文件。

功能：
- 每个 group 创建一个 workdir
- 每个 segment (Trial) 创建一个 game_id
- 通过软链接将原始视频链接到目标目录（不复制）
- 将 JSON 中的相机内外参转换为 intri.yml / extri.yml 格式

用法：
    python tools/convert_project_data.py \
        --json_path work-dirs/example-data.json \
        --output_dir games \
        --data_roots /path/to/data/root

    如果不提供 --data_roots，则使用 JSON 中的 data_roots。
"""

import argparse
import json
import os
import sys
import re

import cv2
import numpy as np


def sanitize_name(name: str) -> str:
    """将中文/特殊字符的名称转换为合法的目录名。"""
    # 去掉括号内容、逗号分隔的附加信息，只保留核心可识别的名称
    # 替换不适合作为目录名的字符
    name = name.replace("/", "_").replace("\\", "_")
    name = name.strip()
    return name


def make_view_id(camera_id: int) -> str:
    """将 camera_id (1-based) 转换为 view_id 字符串 (0-based, 2位补零)。"""
    return f"{camera_id - 1:02d}"


def resolve_file_path(file_path_obj: dict, data_roots: list) -> str:
    """根据 file_path 对象和 data_roots 列表解析出完整的文件路径。"""
    path = file_path_obj["path"]
    if file_path_obj.get("relative_to_data_root", False):
        root_index = file_path_obj.get("data_root_index", 0)
        if root_index < len(data_roots):
            return os.path.join(data_roots[root_index], path)
        else:
            raise ValueError(
                f"data_root_index {root_index} 超出 data_roots 范围 (共 {len(data_roots)} 个)"
            )
    return path


def write_camera_files(cameras: list, k_matrix: list, dist_coeffs: list, save_dir: str):
    """将相机参数写入 intri.yml 和 extri.yml。

    Args:
        cameras: segment 中的 cameras 列表
        k_matrix: 全局共享的内参矩阵 (3x3 list)
        dist_coeffs: 全局共享的畸变系数 (list)
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    intri_path = os.path.join(save_dir, "intri.yml")
    extri_path = os.path.join(save_dir, "extri.yml")

    # 收集有效标定的相机
    valid_cameras = []
    for cam in cameras:
        calib = cam.get("calibration")
        if calib is None:
            continue
        rvec = calib.get("rvec")
        tvec = calib.get("tvec")
        if rvec is None or tvec is None:
            continue
        valid_cameras.append(cam)

    if not valid_cameras:
        print(f"  [警告] 没有有效标定的相机，跳过写入: {save_dir}")
        return []

    view_ids = [make_view_id(cam["camera_id"]) for cam in valid_cameras]

    # 写入 intri.yml
    fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_WRITE)
    fs_intri.write("names", view_ids)
    K = np.array(k_matrix, dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64).reshape(1, -1)

    for cam in valid_cameras:
        vid = make_view_id(cam["camera_id"])
        fs_intri.write(f"K_{vid}", K)
        fs_intri.write(f"dist_{vid}", dist)
    fs_intri.release()

    # 写入 extri.yml
    fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_WRITE)
    fs_extri.write("names", view_ids)
    for cam in valid_cameras:
        vid = make_view_id(cam["camera_id"])
        calib = cam["calibration"]
        rvec = np.array(calib["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(calib["tvec"], dtype=np.float64).reshape(3, 1)
        fs_extri.write(f"R_{vid}", rvec)
        fs_extri.write(f"T_{vid}", tvec)
    fs_extri.release()

    print(f"  已写入相机标定: {intri_path}")
    print(f"  已写入相机标定: {extri_path}")
    return view_ids


def link_videos(cameras: list, data_roots: list, video_dir: str):
    """为每个相机创建视频软链接。"""
    os.makedirs(video_dir, exist_ok=True)
    linked = []

    for cam in cameras:
        file_path_obj = cam.get("file_path")
        if file_path_obj is None:
            continue

        src_path = resolve_file_path(file_path_obj, data_roots)
        src_path = os.path.abspath(src_path)

        if not os.path.exists(src_path):
            print(f"  [警告] 视频文件不存在，跳过: {src_path}")
            continue

        view_id = make_view_id(cam["camera_id"])
        # 保留原始视频扩展名
        _, ext = os.path.splitext(cam["file_name"])
        link_name = f"{view_id}{ext}"
        link_path = os.path.join(video_dir, link_name)

        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            print(f"  [警告] 目标已存在且非软链接，跳过: {link_path}")
            continue

        os.symlink(src_path, link_path)
        print(f"  已链接视频: {link_name} -> {src_path}")
        linked.append((view_id, ext))

    return linked


def make_game_id(group_name: str, segment: dict) -> str:
    """根据 group 名称和 segment 信息生成 game_id。"""
    seg_key = segment["segment_key"].lower()  # e.g. "seg-001"
    return seg_key


def make_workdir_name(group_name: str) -> str:
    """根据 group 名称生成 workdir 目录名。"""
    return sanitize_name(group_name)


def process_project(
    json_path: str, output_dir: str, data_roots: list = None, video_type: str = ".MP4"
):
    """处理 JSON 项目文件，创建 sport-cap 项目目录结构。"""
    with open(json_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    # 使用命令行提供的 data_roots，或回退到 JSON 中的值
    if data_roots is None or len(data_roots) == 0:
        data_roots = project.get("data_roots", [])
    print(f"Data roots: {data_roots}")

    # 全局共享的内参
    intrinsics = project["intrinsics"]
    k_matrix = intrinsics["k_matrix"]
    dist_coeffs = intrinsics["dist_coeffs"]

    groups = project.get("groups", [])
    if not groups:
        print("JSON 中没有 groups 数据")
        return

    summary = []

    for group in groups:
        group_name = group["group_name"]
        workdir_name = make_workdir_name(group_name)
        workdir_path = os.path.join(output_dir, workdir_name)
        print(f"\n{'='*60}")
        print(f"Group: {group_name}")
        print(f"Workdir: {workdir_path}")
        print(f"{'='*60}")

        segments = group.get("segments", [])
        for segment in segments:
            game_id = make_game_id(group_name, segment)
            seg_label = segment.get("segment_label", game_id)
            cameras = segment.get("cameras", [])

            print(f"\n  Segment: {seg_label} -> game_id: {game_id}")
            print(f"  相机数量: {len(cameras)}")

            # 创建目录结构:
            # workdir/videos/<game_id>/
            # workdir/prepare/camera/<game_id>/
            # workdir/prepare/profiles/<game_id>/
            # workdir/prediction/<game_id>/
            # workdir/vis/<game_id>/
            # workdir/output/<game_id>/
            video_dir = os.path.join(workdir_path, "videos", game_id)
            camera_dir = os.path.join(workdir_path, "prepare", "camera", game_id)
            profile_dir = os.path.join(workdir_path, "prepare", "profiles", game_id)
            prediction_dir = os.path.join(workdir_path, "prediction", game_id)
            vis_dir = os.path.join(workdir_path, "vis", game_id)
            output_dir_game = os.path.join(workdir_path, "output", game_id)

            for d in [
                video_dir,
                camera_dir,
                profile_dir,
                prediction_dir,
                vis_dir,
                output_dir_game,
            ]:
                os.makedirs(d, exist_ok=True)

            # 链接视频
            linked = link_videos(cameras, data_roots, video_dir)

            # 写入相机标定文件
            view_ids = write_camera_files(cameras, k_matrix, dist_coeffs, camera_dir)

            if view_ids:
                summary.append(
                    {
                        "workdir": workdir_path,
                        "game_id": game_id,
                        "segment_label": seg_label,
                        "view_list": view_ids,
                        "video_type": video_type,
                    }
                )

    # 打印运行命令摘要
    print(f"\n{'='*60}")
    print("处理完成! 可使用以下命令运行流水线:")
    print(f"{'='*60}")
    for item in summary:
        views = " ".join(item["view_list"])
        print(f"\npython scripts/run.py \\")
        print(f"  --workdir {item['workdir']} \\")
        print(f"  --game_id {item['game_id']} \\")
        print(f"  --view_list {views} \\")
        print(f"  --video_type {item['video_type']} \\")
        print(f"  --time_align")


def main():
    parser = argparse.ArgumentParser(
        description="将 mocap-project JSON 转换为 sport-cap 项目目录结构"
    )
    parser.add_argument(
        "--json_path", type=str, required=True, help="输入的 JSON 项目文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="games", help="输出根目录 (默认: games)"
    )
    parser.add_argument(
        "--data_roots",
        nargs="+",
        default=None,
        help="数据根目录列表，覆盖 JSON 中的 data_roots。"
        "按顺序对应 JSON 中 data_root_index 0, 1, 2...",
    )
    parser.add_argument(
        "--video_type", type=str, default=".MP4", help="视频文件扩展名 (默认: .MP4)"
    )
    args = parser.parse_args()

    process_project(
        json_path=args.json_path,
        output_dir=args.output_dir,
        data_roots=args.data_roots,
        video_type=args.video_type,
    )


if __name__ == "__main__":
    main()
