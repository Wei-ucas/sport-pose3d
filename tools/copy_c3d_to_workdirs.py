from __future__ import annotations

import argparse
import json
import os
import re
import shutil


TRIAL_TO_C3D_FOLDER = {
    1: "JTDZ",
    2: "JDDTDZ",
    3: "FZDTDZ",
}


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()


def extract_group_code(group_name: str) -> str | None:
    match = re.search(r"\(([A-Za-z0-9]+)\)", group_name)
    return match.group(1) if match else None


def infer_trial_index(segment: dict, segment_index: int) -> int | None:
    segment_label = segment.get("segment_label", "")
    match = re.search(r"Trial\s+(\d+)", segment_label, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if segment_index < 3:
        return segment_index + 1
    return None


def copy_group_segment_c3d(
    group: dict,
    workdirs_root: str,
    c3d_root: str,
    overwrite: bool = False,
    dry_run: bool = False,
):
    copied_files = 0
    group_name = group["group_name"]
    workdir_name = sanitize_name(group_name)
    group_code = extract_group_code(group_name)
    if not group_code:
        print(f"[警告] 无法从 group_name 解析缩写: {group_name}")
        return copied_files

    group_c3d_root = os.path.join(c3d_root, group_code)
    if not os.path.isdir(group_c3d_root):
        print(f"[警告] 找不到 C3D group 目录: {group_c3d_root}")
        return copied_files

    for segment_index, segment in enumerate(group.get("segments", [])):
        trial_index = infer_trial_index(segment, segment_index)
        if trial_index not in TRIAL_TO_C3D_FOLDER:
            print(f"[警告] 无法识别 trial index: {group_name} / {segment.get('segment_key')}")
            continue

        src_folder_name = TRIAL_TO_C3D_FOLDER[trial_index]
        src_dir = os.path.join(group_c3d_root, src_folder_name)
        if not os.path.isdir(src_dir):
            print(f"[警告] 找不到 C3D segment 目录: {src_dir}")
            continue

        segment_key = segment["segment_key"]
        dst_dir = os.path.join(workdirs_root, workdir_name, "c3d", segment_key)
        c3d_files = sorted(
            name for name in os.listdir(src_dir) if name.lower().endswith(".c3d")
        )
        if not c3d_files:
            print(f"[警告] 目录中无 c3d 文件: {src_dir}")
            continue

        for file_name in c3d_files:
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            if os.path.exists(dst_path) and not overwrite:
                print(f"[跳过] 已存在: {dst_path}")
                continue

            if dry_run:
                print(f"[模拟复制] {src_path} -> {dst_path}")
            else:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"[复制] {src_path} -> {dst_path}")
            copied_files += 1

    return copied_files


def main():
    parser = argparse.ArgumentParser(
        description="将 Vicon C3D 文件复制到现有 work-dirs 目录结构中"
    )
    parser.add_argument(
        "--project_json",
        type=str,
        required=True,
        help="包含 groups / segments 信息的 mocap-project JSON 文件",
    )
    parser.add_argument(
        "--workdirs_root",
        type=str,
        required=True,
        help="现有 work-dirs 根目录",
    )
    parser.add_argument(
        "--c3d_root",
        type=str,
        default="/badminton/wangwei/data/vicon-data-c3d-only",
        help="Vicon C3D 根目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的目标文件",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="模拟运行，只打印不复制",
    )
    args = parser.parse_args()

    with open(args.project_json, "r", encoding="utf-8") as f:
        project = json.load(f)

    total = 0
    for group in project.get("groups", []):
        total += copy_group_segment_c3d(
            group=group,
            workdirs_root=args.workdirs_root,
            c3d_root=args.c3d_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    action = "计划复制" if args.dry_run else "已复制"
    print(f"\n[完成] 共{action} {total} 个 c3d 文件")


if __name__ == "__main__":
    main()