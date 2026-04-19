"""
将 sport-cap 产出的同步与对齐结果写回 mocap-project JSON，并导出配套结果文件。

video_sync.json 由 sport-cap 的 video_sync_processor 生成，路径结构为：
    <workdir>/prediction/<game_id>/video_sync.json

其中：
    workdir_name  ↔  group["group_name"]  (经 sanitize_name 处理后相同)
    game_id       ↔  segment["segment_key"].lower()

视图与相机ID的对应关系（与 convert_project_data.py 保持一致）：
    camera_id (1-based)  →  view_id = f"{camera_id - 1:02d}"

同步结果将作为 "sync" 字段写入对应的 camera 记录：
    {
        "camera_id": 1,
        ...,
        "sync": {"frame": 93, "time": 1.851...}
    }

用法示例：
    # 自动从 sync_json 路径推断 group / segment
    python tools/sync_project_data.py \\
        --project_json work-dirs/example-data.json \\
        --output_json work-dirs/example-data.synced.json \\
        --sync_json work-dirs/曹恒瑜(CHY),男,11.18测/prediction/seg-001/video_sync.json

    # 批量处理: 扫描一个 workdir 下所有 prediction/*/video_sync.json
    python tools/sync_project_data.py \\
        --project_json work-dirs/example-data.json \\
        --output_json work-dirs/example-data.synced.json \\
        --workdir work-dirs/曹恒瑜(CHY),男,11.18测

    # 批量处理: 扫描整个 work-dirs 下所有 workdir
    python tools/sync_project_data.py \\
        --project_json work-dirs/all_data.json \\
        --output_json work-dirs/all_data.synced.json \\
        --workdirs_root work-dirs

    # 显式指定 group / segment (适用于路径无法自动推断的情况)
    python tools/sync_project_data.py \\
        --project_json work-dirs/example-data.json \\
        --sync_json /some/path/video_sync.json \\
        --group_name "曹恒瑜(CHY),男,11.18测" \\
        --segment_key SEG-001

    # 模拟运行 (不写回文件)
    python tools/sync_project_data.py \\
        --project_json work-dirs/example-data.json \\
        --sync_json .../video_sync.json \\
        --dry_run

    # 在同步后，将各组输出导出到新的结果目录
    python tools/sync_project_data.py \\
        --project_json work-dirs/all_data.json \\
        --workdirs_root work-dirs \\
        --export_root exported-results

脚本还会额外处理：
    1. 将每个 Trial 对应的 c3d 文件名写回 segment 记录中的 "c3d_file_name"
    2. 将 c3d_alignment_report.json 的内容（去除路径字段）写回 segment 记录中的 "c3d_alignment"
    3. 在导出目录中额外导出 c3d 文件
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys


# ── helpers matching convert_project_data.py ─────────────────────────────────

def sanitize_name(name: str) -> str:
    """与 convert_project_data.py 保持完全一致。"""
    name = name.replace("/", "_").replace("\\", "_")
    return name.strip()


def view_id_to_camera_id(view_id: str) -> int:
    """00 → 1, 01 → 2, ..."""
    return int(view_id) + 1


def camera_id_to_view_id(camera_id: int) -> str:
    """1 → '00', 2 → '01', ..."""
    return f"{camera_id - 1:02d}"


# ── project JSON helpers ──────────────────────────────────────────────────────

def load_project(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)
    print(f"[保存] 已创建新的项目文件: {path}")


def derive_output_json_path(project_json: str) -> str:
    """根据输入 JSON 自动生成输出 JSON 路径，不覆盖原文件。"""
    abs_path = os.path.abspath(project_json)
    root, ext = os.path.splitext(abs_path)
    if not ext:
        ext = ".json"
    return f"{root}.synced{ext}"


def find_group(project: dict, group_name: str) -> dict | None:
    """按原始名称或 sanitize 后的名称查找 group。"""
    target = sanitize_name(group_name)
    for g in project.get("groups", []):
        if g["group_name"] == group_name or sanitize_name(g["group_name"]) == target:
            return g
    return None


def find_segment(group: dict, segment_key: str) -> dict | None:
    """按 segment_key (大小写不敏感) 查找 segment。"""
    key_lower = segment_key.lower()
    for s in group.get("segments", []):
        if s["segment_key"].lower() == key_lower:
            return s
    return None


def resolve_group_name(project: dict, workdir_name: str) -> str | None:
    """根据 workdir 目录名反查 project 中的原始 group_name。"""
    for group in project.get("groups", []):
        if sanitize_name(group["group_name"]) == workdir_name:
            return group["group_name"]
    return None


def resolve_segment(project: dict, workdir_name: str, game_id: str):
    """根据 workdir / game_id 定位到 project 中的 group 和 segment。"""
    group_name = resolve_group_name(project, workdir_name) or workdir_name
    group = find_group(project, group_name)
    if group is None:
        return None, None
    segment = find_segment(group, game_id)
    return group, segment


# ── infer group / segment from sync file path ────────────────────────────────

def infer_from_path(sync_path: str):
    """
    从 sync_json 的绝对/相对路径推断 (workdir_name, game_id)。

    期望路径结构：  .../workdir_name/prediction/<game_id>/video_sync.json
    """
    abs_path = os.path.abspath(sync_path)
    parts = abs_path.replace("\\", "/").split("/")

    # 寻找 "prediction" 部分
    try:
        pred_idx = None
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "prediction":
                pred_idx = i
                break
        if pred_idx is None:
            return None, None
        game_id = parts[pred_idx + 1]          # e.g. "seg-001"
        workdir_name = parts[pred_idx - 1]     # directory just before "prediction"
        return workdir_name, game_id
    except IndexError:
        return None, None


def find_segment_c3d_file(workdirs_root: str, workdir_name: str, segment_key: str) -> str | None:
    """查找单个 segment 对应的 c3d 文件。"""
    c3d_dir = os.path.join(workdirs_root, workdir_name, "c3d", segment_key)
    if not os.path.isdir(c3d_dir):
        return None

    c3d_files = [
        os.path.join(c3d_dir, name)
        for name in sorted(os.listdir(c3d_dir))
        if os.path.isfile(os.path.join(c3d_dir, name)) and name.lower().endswith(".c3d")
    ]
    if not c3d_files:
        return None
    if len(c3d_files) > 1:
        print(f"[警告] {c3d_dir} 下存在多个 c3d 文件，将使用第一个: {os.path.basename(c3d_files[0])}")
    return c3d_files[0]


def build_export_c3d_name(segment_key: str, c3d_basename: str) -> str:
    """导出目录中的 c3d 文件名。"""
    return f"{segment_key}_{c3d_basename}"


def resolve_segment_c3d_file_name(segment: dict, c3d_path: str, use_export_name: bool) -> str:
    """统一计算并复用 segment 中记录的 c3d 文件名。"""
    if use_export_name:
        return build_export_c3d_name(segment["segment_key"], os.path.basename(c3d_path))
    return os.path.basename(c3d_path)


def sanitize_alignment_report(report: dict) -> dict:
    """移除对齐报告中的路径字段。"""
    return {
        key: value
        for key, value in report.items()
        if key not in {"k3d_path", "c3d_file_path"}
    }


def update_segment_artifacts(
    project: dict,
    workdirs_root: str,
    workdir_name: str,
    game_id: str,
    dry_run: bool = False,
    use_export_names: bool = False,
) -> int:
    """将 c3d 文件名与对齐结果写入对应的 segment。"""
    group, segment = resolve_segment(project, workdir_name, game_id)
    if group is None:
        print(f"[警告] 写入 trial 附加信息时找不到 group: {workdir_name}，跳过")
        return 0
    if segment is None:
        print(f"[警告] 写入 trial 附加信息时找不到 segment: {workdir_name}/{game_id}，跳过")
        return 0

    updates = 0
    segment_key = segment["segment_key"]
    c3d_path = find_segment_c3d_file(workdirs_root, workdir_name, segment_key)
    if c3d_path is None:
        print(f"[警告] 未找到 c3d 文件: {workdir_name}/{segment_key}")
    else:
        c3d_file_name = resolve_segment_c3d_file_name(
            segment, c3d_path, use_export_names
        )
        print(
            f"  {'[模拟]' if dry_run else '[更新]'} "
            f"segment='{segment_key}' c3d_file_name={c3d_file_name}"
        )
        if not dry_run:
            segment["c3d_file_name"] = c3d_file_name
        updates += 1

    report_path = os.path.join(
        workdirs_root, workdir_name, "output", game_id, "c3d_alignment_report.json"
    )
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        sanitized_report = sanitize_alignment_report(report_data)
        print(
            f"  {'[模拟]' if dry_run else '[更新]'} "
            f"segment='{segment_key}' c3d_alignment 已写入（去除路径字段）"
        )
        if not dry_run:
            segment["c3d_alignment"] = sanitized_report
        updates += 1
    else:
        print(f"[警告] 未找到对齐报告: {report_path}")

    return updates


# ── core patch logic ──────────────────────────────────────────────────────────

def apply_sync(
    project: dict,
    sync_data: dict,
    group_name: str,
    segment_key: str,
    dry_run: bool = False,
) -> int:
    """
    将 sync_data ({view_id: {frame, time}}) 写入对应的 camera 记录。

    Returns:
        更新的相机数量。
    """
    group = find_group(project, group_name)
    if group is None:
        print(f"[错误] 找不到 group: '{group_name}'")
        print("  可用 groups:", [g["group_name"] for g in project.get("groups", [])])
        return 0

    segment = find_segment(group, segment_key)
    if segment is None:
        print(f"[错误] 在 group '{group['group_name']}' 中找不到 segment: '{segment_key}'")
        print("  可用 segments:", [s["segment_key"] for s in group.get("segments", [])])
        return 0

    cameras = segment.get("cameras", [])
    updated = 0

    for view_id, sync_info in sync_data.items():
        cam_id = view_id_to_camera_id(view_id)
        cam = next((c for c in cameras if c["camera_id"] == cam_id), None)
        if cam is None:
            print(f"  [警告] view_id={view_id} (camera_id={cam_id}) 在此 segment 中不存在，跳过")
            continue

        print(
            f"  {'[模拟]' if dry_run else '[更新]'} "
            f"camera_id={cam_id} ({cam.get('camera_name', '')})  "
            f"frame={sync_info.get('frame')}  time={sync_info.get('time'):.6f}"
        )
        if not dry_run:
            cam["sync"] = {
                "frame": int(sync_info["frame"]),
                "time": float(sync_info["time"]),
            }
        updated += 1

    print(
        f"  → group='{group['group_name']}'  segment='{segment['segment_key']}'  "
        f"{'(模拟)' if dry_run else ''} {updated}/{len(sync_data)} 条同步记录已{'处理' if dry_run else '写入'}"
    )
    return updated


# ── collect sync jobs ─────────────────────────────────────────────────────────

def collect_jobs_from_workdir(workdir: str):
    """
    扫描 <workdir>/prediction/*/video_sync.json，
    返回 [(sync_path, workdir_name, game_id), ...]。
    """
    jobs = []
    pred_dir = os.path.join(workdir, "prediction")
    if not os.path.isdir(pred_dir):
        print(f"[错误] prediction 目录不存在: {pred_dir}")
        return jobs

    workdir_name = os.path.basename(os.path.abspath(workdir))
    for game_id in sorted(os.listdir(pred_dir)):
        sync_path = os.path.join(pred_dir, game_id, "video_sync.json")
        if os.path.isfile(sync_path):
            jobs.append((sync_path, workdir_name, game_id))
    return jobs


def collect_jobs_from_workdirs_root(workdirs_root: str):
    """扫描 work-dirs 下所有子目录，收集全部 video_sync.json。"""
    jobs = []
    for name in sorted(os.listdir(workdirs_root)):
        workdir = os.path.join(workdirs_root, name)
        if not os.path.isdir(workdir):
            continue
        jobs.extend(collect_jobs_from_workdir(workdir))
    return jobs


def export_segment_outputs(
    project: dict,
    workdirs_root: str,
    output_root: str,
    jobs,
    dry_run: bool = False,
) -> int:
    """
    反向导出结果文件。

    源文件：
        <workdirs_root>/<workdir_name>/output/<game_id>/k3d.pkl
        <workdirs_root>/<workdir_name>/output/<game_id>/k3d_vis.html
        <workdirs_root>/<workdir_name>/c3d/<SEGMENT_KEY>/*.c3d

    目标文件：
        <output_root>/<workdir_name>/<SEGMENT_KEY>_k3d.pkl
        <output_root>/<workdir_name>/<SEGMENT_KEY>_k3d_vis.html
        <output_root>/<workdir_name>/<SEGMENT_KEY>_<c3d_basename>.c3d
    """
    exported = 0
    seen = set()

    for _, workdir_name, game_id in jobs:
        job_key = (workdir_name, game_id)
        if job_key in seen:
            continue
        seen.add(job_key)

        group, segment = resolve_segment(project, workdir_name, game_id)
        if group is None:
            print(f"[警告] 导出时找不到 group: {workdir_name}，跳过")
            continue
        if segment is None:
            print(f"[警告] 导出时找不到 segment: {workdir_name}/{game_id}，跳过")
            continue

        segment_key = segment["segment_key"]
        src_dir = os.path.join(workdirs_root, workdir_name, "output", game_id)
        dst_dir = os.path.join(output_root, workdir_name)
        file_pairs = [
            ("k3d.pkl", f"{segment_key}_k3d.pkl"),
            ("k3d_vis.html", f"{segment_key}_k3d_vis.html"),
        ]

        for src_name, dst_name in file_pairs:
            src_path = os.path.join(src_dir, src_name)
            if not os.path.isfile(src_path):
                print(f"[警告] 导出源文件不存在: {src_path}")
                continue

            dst_path = os.path.join(dst_dir, dst_name)
            if dry_run:
                print(f"[模拟导出] {src_path} -> {dst_path}")
            else:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"[导出] {src_path} -> {dst_path}")
            exported += 1

        c3d_path = find_segment_c3d_file(workdirs_root, workdir_name, segment_key)
        if c3d_path is None:
            print(f"[警告] 导出源 c3d 不存在: {workdir_name}/{segment_key}")
            continue

        c3d_name = segment.get("c3d_file_name") or resolve_segment_c3d_file_name(
            segment, c3d_path, use_export_name=True
        )
        dst_c3d_path = os.path.join(dst_dir, c3d_name)
        if dry_run:
            print(f"[模拟导出] {c3d_path} -> {dst_c3d_path}")
        else:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(c3d_path, dst_c3d_path)
            print(f"[导出] {c3d_path} -> {dst_c3d_path}")
        exported += 1

    return exported


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将 video_sync.json 同步结果写回 mocap-project JSON"
    )
    parser.add_argument(
        "--project_json",
        type=str,
        required=True,
        help="输入的 mocap-project JSON 文件路径（只读，不会被直接改写）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="输出 JSON 文件路径。默认在原文件旁生成一个 *.synced.json 文件",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--sync_json",
        type=str,
        help="单个 video_sync.json 文件路径；group/segment 从路径自动推断",
    )
    src.add_argument(
        "--workdir",
        type=str,
        help="workdir 根目录；扫描其下所有 prediction/*/video_sync.json",
    )
    src.add_argument(
        "--workdirs_root",
        type=str,
        help="work-dirs 根目录；扫描其下所有 workdir 的 prediction/*/video_sync.json",
    )

    parser.add_argument(
        "--group_name",
        type=str,
        default=None,
        help="显式指定 group_name (覆盖路径推断，仅与 --sync_json 配合使用)",
    )
    parser.add_argument(
        "--segment_key",
        type=str,
        default=None,
        help="显式指定 segment_key (覆盖路径推断，仅与 --sync_json 配合使用)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="模拟运行：打印将要写入的内容，不修改任何文件",
    )
    parser.add_argument(
        "--export_root",
        type=str,
        default=None,
        help="可选。将输出文件导出到该目录，命名为 <workdir>/<SEGMENT_KEY>_k3d.pkl 等",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.project_json):
        print(f"[错误] 项目 JSON 文件不存在: {args.project_json}")
        sys.exit(1)

    output_json = args.output_json or derive_output_json_path(args.project_json)
    if os.path.abspath(output_json) == os.path.abspath(args.project_json):
        print("[错误] output_json 不能与 project_json 相同，请指定新的输出文件路径")
        sys.exit(1)

    project = load_project(args.project_json)
    total_updated = 0
    total_segment_updates = 0

    jobs = []

    # --- 收集 jobs ---
    if args.sync_json:
        # 单文件模式
        if not os.path.isfile(args.sync_json):
            print(f"[错误] sync JSON 文件不存在: {args.sync_json}")
            sys.exit(1)

        workdir_name, game_id = infer_from_path(args.sync_json)

        # 显式参数覆盖
        group_name = args.group_name or workdir_name
        segment_key = args.segment_key or game_id

        if not group_name or not segment_key:
            print(
                "[错误] 无法从路径推断 group_name / segment_key，"
                "请使用 --group_name 和 --segment_key 显式指定"
            )
            sys.exit(1)

        print(f"\n[同步] {args.sync_json}")
        print(f"  group_name  = {group_name}")
        print(f"  segment_key = {segment_key}")

        jobs = [(args.sync_json, workdir_name or sanitize_name(group_name), segment_key.lower())]

        with open(args.sync_json, "r", encoding="utf-8") as f:
            sync_data = json.load(f)

        total_updated += apply_sync(
            project, sync_data, group_name, segment_key, dry_run=args.dry_run
        )

    elif args.workdir:
        # workdir 扫描模式
        jobs = collect_jobs_from_workdir(args.workdir)
        if not jobs:
            print(f"[警告] 在 {args.workdir} 下未找到任何 video_sync.json")
            sys.exit(0)

        workdir_name = os.path.basename(os.path.abspath(args.workdir))
        # 尝试在 project 中找到匹配的 group（优先使用显式 group_name）
        resolved_group_name = args.group_name
        if resolved_group_name is None:
            # 按 sanitize 后的名称查找
            for g in project.get("groups", []):
                if sanitize_name(g["group_name"]) == workdir_name:
                    resolved_group_name = g["group_name"]
                    break
            if resolved_group_name is None:
                resolved_group_name = workdir_name  # fallback: 直接使用目录名查找

        for sync_path, _, game_id in jobs:
            seg_key = args.segment_key or game_id
            print(f"\n[同步] {sync_path}")
            print(f"  group_name  = {resolved_group_name}")
            print(f"  segment_key = {seg_key}")
            with open(sync_path, "r", encoding="utf-8") as f:
                sync_data = json.load(f)
            total_updated += apply_sync(
                project, sync_data, resolved_group_name, seg_key, dry_run=args.dry_run
            )

    else:
        jobs = collect_jobs_from_workdirs_root(args.workdirs_root)
        if not jobs:
            print(f"[警告] 在 {args.workdirs_root} 下未找到任何 video_sync.json")
            sys.exit(0)

        print(f"[扫描] 在 {args.workdirs_root} 下共发现 {len(jobs)} 个 sync 文件")
        for sync_path, workdir_name, game_id in jobs:
            resolved_group_name = resolve_group_name(project, workdir_name) or workdir_name
            print(f"\n[同步] {sync_path}")
            print(f"  group_name  = {resolved_group_name}")
            print(f"  segment_key = {game_id}")
            with open(sync_path, "r", encoding="utf-8") as f:
                sync_data = json.load(f)
            total_updated += apply_sync(
                project,
                sync_data,
                resolved_group_name,
                game_id,
                dry_run=args.dry_run,
            )

    if args.workdirs_root:
        workdirs_source_root = os.path.abspath(args.workdirs_root)
    elif args.workdir:
        workdirs_source_root = os.path.dirname(os.path.abspath(args.workdir))
    else:
        sync_abs = os.path.abspath(args.sync_json)
        workdirs_source_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(sync_abs)))
        )

    seen_jobs = set()
    for _, workdir_name, game_id in jobs:
        job_key = (workdir_name, game_id)
        if job_key in seen_jobs:
            continue
        seen_jobs.add(job_key)
        total_segment_updates += update_segment_artifacts(
            project,
            workdirs_source_root,
            workdir_name,
            game_id,
            dry_run=args.dry_run,
            use_export_names=bool(args.export_root),
        )

    # --- 写回 ---
    if (total_updated > 0 or total_segment_updates > 0) and not args.dry_run:
        save_project(project, output_json)
        print(
            f"[信息] 共写入 {total_updated} 条同步记录，"
            f"更新 {total_segment_updates} 项 trial 附加信息"
        )
        print(f"[信息] 原始项目文件未改动: {args.project_json}")
    elif args.dry_run:
        print(
            f"\n[模拟] 共涉及 {total_updated} 条同步记录，"
            f"{total_segment_updates} 项 trial 附加信息，未写入任何文件"
        )
    else:
        print("\n[信息] 没有更新任何记录")

    if args.export_root:
        if not jobs:
            print("\n[信息] 没有可导出的任务")
            return

        exported = export_segment_outputs(
            project,
            workdirs_source_root,
            os.path.abspath(args.export_root),
            jobs,
            dry_run=args.dry_run,
        )
        action = "计划导出" if args.dry_run else "已导出"
        print(f"\n[导出] 共{action} {exported} 个文件")


if __name__ == "__main__":
    main()
