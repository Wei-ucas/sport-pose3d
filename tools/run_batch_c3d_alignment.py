from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


WORK_DIRS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work-dirs"))
LOG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "c3d_alignment"))
VIDEO_TYPE = ".MP4"
RMSE_WARNING_THRESHOLD = 0.1


def discover_tasks(work_dirs_root: str) -> list[dict]:
    tasks = []
    root = os.path.abspath(work_dirs_root)
    for subject in sorted(os.listdir(root)):
        subject_dir = os.path.join(root, subject)
        if not os.path.isdir(subject_dir):
            continue
        videos_dir = os.path.join(subject_dir, "videos")
        if not os.path.isdir(videos_dir):
            continue
        for game_id in sorted(os.listdir(videos_dir)):
            seg_dir = os.path.join(videos_dir, game_id)
            if not os.path.isdir(seg_dir):
                continue
            view_list = sorted(
                os.path.splitext(name)[0]
                for name in os.listdir(seg_dir)
                if name.endswith(VIDEO_TYPE) and os.path.splitext(name)[0].isdigit()
            )
            if not view_list:
                continue
            tasks.append(
                {
                    "subject": subject,
                    "workdir": subject_dir,
                    "game_id": game_id,
                    "view_list": view_list,
                }
            )
    return tasks


def is_done(task: dict, video_type: str, frame_step: int, max_frame_num: int, pose_model: str) -> bool:
    data_cfg = _make_data_cfg(task, video_type, frame_step, max_frame_num, pose_model)
    return os.path.isfile(data_cfg.find_output_artifact_path("c3d_alignment_report", ".json")) and os.path.isfile(
        data_cfg.find_output_artifact_path("c3d_k3d_vis", ".html")
    )


def _make_data_cfg(task: dict, video_type: str, frame_step: int, max_frame_num: int, pose_model: str):
    from src.data_io.path_config import GamePath

    return GamePath(
        workdir=task["workdir"],
        game_id=task["game_id"],
        view_list=task["view_list"],
        video_type=video_type,
        frame_step=frame_step,
        max_frame_num=max_frame_num,
        pose_model_name=pose_model,
    )


def run_task(task: dict, video_type: str, frame_step: int, max_frame_num: int, confidence_threshold: float, pose_model: str) -> dict:
    from src.processors.c3d_alignment_processor import c3d_alignment_processor
    from src.processors.visualize_c3d_k3d_processor import visualize_c3d_k3d_processor

    label = f"{task['subject']} / {task['game_id']}"
    log_dir = os.path.join(LOG_ROOT, task["subject"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{task['game_id']}.log")
    data_cfg = _make_data_cfg(task, video_type, frame_step, max_frame_num, pose_model)
    report_path = data_cfg.get_output_artifact_path("c3d_alignment_report", ".json")
    html_path = data_cfg.get_output_artifact_path("c3d_k3d_vis", ".html")

    t0 = time.time()
    result_payload = {
        "label": label,
        "subject": task["subject"],
        "game_id": task["game_id"],
        "success": False,
        "error": None,
        "warning": False,
        "warning_message": None,
        "elapsed": 0.0,
        "log_path": log_path,
        "report_path": report_path,
        "html_path": html_path,
        "rmse_after": None,
        "selected_player_id": None,
        "joint_count": None,
        "time_offset_frames": None,
        "time_offset_seconds": None,
    }

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"# {datetime.now().isoformat()}\n")
            lf.write(f"# workdir={task['workdir']}\n")
            lf.write(f"# game_id={task['game_id']}\n")
            lf.write(f"# view_list={' '.join(task['view_list'])}\n")
            lf.write(f"# video_type={video_type} frame_step={frame_step} max_frame_num={max_frame_num}\n")
            lf.write(f"# confidence_threshold={confidence_threshold}\n\n")

            lf.write("[INFO] Running c3d alignment...\n")
            report = c3d_alignment_processor(
                data_path_cfg=data_cfg,
                confidence_threshold=confidence_threshold,
            )
            lf.write(f"[INFO] Alignment report saved: {report_path}\n")
            lf.write("[INFO] Running combined visualization...\n")
            visualize_c3d_k3d_processor(data_path_cfg=data_cfg)
            lf.write(f"[INFO] Visualization saved: {html_path}\n")

            rmse_after = float(report["spatial_alignment"]["rmse_after"])
            warning = rmse_after > RMSE_WARNING_THRESHOLD
            warning_message = None
            if warning:
                warning_message = (
                    f"rmse_after={rmse_after:.6f} exceeds threshold {RMSE_WARNING_THRESHOLD:.3f}"
                )
                lf.write(f"[WARNING] {warning_message}\n")

            result_payload.update(
                {
                    "success": True,
                    "warning": warning,
                    "warning_message": warning_message,
                    "rmse_after": rmse_after,
                    "selected_player_id": report["selected_player_id"],
                    "joint_count": len(report["joint_names"]),
                    "time_offset_frames": int(report["time_alignment"]["c3d_to_k3d_offset_frames"]),
                    "time_offset_seconds": float(report["time_alignment"]["c3d_to_k3d_offset_seconds"]),
                }
            )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n[ERROR] {exc}\n")
                lf.write(tb)
        except OSError:
            pass
        result_payload["error"] = str(exc)

    result_payload["elapsed"] = time.time() - t0
    return result_payload


def write_summary(results: list[dict], summary_dir: str) -> tuple[str, str, str]:
    os.makedirs(summary_dir, exist_ok=True)
    json_path = os.path.join(summary_dir, "summary.json")
    csv_path = os.path.join(summary_dir, "summary.csv")
    warning_path = os.path.join(summary_dir, "warnings.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "subject",
        "game_id",
        "success",
        "warning",
        "warning_message",
        "rmse_after",
        "selected_player_id",
        "joint_count",
        "time_offset_frames",
        "time_offset_seconds",
        "elapsed",
        "report_path",
        "html_path",
        "log_path",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with open(warning_path, "w", encoding="utf-8") as f:
        f.write(f"# generated at {datetime.now().isoformat()}\n")
        for row in results:
            if row.get("warning"):
                f.write(
                    f"[WARNING] {row['subject']} / {row['game_id']}  {row['warning_message']}  log={row['log_path']}\n"
                )
            if row.get("error"):
                f.write(
                    f"[ERROR] {row['subject']} / {row['game_id']}  {row['error']}  log={row['log_path']}\n"
                )

    return json_path, csv_path, warning_path


def main():
    parser = argparse.ArgumentParser(description="Batch-run C3D alignment and combined visualization")
    parser.add_argument("--workdirs_root", type=str, default=WORK_DIRS_ROOT)
    parser.add_argument("--video_type", type=str, default=VIDEO_TYPE)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frame_num", type=int, default=-1)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    parser.add_argument("--filter_subject", nargs="+", default=None, metavar="KEYWORD")
    parser.add_argument("--filter_segment", nargs="+", default=None, metavar="SEG")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, metavar="N")
    parser.add_argument("--pose_model", type=str, default="rtmpose")
    args = parser.parse_args()

    all_tasks = discover_tasks(args.workdirs_root)
    print(f"发现任务总数: {len(all_tasks)}")

    if args.filter_subject:
        keywords = [key.lower() for key in args.filter_subject]
        all_tasks = [
            task for task in all_tasks if any(key in task["subject"].lower() for key in keywords)
        ]
        print(f"过滤受试者后剩余: {len(all_tasks)}")

    if args.filter_segment:
        segment_set = {seg.lower() for seg in args.filter_segment}
        all_tasks = [task for task in all_tasks if task["game_id"].lower() in segment_set]
        print(f"过滤 segment 后剩余: {len(all_tasks)}")

    if not args.force:
        pending = [
            task for task in all_tasks
            if not is_done(task, args.video_type, args.frame_step, args.max_frame_num, args.pose_model)
        ]
        skipped = len(all_tasks) - len(pending)
        if skipped:
            print(f"已完成（跳过）: {skipped}  待处理: {len(pending)}")
    else:
        pending = all_tasks

    if not pending:
        print("所有任务已处理完毕，无需重新运行。")
        return

    print(f"\n待处理任务 ({len(pending)} 个):")
    for i, task in enumerate(pending, 1):
        print(f"  {i:3d}. {task['subject']} / {task['game_id']}  views={task['view_list']}")

    if args.dry_run:
        print("\n[模拟运行] 将执行 C3D 对齐和联合可视化：\n")
        for task in pending:
            print(f"  {task['subject']} / {task['game_id']}")
        return

    from src.utils.logger import config_root_logger

    config_root_logger()

    print(f"\n开始执行 (jobs={args.jobs}) ...\n{'=' * 60}")
    t_start = time.time()
    results = []

    if args.jobs == 1:
        for idx, task in enumerate(pending, 1):
            print(f"[{idx}/{len(pending)}] {task['subject']} / {task['game_id']} ...", flush=True)
            res = run_task(
                task,
                video_type=args.video_type,
                frame_step=args.frame_step,
                max_frame_num=args.max_frame_num,
                confidence_threshold=args.confidence_threshold,
                pose_model=args.pose_model,
            )
            results.append(res)
            if res["success"]:
                status = "✓"
                if res["warning"]:
                    status += f" [warning: rmse_after={res['rmse_after']:.4f}]"
            else:
                status = f"✗ [{res['error']}]"
            print(f"  {status}  {res['elapsed']:.1f}s  log: {res['log_path']}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for task in pending:
                future = executor.submit(
                    run_task,
                    task,
                    args.video_type,
                    args.frame_step,
                    args.max_frame_num,
                    args.confidence_threshold,
                    args.pose_model,
                )
                futures[future] = task
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                try:
                    res = future.result()
                except Exception as exc:  # noqa: BLE001
                    task = futures[future]
                    res = {
                        "subject": task["subject"],
                        "game_id": task["game_id"],
                        "label": f"{task['subject']} / {task['game_id']}",
                        "success": False,
                        "error": str(exc),
                        "warning": False,
                        "warning_message": None,
                        "elapsed": 0.0,
                        "log_path": "(未生成)",
                        "report_path": None,
                        "html_path": None,
                        "rmse_after": None,
                        "selected_player_id": None,
                        "joint_count": None,
                        "time_offset_frames": None,
                        "time_offset_seconds": None,
                    }
                results.append(res)
                if res["success"]:
                    status = "✓"
                    if res["warning"]:
                        status += f" [warning: rmse_after={res['rmse_after']:.4f}]"
                else:
                    status = f"✗ [{res['error']}]"
                print(f"[{done_count}/{len(pending)}] {res['label']}  {status}  {res['elapsed']:.1f}s")

    elapsed_total = time.time() - t_start
    succeeded = [row for row in results if row["success"]]
    warnings = [row for row in results if row.get("warning")]
    failed = [row for row in results if not row["success"]]

    summary_json, summary_csv, warning_log = write_summary(results, LOG_ROOT)

    print(f"\n{'=' * 60}")
    print(f"完成: {len(succeeded)}/{len(results)}  warnings: {len(warnings)}  failures: {len(failed)}  总耗时: {elapsed_total / 60:.1f} min")
    print(f"汇总 JSON: {summary_json}")
    print(f"汇总 CSV: {summary_csv}")
    print(f"告警/错误日志: {warning_log}")

    if warnings:
        print("\nRMSE 告警任务:")
        for row in warnings:
            print(
                f"  !  {row['subject']} / {row['game_id']}  rmse_after={row['rmse_after']:.6f}  log: {row['log_path']}"
            )

    if failed:
        print("\n失败任务:")
        for row in failed:
            print(f"  ✗  {row['subject']} / {row['game_id']}  {row['error']}  log: {row['log_path']}")
        sys.exit(1)


if __name__ == "__main__":
    main()