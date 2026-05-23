"""
批量处理 work-dirs 下所有受试者的所有 seg 数据。

自动发现：
    work-dirs/<subject>/videos/<game_id>/   →  一次 run.py 调用

恢复机制：
    如果 output/<game_id>/k3d.pkl 已存在，则跳过该 (subject, game_id)。

每个任务的日志保存到 logs/<subject>/<game_id>.log。

用法示例：
    # 处理全部数据
    python scripts/run_batch.py

    # 常用模板：切换到新姿态模型，但复用另一个模型已有的 detection bbox
    python scripts/run_batch.py \
        --pose_model vitposepp-h \
        --inherit_detection_from rtmpose-m-halpe26 \
        --view_processes 4 \
        --jobs 1

    # 全量强制重跑，并复用已有 detection bbox
    python scripts/run_batch.py \
        --force \
        --pose_model vitposepp-h \
        --inherit_detection_from rtmpose-m-halpe26 \
        --view_processes 4 \
        --jobs 1

    # 模拟运行，仅打印命令
    python scripts/run_batch.py --dry_run

    # 只处理指定受试者（可多个，支持子字符串匹配）
    python scripts/run_batch.py --filter_subject CHY TWJ

    # 只处理指定 segment（精确匹配）
    python scripts/run_batch.py --filter_segment seg-001 seg-002

    # 强制重新处理（忽略已存在的输出）
    python scripts/run_batch.py --force

    # 指定并行数（默认 1，顺序处理；>1 多进程并行）
    python scripts/run_batch.py --jobs 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_io.path_config import GamePath

# ── 固定参数（与示例命令一致）──────────────────────────────────────────────────
SCRIPT = os.path.join(os.path.dirname(__file__), "run.py")
WORK_DIRS_ROOT = os.path.join(os.path.dirname(__file__), "..", "work-dirs")
COURT_JSON = os.path.join(WORK_DIRS_ROOT, "court.json")
LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "logs")
VIDEO_TYPE = ".MP4"
FIXED_FLAGS = ["--skip_reid", "--skip_analysis", "--time_align"]


# ── 任务发现 ───────────────────────────────────────────────────────────────────


def discover_tasks(work_dirs_root: str, view_list: list[str] = None) -> list[dict]:
    """扫描 work-dirs 下所有 (workdir, game_id, view_list) 任务。"""
    tasks = []
    root = os.path.abspath(work_dirs_root)
    for subject in sorted(os.listdir(root)):
        subject_dir = os.path.join(root, subject)
        if not os.path.isdir(subject_dir):
            continue  # 跳过 .json 等文件
        videos_dir = os.path.join(subject_dir, "videos")
        if not os.path.isdir(videos_dir):
            continue
        for game_id in sorted(os.listdir(videos_dir)):
            seg_dir = os.path.join(videos_dir, game_id)
            if not os.path.isdir(seg_dir):
                continue
            # 收集该 segment 下存在的视图
            find_view_list = sorted(
                f.split(".")[0]
                for f in os.listdir(seg_dir)
                if f.endswith(VIDEO_TYPE) and f.split(".")[0].isdigit()
            )
            if not find_view_list:
                continue
            if view_list is not None:
                # 如果指定了 view_list，检查是否全部存在
                if not all(v in find_view_list for v in view_list):
                    print(f"警告: {subject}/{game_id} 缺失指定视图，跳过")
                    continue
                use_view_list = view_list
            else:
                use_view_list = find_view_list
            tasks.append(
                {
                    "subject": subject,
                    "workdir": subject_dir,
                    "game_id": game_id,
                    "view_list": use_view_list,
                }
            )
    return tasks


def is_done(task: dict, pose_model: str) -> bool:
    """判断任务是否已完成（模型标签对应的 k3d 输出存在）。"""
    data_cfg = GamePath(
        workdir=task["workdir"],
        game_id=task["game_id"],
        view_list=task["view_list"],
        video_type=VIDEO_TYPE,
        pose_model_name=pose_model,
    )
    return os.path.isfile(data_cfg.find_output_artifact_path("k3d", ".pkl"))


# ── 命令构建 ───────────────────────────────────────────────────────────────────


def build_cmd(
    task: dict,
    pose_model: str,
    pose_checkpoint: str = None,
    pose_config: str = None,
    view_processes: int = 1,
    inherit_detection_from: str = None,
    kalman_process_noise: float = 1e-3,
    kalman_measurement_noise: float = 5e-2,
    one_player: bool = False,
    view_list: list[str] = None,
) -> list[str]:
    cmd = [
        sys.executable,
        SCRIPT,
        "--workdir",
        task["workdir"],
        "--game_id",
        task["game_id"],
        "--view_list",
        *task["view_list"],
        "--video_type",
        VIDEO_TYPE,
        "--court",
        os.path.abspath(COURT_JSON),
        "--pose_model",
        pose_model,
        "--view_processes",
        str(view_processes),
        "--kalman_process_noise",
        str(kalman_process_noise),
        "--kalman_measurement_noise",
        str(kalman_measurement_noise),
        *FIXED_FLAGS,
    ]
    if pose_checkpoint:
        cmd.extend(["--pose_checkpoint", pose_checkpoint])
    if pose_config:
        cmd.extend(["--pose_config", pose_config])
    if inherit_detection_from:
        cmd.extend(["--inherit_detection_from", inherit_detection_from])
    if one_player:
        cmd.append("--one_player")
    if view_list:
        cmd.extend(["--view_list", *view_list])
    return cmd


# ── 单任务执行 ─────────────────────────────────────────────────────────────────


def run_task(
    task: dict,
    pose_model: str,
    pose_checkpoint: str = None,
    pose_config: str = None,
    view_processes: int = 1,
    inherit_detection_from: str = None,
    kalman_process_noise: float = 1e-3,
    kalman_measurement_noise: float = 5e-2,
    one_player: bool = False,
    view_list: list[str] = None,
) -> dict:
    """执行单个任务，返回结果信息。任何异常均被捕获并记录到日志，不向上抛出。"""
    log_dir = os.path.join(LOG_ROOT, task["subject"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{task['game_id']}.log")

    cmd = build_cmd(
        task,
        pose_model,
        pose_checkpoint,
        pose_config,
        view_processes,
        inherit_detection_from,
        kalman_process_noise,
        kalman_measurement_noise,
        one_player,
        view_list,
    )
    label = f"{task['subject']} / {task['game_id']}"
    t0 = time.time()

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"# {datetime.now().isoformat()}\n")
            lf.write(f"# cmd: {' '.join(cmd)}\n\n")
            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
        elapsed = time.time() - t0
        success = result.returncode == 0
        if not success:
            # 追加一行明显的失败标记，方便 grep
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n# [FAILED] returncode={result.returncode}\n")
        return {
            "label": label,
            "success": success,
            "returncode": result.returncode,
            "error": None,
            "elapsed": elapsed,
            "log_path": log_path,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        elapsed = time.time() - t0
        tb = traceback.format_exc()
        # 尽可能把异常信息写入日志
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n# [EXCEPTION]\n{tb}\n")
        except OSError:
            pass
        return {
            "label": label,
            "success": False,
            "returncode": None,
            "error": str(exc),
            "elapsed": elapsed,
            "log_path": log_path,
        }


# ── 主逻辑 ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="批量运行 sport-cap 处理流水线")
    parser.add_argument(
        "--dry_run", action="store_true", help="模拟运行：仅打印命令，不执行"
    )
    parser.add_argument(
        "--force", action="store_true", help="强制重新处理已有输出的任务"
    )
    parser.add_argument(
        "--filter_subject",
        nargs="+",
        default=None,
        metavar="KEYWORD",
        help="只处理名称含指定关键词的受试者（子字符串匹配，大小写不敏感）",
    )
    parser.add_argument(
        "--filter_segment",
        nargs="+",
        default=None,
        metavar="SEG",
        help="只处理指定 game_id（精确匹配，大小写不敏感）",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="并行进程数（默认 1，顺序处理）",
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        default="rtmpose",
        choices=[
            "rtmpose",
            "rtmpose-m-halpe26",
            "mediapipe",
            "vitposepp-b",
            "vitposepp-h",
        ],
        help="透传给 run.py 的姿态模型名称。",
    )
    parser.add_argument(
        "--pose_checkpoint",
        type=str,
        default=None,
        help="透传给 run.py 的姿态权重路径。",
    )
    parser.add_argument(
        "--pose_config", type=str, default=None, help="透传给 run.py 的姿态配置路径。"
    )
    parser.add_argument(
        "--view_processes",
        type=int,
        default=1,
        help="透传给 run.py 的按视角进程数；1 为串行，<=0 时自动按视角数启动。",
    )
    parser.add_argument(
        "--inherit_detection_from",
        type=str,
        default=None,
        help="透传给 run.py；复用指定姿态模型已有 detection pkl 中的 bbox。",
    )
    parser.add_argument(
        "--kalman_process_noise",
        type=float,
        default=1e-3,
        help="透传给 run.py 的卡尔曼过程噪声强度。",
    )
    parser.add_argument(
        "--kalman_measurement_noise",
        type=float,
        default=5e-2,
        help="透传给 run.py 的卡尔曼观测噪声强度。",
    )
    parser.add_argument(
        "--one_player",
        action="store_true",
        help="透传给 run.py；在重排序后只保留第一个 player。",
    )
    parser.add_argument(
        "--view_list",
        nargs="+",
        default=None,
        metavar="VIEW",
        help="透传给 run.py 的视角列表，覆盖自动发现的视角。",
    )
    args = parser.parse_args()

    # 校验必要文件
    if not os.path.isfile(COURT_JSON):
        print(f"[错误] court.json 不存在: {COURT_JSON}")
        sys.exit(1)
    if not os.path.isfile(SCRIPT):
        print(f"[错误] run.py 不存在: {SCRIPT}")
        sys.exit(1)

    all_tasks = discover_tasks(WORK_DIRS_ROOT, view_list=args.view_list)
    print(f"发现任务总数: {len(all_tasks)}")

    # 过滤受试者
    if args.filter_subject:
        keywords = [k.lower() for k in args.filter_subject]
        all_tasks = [
            t for t in all_tasks if any(kw in t["subject"].lower() for kw in keywords)
        ]
        print(f"过滤受试者后剩余: {len(all_tasks)}")

    # 过滤 segment
    if args.filter_segment:
        seg_set = {s.lower() for s in args.filter_segment}
        all_tasks = [t for t in all_tasks if t["game_id"].lower() in seg_set]
        print(f"过滤 segment 后剩余: {len(all_tasks)}")

    # 跳过已完成
    if not args.force:
        pending = [t for t in all_tasks if not is_done(t, args.pose_model)]
        skipped = len(all_tasks) - len(pending)
        if skipped:
            print(f"已完成（跳过）: {skipped}  待处理: {len(pending)}")
    else:
        pending = all_tasks

    if not pending:
        print("所有任务已处理完毕，无需重新运行。")
        return

    # 打印任务列表
    print(f"\n待处理任务 ({len(pending)} 个):")
    for i, t in enumerate(pending, 1):
        marker = "[done]" if is_done(t, args.pose_model) and args.force else ""
        print(
            f"  {i:3d}. {t['subject']} / {t['game_id']}  views={t['view_list']}  {marker}"
        )

    if args.dry_run:
        print("\n[模拟运行] 以下命令将被执行:\n")
        for t in pending:
            print(
                "  "
                + " ".join(
                    build_cmd(
                        t,
                        args.pose_model,
                        args.pose_checkpoint,
                        args.pose_config,
                        args.view_processes,
                        args.inherit_detection_from,
                        args.kalman_process_noise,
                        args.kalman_measurement_noise,
                        args.one_player,
                    )
                )
            )
            print()
        return

    # 执行
    print(f"\n开始执行 (jobs={args.jobs}) ...\n{'='*60}")
    results = []
    t_start = time.time()

    if args.jobs == 1:
        for idx, task in enumerate(pending, 1):
            label = f"{task['subject']} / {task['game_id']}"
            print(f"[{idx}/{len(pending)}] {label} ...", flush=True)
            res = run_task(
                task,
                args.pose_model,
                args.pose_checkpoint,
                args.pose_config,
                args.view_processes,
                args.inherit_detection_from,
                args.kalman_process_noise,
                args.kalman_measurement_noise,
                args.one_player,
            )  # 内部已捕获所有异常，不会抛出
            results.append(res)
            if res["success"]:
                status = "✓"
            elif res["error"]:
                status = f"✗ [exception: {res['error']}]"
            else:
                status = f"✗ (rc={res['returncode']})"
            print(f"  {status}  {res['elapsed']:.1f}s  log: {res['log_path']}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.jobs) as exe:
            for task in pending:
                f = exe.submit(
                    run_task,
                    task,
                    args.pose_model,
                    args.pose_checkpoint,
                    args.pose_config,
                    args.view_processes,
                    args.inherit_detection_from,
                    args.kalman_process_noise,
                    args.kalman_measurement_noise,
                    args.one_player,
                )
                futures[f] = task
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                try:
                    res = future.result()
                except Exception as exc:  # noqa: BLE001  — 理论上不应发生
                    orig_task = futures[future]
                    res = {
                        "label": f"{orig_task['subject']} / {orig_task['game_id']}",
                        "success": False,
                        "returncode": None,
                        "error": str(exc),
                        "elapsed": 0.0,
                        "log_path": "(未生成)",
                    }
                results.append(res)
                if res["success"]:
                    status = "✓"
                elif res["error"]:
                    status = f"✗ [exception: {res['error']}]"
                else:
                    status = f"✗ (rc={res['returncode']})"
                print(
                    f"[{done_count}/{len(pending)}] {res['label']}  {status}  {res['elapsed']:.1f}s"
                )

    # ── 汇总 ──
    elapsed_total = time.time() - t_start
    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print(f"完成: {len(succeeded)}/{len(results)}  总耗时: {elapsed_total/60:.1f} min")

    if failed:
        print(f"\n失败任务 ({len(failed)} 个):")
        for r in failed:
            detail = (
                f"exception: {r['error']}" if r["error"] else f"rc={r['returncode']}"
            )
            print(f"  ✗  {r['label']}  {detail}  log: {r['log_path']}")
        # 将失败摘要写入汇总日志，方便复查
        summary_path = os.path.join(LOG_ROOT, "failed_summary.txt")
        os.makedirs(LOG_ROOT, exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(f"\n# batch run {datetime.now().isoformat()}\n")
            for r in failed:
                detail = (
                    f"exception: {r['error']}"
                    if r["error"]
                    else f"rc={r['returncode']}"
                )
                sf.write(f"  {r['label']}  {detail}  log: {r['log_path']}\n")
        print(f"\n失败摘要已追加到: {summary_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
