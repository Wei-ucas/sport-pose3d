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

# ── 固定参数（与示例命令一致）──────────────────────────────────────────────────
SCRIPT = os.path.join(os.path.dirname(__file__), "run.py")
WORK_DIRS_ROOT = os.path.join(os.path.dirname(__file__), "..", "work-dirs")
COURT_JSON = os.path.join(WORK_DIRS_ROOT, "court.json")
LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "logs")
VIDEO_TYPE = ".MP4"
FIXED_FLAGS = ["--skip_reid", "--skip_analysis", "--time_align"]


# ── 任务发现 ───────────────────────────────────────────────────────────────────

def discover_tasks(work_dirs_root: str) -> list[dict]:
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
            view_list = sorted(
                f.split(".")[0]
                for f in os.listdir(seg_dir)
                if f.endswith(VIDEO_TYPE) and f.split(".")[0].isdigit()
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


def is_done(task: dict) -> bool:
    """判断任务是否已完成（output/<game_id>/k3d.pkl 存在）。"""
    k3d_path = os.path.join(task["workdir"], "output", task["game_id"], "k3d.pkl")
    return os.path.isfile(k3d_path)


# ── 命令构建 ───────────────────────────────────────────────────────────────────

def build_cmd(task: dict) -> list[str]:
    return [
        sys.executable, SCRIPT,
        "--workdir", task["workdir"],
        "--game_id", task["game_id"],
        "--view_list", *task["view_list"],
        "--video_type", VIDEO_TYPE,
        "--court", os.path.abspath(COURT_JSON),
        *FIXED_FLAGS,
    ]


# ── 单任务执行 ─────────────────────────────────────────────────────────────────

def run_task(task: dict) -> dict:
    """执行单个任务，返回结果信息。任何异常均被捕获并记录到日志，不向上抛出。"""
    log_dir = os.path.join(LOG_ROOT, task["subject"])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{task['game_id']}.log")

    cmd = build_cmd(task)
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
        "--dry_run", action="store_true",
        help="模拟运行：仅打印命令，不执行"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新处理已有输出的任务"
    )
    parser.add_argument(
        "--filter_subject", nargs="+", default=None, metavar="KEYWORD",
        help="只处理名称含指定关键词的受试者（子字符串匹配，大小写不敏感）"
    )
    parser.add_argument(
        "--filter_segment", nargs="+", default=None, metavar="SEG",
        help="只处理指定 game_id（精确匹配，大小写不敏感）"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="并行进程数（默认 1，顺序处理）"
    )
    args = parser.parse_args()

    # 校验必要文件
    if not os.path.isfile(COURT_JSON):
        print(f"[错误] court.json 不存在: {COURT_JSON}")
        sys.exit(1)
    if not os.path.isfile(SCRIPT):
        print(f"[错误] run.py 不存在: {SCRIPT}")
        sys.exit(1)

    all_tasks = discover_tasks(WORK_DIRS_ROOT)
    print(f"发现任务总数: {len(all_tasks)}")

    # 过滤受试者
    if args.filter_subject:
        keywords = [k.lower() for k in args.filter_subject]
        all_tasks = [
            t for t in all_tasks
            if any(kw in t["subject"].lower() for kw in keywords)
        ]
        print(f"过滤受试者后剩余: {len(all_tasks)}")

    # 过滤 segment
    if args.filter_segment:
        seg_set = {s.lower() for s in args.filter_segment}
        all_tasks = [t for t in all_tasks if t["game_id"].lower() in seg_set]
        print(f"过滤 segment 后剩余: {len(all_tasks)}")

    # 跳过已完成
    if not args.force:
        pending = [t for t in all_tasks if not is_done(t)]
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
        marker = "[done]" if is_done(t) and args.force else ""
        print(f"  {i:3d}. {t['subject']} / {t['game_id']}  views={t['view_list']}  {marker}")

    if args.dry_run:
        print("\n[模拟运行] 以下命令将被执行:\n")
        for t in pending:
            print("  " + " ".join(build_cmd(t)))
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
            res = run_task(task)  # 内部已捕获所有异常，不会抛出
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
                f = exe.submit(run_task, task)
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
                print(f"[{done_count}/{len(pending)}] {res['label']}  {status}  {res['elapsed']:.1f}s")

    # ── 汇总 ──
    elapsed_total = time.time() - t_start
    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print(f"完成: {len(succeeded)}/{len(results)}  总耗时: {elapsed_total/60:.1f} min")

    if failed:
        print(f"\n失败任务 ({len(failed)} 个):")
        for r in failed:
            detail = f"exception: {r['error']}" if r["error"] else f"rc={r['returncode']}"
            print(f"  ✗  {r['label']}  {detail}  log: {r['log_path']}")
        # 将失败摘要写入汇总日志，方便复查
        summary_path = os.path.join(LOG_ROOT, "failed_summary.txt")
        os.makedirs(LOG_ROOT, exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(f"\n# batch run {datetime.now().isoformat()}\n")
            for r in failed:
                detail = f"exception: {r['error']}" if r["error"] else f"rc={r['returncode']}"
                sf.write(f"  {r['label']}  {detail}  log: {r['log_path']}\n")
        print(f"\n失败摘要已追加到: {summary_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
