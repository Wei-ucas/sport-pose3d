import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from typing import Callable
import logging
from src.data_io.path_config import GamePath


def _run_view_task(
    view_id: str,
    data_path_cfg: GamePath,
    task_processor: Callable,
    task_kwargs: dict,
    court_spec: str = None,
):
    # Re-apply court config: spawn子进程重新 import 模块，主进程的 set_default_court 对子进程不可见
    if court_spec is not None:
        from src.utils.constant import load_court, set_default_court

        set_default_court(load_court(court_spec))
    task_processor(
        view_id=view_id,
        data_path_cfg=data_path_cfg,
        **task_kwargs,
    )
    return view_id, os.getpid()


def multiview_processor(
    data_path_cfg: GamePath,
    task_processor: Callable,
    task_type: str,
    num_workers: int = 1,
    court_spec: str = None,
    **kwargs,
):
    logger = logging.getLogger(__name__)
    if num_workers is None:
        num_workers = 1

    if num_workers <= 0:
        num_workers = len(data_path_cfg.view_list)

    if num_workers <= 1 or len(data_path_cfg.view_list) <= 1:
        for view_id in data_path_cfg.view_list:
            logger.info(f"Runing {task_type} view {view_id} data")
            task_processor(view_id=view_id, data_path_cfg=data_path_cfg, **kwargs)
        return

    worker_count = min(num_workers, len(data_path_cfg.view_list))
    logger.info(
        "Runing %s for %d views with %d worker processes",
        task_type,
        len(data_path_cfg.view_list),
        worker_count,
    )
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=mp_context
    ) as executor:
        futures = {}
        for view_id in data_path_cfg.view_list:
            logger.info(f"Submitting {task_type} view {view_id} to worker pool")
            future = executor.submit(
                _run_view_task,
                view_id,
                data_path_cfg,
                task_processor,
                kwargs,
                court_spec,
            )
            futures[future] = view_id

        for future in as_completed(futures):
            view_id = futures[future]
            finished_view_id, pid = future.result()
            logger.info(
                "Finished %s view %s in worker pid=%s",
                task_type,
                finished_view_id,
                pid,
            )
