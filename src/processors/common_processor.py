import os.path
from typing import List, Dict, Optional, Callable
import logging
from src.data_io.path_config import GamePath


def multiview_processor(
        data_path_cfg: GamePath,
        task_processor: Callable,
        task_type: str,
        **kwargs,
):
    logger = logging.getLogger(__name__)
    for view_id in data_path_cfg.view_list:
        logger.info(f"Runing {task_type} view {view_id} data")
        task_processor(
            view_id=view_id,
            data_path_cfg=data_path_cfg,
            **kwargs
        )
