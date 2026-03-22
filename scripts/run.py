import os
from typing import List
import argparse
import logging

from src.utils.logger import config_root_logger
from src.data_io.path_config import GamePath
from src.processors import multiview_processor, detection_processor, reid_processor, triangulation_processor, \
    optimize_processor, analysis_processor, video_sync_processor


def main(
        workdir: str,
        game_id: str,
        view_list: List[str],
        video_type: str = ".mp4",
        frame_step: int = 1,
        time_align: bool = True,
        max_frame_num: int = -1,
        vis_video: bool = False,
):
    logger = logging.getLogger("main")
    data_cfg = GamePath(
        workdir=workdir,
        game_id=game_id,
        view_list=view_list,
        video_type=video_type,
        frame_step=frame_step,
        max_frame_num=max_frame_num
    )

    logger.info(f"Game data path: {data_cfg}")

    # step 0: video synchronization
    if time_align:
        logger.info("Starting video synchronization...")
        video_sync_processor(data_path_cfg=data_cfg)

    # Step 1: detection
    multiview_processor(
        data_path_cfg=data_cfg,
        task_processor=detection_processor,
        task_type="detection",
        vis=vis_video
    )

    # Step 2: reid
    multiview_processor(
        data_path_cfg=data_cfg,
        task_processor=reid_processor,
        task_type="reid",
        vis=vis_video
    )

    # Step 3: triangulation
    triangulation_processor(
        data_path_cfg=data_cfg
    )

    # Step 4: optimization
    optimize_processor(
        data_path_cfg=data_cfg
    )

    # Step 5: analysis
    analysis_processor(
        data_path_cfg=data_cfg,
        time_resolution=0.2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multiview player tracking pipeline.")
    parser.add_argument("--workdir", type=str, required=True, help="Working directory for the game data.")
    parser.add_argument("--game_id", type=str, required=True, help="Game ID for the processing.")
    parser.add_argument("--view_list", nargs="+", required=True, help="List of view IDs to process.")
    parser.add_argument("--video_type", type=str, default=".mp4", help="Type of video files to process.")
    parser.add_argument("--frame_step", type=int, default=1, help="Step size for frame processing.")
    parser.add_argument("--time_align", action='store_true', help="Align video time across views.")
    parser.add_argument("--max_frame_num", type=int, default=-1, help="Maximum number of frames to process.")
    parser.add_argument("--vis_video", action='store_true', help="Visualize video during processing.")

    args = parser.parse_args()

    config_root_logger(log_level=logging.INFO)

    main(
        workdir=args.workdir,
        game_id=args.game_id,
        view_list=args.view_list,
        video_type=args.video_type,
        frame_step=args.frame_step,
        time_align=args.time_align,
        max_frame_num=args.max_frame_num,
        vis_video=args.vis_video
    )
