from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_io.path_config import GamePath
from src.processors.visualize_c3d_k3d_processor import visualize_c3d_k3d_processor
from src.utils.logger import config_root_logger


def infer_view_list(workdir: str, game_id: str, video_type: str):
    video_dir = os.path.join(workdir, "videos", game_id)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    view_list = sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(video_dir)
        if name.endswith(video_type) and os.path.splitext(name)[0].isdigit()
    )
    if not view_list:
        raise ValueError(f"No video views found under {video_dir}")
    return view_list


def main():
    parser = argparse.ArgumentParser(description="Generate combined C3D+K3D HTML visualization")
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--game_id", type=str, required=True)
    parser.add_argument("--video_type", type=str, default=".MP4")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frame_num", type=int, default=-1)
    parser.add_argument("--pose_model", type=str, default="rtmpose")
    args = parser.parse_args()

    config_root_logger()
    view_list = infer_view_list(args.workdir, args.game_id, args.video_type)
    data_cfg = GamePath(
        workdir=args.workdir,
        game_id=args.game_id,
        view_list=view_list,
        video_type=args.video_type,
        frame_step=args.frame_step,
        max_frame_num=args.max_frame_num,
        pose_model_name=args.pose_model,
    )
    visualize_c3d_k3d_processor(data_path_cfg=data_cfg)


if __name__ == "__main__":
    main()