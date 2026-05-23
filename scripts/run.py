import os
import sys
from typing import List
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import config_root_logger
from src.data_io.path_config import GamePath
from src.utils.constant import load_court, set_default_court
from src.processors import (
    multiview_processor,
    detection_processor,
    reid_processor,
    triangulation_processor,
    optimize_processor,
    analysis_processor,
    video_sync_processor,
    collect_processor,
    validate_k3d_length_processor,
    visualize_processor,
)


def main(
    workdir: str,
    game_id: str,
    view_list: List[str],
    video_type: str = ".mp4",
    frame_step: int = 1,
    time_align: bool = True,
    max_frame_num: int = -1,
    vis_video: bool = False,
    skip_reid: bool = False,
    skip_analysis: bool = False,
    court: str = None,
    pose_model: str = "rtmpose",
    pose_checkpoint: str = None,
    pose_config: str = None,
    device: str = "cuda:0",
    view_processes: int = 1,
    inherit_detection_from: str = None,
    kalman_process_noise: float = 1e-3,
    kalman_measurement_noise: float = 5e-2,
    one_player: bool = False,
):
    logger = logging.getLogger("main")

    # 设置球场配置
    if court is not None:
        set_default_court(load_court(court))

    data_cfg = GamePath(
        workdir=workdir,
        game_id=game_id,
        view_list=view_list,
        video_type=video_type,
        frame_step=frame_step,
        max_frame_num=max_frame_num,
        pose_model_name=pose_model,
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
        num_workers=view_processes,
        court_spec=court,
        vis=vis_video,
        pose_model=pose_model,
        pose_checkpoint=pose_checkpoint,
        pose_config=pose_config,
        device=device,
        inherit_detection_from=inherit_detection_from,
    )

    # Step 2: reid
    if not skip_reid:
        multiview_processor(
            data_path_cfg=data_cfg,
            task_processor=reid_processor,
            task_type="reid",
            num_workers=view_processes,
            court_spec=court,
            vis=vis_video,
        )

    # Step 3: triangulation
    triangulation_processor(data_path_cfg=data_cfg, use_reid=not skip_reid)

    # Step 4: optimization
    optimize_processor(
        data_path_cfg=data_cfg,
        use_reid=not skip_reid,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
        one_player=one_player,
    )

    # Step 5: analysis
    if not skip_analysis:
        analysis_processor(data_path_cfg=data_cfg, time_resolution=0.2)

    # Step 6: collect k3d joint data
    collect_processor(data_path_cfg=data_cfg, use_reid=not skip_reid)

    # Step 7: validate collected k3d length against synchronized video length
    validate_k3d_length_processor(data_path_cfg=data_cfg)

    # Step 8: generate HTML 3D visualisation
    visualize_processor(data_path_cfg=data_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multiview player tracking pipeline."
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Working directory for the game data.",
    )
    parser.add_argument(
        "--game_id", type=str, required=True, help="Game ID for the processing."
    )
    parser.add_argument(
        "--view_list", nargs="+", required=True, help="List of view IDs to process."
    )
    parser.add_argument(
        "--video_type", type=str, default=".mp4", help="Type of video files to process."
    )
    parser.add_argument(
        "--frame_step", type=int, default=1, help="Step size for frame processing."
    )
    parser.add_argument(
        "--time_align", action="store_true", help="Align video time across views."
    )
    parser.add_argument(
        "--max_frame_num",
        type=int,
        default=-1,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--vis_video", action="store_true", help="Visualize video during processing."
    )
    parser.add_argument(
        "--skip_reid",
        action="store_true",
        help="Skip the ReID step and use detection results directly for triangulation.",
    )
    parser.add_argument(
        "--skip_analysis", action="store_true", help="Skip the analysis step."
    )
    parser.add_argument(
        "--court",
        type=str,
        default="work-dirs/court.json",
        help="球场配置: 预设名称 (volleyball / badminton) 或 court.json 文件路径。默认为排球场。",
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
        help="姿态模型名称；人体检测统一使用现有 RTMDet(TRT)。",
    )
    parser.add_argument(
        "--pose_checkpoint",
        type=str,
        default=None,
        help="姿态模型权重路径。RTMPose/ViTPose++ 默认使用普通 torch 权重；未提供时优先使用仓库内默认权重，MediaPipe 忽略该参数。",
    )
    parser.add_argument(
        "--pose_config",
        type=str,
        default=None,
        help="姿态模型配置路径。RTMPose/ViTPose++ 默认使用仓库内置配置，MediaPipe 可选读取配置字典。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="检测与姿态推理设备。",
    )
    parser.add_argument(
        "--view_processes",
        type=int,
        default=1,
        help="按视角启动的进程数；1 为串行，<=0 时自动按视角数启动。",
    )
    parser.add_argument(
        "--inherit_detection_from",
        type=str,
        default=None,
        help="复用指定姿态模型已有 detection pkl 中的 bbox，跳过当前流程中的 RTMDet 推理。",
    )
    parser.add_argument(
        "--kalman_process_noise",
        type=float,
        default=1e-3,
        help="轨迹优化中卡尔曼滤波的过程噪声强度。越大越跟随当前观测。",
    )
    parser.add_argument(
        "--kalman_measurement_noise",
        type=float,
        default=5e-2,
        help="轨迹优化中卡尔曼滤波的观测噪声强度。越大越平滑。",
    )
    parser.add_argument(
        "--one_player",
        action="store_true",
        help="在重排序后只保留第一个 player 进入后续优化、收集和可视化流程。",
    )

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
        vis_video=args.vis_video,
        skip_reid=args.skip_reid,
        skip_analysis=args.skip_analysis,
        court=args.court,
        pose_model=args.pose_model,
        pose_checkpoint=args.pose_checkpoint,
        pose_config=args.pose_config,
        device=args.device,
        view_processes=args.view_processes,
        inherit_detection_from=args.inherit_detection_from,
        kalman_process_noise=args.kalman_process_noise,
        kalman_measurement_noise=args.kalman_measurement_noise,
        one_player=args.one_player,
    )
