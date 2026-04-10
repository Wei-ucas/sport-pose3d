import os.path
from typing import List, Dict, Any
import logging
import tqdm
import json

# from parallelbar import progress_map

from src.data_io.path_config import GamePath, read_player_names
from src.structures.frame import Frame
from src.structures.multiview_frame import MvFrame
from src.data_io.loader import FrameInput, MvFrameInput
from src.data_io.saver import FrameSaver, VideoWriter
from src.utils.camera import load_camera
from src.modules.player_triangulation.easymocap_associator import PlayerTriangulator
from src.modules.player_matcher.k3d_match import Player3DMatcher

MAX_MATCHING_BATCH = 5000


def multiview_frame_reader(
    data_path_cfg: GamePath,
    view_time_offset: Dict[int, int] = None,
    use_reid: bool = True,
):
    """
    Generator to Read multiview frames from the specified data path configuration.
    """
    frame_reader_dict = {}
    for view_id in data_path_cfg.view_list:
        if use_reid:
            pkl_path = data_path_cfg.get_reid_path(view_id)
        else:
            pkl_path = data_path_cfg.get_detection_path(view_id)
        frame_reader = FrameInput(
            view_id=view_id,
            pkl_data_path=pkl_path,
            camera_path=data_path_cfg.camera_path,
            frame_downsample_rate=data_path_cfg.frame_step,
            frame_offset=view_time_offset.get(view_id, 0) if view_time_offset else 0,
        )
        frame_reader_dict[view_id] = frame_reader
    frame_id = 0
    while True:
        frame_list = []
        for view_id in data_path_cfg.view_list:
            frame = next(frame_reader_dict[view_id], None)
            if frame is None:
                break
            frame_list.append(frame)

        if len(frame_list) == len(data_path_cfg.view_list):
            yield MvFrame(data_path_cfg.view_list, frame_list, frame_id)
            frame_id += 1
        else:
            break
    return None


def get_video_sync(data_path_cfg: GamePath):
    video_sync_result_path = data_path_cfg.get_video_sync_path()
    if not os.path.exists(video_sync_result_path):
        raise FileNotFoundError(
            f"Video sync result file not found: {video_sync_result_path}"
        )
    video_time_offset = json.load(open(video_sync_result_path, "r"))
    view_frame_offset = {k: v["frame"] for k, v in video_time_offset.items()}
    return view_frame_offset


def triangulation_processor_raw(data_path_cfg: GamePath, use_reid: bool = True):
    """
    Process triangulation for player tracking data.
    """
    logger = logging.getLogger("TriangulationProcessor")
    logger.info("Starting triangulation processing...")

    view_frame_offset = get_video_sync(data_path_cfg)
    logger.info(f"Video sync offsets: {view_frame_offset}")

    raw_3d_save_path = data_path_cfg.get_prediction_save_path(prediction_type="3d-raw")
    if not os.path.exists(raw_3d_save_path):
        os.makedirs(os.path.basename(raw_3d_save_path), exist_ok=True)
        raw_frame_saver = FrameSaver(
            save_path=data_path_cfg.get_prediction_save_path(prediction_type="3d-raw"),
        )
        mv_frame_reader = multiview_frame_reader(
            data_path_cfg, view_time_offset=view_frame_offset, use_reid=use_reid
        )
    else:
        logger.info(f"Raw 3D data already exists at {raw_3d_save_path}")
        return

    camera_params = load_camera(data_path_cfg.camera_path, data_path_cfg.view_list)
    logger.info(f"Loaded camera parameters for views: {data_path_cfg.view_list}")

    player_id2name = data_path_cfg.get_player_info()

    triangulator = PlayerTriangulator(
        camera_params,
        data_path_cfg.view_list,
        pose_conversion="coco17",
        dist_max=0.1,
        num_players=len(player_id2name),
    )

    for mv_frame in tqdm.tqdm(mv_frame_reader, desc="Processing multiview frames"):
        if mv_frame is None:
            continue
        triangulator.process(mv_frame)
        mv_frame.frame_dict = None  # Clear frame_dict to save memory
        mv_frame.frames = None
        if raw_frame_saver is not None:
            raw_frame_saver.save_frame(mv_frame)

    raw_frame_saver.close()
    logger.info(
        f"Triangulation processing completed. Raw 3D data saved to {raw_3d_save_path}"
    )


def triangulation_processor_match(
    data_path_cfg: GamePath,
    run_identity_matching: bool = True,
):
    """
    Process triangulation with matching for player tracking data.
    """
    logger = logging.getLogger("TriangulationProcessor")
    logger.info("Starting triangulation processing with matching...")

    if not os.path.exists(
        data_path_cfg.get_prediction_save_path(prediction_type="3d-raw")
    ):
        logger.error(
            "Raw 3D data not found. Please run triangulation_processor_raw first."
        )
        return

    matched_3d_save_path = data_path_cfg.get_prediction_save_path(prediction_type="3d")
    if not os.path.exists(matched_3d_save_path):
        os.makedirs(os.path.dirname(matched_3d_save_path), exist_ok=True)
        matched_frame_saver = FrameSaver(
            save_path=matched_3d_save_path,
        )
        # video_3d_writer = VideoWriter(
        #     save_path=data_path_cfg.get_vis_path(prediction_type="3d"),
        #     fps=int(0.2 * data_path_cfg.fps),
        # )
        mv_frame_reader = MvFrameInput(
            data_path_cfg.get_prediction_save_path(prediction_type="3d-raw")
        )
    else:
        logger.info(f"Matched 3D data already exists at {matched_3d_save_path}")
        return

    vis_step = int(5 / data_path_cfg.frame_step)

    player_id2name = data_path_cfg.get_player_info()

    matcher = Player3DMatcher(
        track_distance=0.5,
        reid_conf_threshold=0.1,
        reid_dist_threshold=500,
        appear_frame_thr=200,
    )

    frame_list = []

    def dumps_batch():
        logger.info(f"Processing batch of {len(frame_list)} frames for matching...")
        if len(frame_list) == 0:
            return
        matcher.matching_frame_clip(frame_list)
        for i, frame in enumerate(frame_list):
            matched_frame_saver.save_frame(frame)
            # if i % vis_step == 0:
            # video_3d_writer.write(frame.visualize(player_names=player_id2name))
        matcher.tracker.reset()
        frame_list.clear()

    for i, mv_frame in tqdm.tqdm(
        enumerate(mv_frame_reader), desc="Processing multiview frames with matching"
    ):
        if mv_frame is None:
            break
        # Track players across frames (always required)
        matched_frame = matcher.prepare_frame(mv_frame)
        if run_identity_matching:
            frame_list.append(matched_frame)
            if len(frame_list) >= MAX_MATCHING_BATCH:
                dumps_batch()
        else:
            matched_frame_saver.save_frame(matched_frame)

    if run_identity_matching:
        dumps_batch()

    matched_frame_saver.close()
    # video_3d_writer.close()
    logger.info(
        f"Triangulation processing with matching completed. Matched 3D data saved to {matched_3d_save_path}"
    )


def triangulation_processor(data_path_cfg: GamePath, use_reid: bool = True):
    """
    Main function to process triangulation and matching.
    """
    triangulation_processor_raw(data_path_cfg, use_reid=use_reid)
    triangulation_processor_match(data_path_cfg, run_identity_matching=use_reid)
    logging.getLogger("TriangulationProcessor").info(
        "Triangulation processing completed successfully."
    )
