import os.path
from typing import List, Dict, Any
import logging
import tqdm
from src.data_io.path_config import GamePath
from src.structures.frame import Frame
from src.data_io.loader import FrameInput
from src.data_io.saver import FrameSaver, VideoWriter
from src.modules.player_detector.mm_player_detector import PlayerDetector


def detection_processor(
    view_id: str,
    data_path_cfg: GamePath,
    vis: bool = False,
):
    """
    Process a single view for player detection.

    Args:
        view_id (int): The ID of the view to process.
        data_path_cfg (dict): Configuration dictionary containing paths and other settings.
        vis (bool): Whether to visualize the results.

    Returns:
        Frame: The processed frame with detected players.
    """
    if os.path.exists(data_path_cfg.get_prediction_save_path("detection", view_id)):
        logging.info(f"View {view_id} already processed, skipping.")
        return

    frame_reader = FrameInput(
        view_id=view_id,
        video_path = data_path_cfg.get_video_path(view_id),
        camera_path= data_path_cfg.camera_path,
        frame_downsample_rate=data_path_cfg.frame_step
    )

    frame_saver = FrameSaver(
        save_path=data_path_cfg.get_prediction_save_path("detection", view_id),
    )

    if vis:
        video_writer = VideoWriter(
            save_path=data_path_cfg.get_vis_path("detection", view_id),
            fps=data_path_cfg.fps//data_path_cfg.frame_step,
        )
    else:
        video_writer = None

    player_detector = PlayerDetector(
        detection_threshold=0.2,
        tracker_iou_thr=0.2,
        device="cuda:0"
    )

    max_frame = len(frame_reader) // data_path_cfg.frame_step if data_path_cfg.max_frame_num == -1 else data_path_cfg.max_frame_num

    for i in tqdm.tqdm(range(max_frame), desc=f"Processing {view_id}"):
        frame = next(frame_reader)
        if frame is None:
            logging.warning(f"Frame {i} is None, stop.")
            break

        if frame.frame_id % data_path_cfg.frame_step != 0:
            continue

        frame = player_detector.process(frame)

        if vis and video_writer:
            annotated_frame = frame.visualize()
            video_writer.write(annotated_frame)

        frame_saver.save_frame(frame)

    if video_writer:
        video_writer.close()
    frame_saver.close()

    logging.info(f"Processed {view_id} with {len(frame_reader)} frames.")


