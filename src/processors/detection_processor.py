import os.path
from typing import List, Dict, Any
import logging
import tqdm
import numpy as np
from src.data_io.path_config import GamePath
from src.structures.frame import Frame
from src.data_io.loader import FrameInput
from src.data_io.saver import FrameSaver, VideoWriter
from src.modules.player_detector.mm_player_detector import PlayerDetector


def _extract_frame_bboxes(frame: Frame) -> np.ndarray:
    bboxes = []
    for player in frame.players:
        if player.bbox is None:
            continue
        bboxes.append(player.bbox.astype(np.float32, copy=False))
    if not bboxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.stack(bboxes, axis=0)


def detection_processor(
    view_id: str,
    data_path_cfg: GamePath,
    vis: bool = False,
    pose_model: str = "rtmpose",
    pose_checkpoint: str = None,
    pose_config: str = None,
    device: str = "cuda:0",
    inherit_detection_from: str = None,
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

    inherited_detection_path = None
    if inherit_detection_from:
        inherited_detection_path = data_path_cfg.get_detection_path_for_pose_model(
            view_id=view_id,
            pose_model_name=inherit_detection_from,
        )
        if not os.path.exists(inherited_detection_path):
            raise FileNotFoundError(
                f"Inherited detection file not found for view {view_id}: {inherited_detection_path}"
            )
        logging.info(
            "View %s will reuse bbox detections from pose model '%s': %s",
            view_id,
            inherit_detection_from,
            inherited_detection_path,
        )

    frame_reader = FrameInput(
        view_id=view_id,
        pkl_data_path=inherited_detection_path,
        video_path=data_path_cfg.get_video_path(view_id),
        camera_path=data_path_cfg.camera_path,
        frame_downsample_rate=data_path_cfg.frame_step,
    )

    frame_saver = FrameSaver(
        save_path=data_path_cfg.get_prediction_save_path("detection", view_id),
    )

    if vis:
        video_writer = VideoWriter(
            save_path=data_path_cfg.get_vis_path("detection", view_id),
            fps=data_path_cfg.fps // data_path_cfg.frame_step,
        )
    else:
        video_writer = None

    player_detector = PlayerDetector(
        detection_threshold=0.2,
        tracker_iou_thr=0.2,
        device=device,
        pose_model=pose_model,
        pose_checkpoint=pose_checkpoint,
        pose_config=pose_config,
        enable_detector=inherited_detection_path is None,
    )

    if inherited_detection_path is None:
        available_frame_num = len(frame_reader) // data_path_cfg.frame_step
    else:
        available_frame_num = len(frame_reader)
    if data_path_cfg.max_frame_num == -1:
        max_frame = available_frame_num
    else:
        max_frame = min(data_path_cfg.max_frame_num, available_frame_num)

    for i in tqdm.tqdm(range(max_frame), desc=f"Processing {view_id}"):
        frame = next(frame_reader)
        if frame is None:
            logging.warning(f"Frame {i} is None, stop.")
            break

        if frame.frame_id % data_path_cfg.frame_step != 0:
            continue

        inherited_bboxes = None
        if inherited_detection_path is not None:
            inherited_bboxes = _extract_frame_bboxes(frame)
            frame = Frame(
                image=frame.image,
                frame_id=frame.frame_id,
                camera_params=frame.camera_params,
            )

        frame = player_detector.process(frame, inherited_bboxes=inherited_bboxes)

        if vis and video_writer:
            annotated_frame = frame.visualize()
            video_writer.write(annotated_frame)

        frame_saver.save_frame(frame)

    if video_writer:
        video_writer.close()
    frame_saver.close()

    logging.info(f"Processed {view_id} with {len(frame_reader)} frames.")
