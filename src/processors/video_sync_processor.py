import os.path
from typing import List, Dict, Any
import logging
import tqdm
import json

from src.data_io.path_config import GamePath
from src.modules.video_syncer.video_sync import VideoSync


def video_sync_processor(
        data_path_cfg: GamePath,
):
    """
    Process video synchronization for a game.

    Args:
        game_id (str): The ID of the game.
        game_path (GamePath): The path configuration for the game.
        player_profile_path (str): Path to the player profile file.
        player_name_dict_path (str, optional): Path to the player name dictionary file.
        max_frame (int, optional): Maximum number of frames to process. Defaults to -1 (process all).
        step (int, optional): Step size for processing frames. Defaults to 1.

    Returns:
        List[Dict[str, Any]]: List of processed frame data.
    """
    logging.info(f"Processing video synchronization for game {data_path_cfg.game_id}")

    video_sync_result_path = data_path_cfg.get_video_sync_path()
    if os.path.exists(video_sync_result_path):
        logging.info(f"Video sync result already exists at {video_sync_result_path}, skipping processing.")
        return

    video_syncer = VideoSync(
        view_ids=data_path_cfg.view_list,
        video_fps=data_path_cfg.fps,
    )

    video_dict = {
        view_id: data_path_cfg.get_video_path(view_id)
        for view_id in data_path_cfg.view_list
    }

    frame_offset, time_offset = video_syncer.process(video_dict)
    frame_offset_dict = {view: {"frame": int(fo), "time": to} for view, fo, to in
                         zip(data_path_cfg.view_list, frame_offset, time_offset)}

    with open(video_sync_result_path, 'w') as f:
        json.dump(frame_offset_dict, f, indent=4)

    logging.info(f"Video synchronization data saved to {video_sync_result_path}")
