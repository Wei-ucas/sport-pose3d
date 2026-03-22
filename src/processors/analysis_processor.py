import os.path
from typing import List, Dict, Any
import logging

import numpy as np
import tqdm
import json

from src.data_io.path_config import GamePath, read_player_names
from src.data_io.loader import FrameInput, MvFrameInput


def speed_analysis(locations: List[List[float]], time_resolution: float = 0.2, sprint_threshold=4) -> Dict[str, Any]:
    """
    Analyze the speed of a player based on their locations.
    :param locations: List of player locations, each location is a list of [x, y, z] coordinates.
    :param time_resolution: Time resolution in seconds.
    :return: Dictionary containing speed statistics.
    """
    if len(locations) < 2:
        return {}

    locations = np.asarray(locations)
    distances = np.linalg.norm(np.diff(locations, axis=0), axis=1)  # N-1
    distances = np.concatenate([[0], distances])  # N
    speeds = distances / time_resolution  # N

    sprint_ = (speeds[1:-1] > sprint_threshold) * (speeds[1:-1] > speeds[:-2]) * (speeds[1:-1] > speeds[2:])
    sprint_times = int(np.sum(sprint_))

    # distance on speed level
    speed_level_threshold = [0.3, 2, 4, 100]
    distance_speed_level = [0,0,0]
    for i in range(len(speed_level_threshold)-1):
        distance_speed_level[i] = np.sum(distances[(speeds >= speed_level_threshold[i]) & (speeds < speed_level_threshold[i+1])])
        distance_speed_level[i] = np.round(distance_speed_level[i], decimals=1)

    total_distance = np.sum(distance_speed_level)
    # cumulative distance
    cumulative_distance = np.cumsum(distances)  # N

    # acceleration
    accel = np.diff(speeds, prepend=speeds[0]) / time_resolution  # N
    accel = np.round(accel, decimals=1)

    return {
        "speed": np.round(speeds, decimals=1).tolist(),
        "sprint_times": sprint_times,
        "distance_speed_level": distance_speed_level,
        "total_distance": np.round(total_distance, decimals=1).item(),
        "acceleration": accel.tolist(),
        "cumulative_distance": np.round(cumulative_distance, decimals=1).tolist(),
    }


def load_analysis(locations, time_resolution=0.2):
    # Compute the load of a player based on their locations
    if len(locations) < 2:
        return 0.0
    x_y_diff = np.diff(locations, axis=0, prepend=locations[0:1])  # N x 2, difference in x and y coordinates
    x_y_speed = x_y_diff / time_resolution  # N x 2
    x_y_accel = np.diff(x_y_speed, axis=0, prepend=x_y_speed[0:1]) / time_resolution  # N x 2

    unit_load = np.linalg.norm(x_y_accel, axis=1)/100  # N

    speed = np.linalg.norm(x_y_speed, axis=1)  # N
    unit_load[speed < 0.3] = 0  # If speed is less than 0.5, set load to 0

    total_load = np.sum(unit_load)  # Total load over the time period
    accumulated_load = np.cumsum(unit_load)  # Cumulative load over time

    # second load, the sum of load for 1 second, padding to make it even for 1/time_resolution
    padding_length = np.ceil(np.ceil(len(unit_load) * time_resolution) / time_resolution)
    padded_load = np.pad(unit_load, (0, int(padding_length) - len(unit_load)), 'constant', constant_values=0)
    second_load = padded_load.reshape(-1, int(1 / time_resolution)).sum(axis=1)  # Reshape to 1 second intervals

    # load_level: load / (time in minutes)
    running_time = sum(speed >= 0.5) * time_resolution / 60  # in minutes
    load_level = total_load / running_time if running_time > 0 else 0


    return {
        "total_load": np.round(total_load, decimals=1).item(),
        "accumulated_load": np.round(accumulated_load, decimals=1).tolist(),
        "second_load": np.round(second_load, decimals=1).tolist(),
        "load_level": np.round(load_level, decimals=1).item(),
    }


def analysis_processor(data_path_cfg: GamePath,
                       time_resolution: float = 0.2
                       ):
    """
    analysis the statistics of each player
    including:
    player location
    player speed
    player load
    player running distance
    """
    logger = logging.getLogger("analysis_processor")

    if not os.path.exists(data_path_cfg.get_prediction_save_path("3d-opt")):
        logger.error("3d-opt prediction not found, please run 3d-opt first")
        return
    opt_mv_inputs = MvFrameInput(mv_data_path=data_path_cfg.get_prediction_save_path("3d-opt"))

    player_data = {}
    player_id2name = data_path_cfg.get_player_info()

    player_location = {
        int(player_id): [] for player_id in player_id2name.keys()
    }
    for frame in tqdm.tqdm(opt_mv_inputs, desc="Processing player locations"):
        for pid in player_location.keys():
            player_location[pid].append([0, 0, 0])  # Initialize with zeros for all players
        for player_id, loc in frame.player_location.items():
            # if player_id not in player_location:
            #     player_location[player_id] = []
            # player_location[int(player_id)].append(loc)
            player_location[int(player_id)][-1] = loc[:3]  # Update the last location with the current location

    down_sample_step = int(data_path_cfg.fps / data_path_cfg.frame_step * time_resolution)

    # calculate statistics
    for player_id, locations in player_location.items():
        if len(locations) < 2:
            continue
        locations = np.asarray(locations)[::down_sample_step,:2] # N x 3
        # 保留1位小数
        locations = np.round(locations, decimals=1)

        # player_name = player_id2name[player_id].replace("h", "China").replace("g", "Japan")
        player_name = player_id2name[player_id]

        player_data[player_name] = {
            "locations": locations.tolist(),
            # "speed": speed.tolist(),
            # "cumulative_distance": cumulative_distance.tolist(),
            # "total_distance": float(total_distance)
        }
        player_data[player_name].update(load_analysis(locations, time_resolution))
        player_data[player_name].update(speed_analysis(locations, time_resolution))

    player_list = list(player_data.keys())
    all_data = {
        "player_list": player_list,
        "player_data": player_data,
        "time_resolution": time_resolution,
    }

    # save player data
    output_player_data_path = data_path_cfg.get_report_path
    with open(output_player_data_path, 'w') as f:
        json.dump(all_data, f, indent=4)


