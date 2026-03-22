from typing import Dict, List, Union
import numpy as np


from src.utils.keypoints import coco17tobody25, body25tococo17

def player_to_easymocap_format(players: Dict[str, List]) \
        -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Args:
        players (List[Player]): 球员
    Returns:
    """
    annots = {}
    # only use [1,2,5,8,17,18] for 2d keypoints
    # score mask
    score_mask = np.zeros((25,), dtype=np.float32)
    score_mask[[1, 2, 5, 8, 17, 18]] = 1.0
    for camera_id, player_list in players.items():
        annots[camera_id] = []
        for player in player_list:
            k2d = player.pose
            if k2d is not None and k2d.shape[0] == 17:
                k2d = coco17tobody25(k2d)
            assert k2d.shape[0] == 25, f"keypoints should be 25, but got {k2d.shape[0]} for player {player.tracking_id} in camera {camera_id}"
            k2d[:, 2] *= score_mask  # set the score to 0 for other keypoints
            annots[camera_id].append(
                {'keypoints': k2d, 'bbox': player.bbox, 'person_id': player.tracking_id})
    return annots


def camera_to_easymocap_format(cameras: Dict[Union[str, int], np.ndarray], camera_ids: List[str]) -> Dict[
    str, Dict[str, np.ndarray]]:
    """
    Args:
        cameras (Dict[Union[str, int],  np.ndarray]): 相机
    Returns:
    """
    camera_params = {}
    num_camera = len(cameras['K'])
    for i in range(num_camera):
        camera_params[camera_ids[i]] = {'K': cameras['K'][i], 'dist': cameras['dist'][i], 'R': cameras['R'][i],
                                        'T': cameras['T'][i], 'P': cameras['P'][i], "invK": cameras['invK'][i]}

    return camera_params


def easymocap_result_to_player(result:dict):
    keypoint3d = body25tococo17(result['keypoints3d'])
    player_index_in_view = result['indices']
    return keypoint3d, player_index_in_view