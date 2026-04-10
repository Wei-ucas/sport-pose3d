import os
import pickle
import logging

import numpy as np
import tqdm

from src.data_io.path_config import GamePath
from src.data_io.loader import MvFrameInput


def collect_processor(
    data_path_cfg: GamePath,
    use_reid: bool = True,
):
    """
    Collect per-player k3d joint data from the optimized MvFrame pkl file and
    save it as ``{player_id: ndarray(T, k, 4)}`` to ``<output_dir>/k3d.pkl``.

    Args:
        data_path_cfg: GamePath configuration object.
        use_reid: If True, read ``matched_player_k3d``; otherwise ``tracked_player_k3d``.
    """
    output_path = os.path.join(data_path_cfg.output_dir, "k3d.pkl")
    if os.path.exists(output_path):
        logging.getLogger("collect_processor").info(
            f"k3d output {output_path} already exists, skipping."
        )
        return

    logger = logging.getLogger("collect_processor")

    opt_3d_path = data_path_cfg.get_prediction_save_path("3d-opt")
    if not os.path.exists(opt_3d_path):
        logger.error(f"Optimized 3D path {opt_3d_path} does not exist.")
        return

    k3d_field = "matched_player_k3d" if use_reid else "tracked_player_k3d"
    logger.info(f"Collecting k3d data from {opt_3d_path}, field='{k3d_field}'...")

    # First pass: load all frames into memory preserving frame order.
    # frames_data: list of {player_id: ndarray(k, 4)} in frame order.
    frames_data = []
    k_joints = None

    mv_frame_reader = MvFrameInput(opt_3d_path)
    for mv_frame in tqdm.tqdm(mv_frame_reader, desc="Loading frames"):
        if mv_frame is None:
            break
        pid_k3d: dict = getattr(mv_frame, k3d_field, {})
        frames_data.append(dict(pid_k3d))
        # Infer joint count from the first non-empty entry we encounter.
        if k_joints is None:
            for k3d in pid_k3d.values():
                k_joints = k3d.shape[0]
                break

    if not frames_data:
        logger.warning("No frames found in %s.", opt_3d_path)
        return

    if k_joints is None:
        logger.warning("No k3d data found in any frame; nothing to save.")
        return

    T = len(frames_data)

    # Collect all player IDs that appear at least once.
    all_player_ids = set()
    for pid_k3d in frames_data:
        all_player_ids.update(pid_k3d.keys())

    logger.info(
        f"Building k3d arrays: {len(all_player_ids)} players, T={T} frames, k={k_joints} joints."
    )

    # Build result: {player_id: ndarray(T, k, 4)} — zeros where player absent.
    result = {
        pid: np.zeros((T, k_joints, 4), dtype=np.float32) for pid in all_player_ids
    }

    for t, pid_k3d in enumerate(frames_data):
        for pid, k3d in pid_k3d.items():
            result[pid][t] = k3d

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    logger.info(f"k3d data saved to {output_path}.")
