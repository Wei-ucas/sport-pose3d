import json
import logging
import os
import pickle

import cv2

from src.data_io.path_config import GamePath


def _get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def _load_frame_offsets(data_path_cfg: GamePath):
    sync_path = data_path_cfg.get_video_sync_path()
    if not os.path.exists(sync_path):
        return {view_id: 0 for view_id in data_path_cfg.view_list}, None

    with open(sync_path, "r", encoding="utf-8") as f:
        sync_data = json.load(f)
    return {
        view_id: int(sync_data.get(view_id, {}).get("frame", 0))
        for view_id in data_path_cfg.view_list
    }, sync_path


def _load_k3d_length(k3d_path: str) -> int:
    with open(k3d_path, "rb") as f:
        k3d_data = pickle.load(f)

    if not k3d_data:
        return 0

    first_player = next(iter(k3d_data.values()))
    return int(first_player.shape[0])


def validate_k3d_length_processor(data_path_cfg: GamePath):
    """
    Validate that the collected k3d frame length matches the synchronized video length.

    Expected synchronized length is defined as:
        min_i(video_num_frames_i - sync_frame_offset_i)

    For frame_step > 1 or max_frame_num != -1, the expected processed length is further
    adjusted to align with the current pipeline configuration.
    """
    logger = logging.getLogger("validate_k3d_length_processor")

    k3d_path = os.path.join(data_path_cfg.output_dir, "k3d.pkl")
    if not os.path.exists(k3d_path):
        raise FileNotFoundError(f"k3d file not found: {k3d_path}")

    offsets, sync_path = _load_frame_offsets(data_path_cfg)
    k3d_length = _load_k3d_length(k3d_path)

    per_view_stats = {}
    synced_lengths = []
    processed_lengths = []

    for view_id in data_path_cfg.view_list:
        video_path = data_path_cfg.get_video_path(view_id)
        total_frames = _get_video_frame_count(video_path)
        offset = offsets.get(view_id, 0)
        synced_length = max(0, total_frames - offset)

        processed_length = synced_length
        if data_path_cfg.frame_step > 1:
            processed_length = synced_length // data_path_cfg.frame_step
        if data_path_cfg.max_frame_num != -1:
            processed_length = min(processed_length, data_path_cfg.max_frame_num)

        per_view_stats[view_id] = {
            "video_path": video_path,
            "total_frames": total_frames,
            "sync_offset": offset,
            "synced_length": synced_length,
            "processed_length": processed_length,
        }
        synced_lengths.append(synced_length)
        processed_lengths.append(processed_length)

    expected_synced_length = min(synced_lengths) if synced_lengths else 0
    expected_k3d_length = min(processed_lengths) if processed_lengths else 0

    result = {
        "k3d_path": k3d_path,
        "video_sync_path": sync_path,
        "frame_step": data_path_cfg.frame_step,
        "max_frame_num": data_path_cfg.max_frame_num,
        "k3d_length": k3d_length,
        "expected_synced_video_length": expected_synced_length,
        "expected_k3d_length": expected_k3d_length,
        "match": k3d_length == expected_k3d_length,
        "views": per_view_stats,
    }

    report_path = os.path.join(data_path_cfg.output_dir, "k3d_length_check.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        "K3D length check: actual=%d, expected=%d, synced_video_min=%d, report=%s",
        k3d_length,
        expected_k3d_length,
        expected_synced_length,
        report_path,
    )

    if k3d_length != expected_k3d_length:
        raise ValueError(
            "K3D frame length mismatch: "
            f"actual={k3d_length}, expected={expected_k3d_length}, "
            f"synced_video_min={expected_synced_length}. "
            f"See {report_path} for details."
        )
