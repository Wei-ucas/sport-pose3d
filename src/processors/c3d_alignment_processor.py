from __future__ import annotations

import glob
import json
import logging
import os
import pickle

import c3d
import numpy as np

from src.data_io.path_config import GamePath


K3D_JOINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

VICON_MARKER_MAP = {
    "left_shoulder": ["LSHO"],
    "right_shoulder": ["RSHO"],
    "left_elbow": ["LELB"],
    "right_elbow": ["RELB"],
    "left_wrist": ["LWRA", "LWRB"],
    "right_wrist": ["RWRA", "RWRB"],
    "left_hip": ["LASI", "LPSI"],
    "right_hip": ["RASI", "RPSI"],
    "left_knee": ["LKNE"],
    "right_knee": ["RKNE"],
    "left_ankle": ["LANK"],
    "right_ankle": ["RANK"],
}


def _guess_c3d_file_path(data_path_cfg: GamePath) -> str:
    c3d_root = os.path.join(data_path_cfg.workdir, "c3d")
    c3d_dir = os.path.join(c3d_root, data_path_cfg.game_id)
    if not os.path.isdir(c3d_dir) and os.path.isdir(c3d_root):
        target_name = data_path_cfg.game_id.lower()
        for name in os.listdir(c3d_root):
            if name.lower() == target_name:
                c3d_dir = os.path.join(c3d_root, name)
                break
    c3d_files = sorted(glob.glob(os.path.join(c3d_dir, "*.c3d")))
    if not c3d_files:
        raise FileNotFoundError(f"No c3d file found under {c3d_dir}")
    return c3d_files[0]


def _load_k3d_result(k3d_path: str):
    with open(k3d_path, "rb") as f:
        return pickle.load(f)


def _select_longest_valid_player(k3d_result: dict, confidence_threshold: float):
    best_player_id = None
    best_valid_frames = -1
    best_valid_points = -1
    player_stats = {}

    for player_id, player_k3d in k3d_result.items():
        confidence = np.asarray(player_k3d)[..., 3]
        valid_frames = int(np.any(confidence > confidence_threshold, axis=1).sum())
        valid_points = int((confidence > confidence_threshold).sum())
        player_stats[player_id] = {
            "valid_frames": valid_frames,
            "valid_points": valid_points,
        }
        if (
            valid_frames > best_valid_frames
            or (valid_frames == best_valid_frames and valid_points > best_valid_points)
        ):
            best_player_id = player_id
            best_valid_frames = valid_frames
            best_valid_points = valid_points

    if best_player_id is None:
        raise ValueError("No valid player found in k3d result")
    return best_player_id, np.asarray(k3d_result[best_player_id]), player_stats


def _extract_k3d_joint_subset(player_k3d: np.ndarray, joint_names, confidence_threshold: float):
    joints = np.stack([player_k3d[:, K3D_JOINT_INDEX[name], :3] for name in joint_names], axis=1)
    valid = np.stack(
        [player_k3d[:, K3D_JOINT_INDEX[name], 3] > confidence_threshold for name in joint_names],
        axis=1,
    )
    joints = joints.astype(np.float64)
    joints[~valid] = np.nan
    return joints, valid


def _load_c3d_markers(c3d_path: str):
    with open(c3d_path, "rb") as handle:
        reader = c3d.Reader(handle)
        point_rate = float(reader.point_rate)
        labels = [label.strip() for label in reader.point_labels]
        frames = []
        for _, points, _ in reader.read_frames():
            frames.append(np.asarray(points[:, :3], dtype=np.float64))
    if not frames:
        raise ValueError(f"No frames found in c3d file: {c3d_path}")
    return labels, np.stack(frames, axis=0), point_rate


def _extract_c3d_joint_subset(labels, marker_xyz: np.ndarray, joint_names):
    label_to_index = {label: idx for idx, label in enumerate(labels) if label}
    joints = []
    valid_joint_names = []
    for joint_name in joint_names:
        markers = [name for name in VICON_MARKER_MAP[joint_name] if name in label_to_index]
        if not markers:
            continue
        marker_stack = np.stack([marker_xyz[:, label_to_index[name], :] for name in markers], axis=1)
        joint_xyz = np.nanmean(marker_stack, axis=1)
        joints.append(joint_xyz)
        valid_joint_names.append(joint_name)
    if not joints:
        raise ValueError("No overlapping c3d markers found for target joint mapping")
    joint_array = np.stack(joints, axis=1)
    valid = np.isfinite(joint_array).all(axis=2)
    return valid_joint_names, joint_array, valid


def _resample_sequence(data: np.ndarray, target_length: int):
    if len(data) == target_length:
        return data.copy()
    source_x = np.linspace(0.0, 1.0, len(data))
    target_x = np.linspace(0.0, 1.0, target_length)
    flat = data.reshape(len(data), -1)
    out = np.empty((target_length, flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        column = flat[:, i]
        finite = np.isfinite(column)
        if finite.sum() < 2:
            out[:, i] = np.nan
            continue
        out[:, i] = np.interp(target_x, source_x[finite], column[finite])
    return out.reshape((target_length,) + data.shape[1:])


def _motion_signal(joints: np.ndarray):
    velocity = np.diff(joints, axis=0)
    speed = np.linalg.norm(velocity, axis=2)
    signal = np.nanmedian(speed, axis=1)
    signal = np.nan_to_num(signal, nan=0.0)
    if len(signal) >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        signal = np.convolve(signal, kernel, mode="same")
    return signal


def _normalized_correlation(signal_a: np.ndarray, signal_b: np.ndarray, shift: int):
    if shift >= 0:
        start_a = 0
        start_b = shift
    else:
        start_a = -shift
        start_b = 0
    overlap = min(len(signal_a) - start_a, len(signal_b) - start_b)
    if overlap < 10:
        return -np.inf, 0
    a = signal_a[start_a: start_a + overlap]
    b = signal_b[start_b: start_b + overlap]
    if len(a) < 10 or len(b) < 10:
        return -np.inf, 0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return -np.inf, len(a)
    return float(np.dot(a, b) / denom), len(a)


def _estimate_time_offset(k3d_joints: np.ndarray, c3d_joints_resampled: np.ndarray):
    signal_k3d = _motion_signal(k3d_joints)
    signal_c3d = _motion_signal(c3d_joints_resampled)
    max_shift = min(len(signal_k3d), len(signal_c3d)) // 3
    best_shift = 0
    best_score = -np.inf
    best_overlap = 0
    for shift in range(-max_shift, max_shift + 1):
        score, overlap = _normalized_correlation(signal_k3d, signal_c3d, shift)
        if score > best_score:
            best_score = score
            best_shift = shift
            best_overlap = overlap
    return best_shift, best_score, best_overlap


def _collect_correspondences(
    k3d_joints: np.ndarray,
    k3d_valid: np.ndarray,
    c3d_joints: np.ndarray,
    c3d_valid: np.ndarray,
    offset_frames: int,
):
    src_points = []
    dst_points = []
    frame_pairs = []
    for k3d_frame in range(len(k3d_joints)):
        c3d_frame = k3d_frame + offset_frames
        if c3d_frame < 0 or c3d_frame >= len(c3d_joints):
            continue
        valid = k3d_valid[k3d_frame] & c3d_valid[c3d_frame]
        if not np.any(valid):
            continue
        src_points.append(c3d_joints[c3d_frame, valid])
        dst_points.append(k3d_joints[k3d_frame, valid])
        frame_pairs.append((k3d_frame, c3d_frame, int(valid.sum())))
    if not src_points:
        raise ValueError("No valid temporal overlap found between c3d and k3d joints")
    return np.concatenate(src_points, axis=0), np.concatenate(dst_points, axis=0), frame_pairs


def _fit_similarity_transform(src: np.ndarray, dst: np.ndarray):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    s = np.diag(sign)
    rotation = u @ s @ vt
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = float(np.sum(singular_values * sign) / max(src_var, 1e-8))
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def _apply_similarity_transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray):
    reshaped = points.reshape(-1, 3)
    transformed = (scale * (rotation @ reshaped.T)).T + translation
    return transformed.reshape(points.shape)


def _rmse(src: np.ndarray, dst: np.ndarray):
    return float(np.sqrt(np.mean(np.sum((src - dst) ** 2, axis=1))))


def c3d_alignment_processor(
    data_path_cfg: GamePath,
    c3d_file_path: str | None = None,
    confidence_threshold: float = 0.1,
):
    logger = logging.getLogger("c3d_alignment_processor")

    k3d_path = os.path.join(data_path_cfg.output_dir, "k3d.pkl")
    if not os.path.exists(k3d_path):
        raise FileNotFoundError(f"k3d file not found: {k3d_path}")

    c3d_file_path = c3d_file_path or _guess_c3d_file_path(data_path_cfg)
    if not os.path.exists(c3d_file_path):
        raise FileNotFoundError(f"c3d file not found: {c3d_file_path}")

    k3d_result = _load_k3d_result(k3d_path)
    selected_player_id, selected_player_k3d, player_stats = _select_longest_valid_player(
        k3d_result, confidence_threshold
    )

    requested_joint_names = list(K3D_JOINT_INDEX.keys())
    c3d_labels, c3d_markers, c3d_fps = _load_c3d_markers(c3d_file_path)
    valid_joint_names, c3d_joints, c3d_valid = _extract_c3d_joint_subset(
        c3d_labels, c3d_markers, requested_joint_names
    )
    k3d_joints, k3d_valid = _extract_k3d_joint_subset(
        selected_player_k3d, valid_joint_names, confidence_threshold
    )

    k3d_fps = float(data_path_cfg.fps) / max(1, data_path_cfg.frame_step)
    target_c3d_length = max(2, int(round(len(c3d_joints) * k3d_fps / c3d_fps)))
    c3d_joints_resampled = _resample_sequence(c3d_joints, target_c3d_length)
    c3d_valid_resampled = np.isfinite(c3d_joints_resampled).all(axis=2)

    time_offset_frames, time_corr, time_overlap = _estimate_time_offset(
        k3d_joints, c3d_joints_resampled
    )

    src_points, dst_points, frame_pairs = _collect_correspondences(
        k3d_joints,
        k3d_valid,
        c3d_joints_resampled,
        c3d_valid_resampled,
        time_offset_frames,
    )
    scale, rotation, translation = _fit_similarity_transform(src_points, dst_points)
    aligned_src = _apply_similarity_transform(src_points, scale, rotation, translation)

    report = {
        "k3d_path": k3d_path,
        "c3d_file_path": c3d_file_path,
        "selected_player_id": selected_player_id,
        "confidence_threshold": confidence_threshold,
        "joint_names": valid_joint_names,
        "player_stats": player_stats,
        "k3d_fps": k3d_fps,
        "c3d_fps": c3d_fps,
        "k3d_num_frames": int(len(k3d_joints)),
        "c3d_num_frames": int(len(c3d_joints)),
        "c3d_resampled_num_frames": int(len(c3d_joints_resampled)),
        "time_alignment": {
            "c3d_to_k3d_offset_frames": int(time_offset_frames),
            "c3d_to_k3d_offset_seconds": float(time_offset_frames / k3d_fps),
            "correlation": float(time_corr),
            "overlap_frames": int(time_overlap),
        },
        "spatial_alignment": {
            "num_correspondences": int(len(src_points)),
            "scale": float(scale),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "rmse_before": _rmse(src_points, dst_points),
            "rmse_after": _rmse(aligned_src, dst_points),
        },
    }

    result = {
        "joint_names": valid_joint_names,
        "selected_player_id": selected_player_id,
        "k3d_joints": k3d_joints,
        "k3d_valid": k3d_valid,
        "c3d_joints": c3d_joints,
        "c3d_joints_resampled": c3d_joints_resampled,
        "c3d_valid_resampled": c3d_valid_resampled,
        "time_offset_frames": time_offset_frames,
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "frame_pairs": frame_pairs,
    }

    report_path = os.path.join(data_path_cfg.output_dir, "c3d_alignment_report.json")
    result_path = os.path.join(data_path_cfg.output_dir, "c3d_alignment.pkl")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    logger.info(
        "C3D alignment saved: player=%s, joints=%d, offset=%d frames, report=%s",
        selected_player_id,
        len(valid_joint_names),
        time_offset_frames,
        report_path,
    )
    return report