import logging
import numpy as np
from typing import Union, List

from ..base_module import BaseModule
from src.structures.multiview_frame import MvFrame


def interpolate_np_data(data: np.ndarray) -> np.ndarray:
    """Interpolate data in type ndarray, nan value will be set by
    interpolation.

    Args:
        data (np.ndarray):
            Points data in shape [n_frame, n_point, point_dim].

    Returns:
        np.ndarray:
            The interpolation result.
    """
    ret_data = np.apply_along_axis(__interpolate_np_nan__, 0, data)
    return ret_data


def __interpolate_np_nan__(data):
    # True if nan, False otherwise
    nan_mask = np.isnan(data)
    ret_data = np.copy(data)
    try:
        ret_data[nan_mask] = np.interp(
            np.nonzero(nan_mask)[0], np.nonzero(~nan_mask)[0], data[~nan_mask]
        )
    except ValueError:
        pass
    return ret_data


def count_masked_nan(points: np.ndarray, mask: np.ndarray) -> int:
    """Count how many points are nan after the mask applied.

    Args:
        points (np.ndarray):
            In shape [frame_n, person_n, dim].
        mask (np.ndarray):
            In shape [frame_n, person_n].

    Returns:
        int: number of np.nan whose mask is 1.
    """
    squeezed_points = np.sum(points, axis=-1, keepdims=False)
    count = np.count_nonzero(np.logical_and(np.isnan(squeezed_points), mask != 0))
    return count


class KalmanSmoother:

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 5e-2,
        logger: Union[None, str, logging.Logger] = None,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def _smooth_series(self, series: np.ndarray) -> np.ndarray:
        valid_mask = ~np.isnan(series).any(axis=1)
        if valid_mask.sum() < 2:
            return series

        smoothed = series.copy()
        state = np.concatenate([series[valid_mask][0], np.zeros(series.shape[1], dtype=series.dtype)])
        covariance = np.eye(series.shape[1] * 2, dtype=series.dtype)
        transition = np.block([
            [np.eye(series.shape[1], dtype=series.dtype), np.eye(series.shape[1], dtype=series.dtype)],
            [np.zeros((series.shape[1], series.shape[1]), dtype=series.dtype), np.eye(series.shape[1], dtype=series.dtype)],
        ])
        observation = np.concatenate(
            [np.eye(series.shape[1], dtype=series.dtype), np.zeros((series.shape[1], series.shape[1]), dtype=series.dtype)],
            axis=1,
        )
        process_cov = np.eye(series.shape[1] * 2, dtype=series.dtype) * self.process_noise
        measurement_cov = np.eye(series.shape[1], dtype=series.dtype) * self.measurement_noise

        for frame_idx in range(series.shape[0]):
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_cov

            if valid_mask[frame_idx]:
                measurement = series[frame_idx]
                innovation = measurement - observation @ state
                innovation_cov = observation @ covariance @ observation.T + measurement_cov
                kalman_gain = covariance @ observation.T @ np.linalg.pinv(innovation_cov)
                state = state + kalman_gain @ innovation
                covariance = (np.eye(covariance.shape[0], dtype=series.dtype) - kalman_gain @ observation) @ covariance
                smoothed[frame_idx] = state[:series.shape[1]]

        return smoothed

    def smooth(self, trajectories: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        smoothed = trajectories.copy()
        entity_count = trajectories.shape[1]
        for entity_idx in range(entity_count):
            entity_track = trajectories[:, entity_idx].copy()
            entity_valid = valid_mask[:, entity_idx] > 0
            if entity_valid.sum() < 2:
                continue
            entity_track[~entity_valid] = np.nan
            smoothed[:, entity_idx] = self._smooth_series(entity_track)
        return smoothed


class NanInterpolation:

    def __init__(
        self, verbose: bool = True, logger: Union[None, str, logging.Logger] = None
    ) -> None:
        """Assign keypoints3d values by interpolation, replace nan points.

        Args:
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def optimize_trajectory(
        self, trajectories: np.ndarray, trajectory_mask: np.ndarray, **kwargs: dict
    ) -> np.ndarray:
        """Forward function of keypoints3d optimizer.

        Args:
            trajectories (Keypoints): Input keypoints3d.
        kwargs:
            Redundant keyword arguments to be
            ignored.

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        # if keypoints3d.dtype == 'numpy':
        keypoints3d_np = trajectories
        frame_number, person_number, _ = keypoints3d_np.shape
        # else:
        #     keypoints3d_np = keypoints3d.to_numpy()
        #     self.logger.warning(
        #         'NanInterpolation only support numpy kps for now,' +
        #         ' the input kps has been converted to numpy.')
        total_nan_count = 0
        interp_nan_count = 0
        ret_keypoints3d = keypoints3d_np.copy()
        ret_kps_arr = ret_keypoints3d
        for person_idx in range(person_number):
            kps_arr = keypoints3d_np[:, person_idx, ...]
            mask = trajectory_mask[:, person_idx, ...]
            kps_interp = interpolate_np_data(kps_arr)
            ret_kps_arr[:, person_idx, ...] = kps_interp
            # record nan
            input_nan_count = count_masked_nan(kps_arr, mask)
            output_nan_count = count_masked_nan(kps_interp, mask)
            total_nan_count += input_nan_count
            interp_nan_count += input_nan_count - output_nan_count
        ret_keypoints3d = ret_kps_arr
        self.logger.info(
            f"How many nans are found after mask: {total_nan_count}"
            + f"How many nans are interpolated: {interp_nan_count}"
        )
        return ret_keypoints3d


class TrajectoryOptimizer:

    def __init__(
        self, n_max_frame: int = 9, logger: Union[None, str, logging.Logger] = None
    ) -> None:
        """Look for kps3d that deviate from the trajectory, and replace it by
        interpolation.

        Args:
            n_max_frame (int, optional):
                Find the maximum range of valid points. Defaults to 9.
            verbose (bool, optional):
                Whether to log info.
                Defaults to True.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.n_max_frame = n_max_frame
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def optimize_trajectory(
        self, trajectories: np.ndarray, trajectory_masks: np.ndarray, **kwargs: dict
    ) -> np.ndarray:
        """Forward function of keypoints3d optimizer.

        Args:
            trajectories (Keypoints): Input keypoints3d. n frame x n person x 3
        kwargs:
            Redundant keyword arguments to be
            ignored.

        Returns:
            Keypoints: The optimized keypoints3d.
        """
        trajectories_np = trajectories[:, :, None]
        ret_trajectories = trajectories_np.copy()
        ret_location_arr = ret_trajectories
        frame_number, person_number, _, _ = trajectories_np.shape
        for person_idx in range(person_number):
            location_arr = trajectories_np[:, person_idx]
            location_mask = trajectory_masks[:, person_idx]
            optimized_kps3d = self.check_kps3d(location_arr, location_mask)
            ret_location_arr[:, person_idx] = optimized_kps3d
        return ret_location_arr[:, :, 0]

    def check_kps3d(self, kps3d_arr: np.ndarray, kps3d_mask: np.ndarray) -> np.ndarray:
        kps3d = kps3d_arr[..., :2]
        n_frame = kps3d_arr.shape[0]
        n_kps3d = kps3d_arr.shape[1]
        # person_nan = np.where(np.sum(kps3d_mask, axis=1) == 0)[0]
        # kps3d[person_nan] = np.nan

        for frame_idx in range(2, n_frame - 1):
            for kps3d_idx in range(n_kps3d):
                if np.isnan(kps3d[frame_idx, kps3d_idx]).all():
                    continue
                calc_curr_dist = True
                for i in range(1, self.n_max_frame):
                    if frame_idx - i < 1:
                        break
                    curr_dist = (
                        np.linalg.norm(
                            kps3d[frame_idx, kps3d_idx]
                            - kps3d[frame_idx - i, kps3d_idx],
                            ord=2,
                        )
                        / i
                    )
                    if curr_dist > 0:
                        for j in range(
                            frame_idx - i - 1, frame_idx - i - self.n_max_frame, -1
                        ):
                            if not np.isnan(kps3d[j, kps3d_idx]).all():
                                dist_threshold = (
                                    2
                                    * np.linalg.norm(
                                        kps3d[frame_idx - i, kps3d_idx]
                                        - kps3d[j, kps3d_idx],
                                        ord=2,
                                    )
                                    / (frame_idx - i - j)
                                )
                                if curr_dist > dist_threshold:
                                    kps3d[frame_idx, kps3d_idx] = np.nan
                                calc_curr_dist = False
                                break
                    if not calc_curr_dist:
                        break
        kps3d_score = kps3d_arr[..., 2:3]
        # kps3d_score[person_nan] = np.nan
        return np.concatenate((kps3d, kps3d_score), axis=-1)


class K3dFilter:

    def __init__(self, logger: Union[None, str, logging.Logger] = None):
        """Filter the keypoints3d by several criteria."""
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        # self.parent_ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14], dtype=int)
        # self.limb_length_threshold = 0.9
        self.x_range = [-2, 29.1]
        self.y_range = [-2, 16.0]
        self.z_range = [0.0, 5.0]

        # self.torso_index = np.array([[5,6], [11,12]])
        # self.torso_length_range = (0.0,0.9)

        # self.shoulder_index = [5, 6]
        # self.waist_index = [11, 12]
        # self.shoulder_waist_length_range = (0.1, 0.5)
        #
        # self.min_num_kps = 7

    # def torso_length_filter(self, k3ds):
    #     torso_length = np.linalg.norm(k3ds[:,:,self.torso_index[0]] - k3ds[:,:,self.torso_index[1]], axis=-1)
    #     torsor_invalid = np.logical_or(torso_length.min(-1) < self.torso_length_range[0], torso_length.max(-1) > self.torso_length_range[1], np.isnan(torso_length).any(-1))
    #
    #     shoulder_waist_length = np.linalg.norm(k3ds[:,:,self.torso_index[:,0]] - k3ds[:,:,self.torso_index[:,1]], axis=-1)
    #     shoulder_waist_invalid = np.logical_or(shoulder_waist_length.min(-1) < self.shoulder_waist_length_range[0], shoulder_waist_length.max(-1) > self.shoulder_waist_length_range[1], np.isnan(shoulder_waist_length).any(-1))
    #
    #     invalid_k3ds = np.logical_or(torsor_invalid, shoulder_waist_invalid)
    #
    #     # invalid_k3ds = np.any(invalid, axis=-1)
    #     if np.any(invalid_k3ds):
    #         print(f'torso length filter: {np.sum(invalid_k3ds)} invalid k3ds')
    #     k3ds[invalid_k3ds] = np.nan
    #     return k3ds

    # def num_keypoints_filter(self, k3ds):
    #     invalid_num_k3d = np.sum(np.isnan(k3ds).sum(-1) > 0, axis=-1) > self.min_num_kps
    #     if np.any(invalid_num_k3d):
    #         print(f'num keypoints filter: {np.sum(invalid_num_k3d)} invalid k3ds')
    #     k3ds[invalid_num_k3d] = np.nan
    #     return k3ds
    #
    # def limb_length_filter(self, k3ds):
    #     # k3ds  n_frame x n_person x n_kps x 3
    #     parent_k3ds = k3ds[:, :, self.parent_ids]
    #     limb_length = np.linalg.norm(k3ds - parent_k3ds, axis=-1)
    #     invalid = limb_length > self.limb_length_threshold
    #     invalid_k3ds = np.any(invalid, axis=-1)
    #     if np.any(invalid):
    #         print(f'limb length filter: {np.sum(invalid_k3ds)} invalid k3ds')
    #     k3ds[invalid_k3ds] = np.nan
    #     return k3ds

    def court_filter(self, k3ds):
        invalid = np.logical_or(
            np.logical_or(
                k3ds[..., 0] < self.x_range[0], k3ds[..., 0] > self.x_range[1]
            ),
            np.logical_or(
                k3ds[..., 1] < self.y_range[0], k3ds[..., 1] > self.y_range[1]
            ),
            # np.logical_or(k3ds[..., 2] < self.z_range[0], k3ds[..., 2] > self.z_range[1])
        )
        invalid_k3ds = np.any(invalid, axis=-1)
        if np.any(invalid):
            print(f"court filter: {np.sum(invalid_k3ds)} invalid k3ds")
        k3ds[invalid_k3ds] = np.nan
        return k3ds

    def filter(self, k3ds):
        k3ds = self.court_filter(k3ds)
        k3ds = self.continuous_filter(k3ds)
        return k3ds

    def continuous_filter(self, k3ds):
        n_frame, n_person, _ = k3ds.shape
        pre_kpt = k3ds[0, :].copy()
        num_missing = np.zeros((k3ds.shape[1]))
        for i in range(1, n_frame):
            for j in range(n_person):
                if np.isnan(pre_kpt[j]).sum() > 0:
                    if np.isnan(k3ds[i, j]).sum() > 0:
                        continue
                    pre_kpt[j] = k3ds[i, j].copy()
                    continue
                # if i==11310 and j==7:
                #     print('debug')
                dist = np.linalg.norm(k3ds[i, j] - pre_kpt[j], axis=-1)  # a number
                # ignore nan
                dist = dist[~np.isnan(dist)]
                if dist.shape[0] == 0:
                    num_missing[j] += 1
                    continue
                if dist.min() > 2.0 and num_missing[j] < 5:
                    k3ds[i, j] = np.nan
                    num_missing[j] += 1
                else:
                    pre_kpt[j] = k3ds[i, j].copy()
                    num_missing[j] = 0
        return k3ds


class PlayerOptimizer(BaseModule):

    def __init__(
        self,
        trajectory_range: int = 5,
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 5e-2,
        one_player: bool = False,
    ) -> None:
        super().__init__("PlayerOptimizer")
        self.traj_optim = TrajectoryOptimizer(
            n_max_frame=trajectory_range, logger=self.logger
        )
        self.nan_interp = NanInterpolation(logger=self.logger)
        self.filter = K3dFilter(logger=self.logger)
        self.kalman_smoother = KalmanSmoother(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
            logger=self.logger,
        )
        self.one_player = one_player

    def _select_fields(self, use_reid: bool):
        if use_reid:
            return "matched_player_location", "matched_player_k3d"
        return "tracked_player_location", "tracked_player_k3d"

    def _collect_player_ids(
        self,
        mv_frames: List[MvFrame],
        location_field: str,
        k3d_field: str,
        player_ids=None,
    ):
        if player_ids is not None:
            return list(player_ids)

        # Count occurrences of each ID across all frames
        id_counts = {}
        for mv_frame in mv_frames:
            # for pid in getattr(mv_frame, location_field, {}).keys():
            #     id_counts[pid] = id_counts.get(pid, 0) + 1
            for pid in getattr(mv_frame, k3d_field, {}).keys():
                id_counts[pid] = id_counts.get(pid, 0) + 1

        # Sort by count descending (most frequent first), assign new IDs starting from 1
        sorted_ids = sorted(id_counts, key=lambda x: id_counts[x], reverse=True)
        if len(sorted_ids) == 0:
            self.logger.warning(
                "No player IDs found in mv_frames for fields "
                f"{location_field} and {k3d_field}. No optimization will be applied."
            )
            raise ValueError("No player IDs found in mv_frames for optimization.")
        # if the most frequent id appears less than len(mv_frames) - 100, throw error
        if id_counts[sorted_ids[0]] < len(mv_frames) - 100:
            self.logger.error(
                f"Most frequent player ID {sorted_ids[0]} appears only {id_counts[sorted_ids[0]]} times, which is less than total frames {len(mv_frames)} - 100. The ID remapping may be unreliable."
            )
            # raise ValueError("Most frequent player ID appears too few times, ID remapping may be unreliable.")

        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids, start=1)}
        self.logger.info(f"Player ID remapping (old->new, sorted by occurrence): {id_mapping}")

        # Remap IDs in all mv_frames for both fields
        for mv_frame in mv_frames:
            for field in (location_field, k3d_field):
                old_dict = getattr(mv_frame, field, {})
                if old_dict:
                    setattr(mv_frame, field, {id_mapping[pid]: val for pid, val in old_dict.items()})

        return list(range(1, len(sorted_ids) + 1))

    def _keep_only_player_ids(
        self,
        mv_frames: List[MvFrame],
        player_ids: List[int],
    ) -> None:
        selected_ids = set(player_ids)
        fields_to_filter = [
            "matched_player_location",
            "matched_player_k3d",
            "tracked_player_location",
            "tracked_player_k3d",
            "tracked_player_reid_info",
        ]
        for mv_frame in mv_frames:
            for field_name in fields_to_filter:
                field_value = getattr(mv_frame, field_name, None)
                if isinstance(field_value, dict):
                    setattr(
                        mv_frame,
                        field_name,
                        {
                            player_id: value
                            for player_id, value in field_value.items()
                            if player_id in selected_ids
                        },
                    )

    def _optimize_locations(
        self, mv_frames: List[MvFrame], player_ids: List[int], location_field: str
    ):
        if len(player_ids) == 0:
            return
        n_frame = len(mv_frames)
        mf_mp_loc = {pid: np.zeros((n_frame, 3)) for pid in player_ids}
        for f, mv_frame in enumerate(mv_frames):
            frame_locations = getattr(mv_frame, location_field, {})
            for player_id in player_ids:
                if player_id in frame_locations:
                    mf_mp_loc[player_id][f] = frame_locations[player_id]

        mf_mp_loc = np.stack(
            [mf_mp_loc[pid] for pid in player_ids], axis=1
        )  # n_frame, n_person, 3
        mf_mp_loc[mf_mp_loc == 0] = np.nan
        mf_mp_loc = self.filter.filter(mf_mp_loc)

        loc_mask = np.ones_like(mf_mp_loc[..., 0])
        mf_mp_loc = self.traj_optim.optimize_trajectory(mf_mp_loc, loc_mask)
        mf_mp_loc = self.nan_interp.optimize_trajectory(mf_mp_loc, loc_mask)
        mf_mp_loc = self.filter.filter(mf_mp_loc)
        mf_mp_loc = self.nan_interp.optimize_trajectory(mf_mp_loc, loc_mask)
        mf_mp_loc = self.kalman_smoother.smooth(mf_mp_loc, loc_mask)
        mf_mp_loc[np.isnan(mf_mp_loc)] = 0

        optimized_mv_frames = []
        for i, mv_frame in enumerate(mv_frames):
            target_locations = getattr(mv_frame, location_field, {})
            for j, pid in enumerate(player_ids):
                target_locations[pid] = mf_mp_loc[i, j]
            setattr(mv_frame, location_field, target_locations)
            optimized_mv_frames.append(mv_frame)
        return optimized_mv_frames

    def _optimize_k3d(
        self, mv_frames: List[MvFrame], player_ids: List[int], k3d_field: str
    ):
        if len(player_ids) == 0:
            return

        n_frame = len(mv_frames)
        n_kps = None
        for mv_frame in mv_frames:
            frame_k3d = getattr(mv_frame, k3d_field, {})
            for k3d in frame_k3d.values():
                if isinstance(k3d, np.ndarray) and k3d.ndim == 2 and k3d.shape[1] >= 4:
                    n_kps = k3d.shape[0]
                    break
            if n_kps is not None:
                break

        if n_kps is None:
            return

        mf_mp_k3d = np.zeros((n_frame, len(player_ids), n_kps, 4), dtype=np.float32)
        for f, mv_frame in enumerate(mv_frames):
            frame_k3d = getattr(mv_frame, k3d_field, {})
            for j, pid in enumerate(player_ids):
                if pid in frame_k3d and isinstance(frame_k3d[pid], np.ndarray):
                    mf_mp_k3d[f, j] = frame_k3d[pid]

        conf = mf_mp_k3d[..., 3:4]
        xyz = mf_mp_k3d[..., :3]
        xyz[conf[..., 0] <= 0] = np.nan

        xyz_flat = xyz.reshape(n_frame, -1, 3)
        xyz_mask = np.ones_like(xyz_flat[..., 0])
        xyz_flat = self.traj_optim.optimize_trajectory(xyz_flat, xyz_mask)
        xyz_flat = self.nan_interp.optimize_trajectory(xyz_flat, xyz_mask)
        xyz_flat = self.nan_interp.optimize_trajectory(xyz_flat, xyz_mask)
        xyz_flat = self.kalman_smoother.smooth(xyz_flat, xyz_mask)
        xyz_flat[np.isnan(xyz_flat)] = 0
        xyz = xyz_flat.reshape(n_frame, len(player_ids), n_kps, 3)

        conf[conf <= 0] = 0.1

        optimized_mv_frames = []
        for f, mv_frame in enumerate(mv_frames):
            target_k3d = getattr(mv_frame, k3d_field, {})
            for j, pid in enumerate(player_ids):
                # if pid in target_k3d:
                target_k3d[pid] = np.concatenate([xyz[f, j], conf[f, j]], axis=-1)
            setattr(mv_frame, k3d_field, target_k3d)
            optimized_mv_frames.append(mv_frame)

        return optimized_mv_frames
            
            

    def optimize(
        self, mv_frames: List[MvFrame], player_ids=None, use_reid: bool = True
    ):
        location_field, k3d_field = self._select_fields(use_reid)
        player_ids = self._collect_player_ids(
            mv_frames, location_field, k3d_field, player_ids
        )

        if self.one_player and len(player_ids) > 0:
            player_ids = player_ids[:1]
            self._keep_only_player_ids(mv_frames, player_ids)

        if not use_reid:
            optimized_mv_frames = self._optimize_k3d(mv_frames, player_ids, k3d_field)
        else:
            optimized_mv_frames = self._optimize_locations(mv_frames, player_ids, location_field)

        # backward compatibility for downstream analysis/visualization modules
        for mv_frame in optimized_mv_frames:
            mv_frame.player_location = getattr(mv_frame, location_field, {})

        return optimized_mv_frames, player_ids
