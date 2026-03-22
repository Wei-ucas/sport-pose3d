from typing import List, Tuple

from ..base_module import BaseModule
from src.structures.multiview_frame import MvFrame
from src.utils.keypoints import coco17tolocation
import numpy as np


def build_k3d_tracker(track_distance):
    from ..player_trackers.k3d_tracker.pose_tracker import K3dSort
    tracker = K3dSort(
        max_age=20,
        dist_threshold=track_distance,  # 影响匹配的结果， 关键
        min_hits=3,
    )
    return tracker


def trajectory_nms(reid_player_candidates: List[Tuple[np.ndarray, float]]):
    sorted_candidates = sorted(reid_player_candidates, key=lambda x: x[1], reverse=True)
    keep_index = []
    rest_index = list(range(len(sorted_candidates)))
    remove_index = []
    current_overlap = np.zeros(len(sorted_candidates[0][0]))
    for i, (kps, confidence) in enumerate(sorted_candidates):
        if i in rest_index:
            keep_index.append(i)
            rest_index.remove(i)
            current_overlap += (kps[:, 2] != 0).astype('int')
            current_rest = []
            for j in rest_index:
                # overlap = (candidate_kps[:, 0, 0] != 0).astype('int') + (kps[:, 0, 0] != 0).astype('int')
                # overlap = (sorted_candidates[j][0][:, 0, 0] != 0).astype('int') + (kps[:, 0, 0] != 0).astype('int')
                overlap = current_overlap + (sorted_candidates[j][0][:, 2] != 0).astype('int')
                n_overlap = (overlap > 1).sum()
                # if overlap.max() <= 1:
                if n_overlap <= 0:
                    current_rest.append(j)
                    continue
                    # current_overlap = overlap
                elif n_overlap > 10:
                    # rest_index.remove(j)
                    remove_index.append(j)
                else:  # remove the overlap frame
                    overlap_index = np.where(overlap > 1)[0]
                    sorted_candidates[j][0][overlap_index] = 0
                    current_rest.append(j)
            rest_index = current_rest
    keep_candidates = [sorted_candidates[i][0] for i in keep_index]
    remove_candidates = [sorted_candidates[i][0] for i in remove_index]
    return keep_candidates, remove_candidates


class Player3DMatcher(BaseModule):

    def __init__(self,
                 track_distance=2.0,
                 reid_conf_threshold=0.9,
                 # reid_ratio=0.05,
                 reid_dist_threshold=500,
                 appear_frame_thr=200
                 ):
        super().__init__("Player3DMatcher")
        self.tracker = build_k3d_tracker(track_distance)
        self.logger.info(f"Player3DMatcher initialized with track_distance={track_distance}, ")
        self.reid_conf_threshold = reid_conf_threshold
        self.reid_dist_threshold = reid_dist_threshold
        self.appear_frame_thr = appear_frame_thr

    def filter_k3d(self, k3d_dets):
        # remove duplicate detections
        k3d_dist_mat = np.linalg.norm(k3d_dets[:, None, 3:7, :3] - k3d_dets[None, :, 3:7, :3], axis=-1)
        k3d_dist_mat += np.eye(len(k3d_dets))[:, :, None] * 1000
        duplicate = np.where((k3d_dist_mat < 0.2).sum(-1) > 2)
        # for duplicate dets, keep the one with higher score
        for i, j in zip(duplicate[0], duplicate[1]):
            if k3d_dets[i, :, 3].mean() > k3d_dets[j, :, 3].mean():
                k3d_dets[j] = 0
            else:
                k3d_dets[i] = 0
        valid = np.where(k3d_dets[:, :, 3].max(-1) > 0)[0]
        return valid

    def prepare_frame(self, mv_frame: MvFrame):
        k3d_dets = mv_frame.untracked_players_k3d
        if len(k3d_dets) == 0:
            return mv_frame
        reid = mv_frame.untracked_reid_info
        k3d_dets = np.array(k3d_dets)
        reid = np.array(reid)
        valid_index = self.filter_k3d(k3d_dets)
        k3d_dets = k3d_dets[valid_index]
        reid = reid[valid_index]

        mv_frame.untracked_players_k3d = []

        match_ids, det_index = self.tracker.update(k3d_dets, reid)
        for i, det_idx in enumerate(det_index):
            # mv_frames.player_k3d[match_ids[i]] = k3d_dets[det_idx]
            mv_frame.track3d_player_location[match_ids[i]] = coco17tolocation(k3d_dets[det_idx])
            mv_frame.tracked_player_reid_info[match_ids[i]] = reid[det_idx]
        return mv_frame

    def matching_frame_clip(self, mv_frames: List[MvFrame]):
        reid_players = {}
        no_name_players = []
        tracking_players = {}
        tracking_players_reid = {}
        for i, frame in enumerate(mv_frames):
            for player_id, player_loc in frame.track3d_player_location.items():
                if player_id not in tracking_players:
                    tracking_players[player_id] = np.zeros((len(mv_frames), 3))
                    tracking_players_reid[player_id] = np.ones((len(mv_frames), 2))
                    tracking_players_reid[player_id][:, 0] = -1
                    tracking_players_reid[player_id][:, 1] = 10000
                tracking_players[player_id][i] = player_loc
                tracking_players_reid[player_id][i] = frame.tracked_player_reid_info[player_id]
        tracking_player_appear = {}
        for player_id in tracking_players.keys():
            tracking_player_appear[player_id] = (tracking_players[player_id][:, 2] != 0).sum(0)
        sorted_tracking_player_appear = sorted(list(tracking_player_appear.keys()),
                                               key=lambda x: tracking_player_appear[x], reverse=True)

        reid_player_candidates = {}
        # for player_id in tracking_players.keys():
        for player_id in sorted_tracking_player_appear:
            kps = np.array(tracking_players[player_id])
            n_frame_player_appear = tracking_player_appear[player_id]
            reid = np.array(tracking_players_reid[player_id])
            pid = reid[:, 0]
            cost = reid[:, 1]
            valid_flag = cost < self.reid_dist_threshold
            valid = np.where(valid_flag)
            valid_ids = pid[valid].astype(int)
            # get the most frequent id
            if len(valid_ids) > 0 and n_frame_player_appear > 25:
                id_count = np.bincount(valid_ids)
                # reid_id = np.argmax(id_count)
                # max_count = id_count[reid_id]
                # chose the id with the highest confidence and the most frequent
                id_confidence = np.zeros(len(id_count))
                for i in range(len(id_count)):
                    id_confidence[i] = self.reid_dist_threshold / np.mean(cost[pid == i]) if id_count[i] > 0 else 0
                reid_id = np.argmax(id_count * id_confidence)
                max_count = id_count[reid_id]

                # mean_cost = np.mean(cost[pid==reid_id])
                mean_cost = np.mean(cost[pid == reid_id])
                match_confidence = max_count / sum(id_count) * (self.reid_dist_threshold / mean_cost) ** 2 * np.sqrt(
                    len(valid_ids) / self.appear_frame_thr if len(
                        valid_ids) < self.appear_frame_thr * 2 else 2) * np.sqrt(
                    len(valid_ids) / n_frame_player_appear)
                if match_confidence < self.reid_conf_threshold:
                    # reid_id = -1
                    no_name_players.append(kps)
                    continue
                if reid_id not in reid_player_candidates:
                    # reid_players[reid_id] = kps
                    reid_player_candidates[reid_id] = [(kps, match_confidence)]
                else:
                    reid_player_candidates[reid_id].append((kps, match_confidence))
            else:
                no_name_players.append(kps)
            #     reid_id = -1
        for reid_id in reid_player_candidates.keys():
            # reid_players[reid_id] = reid_player_candidates[reid_id][0][0]
            keep_candidates, remove_candidates = trajectory_nms(reid_player_candidates[reid_id])
            # reid_player_candidates[reid_id] = keep_candidates[0]
            reid_players[reid_id] = keep_candidates[0]
            for i in range(1, len(keep_candidates)):
                if ((reid_players[reid_id][:, 2] != 0).astype('int')
                    + (keep_candidates[i][:, 2] != 0).astype('int')).max() > 1:
                    self.logger.warn(f"Player {reid_id} has overlapping frames, merging them.")
                reid_players[reid_id] += keep_candidates[i]
            for candidate_kps in remove_candidates:
                no_name_players.append(candidate_kps)

        for i, frame in enumerate(mv_frames):
            for player_id, player_loc in reid_players.items():
                frame.player_location[player_id] = player_loc[i]
            for k3d in no_name_players:
                frame.no_name_player_location.append(k3d[i])
            frame.track3d_player_location = {}
            frame.tracked_player_reid_info = {}
            frame.untracked_players_k3d = []
            frame.untracked_reid_info = []
        return mv_frames
