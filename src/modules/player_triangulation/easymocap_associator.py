from typing import List, Tuple, Dict, Any, Union
import os
import sys
import numpy as np

from easydict import EasyDict as edict

from src.utils.camera import undistort_points
from src.utils.common import linear_assignment
from src.structures.multiview_frame import MvFrame
from src.structures.player import Player
from ..base_module import BaseModule

from .utils import (
    player_to_easymocap_format,
    camera_to_easymocap_format,
    easymocap_result_to_player,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .easymocap.affinity.affinity import ComposedAffinity
from .easymocap.assignment.associate import simple_associate
from .easymocap.assignment.group import PeopleGroup


class Affinity(ComposedAffinity):
    affinity_cfg = edict(
        {
            "aff_min": 0.2,
            "svt_py": False,
            "aff_funcs": edict(
                {"easymocap.affinity.ray.Affinity": edict({"MAX_DIST": 0.1})}
            ),
            "svt_args": edict(
                {
                    "debug": 0,
                    "log": 0,
                    "maxIter": 20,
                    "w_sparse": 0.1,
                    "w_rank": 50,
                    "tol": 0.0001,
                }
            ),
            "vis_aff": False,
            "vis_res": False,
            "vis_pair": False,
        }
    )

    def __init__(self, cameras, basenames):
        super().__init__(cameras, basenames, self.affinity_cfg)


class PlayerTriangulator(BaseModule):
    associate_cfg = edict(
        {
            "debug": False,
            "log": False,
            "body": "body25",
            "max_repro_error": 0.1,
            "min_views": 4,
            "criterions": edict(
                {
                    "easymocap.assignment.criterion.BaseCrit": edict(
                        {"min_conf": 0.1, "min_joints": 5}
                    ),
                    "easymocap.assignment.criterion.CritLenTorso": edict(
                        {
                            "src": 2,
                            "dst": 5,
                            "min_torso_length": 0.1,
                            "max_torso_length": 1,
                            "min_conf": 0.1,
                        }
                    ),
                    # 'easymocap.assignment.criterion.CritMinMax': edict({'max_human_length': 2.5, 'min_conf': 0.001}),
                    "easymocap.assignment.criterion.CritWithTorso": edict(
                        {"torso_idx": [1, 17, 18], "min_conf": 0.3}
                    ),
                    "easymocap.assignment.criterion.CritLimbLength": edict(
                        {"max_rate": 1.0, "body_type": "body25", "min_conf": 0.1}
                    ),
                }
            ),
        }
    )

    def __init__(
        self,
        cameras_params,
        camera_ids,
        pose_conversion="coco17",
        dist_max=25,
        num_players=10,
    ):
        super().__init__("PlayerTriangulator")
        self.camera_ids = camera_ids
        if cameras_params is not None:
            self.cameras_params = cameras_params
            self.cameras = camera_to_easymocap_format(cameras_params, camera_ids)
            self.affinity = Affinity(self.cameras, camera_ids)
            self.Pall = np.stack([self.cameras[cam]["P"] for cam in self.camera_ids])
            self.logger.info(f"Loaded cameras: {self.camera_ids}")
        else:
            self.cameras = None
        self.dist_max = dist_max
        assert (
            pose_conversion == "coco17"
        ), "Only coco17 pose conversion is supported in EasyMocapAssociate"
        self.pose_conversion = pose_conversion
        self.num_target_player = num_players

    def process(self, mv_frame: MvFrame) -> MvFrame:
        camera_params = mv_frame.camera_params
        if self.cameras is None:
            assert (
                camera_params is not None
            ), "Camera parameters must be provided if cameras are not initialized."
            cameras = camera_to_easymocap_format(camera_params, self.camera_ids)
            self.affinity = Affinity(cameras, self.camera_ids)
            self.Pall = np.stack([cameras[cam]["P"] for cam in self.camera_ids])
            self.logger.info("Camera updated.")
        mv_mp_player_list: Dict[str, List[Player]] = {
            cam: mv_frame.frame_dict[cam].players for cam in self.camera_ids
        }
        camera_annots = player_to_easymocap_format(mv_mp_player_list)
        all_players = []  # all players in all frames
        for frame in mv_frame.frames:
            all_players.extend(frame.players)
        results = self(camera_annots, mv_frame.images)

        reid_distances = []

        for result in results:
            # players = [all_players[idx] for idx in result['indices']]
            # keypoint3d = body25tococo17(result['keypoints3d'])
            keypoint3d, player_indexes = easymocap_result_to_player(result)
            players = [all_players[idx] for idx in player_indexes]

            # player_ids = [player.player_id for player in players if player.player_id not in [None, -1]]
            # player_reid_confs = [player.reid_confidence for player in players if player.player_id not in [None, -1]]
            # get the most frequent player id
            # player_id = max(set(player_ids), key=player_ids.count) if len(player_ids) > 0 else None
            # player_id = player_ids[np.argmin(player_reid_confs)] if len(player_ids) > 0 else -1
            # conf = min(player_reid_confs) if len(player_reid_confs) > 0 else 10000

            distances = []
            for player in players:
                dist = getattr(player, "reid_distances", [])
                if len(dist) == 0:
                    dist = np.ones(self.num_target_player) * 1000
                distances.append(dist)
            distances = np.stack(distances)
            reid_distances.append(
                distances.min(axis=0)
            )  # for each player, use the minimum distance across all cameras

            mv_frame.untracked_players_k3d.append(keypoint3d)
            mv_frame.untracked_reid_info.append([-1, 1000])

        if len(reid_distances) > 0:
            reid_distances = np.stack(reid_distances)
            if reid_distances.size > 0 and reid_distances.shape[1] > 0:
                match_indices = linear_assignment(reid_distances)
                for i, j in match_indices:
                    if reid_distances[i, j] < 1000:
                        mv_frame.untracked_reid_info[i] = [j, reid_distances[i, j]]

        return mv_frame

    def __call__(self, camera_annots: dict, images: dict = None):
        """

        Args:
            annots: dict, {camera_id: [{keypoints: 25x3, bbox: 4x1, isKeyframe, bbox, person_id}]}
            images: dict, {camera_id: image}

        Returns:
            results: list, [{id: pid, keypoints3d: 25x4, kptsRepro: 25x3}]
        """
        camera_annots = self.undistort_keypoints(camera_annots)
        results = self.associate_triangulate(camera_annots, images)
        return results

    def undistort_keypoints(self, camera_annots):
        for camera_id, annots in camera_annots.items():
            for p in annots:
                if np.sum(p["keypoints"][:, -1]) > 0:
                    p["keypoints"] = undistort_points(
                        p["keypoints"],
                        self.cameras[camera_id]["K"],
                        self.cameras[camera_id]["dist"],
                    )
        return camera_annots

    def associate_triangulate(self, camera_annots, images):

        annots = [camera_annots[idx] for idx in self.camera_ids]
        # annots = self.keypoint_filter(annots)
        affinity, dimGroups = self.affinity(annots, images)
        associate_results = simple_associate(
            annots,
            affinity,
            dimGroups,
            self.Pall,
            PeopleGroup(Pall=self.Pall, cfg=edict({})),
            cfg=self.associate_cfg,
            images=list(images.values()),
        )
        results = []
        for ik, (pid, people) in enumerate(associate_results.items()):
            # self.get_reproject_error(people.keypoints3d, camera_annots)
            # d = people.keypoints3d[0, 0] ** 2 + people.keypoints3d[0, 1] ** 2 + people.keypoints3d[0, 2] ** 2
            result = {
                "id": pid,
                "keypoints3d": people.keypoints3d,
                "bbox": people.info["bbox"],
                "kptsRepro": people.info["kptsRepro"],
                "indices": people.indices,
            }
            results.append(result)
        return results
