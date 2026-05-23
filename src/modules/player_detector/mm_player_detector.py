import logging
import numpy as np
from src.structures.frame import Frame
from src.structures.player import Player

from ..base_module import BaseModule
from .bbox_filter import BboxFilter
from .pose_estimator import PlayerPoseEstimator


class PlayerDetector(BaseModule):
    """
    Player detection and pose detection module
    """

    def __init__(
        self,
        detection_threshold: float = 0.2,
        tracker_iou_thr: float = 0.2,
        device: str = "cuda:0",
        pose_model: str = "rtmpose",
        pose_checkpoint: str = None,
        pose_config: str = None,
        enable_detector: bool = True,
    ):
        """
        Args:
        """
        super().__init__("PlayerDetector")
        pose_model_key = pose_model.lower()
        default_pose_checkpoint = pose_checkpoint
        default_pose_config = pose_config
        if pose_model_key == "rtmpose":
            default_pose_checkpoint = (
                default_pose_checkpoint
                or "cpks/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
            )
            default_pose_config = (
                default_pose_config
                or "src/modules/player_detector/configs/rtmpose-m_8xb256-420e_coco-256x192.py"
            )
        elif pose_model_key in {"rtmpose-m-halpe26", "rtmpose-halpe26"}:
            default_pose_checkpoint = (
                default_pose_checkpoint
                or "cpks/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"
            )
            default_pose_config = (
                default_pose_config
                or "src/modules/player_detector/configs/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
            )
        elif pose_model_key in {"vitposepp-b", "vitpose++-b", "vitpose-b", "vitpose"}:
            default_pose_config = (
                default_pose_config
                or "src/modules/player_detector/configs/vitposepp_b_wholebody_256x192.py"
            )
            default_pose_checkpoint = default_pose_checkpoint or "cpks/vitpose_base.pth"
        elif pose_model_key in {"vitposepp-h", "vitpose++-h", "vitpose-h"}:
            default_pose_config = (
                default_pose_config
                or "src/modules/player_detector/configs/vitposepp_h_wholebody_256x192.py"
            )
            default_pose_checkpoint = (
                default_pose_checkpoint or "cpks/vitpose _huge.pth"
            )

        self.model = PlayerPoseEstimator(
            pose_model=pose_model,
            pose_config=default_pose_config,
            pose_checkpoint=default_pose_checkpoint,
            detector_checkpoint="cpks/trt/rtmdet-m-960",
            pred_pose=True,
            tracker_cfg=dict(
                type="ocsort",
                det_thresh=0.2,
                max_age=10,
                min_hits=2,
                iou_threshold=tracker_iou_thr,
                delta_t=3,
                asso_func="diou",
                inertia=0.5,
                use_byte=True,
            ),
            bbox_score_thr=detection_threshold,
            device=device,
            bbox_filter=BboxFilter(camera_params=None),
            enable_detector=enable_detector,
        )
        self.logger.info(
            "PlayerDetector initialized with pose model '%s' (rtmdet enabled=%s).",
            pose_model,
            enable_detector,
        )

    def process(self, frame: Frame, inherited_bboxes: np.ndarray = None) -> Frame:
        image: np.ndarray = frame.image
        camera_params = frame.camera_params
        external_bboxes_list = None
        if inherited_bboxes is not None:
            external_bboxes_list = [inherited_bboxes]
        track_results = self.model(
            [image],
            camera_params_list=[camera_params],
            external_bboxes_list=external_bboxes_list,
        )[0]
        detected_players = []
        for i, box in enumerate(track_results["bboxes"]):
            score = track_results["bbox_scores"][i]
            bbox = np.array([box[0], box[1], box[2], box[3], score])
            pose = track_results["keypoints"][i]
            track_id = track_results["bbox_ids"][i]
            player = Player(tracking_id=track_id, bbox=bbox, pose=pose)
            detected_players.append(player)
        frame.add_player(detected_players)
        return frame
