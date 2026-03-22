import logging
import numpy as np
from src.structures.frame import Frame
from src.structures.player import Player

from ..base_module import BaseModule
from .mm_pose import RTMPoseEstimator
from .bbox_filter import BboxFilter


class PlayerDetector(BaseModule):
    """
    Player detection and pose detection module
    """

    def __init__(self, detection_threshold: float = 0.2, tracker_iou_thr: float = 0.2, device: str = "cuda:0"):
        """
        Args:
        """
        super().__init__("PlayerDetector")
        self.model = RTMPoseEstimator(
            model_type='trt',
            pose_cfg="src/modules/player_detector/configs/rtmpose-m_8xb256-420e_coco-256x192.py",
            pose_checkpoint="cpks/trt/rtmpose-m-fp16",
            detector_cfg="src/modules/player_detector/configs/rtmdet_m_640-8xb32_coco-person.py",
            detector_checkpoint="cpks/trt/rtmdet-m-960",
            pred_pose=True,
            tracker_cfg=dict(
                type="ocsort",
                det_thresh=0.2, max_age=10, min_hits=2,
                iou_threshold=tracker_iou_thr, delta_t=3, asso_func="diou", inertia=0.5, use_byte=True
            ),
            bbox_score_thr=detection_threshold,
            bbox_nms_thr=0.7,  # TRT模型不生效
            det_batch_size=12,  # TRT模型不生效
            pose_batch_size=64,  # TRT模型不生效
            device=device,
            bbox_filter=BboxFilter(
                camera_params=None
            )
        )
        self.logger.info("PlayerDetector initialized with RTMPoseEstimator model.")

    def process(self, frame: Frame) -> Frame:
        image: np.ndarray = frame.image
        camera_params = frame.camera_params
        track_results = self.model([image], camera_params_list=[camera_params])[0]
        detected_players = []
        for i, box in enumerate(track_results['bboxes']):
            score = track_results['bbox_scores'][i]
            bbox = np.array([box[0], box[1], box[2], box[3], score])
            pose = track_results['keypoints'][i]
            track_id = track_results['bbox_ids'][i]
            player = Player(tracking_id=track_id, bbox=bbox, pose=pose)
            detected_players.append(player)
        frame.add_player(detected_players)
        return frame
