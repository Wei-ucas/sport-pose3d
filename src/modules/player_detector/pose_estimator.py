import os
import runpy
import warnings
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.utils.camera import get_basketball_outer_range
from src.utils.keypoints import (
    coco17tobody25,
    mediapipe33tobody25,
    wholebody133tobody25,
)


def init_rtmdet_trt(model_path: str, device: str = "cuda:0"):
    from mmdeploy_runtime import Detector

    return Detector(
        model_path=model_path,
        device_name=device.split(":")[0],
        device_id=int(device.split(":")[-1]),
    )


def init_rtmpose_trt(model_path: str, device: str = "cuda:0"):
    from mmdeploy_runtime import PoseDetector

    return PoseDetector(
        model_path=model_path,
        device_name=device.split(":")[0],
        device_id=int(device.split(":")[-1]),
    )


def build_tracker(tracker_cfg: Dict):
    if tracker_cfg["type"] == "ocsort":
        from src.modules.player_trackers.ocsort.ocsort import OCSort

        tracker_cfg = tracker_cfg.copy()
        tracker_cfg.pop("type")
        return OCSort(**tracker_cfg)
    raise NotImplementedError


def _crop_from_bbox(image: np.ndarray, bbox: np.ndarray, padding_ratio: float = 0.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    crop_x1 = int(max(0, np.floor(x1 - pad_x)))
    crop_y1 = int(max(0, np.floor(y1 - pad_y)))
    crop_x2 = int(min(w, np.ceil(x2 + pad_x)))
    crop_y2 = int(min(h, np.ceil(y2 + pad_y)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return image, 0, 0
    return image[crop_y1:crop_y2, crop_x1:crop_x2], crop_x1, crop_y1


def _load_pose_config(config_path: Optional[str], variable_names: List[str]) -> Dict:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Pose config not found: {config_path}")
    namespace = runpy.run_path(config_path)
    for variable_name in variable_names:
        config = namespace.get(variable_name)
        if isinstance(config, dict):
            return config
    return {}


class BasePoseBackend:
    model_name = "unknown"

    def infer(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RTMPoseTorchBackend(BasePoseBackend):
    model_name = "rtmpose"

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
        output_format: str = "body25",
        model_name: str = "rtmpose",
    ):
        from mmpose.apis import inference_topdown, init_model

        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(f"RTMPose config not found: {config_path}")
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "RTMPose checkpoint not found. Provide --pose_checkpoint with the downloaded weight file."
            )

        self.model_name = model_name
        self.output_format = output_format
        self.pose_estimator = init_model(config_path, checkpoint_path, device=device)
        self.inference_topdown = inference_topdown

    def _format_pose(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        pose = np.concatenate([keypoints[:, :2], scores[:, None]], axis=1).astype(
            np.float32
        )
        if self.output_format == "body25":
            return coco17tobody25(pose).astype(np.float32)
        return pose

    def infer(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if bboxes.shape[0] == 0:
            output_kpts = 25 if self.output_format == "body25" else 26
            return np.zeros((0, output_kpts, 3), dtype=np.float32)

        pose_results = self.inference_topdown(
            self.pose_estimator,
            image,
            bboxes=bboxes[:, :4],
            bbox_format="xyxy",
        )

        formatted = []
        for data_sample in pose_results:
            pred_instances = data_sample.pred_instances
            keypoints = np.asarray(pred_instances.keypoints, dtype=np.float32)
            if keypoints.ndim == 3:
                keypoints = keypoints[0]

            if hasattr(pred_instances, "keypoint_scores"):
                scores = np.asarray(pred_instances.keypoint_scores, dtype=np.float32)
            elif hasattr(pred_instances, "keypoints_visible"):
                scores = np.asarray(pred_instances.keypoints_visible, dtype=np.float32)
            else:
                scores = np.ones((keypoints.shape[0],), dtype=np.float32)

            if scores.ndim == 2:
                scores = scores[0]

            formatted.append(self._format_pose(keypoints, scores))

        if not formatted:
            output_kpts = 25 if self.output_format == "body25" else 26
            return np.zeros((0, output_kpts, 3), dtype=np.float32)
        return np.stack(formatted, axis=0)


class ViTPoseWholeBodyBackend(BasePoseBackend):
    model_name = "vitposepp-b"

    def __init__(
        self,
        config_path: Optional[str],
        checkpoint_path: str,
        device: str = "cuda:0",
        variant_name: str = "ViTPose++",
        model_name: str = "vitposepp-b",
    ):
        from .vitpose_repo import ViTPoseRepoWholeBodyInferencer

        self.model_name = model_name

        if config_path is not None and not os.path.isfile(config_path):
            raise FileNotFoundError(f"ViTPose config not found: {config_path}")
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"{variant_name} checkpoint not found. Provide --pose_checkpoint with the downloaded weight file."
            )

        self.inferencer = ViTPoseRepoWholeBodyInferencer(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )

    def infer(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if bboxes.shape[0] == 0:
            return np.zeros((0, 25, 3), dtype=np.float32)

        wholebody133 = self.inferencer.infer(image=image, bboxes=bboxes[:, :4])
        body25_results = [
            wholebody133tobody25(points).astype(np.float32) for points in wholebody133
        ]

        if not body25_results:
            return np.zeros((0, 25, 3), dtype=np.float32)
        return np.stack(body25_results, axis=0)


class MediaPipeBody25Backend(BasePoseBackend):
    model_name = "mediapipe"

    def __init__(self, config_path: Optional[str] = None):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is not installed. Install the mediapipe package in the active Python environment."
            ) from exc

        config = _load_pose_config(
            config_path, ["MEDIAPIPE_POSE_CONFIG", "MEDIAPIPE_MODEL"]
        )
        self.padding_ratio = float(config.get("padding_ratio", 0.25))
        self.min_visibility = float(config.get("min_visibility", 0.0))
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=bool(config.get("static_image_mode", True)),
            model_complexity=int(config.get("model_complexity", 1)),
            smooth_landmarks=bool(config.get("smooth_landmarks", False)),
            enable_segmentation=bool(config.get("enable_segmentation", False)),
            min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )

    @staticmethod
    def _landmark_score(landmark) -> float:
        visibility = float(getattr(landmark, "visibility", 1.0))
        presence = float(getattr(landmark, "presence", 1.0))
        if presence <= 0.0:
            return float(np.clip(visibility, 0.0, 1.0))
        return float(np.clip(visibility * presence, 0.0, 1.0))

    def infer(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if bboxes.shape[0] == 0:
            return np.zeros((0, 25, 3), dtype=np.float32)

        results = []
        for bbox in bboxes:
            crop, offset_x, offset_y = _crop_from_bbox(
                image, bbox, padding_ratio=self.padding_ratio
            )
            keypoints = np.zeros((25, 3), dtype=np.float32)
            if crop.size == 0:
                results.append(keypoints)
                continue

            pose_result = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if pose_result.pose_landmarks is not None:
                crop_h, crop_w = crop.shape[:2]
                mediapipe_points = np.zeros((33, 3), dtype=np.float32)
                for landmark_id, landmark in enumerate(
                    pose_result.pose_landmarks.landmark
                ):
                    score = self._landmark_score(landmark)
                    if score < self.min_visibility:
                        score = 0.0
                    mediapipe_points[landmark_id, 0] = landmark.x * crop_w + offset_x
                    mediapipe_points[landmark_id, 1] = landmark.y * crop_h + offset_y
                    mediapipe_points[landmark_id, 2] = score
                keypoints = mediapipe33tobody25(mediapipe_points).astype(np.float32)
            results.append(keypoints)

        return np.stack(results, axis=0)


class PlayerPoseEstimator:
    def __init__(
        self,
        pose_model: str,
        detector_checkpoint: str,
        pose_checkpoint: Optional[str] = None,
        pose_config: Optional[str] = None,
        pred_pose: bool = True,
        tracker_cfg: Optional[Dict] = None,
        bbox_score_thr: float = 0.3,
        device: str = "cuda:0",
        bbox_filter=None,
        enable_detector: bool = True,
    ):
        self.pose_model = pose_model
        self.device = device
        self.bbox_score_thr = bbox_score_thr
        self.pred_pose = pred_pose
        self.detector_checkpoint = detector_checkpoint
        self.enable_detector = enable_detector
        self.detector = (
            init_rtmdet_trt(detector_checkpoint, device=device)
            if enable_detector
            else None
        )
        self.pose_backend = self._build_pose_backend(
            pose_model=pose_model,
            pose_checkpoint=pose_checkpoint,
            pose_config=pose_config,
            device=device,
        )
        self.tracker = build_tracker(tracker_cfg) if tracker_cfg is not None else None
        self.bbox_filter = bbox_filter
        if bbox_filter is None:
            warnings.warn(
                "bbox filter is None, all bboxes will be used for pose estimation"
            )

        self.court_image_range = None

    def _build_pose_backend(
        self,
        pose_model: str,
        pose_checkpoint: Optional[str],
        pose_config: Optional[str],
        device: str,
    ) -> BasePoseBackend:
        model_key = pose_model.lower()
        if model_key == "rtmpose":
            if not pose_checkpoint:
                raise ValueError("RTMPose requires a pose checkpoint path.")
            return RTMPoseTorchBackend(
                config_path=pose_config,
                checkpoint_path=pose_checkpoint,
                device=device,
                output_format="body25",
                model_name="rtmpose",
            )
        if model_key in {"rtmpose-m-halpe26", "rtmpose-halpe26"}:
            if not pose_checkpoint:
                raise ValueError("RTMPose halpe26 requires a pose checkpoint path.")
            return RTMPoseTorchBackend(
                config_path=pose_config,
                checkpoint_path=pose_checkpoint,
                device=device,
                output_format="halpe26",
                model_name="rtmpose-m-halpe26",
            )
        if model_key in {"vitposepp-b", "vitpose++-b", "vitpose-b", "vitpose"}:
            return ViTPoseWholeBodyBackend(
                config_path=pose_config,
                checkpoint_path=pose_checkpoint,
                device=device,
                variant_name="ViTPose++-B",
                model_name="vitposepp-b",
            )
        if model_key in {"vitposepp-h", "vitpose++-h", "vitpose-h"}:
            return ViTPoseWholeBodyBackend(
                config_path=pose_config,
                checkpoint_path=pose_checkpoint,
                device=device,
                variant_name="ViTPose++-H",
                model_name="vitposepp-h",
            )
        if model_key in {"mediapipe", "mediapipe-pose"}:
            return MediaPipeBody25Backend(config_path=pose_config)
        raise ValueError(f"Unsupported pose model: {pose_model}")

    def preprocess(self, image_list: List[np.ndarray], camera_params_list: List[Dict]):
        if self.court_image_range is None:
            self.court_image_range = []
            for i, param in enumerate(camera_params_list):
                court_range = get_basketball_outer_range(param)
                self.court_image_range.append(
                    [
                        int(max(court_range[0] - 10, 0)),
                        int(max(court_range[1] - 10, 0)),
                        int(min(court_range[2] + 10, image_list[i].shape[1])),
                        int(min(court_range[3] + 10, image_list[i].shape[0])),
                    ]
                )

        cropped_image = []
        for i, image in enumerate(image_list):
            cropped_image.append(
                image[
                    self.court_image_range[i][1] : self.court_image_range[i][3],
                    self.court_image_range[i][0] : self.court_image_range[i][2],
                ]
            )
        return cropped_image

    def postprocess(self, results):
        for i, result in enumerate(results):
            if result["bboxes"].shape[0] == 0:
                continue
            offset_x = self.court_image_range[i][0]
            offset_y = self.court_image_range[i][1]
            result["bboxes"][:, 0] += offset_x
            result["bboxes"][:, 1] += offset_y
            result["bboxes"][:, 2] += offset_x
            result["bboxes"][:, 3] += offset_y

            result["keypoints"][:, :, 0] += offset_x
            result["keypoints"][:, :, 1] += offset_y
        return results

    @staticmethod
    def _build_head_bboxes(keypoints: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if keypoints.shape[1] == 25:
            head_indices = [0, 15, 16, 17, 18]
        elif keypoints.shape[1] == 26:
            head_indices = [0, 1, 2, 3, 4, 17, 18]
        else:
            head_indices = list(range(min(5, keypoints.shape[1])))
        head_bboxes = bboxes[:, :5].copy()
        for idx in range(len(keypoints)):
            head_points = keypoints[idx, head_indices]
            valid = head_points[:, 2] > 0.05
            if np.any(valid):
                head_bboxes[idx, 0] = head_points[valid, 0].min()
                head_bboxes[idx, 1] = head_points[valid, 1].min()
                head_bboxes[idx, 2] = head_points[valid, 0].max()
                head_bboxes[idx, 3] = head_points[valid, 1].max()
        return head_bboxes

    @staticmethod
    def _as_bbox_array(bboxes) -> np.ndarray:
        if bboxes is None:
            return np.zeros((0, 5), dtype=np.float32)

        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]
        if bboxes.shape[1] < 4:
            raise ValueError(f"bbox shape should be Nx4/Nx5, but got {bboxes.shape}")
        if bboxes.shape[1] == 4:
            scores = np.ones((bboxes.shape[0], 1), dtype=np.float32)
            bboxes = np.concatenate([bboxes[:, :4], scores], axis=1)
        else:
            bboxes = bboxes[:, :5]
        return bboxes.astype(np.float32, copy=False)

    def _prepare_external_bboxes(
        self, external_bboxes_list, image_list, camera_params_list
    ):
        det_bboxes = []
        results = []
        valid_ids = []

        for vid, raw_bboxes in enumerate(external_bboxes_list):
            bboxes = self._as_bbox_array(raw_bboxes).copy()
            if bboxes.shape[0] > 0:
                offset_x = self.court_image_range[vid][0]
                offset_y = self.court_image_range[vid][1]
                image_h, image_w = image_list[vid].shape[:2]

                bboxes[:, 0] -= offset_x
                bboxes[:, 1] -= offset_y
                bboxes[:, 2] -= offset_x
                bboxes[:, 3] -= offset_y

                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, image_w - 1)
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, image_h - 1)
                bboxes[:, 2] = np.clip(bboxes[:, 2], 0, image_w - 1)
                bboxes[:, 3] = np.clip(bboxes[:, 3], 0, image_h - 1)

                valid_bbox = np.logical_and(
                    bboxes[:, 2] > bboxes[:, 0], bboxes[:, 3] > bboxes[:, 1]
                )
                bboxes = bboxes[valid_bbox]

            if len(bboxes) > 0 and self.bbox_filter is not None:
                assert (
                    camera_params_list is not None
                ), "camera params is required for bbox filter"
                valid_idx = self.bbox_filter.filter(
                    bboxes, camera_params_list[vid], self.court_image_range[vid]
                )
                bboxes = bboxes[valid_idx]

            results.append(
                {
                    "bboxes": bboxes,
                    "bbox_scores": (
                        bboxes[:, -1]
                        if len(bboxes) > 0
                        else np.zeros((0,), dtype=np.float32)
                    ),
                }
            )
            det_bboxes.append(bboxes[:, :4])

            if bboxes.shape[0] > 0:
                valid_ids.append(vid)
            else:
                results[vid].update(
                    {
                        "bbox_ids": np.zeros((0,), dtype=np.float32),
                        "keypoints": np.zeros((0, 25, 3), dtype=np.float32),
                        "keypoint_scores": np.zeros((0, 25), dtype=np.float32),
                    }
                )

        return det_bboxes, results, valid_ids

    def infer(
        self,
        image_list: List[np.ndarray],
        camera_params_list: List[Dict] = None,
        external_bboxes_list=None,
    ):
        assert len(image_list) == 1, "only support batch size 1"

        image_list = self.preprocess(image_list, camera_params_list)
        if external_bboxes_list is not None:
            assert len(external_bboxes_list) == len(
                image_list
            ), "external bboxes should match image batch size"
            det_bboxes, results, valid_ids = self._prepare_external_bboxes(
                external_bboxes_list,
                image_list,
                camera_params_list,
            )
        else:
            if self.detector is None:
                raise RuntimeError(
                    "RTMDet detector is disabled; provide external_bboxes_list to skip detection."
                )

            detection_results = self.detector.batch(image_list)

            det_bboxes = []
            results = []
            valid_ids = []
            for vid, pred in enumerate(detection_results):
                bboxes = pred[0]
                scores = bboxes[:, -1]
                bboxes = bboxes[scores > self.bbox_score_thr]
                if len(bboxes) > 0 and self.bbox_filter is not None:
                    assert (
                        camera_params_list is not None
                    ), "camera params is required for bbox filter"
                    valid_idx = self.bbox_filter.filter(
                        bboxes, camera_params_list[vid], self.court_image_range[vid]
                    )
                    bboxes = bboxes[valid_idx]

                results.append(
                    {
                        "bboxes": bboxes,
                        "bbox_scores": (
                            bboxes[:, -1]
                            if len(bboxes) > 0
                            else np.zeros((0,), dtype=np.float32)
                        ),
                    }
                )
                det_bboxes.append(bboxes[:, :4])

                if bboxes.shape[0] > 0:
                    valid_ids.append(vid)
                else:
                    results[vid].update(
                        {
                            "bbox_ids": np.zeros((0,), dtype=np.float32),
                            "keypoints": np.zeros((0, 25, 3), dtype=np.float32),
                            "keypoint_scores": np.zeros((0, 25), dtype=np.float32),
                        }
                    )

        if not self.pred_pose:
            return self.postprocess(results)

        for vid in valid_ids:
            bboxes = results[vid]["bboxes"].copy()
            pred_keypoints = self.pose_backend.infer(image_list[vid], det_bboxes[vid])

            if self.bbox_filter is not None and len(pred_keypoints) > 0:
                valid_idx = self.bbox_filter.remove_duplicate_k2d(pred_keypoints.copy())
                pred_keypoints = pred_keypoints[valid_idx]
                bboxes = bboxes[valid_idx]

            pred_keypoints_scores = (
                pred_keypoints[:, :, 2]
                if len(pred_keypoints) > 0
                else np.zeros((0, 25), dtype=np.float32)
            )
            if len(pred_keypoints_scores) != 0 and pred_keypoints_scores.max() > 1:
                pred_keypoints_scores = pred_keypoints_scores / 2
                pred_keypoints[:, :, 2] = pred_keypoints_scores

            if self.tracker is not None and len(bboxes) > 0:
                head_bboxes = self._build_head_bboxes(pred_keypoints, bboxes)
                tracked_bboxes, det_ids = self.tracker.update(
                    head_bboxes.astype(np.float32), (1, 1), (1, 1)
                )
                if isinstance(tracked_bboxes, List) and len(tracked_bboxes) > 0:
                    tracked_bboxes = np.stack(
                        [track.result for track in tracked_bboxes]
                    )
                if isinstance(tracked_bboxes, np.ndarray) and tracked_bboxes.size > 0:
                    bbox_ids = tracked_bboxes[:, -1]
                else:
                    bbox_ids = np.zeros((0,), dtype=np.float32)
                    det_ids = np.zeros((0,), dtype=np.int64)
            else:
                bbox_ids = np.zeros(bboxes.shape[0], dtype=np.float32) - 1
                det_ids = np.arange(bboxes.shape[0], dtype=np.int64)

            if len(det_ids) > 0:
                ordered_bboxes = np.concatenate(
                    [bboxes[det_ids], bbox_ids[:, None]], axis=1
                )
                ordered_keypoints = pred_keypoints[det_ids]
                ordered_scores = pred_keypoints_scores[det_ids]
            else:
                ordered_bboxes = np.zeros((0, 6), dtype=np.float32)
                ordered_keypoints = np.zeros((0, 25, 3), dtype=np.float32)
                ordered_scores = np.zeros((0, 25), dtype=np.float32)

            results[vid].update(
                {
                    "bboxes": ordered_bboxes[:, :-1],
                    "bbox_ids": bbox_ids,
                    "keypoints": ordered_keypoints,
                    "keypoint_scores": ordered_scores,
                }
            )

        return self.postprocess(results)

    def __call__(
        self,
        image_list: List[np.ndarray],
        camera_params_list: List[Dict],
        external_bboxes_list=None,
    ):
        return self.infer(
            image_list, camera_params_list, external_bboxes_list=external_bboxes_list
        )
