import warnings

from mmpose.structures import merge_data_samples
from mmengine.structures import InstanceData
import numpy as np
from typing import List, Dict, Union

from src.utils.camera import get_basketball_outer_range


def init_mmpose_trt(det_model_path, pose_model_path, device='cuda'):
    from mmdeploy_runtime import Detector, PoseDetector
    # create object detector
    detector = Detector(
        model_path=det_model_path, device_name=device.split(':')[0], device_id=int(device.split(':')[-1]))
    # create pose detector
    pose_detector = PoseDetector(
        model_path=pose_model_path, device_name=device.split(':')[0], device_id=int(device.split(':')[-1]))

    return detector, pose_detector


# def init_mmpose(det_config, det_checkpoint, pose_config, pose_checkpoint, device):
#     from mmpose.apis.inferencers import Pose2DInferencer
#     from mmdet.apis.det_inferencer import DetInferencer
#     detector = DetInferencer(model=det_config, weights=det_checkpoint, device=device, show_progress=False)
#     pose_estimator = Pose2DInferencer(model=pose_config, weights=pose_checkpoint, device=device,
#                                       det_model="whole-image"
#                                       )
#
#     return detector, pose_estimator


def postprocess(preds, image_bbox_id_list, bbox_track_id_list=None):
    """Merge the data_samples from each image into a single data_sample."""

    data_samples = []
    for image_id, bbox_ids in enumerate(image_bbox_id_list):
        # data_samples.append(merge_data_samples(preds[bbox_ids]))
        image_data_samples = [preds[i] for i in bbox_ids]
        # using sigmoid to make the keypoint score between 0 and 1
        for bbox_i, data_sample in enumerate(image_data_samples):
            if bbox_track_id_list is not None:
                bbox_id = bbox_track_id_list[image_id][bbox_i]
                data_sample.pred_instances.bbox_ids = np.array([bbox_id])
            data_sample.pred_instances.keypoint_scores = data_sample.pred_instances.keypoints_visible
        data_samples.append(merge_data_samples(image_data_samples))
    return data_samples


def data_sample_to_dict(data_sample: List[InstanceData]) -> List[Dict]:
    results = []
    for instance_data in data_sample:
        keypoints = np.concatenate(
            [instance_data.pred_instances.keypoints, instance_data.pred_instances.keypoints_visible[:, :, None]],
            axis=2)
        result = {
            "bboxes": instance_data.pred_instances.bboxes,
            "bbox_scores": instance_data.pred_instances.bbox_scores,
            "bbox_ids": instance_data.pred_instances.bbox_ids,
            "keypoints": keypoints,
            "keypoint_scores": instance_data.pred_instances.keypoints_visible
        }
        results.append(result)
    return results


def preprocess(model,
               inputs,
               batch_size,
               bboxes,
               **kwargs):
    """Process the inputs into a model-feedable format.

    Args:
        inputs (InputsType): Inputs given by user.
        batch_size (int): batch size. Defaults to 1.

    Yields:
        Any: Data processed by the ``pipeline`` and ``collate_fn``.
        List[str or np.ndarray]: List of original inputs in the batch
    """
    all_data_infos = []
    all_inputs = []
    image_bbox_id_list = []  # for each image, list all bbox ids
    bbox_track_id_list = []  # for each bbox, list all track ids
    for i, input in enumerate(inputs):
        bbox = bboxes[i] if bboxes is not None else []
        if len(bbox) == 0:
            image_bbox_id_list.append([])
            continue
        data_infos = preprocess_single(model,
                                       input, index=i, bboxes=bbox, **kwargs)
        bbox_track_id_list.append(list(bbox[:, -1]))
        # only supports inference with batch size 1
        # yield model.collate_fn(data_infos), [input]
        current_len = len(all_data_infos)
        all_inputs += [input] * len(data_infos)
        image_bbox_id_list.append([j + current_len for j in range(len(data_infos))])
        all_data_infos += data_infos
    return model.collate_fn(all_data_infos), all_inputs, image_bbox_id_list, bbox_track_id_list


def build_tracker(tracker_cfg: Dict):
    if tracker_cfg['type'] == 'ocsort':
        from src.modules.player_trackers.ocsort.ocsort import OCSort
        tracker_cfg.pop('type')
        return OCSort(**tracker_cfg)
    else:
        raise NotImplementedError


def preprocess_single(pose_estimator,
                      input,
                      index: int,
                      bbox_thr: float = 0.3,
                      nms_thr: float = 0.3,
                      bboxes: Union[List[List], List[np.ndarray],
                      np.ndarray] = []):
    """Process a single input into a model-feedable format.

    Args:
        input (InputType): Input given by user.
        index (int): index of the input
        bbox_thr (float): threshold for bounding box detection.
            Defaults to 0.3.
        nms_thr (float): IoU threshold for bounding box NMS.
            Defaults to 0.3.

    Yields:
        Any: Data processed by the ``pipeline`` and ``collate_fn``.
    """
    if isinstance(input, str):
        data_info = dict(img_path=input)
    else:
        data_info = dict(img=input, img_path=f'{index}.jpg'.rjust(10, '0'))
    data_info.update(pose_estimator.model.dataset_meta)

    if pose_estimator.cfg.data_mode == 'topdown':
        # bboxes = []
        if pose_estimator.detector is not None:
            raise NotImplementedError

        data_infos = []
        if len(bboxes) > 0:
            for bbox in bboxes:
                inst = data_info.copy()
                inst['bbox'] = bbox[None, :4]
                inst['bbox_score'] = bbox[4:5]
                data_infos.append(pose_estimator.pipeline(inst))
        else:
            inst = data_info.copy()
            h, w = input.shape[:2]
            inst['bbox'] = np.array([[0, 0, w, h]], dtype=np.float32)
            inst['bbox_score'] = np.ones(0, dtype=np.float32)
            data_infos.append(pose_estimator.pipeline(inst))

    else:
        data_infos = [pose_estimator.pipeline(data_info)]

    return data_infos


class RTMPoseEstimator():
    """

    Args:
        model_type (str, ): model type, pytorch or trt
        pose_cfg (str, optional): pose model config file path. Defaults to None.
        pose_checkpoint (str, optional): pose model checkpoint file path. Defaults to None.
        detector_cfg (str, optional): detector model config file path. Defaults to None.
        detector_checkpoint (str, optional): detector model checkpoint file path. Defaults to None.
        bbox_score_thr (float, optional): bbox score threshold. Defaults to 0.3.
        bbox_nms_thr (float, optional): bbox nms threshold. Defaults to 0.3.
        device (str, optional): device name. Defaults to 'cuda:0'.
    """

    def __init__(self,
                 model_type: str,
                 pose_cfg: str = None,
                 pose_checkpoint: str = None,
                 detector_cfg: str = None,
                 detector_checkpoint: str = None,
                 pred_pose=False,
                 tracker_cfg: Dict = None,
                 bbox_score_thr: float = 0.3,
                 bbox_nms_thr: float = 0.3,
                 det_batch_size: int = 6,
                 pose_batch_size: int = 6,
                 device='cuda:0',
                 bbox_filter=None,
                 ):

        self.model_type = model_type
        self.pose_cfg = pose_cfg
        self.pose_checkpoint = pose_checkpoint
        self.detector_cfg = detector_cfg
        self.detector_checkpoint = detector_checkpoint
        self.device = device
        self.bbox_score_thr = bbox_score_thr
        self.bbox_nms_thr = bbox_nms_thr
        self.det_batch_size = det_batch_size
        self.pose_batch_size = pose_batch_size
        self.tracker_cfg = tracker_cfg
        self.pred_pose = pred_pose

        if self.model_type == 'pytorch':
            raise NotImplementedError
        elif self.model_type == 'trt':
            assert self.detector_checkpoint is not None, 'detector checkpoint path is None'
            assert self.pose_checkpoint is not None, 'pose checkpoint path is None'
            self.detector, self.pose_estimator = init_mmpose_trt(self.detector_checkpoint,
                                                                 self.pose_checkpoint, device=device)

        if self.tracker_cfg is not None:
            self.tracker = build_tracker(self.tracker_cfg)
        else:
            self.tracker = None

        self.bbox_filter = bbox_filter
        if bbox_filter is None:
            warnings.warn("bbox filter is None, all bboxes will be used for pose estimation")

        self.court_image_range = None

    def preprocess(self, image_list: List[np.ndarray], camera_params_list: List[Dict]):
        # crop image with court range
        if self.court_image_range is None:
            # get the court range
            self.court_image_range = []
            for i, param in enumerate(camera_params_list):
                court_range = get_basketball_outer_range(param)
                self.court_image_range.append([
                    int(max(court_range[0] - 10, 0)),  # x1
                    int(max(court_range[1] - 10, 0)),  # y1
                    int(min(court_range[2] + 10, image_list[i].shape[1])),  # x2
                    int(min(court_range[3] + 10, image_list[i].shape[0]))  # y2
                ])

        cropped_image = []
        for i, image in enumerate(image_list):
            cropped_image.append(
                image[
                self.court_image_range[i][1]:self.court_image_range[i][3],
                self.court_image_range[i][0]:self.court_image_range[i][2]
                ]
            )
        return cropped_image

    def postprocess(self, results):
        # postprocess the results to match the original image size
        for result in results:
            if result["bboxes"].shape[0] == 0:
                continue
            result["bboxes"][:, 0] += self.court_image_range[0][0]
            result["bboxes"][:, 1] += self.court_image_range[0][1]
            result["bboxes"][:, 2] += self.court_image_range[0][0]
            result["bboxes"][:, 3] += self.court_image_range[0][1]

            result["keypoints"][:, :, 0] += self.court_image_range[0][0]
            result["keypoints"][:, :, 1] += self.court_image_range[0][1]
        return results

    def trt_inference(self, image_list: List[np.ndarray], camera_params_list: List[Dict] = None):
        assert len(image_list) == 1, "only support batch size 1"

        image_list = self.preprocess(image_list, camera_params_list)

        detection_results = self.detector.batch(image_list)
        det_bboxes = []
        results = []
        valid_ids = []
        for vid, pred in enumerate(detection_results):
            bboxes = pred[0]
            scores = bboxes[:, -1]
            bboxes = bboxes[scores > self.bbox_score_thr]
            if len(bboxes) > 0 and self.bbox_filter is not None:
                assert camera_params_list is not None, "camera params is required for bbox filter"
                valid_idx = self.bbox_filter.filter(bboxes, camera_params_list[vid], self.court_image_range[vid])
                bboxes = bboxes[valid_idx]
            results.append({
                "bboxes": bboxes,
                "bbox_scores": bboxes[:, -1],
            })
            bboxes = bboxes[:, :-1]
            if bboxes.shape[0] > 0:
                valid_ids.append(vid)
            else:
                results[vid].update({
                    "keypoints": np.zeros((0, 0, 0)),
                    "keypoint_scores": np.zeros((0,))
                })

            # bboxes[:,2:] += bboxes[:,:2]
            det_bboxes.append(bboxes[:, :4])
            if not self.pred_pose:
                return results

        keypoints_results = self.pose_estimator.batch([image_list[vid] for vid in valid_ids],
                                                      [det_bboxes[vid] for vid in valid_ids])
        for vid, keypoints_result in zip(valid_ids, keypoints_results):
            bboxes = results[vid]["bboxes"].copy()

            if self.bbox_filter is not None:
                valid_idx = self.bbox_filter.remove_duplicate_k2d(keypoints_result)
                keypoints_result = keypoints_result[valid_idx]
                bboxes = bboxes[valid_idx]

            pred_keypoints = keypoints_result[:, :, :3]
            pred_keypoints_scores = keypoints_result[:, :, 2]

            # refine bbox with keypoints, exclude the hand keypoints 7,8,9,10
            min_x = np.stack([pred_keypoints[:, :7, 0].min(axis=1), pred_keypoints[:, 11:, 0].min(axis=1)], axis=1).min(
                axis=1)
            max_x = np.stack([pred_keypoints[:, :7, 0].max(axis=1), pred_keypoints[:, 11:, 0].max(axis=1)], axis=1).max(
                axis=1)

            bboxes[:, 0] = min_x - 10
            bboxes[:, 2] = max_x + 10

            if self.tracker is not None:

                # use head bbox for tracking
                head_keypoints = pred_keypoints[:, [3, 4, 5, 6], :]  # 3,4,5,6 are head keypoints
                head_bboxes = np.concatenate(
                    [head_keypoints[:, :, 0].min(axis=1, keepdims=True),
                     head_keypoints[:, :, 1].min(axis=1, keepdims=True),
                     head_keypoints[:, :, 0].max(axis=1, keepdims=True),
                     head_keypoints[:, :, 1].max(axis=1, keepdims=True),
                     bboxes[:, 4:5]],  # score
                    axis=1
                )

                # tracked_bboxes, det_ids = self.tracker.update(bboxes.astype(np.float32), (1, 1), (1, 1))
                tracked_bboxes, det_ids = self.tracker.update(head_bboxes.astype(np.float32), (1, 1), (1, 1))
                if isinstance(tracked_bboxes, List):
                    tracked_bboxes = np.stack([t.result for t in tracked_bboxes])
                bbox_ids = tracked_bboxes[:, -1]
                # bboxes = tracked_bboxes[:, :5]
            else:
                bbox_ids = np.zeros(bboxes.shape[0]) - 1
                det_ids = np.arange(bboxes.shape[0])

            # bboxes = np.concatenate([bboxes, bbox_ids[:, None]], axis=1)  # x1, y1, x2, y2, score, id
            bboxes = np.concatenate([bboxes[det_ids], bbox_ids[:, None]], axis=1)  # x1, y1, x2, y2, score, id
            pred_keypoints = pred_keypoints[det_ids]
            pred_keypoints_scores = pred_keypoints_scores[det_ids]

            if len(pred_keypoints_scores) != 0 and pred_keypoints_scores.max() > 1:
                pred_keypoints_scores /= 2
            results[vid].update({
                "bboxes": bboxes[:, :-1],
                "bbox_ids": bbox_ids,
                "keypoints": pred_keypoints,
                "keypoint_scores": pred_keypoints_scores
            })
        # postprocess the results
        results = self.postprocess(results)
        return results

    def __call__(self, image_list: list, camera_params_list: List[Dict]):
        """inference pose estimator

        Args:
            image_list (list): image list of numpy array

        Returns:
            list: pose result
        """

        if self.model_type == 'trt':
            pose_results = self.trt_inference(image_list, camera_params_list)
        else:
            raise NotImplementedError

        return pose_results


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__dict__[key] = DictToObject(value)
            else:
                self.__dict__[key] = value
