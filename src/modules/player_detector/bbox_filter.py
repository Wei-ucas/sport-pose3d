import numpy as np

from src.utils.camera import image2ground
from typing import Dict, List


class BboxFilter:
    """
    Filter the bbox with the ground plane, bbox not in the court will be filtered out.
    """

    def __init__(self,camera_params=None):
        if camera_params is not None:
            self.set_camera_params(camera_params)
        self.court_range_x = [-0.5, 28.5]
        self.court_range_y = [-0.5, 15.5]

    def set_camera_params(self, camera_params):
        self.camera_params = camera_params
        self.camera_params['inv_P'] = np.linalg.inv(self.camera_params['RT'][:, [0, 1, 3]]) @ self.camera_params[
            'invK']
        self.view_id = 0

    def filter(self, bboxes: np.ndarray, camera_params: Dict[str, np.ndarray] = None,
               court_crop_range: List[float] = None) -> List[bool]:
        """
        Filter the bbox with the ground plane, bbox not in the court will be filtered out.
        Args:
            bboxes: np.ndarray, the person bbox
            view_id: int, the view id of the camera
            camera_params: Dict, if None, use previous camera params
        Returns:
            List[Dict], the filtered person dict
        """
        bboxes = bboxes.copy()
        bboxes[:, 0::2] += court_crop_range[0]
        bboxes[:, 1::2] += court_crop_range[1]
        if camera_params is not None:
            self.set_camera_params(camera_params)
            # print("update camera")
        bboxes_bottom_center = np.concatenate([(bboxes[:, :1] + bboxes[:, 2:3]) / 2, bboxes[:, 3:4]], axis=1)
        ground_points = image2ground(bboxes_bottom_center, self.camera_params)
        valid_idx = np.logical_and(np.logical_and(
            np.logical_and(ground_points[:, 0] > self.court_range_x[0], ground_points[:, 0] < self.court_range_x[1]),
            np.logical_and(ground_points[:, 1] > self.court_range_y[0], ground_points[:, 1] < self.court_range_y[1]),
        ), bboxes_bottom_center[:, 1] < (court_crop_range[3] - 5))
        return valid_idx

    def remove_duplicate_k2d(self, k2d_dets: np.ndarray):
        k3d_dist_mat = np.zeros((len(k2d_dets), len(k2d_dets)))
        k3d_dist_mat = np.linalg.norm(k2d_dets[:, None, :, :2] - k2d_dets[None, :, :, :2], axis=-1)
        k3d_dist_mat += np.eye(len(k2d_dets))[:, :, None] * 1000
        duplicate = np.where((k3d_dist_mat < 3).sum(-1) > 10)
        for i, j in zip(duplicate[0], duplicate[1]):
            if k2d_dets[i, :, 2].mean() > k2d_dets[j, :, 2].mean():
                k2d_dets[j] = 0
            else:
                k2d_dets[i] = 0
        valid_index = np.where(k2d_dets[:, 0, 2] > 0)[0]
        return valid_index
