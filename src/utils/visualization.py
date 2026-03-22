from typing import Union

import numpy as np
import cv2
import random

from .keypoints import joints_dict


def cv_draw_joints(im, kpt, vis, flip_pair_ids, color_left=(255, 0, 0), color_right=(0, 255, 0), radius=2):
    for ipt in range(0, kpt.shape[0]):
        if vis[ipt, 0]:
            cv2.circle(im, (int(kpt[ipt, 0] + 0.5), int(kpt[ipt, 1] + 0.5)), radius, color_left, -1)
    for i in range(0, flip_pair_ids.shape[0]):
        id = flip_pair_ids[i][0]
        if vis[id, 0]:
            cv2.circle(im, (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)), radius, color_right, -1)


def cv_draw_joints_parent(im, kpt, vis, parent_ids, color=(0, 0, 255), thickness=1):
    for i in range(0, len(parent_ids)):
        id = parent_ids[i]
        if vis[id, 0] and vis[i, 0]:
            cv2.line(im, (int(kpt[i, 0] + 0.5), int(kpt[i, 1] + 0.5)), (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)),
                     color, thickness=thickness)


def visualize_joints2d(image: np.ndarray,
                       joints: np.ndarray,
                       convention: str = 'aqa',
                       joints_mask: np.ndarray = None,
                       color_left: tuple = (255, 0, 0),
                       color_right: tuple = (0, 255, 0),
                       threshold: float = 0.5,
                       name: str = None,
                       radius: int = 4) -> np.ndarray:
    '''
    plot 2d joints on image
    Args:
        image (np.ndarray): [h, w, c]
        joints (np.ndarray): [n_joints, 2]
        convention (str): 'aqa' or 'openpose_25'
        joints_mask (np.ndarray): [n_joints, 1]
        color_left (tuple): BGR format
        color_right (tuple): BGR format
        radius (int): the radius of joint
        name (str): the name of the person
    Returns:
        image (np.ndarray): [h, w, c]
    '''

    if joints_mask is None:
        if joints.shape[-1] > 2:
            joints_mask = joints[..., 2:] > threshold
        else:
            joints_mask = np.ones_like(joints)
    if convention in joints_dict.keys():
        flip_pairs, parent_ids = joints_dict[convention]
    else:
        raise NotImplementedError
    if len(joints.shape) == 3:  # multi person
        for i in range(joints.shape[0]):
            visualize_joints2d(image, joints[i], convention=convention, joints_mask=joints_mask[i])
        return image
    cv_draw_joints(image, joints, joints_mask, flip_pairs, color_left=color_left, color_right=color_right,
                   radius=radius)
    cv_draw_joints_parent(image, joints, joints_mask, parent_ids)
    if name is not None:
        cv2.putText(image, name, (int(joints[0, 0] + 0.5), int(joints[0, 1] + 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color_left, 2)
    return image


# def visualize_joints3d(keypoints_arr: np.ndarray,
#                        camera: Union[list, dict],
#                        background_arr: Union[list, np.ndarray, None] = None,
#                        camera_width: int = None,
#                        camera_height: int = None,
#                        width: int = None,
#                        height: int = None,
#                        name: str = None,
#                        color: tuple = (0, 0, 255),
#                        convention: str = 'aqa'):
#     """
#     visualize 3d joints on background
#     Args:
#
#         keypoints_arr: np.ndarray, [n_frame, n_joints, 4]
#         camera: dict or list of dict, list means multi-view, camera format: easymocap, {"K":,"invK":,"R":,"T":}
#         background_arr: np.ndarray, [n_camera, n_frame, h, w, c] or list of np.ndarray
#          background_arr should be multi-view too
#         camera_width: camera width for the intri and extri
#         camera_height: camera height for the intri and extri
#         height: image height
#         width: image width
#         convention: 'aqa' or 'openpose_25'
#
#     Returns:
#         images: np.ndarray, [n_camera, n_frame, h, w, c]
#     """
#     joints = copy.deepcopy(keypoints_arr)
#     assert len(joints.shape) == 3
#     assert joints.shape[2] >= 3
#
#     if color is not None:
#         color_left = color
#         color_right = color_left
#     else:
#         color_left = (255, 0, 0)
#         color_right = (0, 255, 0)
#
#     n_frame, n_kpts, _ = joints.shape
#
#     if isinstance(camera, dict):
#         camera = [camera]
#
#     if background_arr is None:
#         assert width is not None and height is not None
#         background_arr = np.zeros((len(camera), n_frame, height, width, 3), dtype=np.uint8)
#
#     # copy background_arr to [1, 1, h, w, c]
#     if len(background_arr.shape) == 3:
#         background_arr = np.expand_dims(background_arr, axis=0)
#         background_arr = np.expand_dims(background_arr, axis=0)
#
#     assert background_arr.shape[0] == len(camera)
#     assert background_arr.shape[1] == n_frame
#
#     if joints.shape[2] > 3:
#         joints_score = joints[..., 3:]
#         joints = joints[..., :3]
#     else:
#         joints_score = [None] * joints.shape[0]
#
#     for i_cam, cam in enumerate(camera):  # multi view
#         assert isinstance(cam, dict)
#         assert "K" in cam.keys()
#         assert "R" in cam.keys()
#         assert "T" in cam.keys()
#         assert "dist" in cam.keys()
#
#         view_points2d = camera_project(joints.reshape((n_frame * n_kpts, 3)), cam)
#         view_points2d = view_points2d.reshape((n_frame, n_kpts, 2))
#
#         if camera_width is not None and camera_height is not None:
#             view_points2d[..., 0] = view_points2d[..., 0] / camera_width * background_arr.shape[3]
#             view_points2d[..., 1] = view_points2d[..., 1] / camera_height * background_arr.shape[2]
#         # views_points.append(view)
#
#         for i_frame, joint in enumerate(view_points2d):  # multi frame
#             # image = np.zeros((height, width, 3), np.uint8)
#             # ignore the joints out of the image
#             # if joint.min() < 0 or joints[:, 0].max() > background_arr.shape[3] or joints[:, 1].max() > \
#             #         background_arr.shape[2]:
#             #     continue
#             visualize_joints2d(background_arr[i_cam, i_frame], joint.astype("float32"), convention=convention,
#                                joints_mask=joints_score[i_frame], name=name, color_left=color_left,
#                                color_right=color_right)
#             # multi_frame_images.append(background_arr[i_cam, i_frame])
#     return background_arr


# class BasketballCourt2d:
#
#     def __init__(self):
#
#         self.base_court_background = cv2.imread("modules/utils/tactical-board.png")
#         self.image_court_corner = [
#             [102]
#         ]


def get_random_color(idx):
    """
    get random color
    Args:
        idx: index
    Returns:
        color: (r, g, b)
    """
    random.seed(int(idx))
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

# def debug_triagulation(images, keypoints2d, Vused, kptsRepro=None):
#     # from modules.utils.visualization import visualize_joints2d
#     import matplotlib.pyplot as plt
#     for i, v in enumerate(Vused):
#         plt.figure(figsize=(images[v].shape[1] / 100, images[v].shape[0] / 100))
#         kps = keypoints2d[i]
#         image = images[v].copy()
#         visualize_joints2d(image, joints=kps, convention='openpose_25', threshold=0.1)
#         if kptsRepro is not None:
#             kpts = kptsRepro[i]
#             visualize_joints2d(image, joints=kpts, convention='openpose_25', radius=1, threshold=0.1)
#         plt.imshow(image[:, :, ::-1])
#         plt.show()
