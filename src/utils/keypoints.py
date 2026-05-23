import numpy as np

BODY25_IN_WHOLEBODY133 = [0,
                          -1,  # Neck
                          6, 8, 10,
                          5, 7, 9,
                          -1,  # MidHip
                          12, 14, 16,
                          11, 13, 15,
                          2, 1,
                          4, 3,
                          17, 18, 19,
                          20, 21, 22
                          ]

LHADN21_IN_WHOLEBODY133 = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                           102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
RHADN21_IN_WHOLEBODY133 = [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                           124, 125, 126, 127, 128, 129, 130, 131, 132]


def wholebody133tobody25(points2d):
    dim = 3
    if len(points2d.shape) == 2:
        points2d = points2d[None, :, :]
        dim = 2
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, :, :2] = points2d[:, BODY25_IN_WHOLEBODY133, :2]
    kpts[:, :, 2:3] = points2d[:, BODY25_IN_WHOLEBODY133, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)

    if dim == 2:
        kpts = kpts[0]
    return kpts


def wholebody133tobody25hand(points2d):
    body_kpts = wholebody133tobody25(points2d)
    # ignore foot
    body_kpts[:, 19:22, :] = 0
    body_kpts[:, 22:25, :] = 0

    lh_kpts = np.zeros((points2d.shape[0], 21, 3))
    rh_kpts = np.zeros((points2d.shape[0], 21, 3))
    lh_kpts[:, :, :2] = points2d[:, LHADN21_IN_WHOLEBODY133, :2]
    lh_kpts[:, :, 2:3] = points2d[:, LHADN21_IN_WHOLEBODY133, 2:3]
    rh_kpts[:, :, :2] = points2d[:, RHADN21_IN_WHOLEBODY133, :2]
    rh_kpts[:, :, 2:3] = points2d[:, RHADN21_IN_WHOLEBODY133, 2:3]

    body25hand_kpts = np.concatenate((body_kpts, lh_kpts, rh_kpts), axis=1)
    return body25hand_kpts


COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
BODY25_IN_HALPE26 = [0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 20, 22, 24, 21, 23, 25]
BODY25_IN_MEDIAPIPE33 = [0,
                         -1,  # Neck
                         12, 14, 16,
                         11, 13, 15,
                         -1,  # MidHip
                         24, 26, 28,
                         23, 25, 27,
                         5, 2,
                         8, 7,
                         31, 31, 29,
                         32, 32, 30]


def coco17tobody25(points2d):
    dim = 3
    if len(points2d.shape) == 2:
        points2d = points2d[None, :, :]
        dim = 2
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    if dim == 2:
        kpts = kpts[0]
    return kpts


def mediapipe33tobody25(points2d):
    dim = 3
    if len(points2d.shape) == 2:
        points2d = points2d[None, :, :]
        dim = 2
    kpts = np.zeros((points2d.shape[0], 25, 3), dtype=points2d.dtype)
    valid_body25_ids = [idx for idx, source_idx in enumerate(BODY25_IN_MEDIAPIPE33) if source_idx >= 0]
    source_ids = [BODY25_IN_MEDIAPIPE33[idx] for idx in valid_body25_ids]
    kpts[:, valid_body25_ids, :2] = points2d[:, source_ids, :2]
    kpts[:, valid_body25_ids, 2:3] = points2d[:, source_ids, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    if dim == 2:
        kpts = kpts[0]
    return kpts


def halpe26tobody25(points2d):
    last_dim = points2d.shape[-1]
    dim = 3
    if len(points2d.shape) == 2:
        points2d = points2d[None, :, :]
        dim = 2
    kpts = np.zeros((points2d.shape[0], 25, last_dim), dtype=points2d.dtype)
    kpts[:, :, :-1] = points2d[:, BODY25_IN_HALPE26, :-1]
    kpts[:, :, -1:] = points2d[:, BODY25_IN_HALPE26, -1:]

    # Rebuild Body25 neck and mid-hip from shoulder/hip pairs for consistency
    # with the other Body25 converters used in the pipeline.
    kpts[:, 1, :-1] = kpts[:, [2, 5], :-1].mean(axis=1)
    kpts[:, 1, -1] = kpts[:, [2, 5], -1].min(axis=1)
    kpts[:, 8, :-1] = kpts[:, [9, 12], :-1].mean(axis=1)
    kpts[:, 8, -1] = kpts[:, [9, 12], -1].min(axis=1)

    neck_fallback = points2d[:, 18]
    neck_missing = kpts[:, 1, -1] <= 0
    if np.any(neck_missing):
        kpts[neck_missing, 1] = neck_fallback[neck_missing]

    midhip_fallback = points2d[:, 19]
    midhip_missing = kpts[:, 8, -1] <= 0
    if np.any(midhip_missing):
        kpts[midhip_missing, 8] = midhip_fallback[midhip_missing]

    if dim == 2:
        kpts = kpts[0]
    return kpts


def body25tococo17(points):
    last_dim = points.shape[-1]
    dim = 3
    if len(points.shape) == 2:
        points = points[None, :, :]
        dim = 2
    kpts = np.zeros((points.shape[0], 17, last_dim))
    kpts[:, :, :-1] = points[:, COCO17_IN_BODY25, :-1]
    kpts[:, :, -1:] = points[:, COCO17_IN_BODY25, -1:]
    if dim == 2:
        kpts = kpts[0]
    return kpts


def body25tolocation(k3d):
    assert k3d.shape[0] == 25, "k3d should have 25 joints"
    referent_joints = k3d[[1, 2, 5, 8]]
    referent_joints_mean = np.sum(referent_joints[:, :2] * referent_joints[:, 3:4], axis=0) / (np.sum(referent_joints[:, 3:4]) + 1e-6)
    confidence = np.mean(referent_joints[:, 3])
    return np.concatenate([referent_joints_mean, np.array([confidence])], axis=0)


def coco17tolocation(k3d):
    assert k3d.shape[0] == 17, "k3d should have 17 joints"
    # use the points 3,4,5,6
    referent_joints = k3d[[3, 4, 5, 6]]
    # get the mean by confidence
    referent_joints_mean = np.sum(referent_joints[:,:2] * referent_joints[:,3:4], axis=0) / (np.sum(referent_joints[:,3:4]) + 1e-6)
    confidence = np.mean(referent_joints[:, 3])
    return np.concatenate([referent_joints_mean, np.array([confidence])], axis=0)


def keypoints3dtolocation(k3d):
    if k3d.shape[0] == 25:
        return body25tolocation(k3d)
    if k3d.shape[0] == 17:
        return coco17tolocation(k3d)
    raise AssertionError(f"k3d should have 17 or 25 joints, but got {k3d.shape[0]}")


s_body25_flip_pairs = np.array(
    [[2, 5], [3, 6], [4, 7], [9, 12], [10, 13], [11, 14], [15, 16], [17, 18], [22, 19], [23, 20], [24, 21]], dtype=int)
s_body25_parent_ids = np.array([0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11],
                               dtype=int)

s_coco_flip_pairs = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=int)
s_coco_parent_ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14], dtype=int)

s_halpe26_flip_pairs = np.array(
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [20, 21], [22, 23], [24, 25]],
    dtype=int,
)
s_halpe26_parent_ids = np.array([18, 0, 0, 1, 2, 18, 18, 5, 6, 7, 8, 19, 19, 11, 12, 13, 14, 0, 19, 19, 15, 16, 15, 16, 15, 16], dtype=int)

joints_dict = {
    # 'aqa': (s_aqa_flip_pairs, s_aqa_parent_ids),
    'openpose_25': (s_body25_flip_pairs, s_body25_parent_ids),
    # 'openpose_25_hand': (s_body25hand_flip_pairs, s_body25hand_parent_ids)
    "coco": (s_coco_flip_pairs, s_coco_parent_ids),
    "halpe26": (s_halpe26_flip_pairs, s_halpe26_parent_ids),
}
