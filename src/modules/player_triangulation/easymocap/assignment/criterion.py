'''
  @ Date: 2021-05-28 16:36:45
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:48:57
  @ FilePath: /EasyMocapRelease/easymocap/assignment/criterion.py
'''
import numpy as np

class BaseCrit:
    def __init__(self, min_conf, min_joints=3) -> None:
        self.min_conf = min_conf
        self.min_joints = min_joints
        self.name = self.__class__.__name__

    def __call__(self, keypoints3d, **kwargs):
        # keypoints3d: (N, 4)
        conf = keypoints3d[..., -1]
        conf[conf<self.min_conf] = 0
        idx = keypoints3d[..., -1] > self.min_conf
        return len(idx) > self.min_joints

class CritWithTorso(BaseCrit):
    def __init__(self, torso_idx, min_conf, **kwargs) -> None:
        super().__init__(min_conf)
        self.idx = torso_idx
        self.min_conf = min_conf
    
    def __call__(self, keypoints3d, **kwargs) -> bool:
        self.log = '{}'.format(keypoints3d[self.idx, -1])
        return (keypoints3d[self.idx, -1] > self.min_conf).all()

class CritLenTorso(BaseCrit):
    def __init__(self, src, dst, min_torso_length, max_torso_length, min_conf) -> None:
        super().__init__(min_conf)
        self.src = src
        self.dst = dst
        self.min_torso_length = min_torso_length
        self.max_torso_length = max_torso_length
    
    def __call__(self, keypoints3d, **kwargs):
        """length of torso"""
        # eps = 0.1
        # MIN_TORSO_LENGTH = 0.3
        # MAX_TORSO_LENGTH = 0.8
        if (keypoints3d[[self.src, self.dst], -1] < self.min_conf).all():
            # low confidence, skip
            return True
        length = np.linalg.norm(keypoints3d[self.dst, :3] - keypoints3d[self.src, :3])
        self.log = '{}: {:.3f}'.format(self.name, length)
        if length < self.min_torso_length or length > self.max_torso_length:
            return False
        return True

class CritRange(BaseCrit):
    def __init__(self, minr, maxr, rate_inlier, min_conf) -> None:
        super().__init__(min_conf)
        self.min = minr
        self.max = maxr
        self.rate = rate_inlier
    
    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        crit = (k3d[:, 0] > self.min[0]) & (k3d[:, 0] < self.max[0]) &\
        (k3d[:, 1] > self.min[1]) & (k3d[:, 1] < self.max[1]) &\
        (k3d[:, 2] > self.min[2]) & (k3d[:, 2] < self.max[2])
        self.log = '{}: {}'.format(self.name, k3d)
        return crit.sum()/crit.shape[0] > self.rate

class CritMinMax(BaseCrit):  
    def __init__(self, max_human_length, min_conf) -> None:
        super().__init__(min_conf)
        self.max_human_length = max_human_length

    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        if sum(idx) == 0:
            return False
        mink = np.min(k3d, axis=0)
        maxk = np.max(k3d, axis=0)
        length = max(np.abs(maxk - mink))
        self.log = '{}: {:.3f}'.format(self.name, length)
        return length < self.max_human_length

class CritLimbLength(BaseCrit):
    def __init__(self, body_type, max_rate, min_conf) -> None:
        super().__init__(min_conf)
        self.body_type = body_type
        self.max_rate= max_rate
        from ..dataset.config import CONFIG
        config = CONFIG[body_type]
        self.skeleton = config['skeleton']
    
    def __call__(self, keypoints3d, **kwargs):
        valid = True
        for (i, j), info in self.skeleton.items():
            if keypoints3d[i, 3] < self.min_conf or keypoints3d[j, 3] < self.min_conf:
                continue
            l_mean = info['mean']
            l_est = np.linalg.norm(keypoints3d[i, :3] - keypoints3d[j, :3])
            if l_mean > 0.15: # 超过十五厘米的 用均值判断
                l_std = info['std']
                rate = abs(l_est - l_mean)/l_mean
                if rate > self.max_rate:
                    valid = False
                    break
            else:
                if l_est > 0.3:
                    valid = False
                    break
        return valid


class CritMaxJumpHeight(BaseCrit):
    def __init__(self, foot_id,max_jump_height=0.8, min_conf=0.1) -> None:
        super().__init__(min_conf)
        self.foot_id = foot_id
        self.max_jump_height = max_jump_height

    def __call__(self, keypoints3d, **kwargs):
        if (keypoints3d[self.foot_id, -1] < self.min_conf).all():
            return True
        if (keypoints3d[self.foot_id, 2]>self.max_jump_height).any() or (keypoints3d[self.foot_id, 2]<0).any():
            return False
        return True


class CritTorsoWidth(BaseCrit):
    def __init__(self, torso_pairs, min_conf, min_width, max_width) -> None:
        super().__init__(min_conf)
        self.torso_pairs = torso_pairs
        self.min_width = min_width
        self.max_width = max_width

    def __call__(self, keypoints3d, **kwargs):
        for idx in self.torso_pairs:
            if (keypoints3d[idx, -1] < self.min_conf).any():
                return True
        widths = []
        vectors = []
        for torso_idx in self.torso_pairs:
            vec = keypoints3d[torso_idx[0], :3] - keypoints3d[torso_idx[1], :3]
            width = np.linalg.norm(vec)
            vectors.append(vec)
            widths.append(width)
            if width < self.min_width or width > self.max_width:
                return False

        # check if the two vectors are parallel, and length is similar
        vec1 = vectors[0]
        vec2 = vectors[1]
        if np.dot(vec1, vec2) < 0:
            return False
        if abs(widths[0] - widths[1]) > self.min_width * 2:
            return False
        return True