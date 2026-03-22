"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np

from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# def iou_batch(bb_test, bb_gt):
#   """
#   From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
#   """
#   bb_gt = np.expand_dims(bb_gt, 0)
#   bb_test = np.expand_dims(bb_test, 1)
#
#   xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#   yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#   xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#   yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#   w = np.maximum(0., xx2 - xx1)
#   h = np.maximum(0., yy2 - yy1)
#   wh = w * h
#   o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
#     + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
#   return(o)


# def convert_bbox_to_z(bbox):
#   """
#   Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#     [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#     the aspect ratio
#   """
#   w = bbox[2] - bbox[0]
#   h = bbox[3] - bbox[1]
#   x = bbox[0] + w/2.
#   y = bbox[1] + h/2.
#   s = w * h    #scale is just area
#   r = w / float(h)
#   return np.array([x, y, s, r]).reshape((4, 1))


# def convert_x_to_bbox(x,score=None):
#   """
#   Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
#     [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
#   """
#   w = np.sqrt(x[2] * x[3])
#   h = x[2] / w
#   if(score==None):
#     return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
#   else:
#     return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
def convert_k3d_to_x(k3d, keypoints_converstion='coco'):
    """
    Takes a 3D keypoints in the form [x1,y1,z1,x2,y2,z2,...] and returns it in the form
      [x,y,z] where x,y,z is the mid hip position
    """
    if keypoints_converstion == 'coco':
        mid_hip = (k3d[11] + k3d[12]) / 2
        nose = k3d[0]
        nose_mid_hip = nose - mid_hip
    else:
        raise NotImplementedError
    return np.concatenate([mid_hip, nose_mid_hip], axis=-1).reshape((6,1))


def convert_k3ds_to_x(k3ds, keypoints_converstion='coco'):
    # k3ds n x n_kps * 3
    if keypoints_converstion == 'coco':
        neck = (k3ds[:, 5] + k3ds[:, 6]) / 2
        # nose = k3ds[:, 0]
        nose = (k3ds[:, 3] + k3ds[:, 4]) / 2
        nose_mid_hip = nose - neck
        scores = (neck[:, 3] + nose[:, 3])/2
        neck = neck[:, :3]
        nose_mid_hip = nose_mid_hip[:, :3]
    else:
        raise NotImplementedError
    return np.concatenate([neck, nose_mid_hip], axis=-1).reshape(-1, 6), scores


def dist_batch(x_test, x_gt, x_reid=None, x_gt_reid=None, position_weight=1, orientation_weight=1, limb_weight=1, reid_temp=300):
    # x_test: n x 6, x_gt: mx6  return n x m
    # x:[[mid_hip[0], mid_hip[1], mid_hip[2], nose_mid_hip[0], nose_mid_hip[1], nose_mid_hip[2]]]
    x_test = x_test.reshape(-1,1, 2, 3)
    x_gt = x_gt.reshape(1,-1, 2, 3)
    position_dist = np.linalg.norm(x_test[:,:,0] - x_gt[:,:,0], axis=-1)
    # orientation_dist = 1 - cos_sim
    orientation_dist = 1 - np.sum(x_test[:,:,1] * x_gt[:,:,1], axis=-1)/np.linalg.norm(x_test[:,:,1], axis=-1)/np.linalg.norm(x_gt[:,:,1], axis=-1)
    # limb_dist = np.linalg.norm(x_test[:,:,1] - x_gt[:,:,1], axis=-1)
    limb_dist = np.abs(np.linalg.norm(x_test[:,:,1], axis=-1) -  np.linalg.norm(x_gt[:,:,1], axis=-1))
    reid_dist = limb_dist * 0

    space_dist = position_dist * position_weight + orientation_dist * orientation_weight + limb_dist * limb_weight
    # a = (space_dist < 0.4).astype(np.int32)
    # ignore_gt = np.where(a.sum(0) > 1)[0]
    # ignore_test = np.where(a.sum(1) > 1)[0]
    # ignore_gt_test = []
    # for it in ignore_test:
    #     ignore_gt_test+=list(np.where(a[it] == 1)[0])
    # ignore_gt = list(set(ignore_gt_test)) + list(ignore_gt)

    # if len(ignore_gt) > 0 or len(ignore_test) > 0:
    #     print("player occulation detected!")
    # space_dist[ignore_test] = 1000
    # space_dist[:, ignore_gt] = 1000

    # temp = 200
    if x_reid is not None and x_gt_reid is not None:
        for i in range(len(reid_dist)):
            for j in range(len(reid_dist[0])):
                if x_reid[i][1] < 1000 and x_gt_reid[j][1] < 1000 and  x_reid[i][0] != x_gt_reid[j][0] and x_reid[i][0] != -1 and x_gt_reid[j][0] != -1:
                    reid_dist[i,j] += reid_temp/(x_reid[i][1] + x_gt_reid[j][1])
    total_dist = space_dist * 0.5 + reid_dist



    a = (total_dist < 0.1).astype(np.int32)
    ignore_gt = np.where(a.sum(0) > 5)[0]
    ignore_test = np.where(a.sum(1) > 1)[0]
    # ignore_gt_test = []
    # for it in ignore_test:
    #     ignore_gt_test+=list(np.where(a[it] == 1)[0])
    # ignore_gt = list(set(ignore_gt_test)) + list(ignore_gt)
    #
    # if len(ignore_gt) > 0 or len(ignore_test) > 0:
    #     print("player overlapping detected!")
    # total_dist[ignore_test] = 1000
    # total_dist[:, ignore_gt] = 1000


    return total_dist, ignore_gt





class KalmanK3DTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, k3d, keypoints_convention='coco'):
        """
        Initialises a tracker using initial keypoints.
        x = [x,y,z of the mid hip, x,y,z of the nose - x,y,z of the mid hip]
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=9, dim_z=6)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1],
             ])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0],
             ])

        self.kf.R[3:, 3:] *= 10.
        self.kf.P[6:, 6:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[6:, 6:] *= 0.01

        # self.kf.x[:4] = convert_bbox_to_z(bbox)
        if k3d.shape[0] > 6:
            k3d = convert_k3d_to_x(k3d, keypoints_convention)
        self.kf.x[:6] = k3d.reshape((6, 1))
        self.time_since_update = 0
        self.id = KalmanK3DTracker.count
        KalmanK3DTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.current_det_index = -1

        self.reid = [-1, 10000]

    def update(self, k3d):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if k3d.shape[0] > 6:
            k3d = convert_k3d_to_x(k3d)
        self.kf.update(k3d)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if ((self.kf.x[6] + self.kf.x[2]) <= 0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:6].reshape((1, 6)))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        # return convert_x_to_bbox(self.kf.x)
        return self.kf.x[:6].reshape((1, 6))


def associate_detections_to_trackers(detections, trackers, detection_reid=None, trackers_reid=None, dist_threshold=1.0):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int), []

    # iou_matrix = iou_batch(detections, trackers)
    dist_matrix, ignore_gt = dist_batch(detections, trackers, detection_reid, trackers_reid)

    if min(dist_matrix.shape) > 0:
        a = (dist_matrix < dist_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(dist_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (dist_matrix[m[0], m[1]] > dist_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), ignore_gt


class K3dSort(object):
    def __init__(self, det_thresh=-1, max_age=30, min_hits=3, dist_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh

    def reset(self):
        self.trackers = []
        self.frame_count = 0

    def update(self, output_results, reid_results=None):
        """
        Params:
          dets - a numpy array of k3ds, n x n_kps * 4
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # post_process detections
        # output_results = output_results.cpu().numpy()
        # scores = output_results[:, 4] * output_results[:, 5]
        # bboxes = output_results[:, :4]  # x1y1x2y2
        # img_h, img_w = img_info[0], img_info[1]
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale
        dets, scores = convert_k3ds_to_x(output_results)
        # dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        if reid_results is not None:
            dets_reid = reid_results[remain_inds]
        else:
            dets_reid = None
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        trks_reid = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            # trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
            trks_reid.append(self.trackers[t].reid)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks , ignore_gt = associate_detections_to_trackers(dets, trks, dets_reid, trks_reid, self.dist_threshold)
        for ignore_t in ignore_gt:
            self.trackers[ignore_t].time_since_update = self.max_age+1

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].current_det_index = m[0]

            if dets_reid is not None and dets_reid[m[0],0]!=-1 and (self.trackers[m[1]].reid[0] == -1 \
                    or self.trackers[m[1]].reid[1] > dets_reid[m[0], 1] \
                    or self.trackers[m[1]].reid[0]!= dets_reid[m[0], 0]):
                self.trackers[m[1]].reid = dets_reid[m[0]]

        for m in unmatched_trks:
            self.trackers[m].current_det_index = -1

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # trk = KalmanBoxTracker(dets[i, :])
            trk = KalmanK3DTracker(dets[i, :])
            self.trackers.append(trk)
            trk.current_det_index = i

        track_ids = []
        det_index = []

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                # ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
                track_ids.append(trk.id+1)
                det_index.append(trk.current_det_index)
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(track_ids) > 0):
            # return np.concatenate(ret)
            return track_ids, det_index
        # return np.empty((0, 5))
        return [], []
