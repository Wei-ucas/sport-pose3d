from typing import Optional, List, Union, Dict, Tuple
import numpy as np
import torch

from src.modules.base_module import BaseModule
from src.structures.frame import Frame
from .utils import linear_assignment, ious, crop_player_photo, load_player_profiles


def build_reid_module(match_threshold=500,
                      config_file_path='src/modules/player_matcher/SOLIDER_ReID/configs/market/swin_base.yml',
                      checkpoint='cpks/reid_msmt17_transformer_120.pth'):
    from .solider_reid import ReID
    return ReID(
        match_threshold=match_threshold,
        config_file=config_file_path,
        checkpoint=checkpoint
    )


class PlayerReIDMatch(BaseModule):

    def __init__(self, profile_file_path: str, reid_match_threshold=500, max_gallery_profile_num: int = 20,
                 track_link_overlap_threshold: float = 0.5, valid_photo_threshold: float = 0.7,
                 min_query_photo_num: int = 2,
                 max_query_photo_num: int = 10,
                 video_fps: int = 25
                 ) -> None:
        """
        Process tracking results, break the tracking id when the bbox is overlapped, and match the tracking sequence with the reid profile
        Args:
            reid_module: The reid module
            track_link_iou_threshold: 当一个bbox和其他bbox的iou大于该阈值时，认为是重叠状态，跟踪的ID应当断开
            valid_photo_threshold: the score threshold to determine the bbox is valid to compute the reid
            min_query_photo_num: minimum bbox number of a tracking sequence to compute the reid
            max_query_photo_num: maximum bbox number of a tracking sequence to compute the reid
        """
        super().__init__("PlayerReidMatch")
        self.reid_match_threshold = reid_match_threshold
        self.REID = build_reid_module(match_threshold=reid_match_threshold)
        self.logger.info("PlayerReidMatch initialized with ReID module.")

        self.profile_file_path = profile_file_path
        self.track_link_overlap_threshold = track_link_overlap_threshold
        self.valid_photo_threshold = valid_photo_threshold
        self.max_gallery_profile_num = max_gallery_profile_num
        self.min_query_photo_num = min_query_photo_num
        self.max_query_photo_num = max_query_photo_num
        self.video_fps = video_fps

        self.player_photo_min_step = int(0.2 * self.video_fps)  # 至少每隔0.2秒取一次照片
        self.player_max_disappear_frame = int(0.5 * self.video_fps)  # 每个ID最多消失0.5秒
        self.init()

    def init(self):
        # init cache variables
        self.re_tracking_id_dict: Dict[int, int] = {}  # 将旧ID映射为新跟踪ID
        self.tracking_overlaped: Dict[int, bool] = {}  # 旧ID是否是重叠状态
        self.tracking_player_photos: Dict[int, List[np.ndarray]] = {}  # 每个新跟踪ID对应的跟踪序列照片
        self.tracking_player_photos_frameid: Dict[int, List[int]] = {}  # 每个新跟踪ID对应的跟踪序列照片的帧id
        self.tracking_player_disappear_frame_num: Dict[int, int] = {}  # 每个新跟踪ID对应的消失帧数

        self.reid_to_tracking: Dict[int, List[int]] = {}  # reid id to tracking ids
        self.tracking_to_reid: Dict[int, Tuple[int, np.ndarray]] = {}  # tracking id to (reid id, distance)
        self.reid_profile: Dict[int, List[np.ndarray]] = {}  # reid id to profile images
        self.reid_features: Dict[int, torch.Tensor] = {}  # reid id to features
        self.reid_to_name: Dict[int, str] = {}  # reid id to player name

        # self.reid_end_frame: Dict[int, int] = {}
        # self.reid_bbox_seq: Dict[int, List[np.ndarray]] = {}
        # self.tracking_start_frame: Dict[int, int] = {}
        self.tracking_bbox_seq: Dict[int, List[np.ndarray]] = {}

        self.frame_count = 0
        self.current_max_tracking_id = -1

        # load player profiles
        self.init_player_profile(
            profile_file_path=self.profile_file_path,
            max_gallery_profile_num=self.max_gallery_profile_num
        )

    def init_player_profile(self,
                            profile_file_path: str,
                            max_gallery_profile_num: int = 20):
        player_profiles = load_player_profiles(profile_file_path)
        for player_name, profile in player_profiles.items():
            rid = profile['tmp_player_id']
            if len(profile['profile_image']) == 0:
                continue
            self.reid_to_name[rid] = player_name if '#' not in player_name else player_name.split('#')[1]
            if len(profile['profile_image']) > max_gallery_profile_num:
                self.reid_profile[rid] = profile['profile_image'][
                                         ::len(profile['profile_image']) // max_gallery_profile_num]
            self.reid_profile[rid] = profile['profile_image'][:max_gallery_profile_num]
            self.reid_features[rid] = self.REID.extract_features(self.reid_profile[rid])
            # self.reid_end_frame[rid] = -1
            self.reid_to_tracking[rid] = []
        self.logger.info(
            f"Loaded {len(self.reid_profile)} player profiles from {profile_file_path}."
        )

    def get_new_tracking_id(self):
        self.current_max_tracking_id += 1
        return self.current_max_tracking_id

    def match_frame(self, frame: Frame) -> Frame:

        players = frame.players
        tracking_ids = [player.tracking_id for player in players]
        bboxes = [player.bbox for player in players]
        if len(bboxes) == 0:
            return frame
        bboxes = torch.from_numpy(np.stack(bboxes)).to(torch.device("cuda:0"))[:, :4]
        iou_mats = ious(bboxes, bboxes)

        for i in range(len(players)):
            iou_mats[i, i] = 0
            max_iou = iou_mats[i].max().item()
            if max_iou > self.track_link_overlap_threshold:  # 目标被重叠
                # 已分配新ID
                if (
                        tracking_ids[i] in self.tracking_overlaped.keys() and
                        self.tracking_overlaped[tracking_ids[i]]
                ):
                    # players[i].tracking_id = self.re_tracking_id_dict[tracking_ids[i]]
                    pass
                else:
                    # 未分配新ID
                    # should assign a new id
                    self.tracking_overlaped[tracking_ids[i]] = True
                    new_id = self.get_new_tracking_id()
                    self.re_tracking_id_dict[tracking_ids[i]] = new_id
                # the bbox is overlapped, confidence should be lower
            else:
                self.tracking_overlaped[tracking_ids[i]] = False  # 消除重叠状态，以备下次发生重叠时重新分配ID
                if tracking_ids[i] not in self.re_tracking_id_dict.keys():
                    self.re_tracking_id_dict[tracking_ids[i]] = self.get_new_tracking_id()

            # 防止新ID太长，限制最长为200
            if (self.re_tracking_id_dict[tracking_ids[i]] in self.tracking_bbox_seq.keys()
                    and len(self.tracking_bbox_seq[self.re_tracking_id_dict[tracking_ids[i]]]) >= 200
            ):
                # 强制断开，即映射一个新ID
                self.re_tracking_id_dict[tracking_ids[i]] = self.get_new_tracking_id()
            players[i].bbox[-1] = 1 - max_iou
            players[i].tracking_id = self.re_tracking_id_dict[tracking_ids[i]]

        self.reid(frame)
        return frame

    def get_new_id(self):
        if len(self.reid_to_tracking) == 0:
            return 0
        return max(self.reid_to_tracking.keys()) + 1

    def get_candidate_id(self):
        candidate_id = []
        for rid in self.reid_to_tracking.keys():
            candidate_id.append(rid)
        return candidate_id

    def reid(self, frame: Frame) -> None:
        players = frame.players
        tracking_ids = [player.tracking_id for player in players]
        bboxes = [player.bbox for player in players]

        for i, tid in enumerate(tracking_ids):
            if tid not in self.tracking_player_photos.keys():
                self.tracking_player_photos[tid] = []
                self.tracking_player_photos_frameid[tid] = []
                # self.tracking_start_frame[tid] = self.frame_count
                self.tracking_bbox_seq[tid] = []
            if bboxes[i][-1] > self.valid_photo_threshold:
                if (len(self.tracking_player_photos_frameid[tid]) == 0 or
                        self.frame_count - self.tracking_player_photos_frameid[tid][
                            -1] >= self.player_photo_min_step):  # 间隔几帧再取
                    photo = crop_player_photo(frame.image, bboxes, i)
                    self.tracking_player_photos[tid].append(photo)
                    self.tracking_player_photos_frameid[tid].append(self.frame_count)
                # photo = crop_profile(frame.image, bboxes, i)
                # self.tracking_seq_profile[tid].append(photo)
            self.tracking_player_disappear_frame_num[tid] = 0  # 消失计数归零
            self.tracking_bbox_seq[tid].append(bboxes[i])

        # cache_frame_num_list = self.cache_frame_num.keys()
        cache_frame_num_list = list(self.tracking_player_disappear_frame_num.keys())
        for idx in cache_frame_num_list:
            if idx not in tracking_ids:
                self.tracking_player_disappear_frame_num[idx] += 1  # 消失计数+1
                if self.tracking_player_disappear_frame_num[idx] > self.player_max_disappear_frame:
                    self.match(idx)
            # else:
            #     self.tracking_player_disappear_frame_num[idx] = 0
        self.frame_count += 1

    def select_tracking_profile(self, tracking_id):
        if len(self.tracking_player_photos[tracking_id]) < self.min_query_photo_num:
            return None
        if len(self.tracking_player_photos[tracking_id]) < self.max_query_photo_num:
            return self.tracking_player_photos[tracking_id]
        ### equal step select
        return self.tracking_player_photos[tracking_id][
               ::len(self.tracking_player_photos[tracking_id]) // self.max_query_photo_num]

    def match(self, idx):
        # rest_tracking_id = list(self.tracking_seq_profile.keys())
        # for idx in rest_tracking_id:
        if len(self.tracking_player_photos[idx]) < self.min_query_photo_num:
            self.tracking_player_disappear_frame_num.pop(idx)
            self.tracking_player_photos.pop(idx)
            return

        candidate_id = self.get_candidate_id()
        candidate_features = [self.reid_features[cid] for cid in candidate_id]
        tracking_profile = self.select_tracking_profile(idx)

        match_idx, _, dist = self.REID.seq_reid(candidate_features, tracking_profile)

        if match_idx != -1:
            rid = candidate_id[match_idx]
            self.reid_to_tracking[rid].append(idx)

            self.tracking_player_disappear_frame_num.pop(idx)
            self.tracking_player_photos.pop(idx)
            self.tracking_to_reid[idx] = (rid, dist)
        else:

            self.tracking_player_disappear_frame_num.pop(idx)
            self.tracking_player_photos.pop(idx)

    def reid_assign(self, frame: Frame) -> Frame:
        players = frame.players
        reid_distances = []
        default_reid_distance = np.ones(len(self.reid_profile.keys())) * 10000
        # distance for each player is 8 number to each reid
        for p in players:
            p.reid_distances = default_reid_distance
            if p.tracking_id in self.tracking_to_reid.keys():
                # p.player_id = self.tracking_to_reid[p.tracking_id][0]
                # p.reid_confidence = self.tracking_to_reid[p.tracking_id][1]
                reid_distances.append(self.tracking_to_reid[p.tracking_id][1])
                # print(self.tracking_to_reid[p.tracking_id][1])
            else:
                # assert False, "tracking id not in reid dict"
                reid_distances.append(default_reid_distance)
        # print(reid_distances)
        reid_distances = np.array(reid_distances)
        if len(reid_distances) == 0:
            return frame
        reid_cost = reid_distances / self.reid_match_threshold
        matched_indices = linear_assignment(reid_cost)
        for i, j in matched_indices:
            players[i].player_id = j if reid_distances[i][j] < 1000 else -1
            players[i].player_name = self.reid_to_name.get(j, "Unknown")
            players[i].reid_confidence = reid_distances[i][j]
            players[i].reid_distances = reid_distances[i]

        return frame

    def match_rest(self):
        rest_tracking_id = list(self.tracking_player_photos.keys())
        for idx in rest_tracking_id:
            self.match(idx)

    def reid_frame(self, frame: Frame) -> Frame:
        return self.reid_assign(frame)
