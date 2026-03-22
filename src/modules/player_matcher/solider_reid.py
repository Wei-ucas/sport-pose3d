from typing import List, Dict, Union
import torch
from torch import nn
import torchvision.transforms as T
import numpy as np
from PIL import Image

from .SOLIDER_ReID.config import cfg
from .SOLIDER_ReID.model import make_model
from .SOLIDER_ReID.utils.metrics import euclidean_distance


class ReID:
    def __init__(self, config_file: str, checkpoint: str, re_ranking: bool = False, semantic_weight: float = 0.2,
                 device='cuda:0', match_threshold=500):
        super().__init__()
        # init reid model
        if config_file is not None:
            cfg.merge_from_file(config_file)
        opts = [
            'TEST.WEIGHT', checkpoint,
            'TEST.RE_RANKING', str(re_ranking),
            'MODEL.SEMANTIC_WEIGHT', str(semantic_weight)
        ]
        cfg.merge_from_list(opts)
        cfg.freeze()
        self.cfg = cfg
        self.device = device
        self.model = make_model(cfg, num_class=751, camera_num=6, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        if cfg.TEST.WEIGHT != '':
            self.model.load_param(cfg.TEST.WEIGHT)
        self.model = self.model.to(self.device)
        self.model.eval()
        # transforms for data pre-processing
        # self.transforms = T.Compose([
        #     T.Resize(cfg.INPUT.SIZE_TEST),
        #     T.ToTensor(),
        #     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        # ])
        self.transforms = nn.Sequential(
            T.Resize(cfg.INPUT.SIZE_TEST),
            # T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ).to(self.device)

        self.match_threshold = match_threshold

    def seq_euclidean_distance(self, qf, gf, return_matrix=False):
        m = qf.shape[0]
        n = gf.shape[0]
        dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_mat.addmm_(1, -2, qf, gf.t())
        if return_matrix:
            return dist_mat
        sq_dist = dist_mat.min(dim=0)[0]
        dist = sq_dist.cpu().numpy()
        return dist

    def extract_features(self, images: List[np.ndarray]):
        """
        Args:
            images: List of images
        Returns:
            List of feature tensors
        """
        features = []
        for img in images:
            img = torch.from_numpy(img).to(self.device)[:, :, [2, 1, 0]] / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = self.transforms(img)
            features.append(img)
        features = torch.concatenate(features, dim=0)
        with torch.no_grad():
            features, _ = self.model(features)
        return features

    def seq_reid(self, candidate_features: List[torch.Tensor], query_photos: List[np.ndarray]):
        """
        Args:
            candidate_features: List of feature tensors
            query_photos: List of profile images
        Returns:
            List of matched player ids
        """
        if len(candidate_features) == 0:
            return -1, None, np.array([])
        seq_features = self.extract_features(query_photos)

        dist = np.ones((len(query_photos), len(candidate_features))) * 10000
        for i, c_feature in enumerate(candidate_features):
            d = self.seq_euclidean_distance(c_feature, seq_features)
            dist[:, i] = d

        dist_to_each_candidate = dist.mean(axis=0)
        min_index = dist_to_each_candidate.argmin()
        min_dist = dist_to_each_candidate.min()

        if min_dist < self.match_threshold:
            # return min_index, selected_seq_feature, min_dist
            return min_index, None, dist_to_each_candidate
        else:
            # return -1, seq_features, min_dist
            return -1, None, dist_to_each_candidate
