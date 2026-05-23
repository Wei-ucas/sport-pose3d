from typing import Any
import numpy as np
import cv2
from typing import Union, List, Dict, Tuple

from .player import Player
from src.utils.visualization import visualize_joints2d, get_random_color


class Frame:
    """单帧"""

    def __init__(self,
                 image: np.ndarray,
                 frame_id: int,
                 players: List[Player] = None,
                 camera_params: Dict[str, Any] = None
                 ):
        """

        Args:
            image(np.ndarray): 图像, np.ndarray, shape(h, w, c)
            frame_id(int): 帧id
            players(List[Player]): 球员列表
            # actions(List[Action]): 动作列表
            # tactics(List[Tactic]): 战术列表
            # teams(List[Team]): 球队列表
            court(Court): 球场信息
        """
        self.image = image
        self.frame_id = frame_id

        self.players: List[Player] = []
        self.tracking_overlaps: List[List[int]] = []  # 检测到的球员之间的重叠关系

        self.camera_params = camera_params

        # 以下字典用于快速查找
        self.tracking_id2player: Dict[int, Player] = {}  # 在ReID前，只有tracking_id，没有player_id
        self.player_id2player: Dict[int, Player] = {}

        if players is not None:
            self.add_player(players)

    def rebuild_dict(self):
        self.player_id2player = {player.player_id: player for player in self.players}
        self.tracking_id2player = {player.tracking_id: player for player in self.players}

    def get_player(self, player_id: int) -> List[Player]:
        """通过player_id获取球员对象
        Args:
            player_id(int): 球员id, 只能是球员id，不能是追踪id
        """
        player_list = []
        for player in self.players:
            if player.player_id == player_id:
                player_list.append(player)
        return player_list

    def add_player(self, player: Union[Player, List[Player]]):
        if isinstance(player, list):
            for p in player:
                self.players.append(p)
                self.tracking_id2player[p.tracking_id] = p
                if p.player_id is not None:
                    self.player_id2player[p.player_id] = p
        else:
            self.players.append(player)
            self.tracking_id2player[player.tracking_id] = player
            if player.player_id is not None:
                self.player_id2player[player.player_id] = player

    def set_player(self,
                   tracking_id: int = None,
                   player_id: int = None,
                   team_id: int = None,
                   bbox: np.ndarray = None,
                   pose: np.ndarray = None,
                   location: np.ndarray = None,
                   ball_holding: bool = None):
        """
        设置球员信息，当球员已经初始化后，可以通过tracking_id或player_id来设置球员信息
        Args:
            team_id (int): 球队id
            tracking_id (int): 跟踪ID，在进行检测跟踪时就能得到
            player_id: ReID后的球员ID，整场比赛中球员ID不会改变，初始化时可以为空，开始时利用只能利用tracking_id区分球员
            bbox (np.ndarray (5,)): 球员在图像中的位置，2d图片坐标，[x1, y1, x2, y2, score]
            pose (np.ndarray (17, 3)): 球员的姿态，16个关键点的2d图片坐标，[[x, y, score],...]
                        pose (np.ndarray (N, 3)): 球员的姿态关键点，[[x, y, score],...]
            location (np.ndarray (3,)): 球员的位置，3d世界坐标，[x, y, z], 单位m,一般是球员的脚底中心点 （考虑将球场平面作为z=0平面）
            ball_holding (bool): 是否持球
        """
        if tracking_id is not None:
            if tracking_id in self.tracking_id2player:
                player = self.tracking_id2player[tracking_id]
                if player_id is not None:
                    player.player_id = player_id
                    self.player_id2player[player_id] = player
                    # if team_id is None:
                    #     team_id = self.player_id2team_id[player_id]
                player.set(team_id=team_id, bbox=bbox, pose=pose, location=location, ball_holding=ball_holding)
            else:
                player = Player(tracking_id=tracking_id, player_id=player_id, bbox=bbox, pose=pose, location=location,
                                ball_holding=ball_holding)
                self.add_player(player)
        else:
            assert player_id is not None, "player_id 和 tracking_id 不能同时为空"
            if player_id in self.player_id2player:
                player = self.player_id2player[player_id]
                player.set(bbox=bbox, pose=pose, location=location, ball_holding=ball_holding)
            else:
                raise ValueError(f"player_id {player_id} not found")

    def remove_image(self):
        """移除图像，释放内存"""
        self.image = None

    def visualize(self, show_pose=True) -> np.ndarray:
        """可视化当前帧"""
        image = self.image.copy()
        # 可视化球员
        for player in self.players:
            if player.bbox is None:
                continue
            if player.player_id is not None and player.player_id != -1:
                color = get_random_color(player.player_id)
                x1, y1, x2, y2, score = player.bbox
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image,
                            f"{player.player_name}",
                            (int(x1), int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            # team_colors[player.team_id],
                            color,
                            1)

            else:
                x1, y1, x2, y2, score = player.bbox
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image,
                        f"{player.tracking_id}",
                        (int(x2), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1)
            if show_pose and player.pose is not None:
                    if player.pose.shape[0] == 25:
                        convention = 'openpose_25'
                    elif player.pose.shape[0] == 26:
                        convention = 'halpe26'
                    else:
                        convention = 'coco'
                    image = visualize_joints2d(image, player.pose, convention=convention, threshold=0.3)

        return image
