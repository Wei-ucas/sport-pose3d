from typing import List, Dict
import os
import cv2
import json

# data format
"""
workdir/
videos/
    game_id/
        01.mp4
        02.mp4
        ...
prepare/
    camera/
        game_id/
            intri.yml
            extri.yml
        ...
    profiles/
        game_id/
            01.json # annotation for 01.mp4
            profiles.pkl # player profiles
prediction/
    game_id/
        01_detection_s1_2000.pkl  # {view}_{type}_s{frame step}_{max frame}.pkl
        ...
        3d_2000_v4.pkl  # 3d position of players, {max frame}_v{number of views}.pkl
vis/
    game_id/
        3d_2000_v4.mp4  # visualization of 3d position
output/
    game_id/
        report.pkl
        report.json
        ...
"""


class GamePath:
    def __init__(self, workdir: str, game_id: str, view_list: List[str],
                 video_type: str = ".mp4", frame_step: int = 1,
                 max_frame_num: int = -1):
        self.workdir = workdir
        self.game_id = game_id
        self.view_list = view_list
        self.video_type = video_type
        self.frame_step = frame_step
        self.max_frame_num = max_frame_num

        self.video_dir = os.path.join(workdir, "videos", game_id)
        assert os.path.exists(self.video_dir), f"Video directory {self.video_dir} does not exist."

        self.prepare_dir = os.path.join(workdir, "prepare")
        assert os.path.exists(self.prepare_dir), f"Prepare directory {self.prepare_dir} does not exist."

        self.prediction_dir = os.path.join(workdir, "prediction", game_id)
        os.makedirs(self.prediction_dir, exist_ok=True)

        self.vis_dir = os.path.join(workdir, "vis", game_id)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.output_dir = os.path.join(workdir, "output", game_id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.fps = self._get_fps()

    def __repr__(self):
        return f"GamePath(workdir={self.workdir}, game_id={self.game_id}, views={self.view_list}, " \
               f"video_type={self.video_type}, frame_step={self.frame_step}, max_frame_num={self.max_frame_num})"

    def _get_fps(self) -> float:
        """Get the frames per second (fps) of the video."""
        video_path = os.path.join(self.video_dir, f"{self.view_list[0]}{self.video_type}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    @property
    def camera_path(self):
        return os.path.join(self.prepare_dir, "camera", self.game_id)

    @property
    def get_profile_path(self) -> str:
        """Get the profile path for a specific view."""
        return os.path.join(self.prepare_dir, "profiles", self.game_id, f"profiles.pkl")

    def get_video_path(self, view_id: str) -> str:
        """Get the video path for a specific view."""
        return os.path.join(self.video_dir, f"{view_id}{self.video_type}")

    def get_detection_path(self, view_id: str) -> str:
        """Get the detection path for a specific view."""
        return os.path.join(self.prediction_dir, f"{view_id}_detection_s{self.frame_step}_{self.max_frame_num}.pkl")

    def get_reid_path(self, view_id: str) -> str:
        """Get the reid path for a specific view."""
        return os.path.join(self.prediction_dir, f"{view_id}_reid_s{self.frame_step}_{self.max_frame_num}.pkl")

    def get_video_sync_path(self) -> str:
        return os.path.join(self.prediction_dir, "video_sync.json")

    # @property
    # def get_3d_pre_path(self) -> str:
    #     return os.path.join(self.prediction_dir, f"3d_{self.max_frame_num}_v{len(self.view_list)}-pre.pkl")
    #
    # @property
    # def get_3d_path(self) -> str:
    #     """Get the 3D path for the game."""
    #     return os.path.join(self.prediction_dir, f"3d_{self.max_frame_num}_v{len(self.view_list)}.pkl")
    #
    # @property
    # def get_3d_opt_path(self) -> str:
    #     """Get the optimized 3D path for the game."""
    #     return os.path.join(self.prediction_dir, f"3d_{self.max_frame_num}_v{len(self.view_list)}-opt.pkl")
    #
    # @property
    # def get_detection_vis_path(self) -> str:
    #     """Get the detection visualization path for the game."""
    #     return os.path.join(self.vis_dir, f"detection_{self.max_frame_num}_v{len(self.view_list)}.mp4")
    #
    # @property
    # def get_reid_vis_path(self) -> str:
    #     """Get the reid visualization path for the game."""
    #     return os.path.join(self.vis_dir, f"reid_{self.max_frame_num}_v{len(self.view_list)}.mp4")
    #
    # @property
    # def get_3d_vis_path(self) -> str:
    #     """Get the 3D visualization path for the game."""
    #     return os.path.join(self.vis_dir, f"3d_{self.max_frame_num}_v{len(self.view_list)}.mp4")
    #
    # @property
    # def get_3d_opt_vis_path(self) -> str:
    #     """Get the optimized 3D visualization path for the game."""
    #     return os.path.join(self.vis_dir, f"3d_{self.max_frame_num}_v{len(self.view_list)}-opt.mp4")

    @property
    def get_report_path(self) -> str:
        """Get the report path for the game."""
        return os.path.join(self.output_dir, "report.json")

    def get_prediction_save_path(self, prediction_type: str, view_id: str = None) -> str:
        """
        Get the save path for a specific prediction type and view ID.
        :param prediction_type: Type of prediction (e.g., 'detection', 'reid', '3d').
        :param view_id: Optional view ID for specific predictions.
        :return: Full path to save the prediction file.
        """
        if view_id:
            return os.path.join(self.prediction_dir,
                                f"{view_id}_{prediction_type}_s{self.frame_step}_{self.max_frame_num}.pkl")
        return os.path.join(self.prediction_dir, f"{prediction_type}_{self.max_frame_num}_v{len(self.view_list)}.pkl")

    def get_vis_path(self, prediction_type: str, view_id: str = None) -> str:
        """
        Get the visualization path for a specific type of visualization.
        :param view_id:
        :param prediction_type:
        :return: Full path to save the visualization video.
        """
        if view_id:
            return os.path.join(self.vis_dir,
                                f"{prediction_type}_{view_id}_s{self.frame_step}_{self.max_frame_num}.mp4")
        return os.path.join(self.vis_dir, f"{prediction_type}_{self.max_frame_num}_v{len(self.view_list)}.mp4")

    def get_player_info(self) -> Dict[int, str]:
        if not os.path.exists(self.get_profile_path):
            return {}
        return read_player_names(self.get_profile_path)


def read_player_names(player_profile_path: str, player_name_dict_path: str = None) -> Dict[int, str]:
    player_id2number = {}
    player_id2name = {}
    import pickle
    with open(player_profile_path, "rb") as f:
        player_profiles = pickle.load(f)
    if player_name_dict_path is not None:
        player_name_dict = json.load(open(player_name_dict_path))
    else:
        player_name_dict = None

    for player_team_number in player_profiles.keys():
        player_id = player_profiles[player_team_number]["tmp_player_id"]
        player_team, player_number = player_team_number.split("#")
        player_ab_number = f'{player_team}#{player_number}'
        player_id2number[player_id] = player_number

        if player_name_dict is not None:
            player_name = player_name_dict[player_team_number]
        else:
            player_name = player_ab_number
        player_id2name[player_id] = player_name
    # return player_id2number, player_id2name
    return player_id2name
