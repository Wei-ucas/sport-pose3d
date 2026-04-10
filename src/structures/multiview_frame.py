from typing import List, Dict, Union
import numpy as np
import cv2

from .frame import Frame
from .player import Player

from src.utils import constant as _court_module


class MvFrame:
    """
    Represents a multi-view frame containing multiple views and their associated players.
    """

    def __init__(
        self,
        view_ids: List[Union[int, str]],
        frames: List[Frame] = None,
        frame_id: int = None,
    ):
        """
        Initializes an MvFrame instance.

        Args:
            view_ids (List[Union[int, str]]): List of view IDs.
            frames (List[Frame], optional): List of Frame objects for each view.
            frame_id (int, optional): Frame ID.
            camera_params (Dict[str, np.ndarray], optional): Camera parameters for each view.
        """
        self.num_views = len(view_ids)
        assert (
            frames is None or len(frames) == self.num_views
        ), "num_views should match the length of frames"
        self.view_ids = view_ids
        self.frames = frames
        self.frame_dict: Dict[Union[int, str], Frame] = {
            view_id: frame for view_id, frame in zip(view_ids, frames)
        }

        self.images = {
            view_id: frame.image for view_id, frame in self.frame_dict.items()
        }
        for frame in self.frame_dict.values():
            frame.remove_image()

        for view_id, frame in self.frame_dict.items():
            for player in frame.players:
                player.view_id = view_id

        self.frame_id = frame_id
        self.camera_params = self._get_camera_params_from_frames(view_ids, frames)

        # untracked -> tracked -> matched/unmatched

        self.untracked_players_k3d = []  # before reid, unmatched players
        self.untracked_reid_info = []  # before reid, unmatched players' reid distances
        self.tracked_player_k3d = {}  # players that are being tracked
        self.tracked_player_location = {}  # players that are being tracked
        self.tracked_player_reid_info = (
            {}
        )  # players that are being tracked with reid distances
        self.matched_player_k3d = {}  # player_id -> k3d
        self.matched_player_location = {}
        self.unmatched_player_k3d = []  # cannot be matched with any player name
        self.unmatched_player_location = []  # cannot be matched with any player name

        # unmatched -> tracked -> player/no_name_player_location

    def _get_camera_params_from_frames(
        self, view_ids: List[Union[int, str]], frames: List[Frame]
    ) -> Dict[str, np.ndarray]:
        """
        Extracts camera parameters from the frames.

        Args:
            view_ids (List[Union[int, str]]): List of view IDs.
            frames (List[Frame]): List of Frame objects.

        Returns:
            Dict[str, np.ndarray]: Camera parameters aggregated across views.
        """
        camera_params = {}
        for view_id, frame in zip(view_ids, frames):
            for key, value in frame.camera_params.items():
                if isinstance(value, (np.ndarray, list)):
                    camera_params.setdefault(key, []).append(value)

        for key in camera_params.keys():
            if key not in {"dist", "view_names"}:
                camera_params[key] = np.array(camera_params[key])
        camera_params["view_names"] = view_ids
        return camera_params

    def get_mv_player(self, player_id: int) -> List[List[Player]]:
        """
        Retrieves a player across multiple views by their ID.

        Args:
            player_id (int): Player ID.

        Returns:
            List[List[Player]]: List of players with the given ID across views.
        """
        return [frame.get_player(player_id) for frame in self.frames]

    def remove_image(self) -> None:
        """Removes images from all frames to save memory."""
        for frame in self.frames:
            frame.image = None

    def visualize_player_k3d(
        self, camera, player_names: Dict[int, str] = None
    ) -> np.ndarray:
        """
        Visualizes 3D players on a court background.

        Args:
            camera: Camera object for projection.
            player_names (Dict[int, str], optional): Mapping of player IDs to names.
            if player_names is None, only visualize the tracked_players_k3d without names

        Returns:
            np.ndarray: Image with visualized players.
        """
        raise NotImplementedError("visualize_player_k3d is not implemented yet")

    def visualize_location(
        self, player_names: Dict[int, str] = None, color_dict: Dict[str, tuple] = None
    ) -> np.ndarray:
        raise NotImplementedError("visualize_location is not implemented yet")

    @staticmethod
    def _draw_player_position(
        background_image: np.ndarray, k3d: np.ndarray, color: tuple, name: str
    ) -> np.ndarray:
        """
        Draws a player's position on the court.

        Args:
            background_image (np.ndarray): Background image.
            k3d (np.ndarray): 3D keypoints of the player.
            color (tuple): Color for the player.
            name (str): Player's name.

        Returns:
            np.ndarray: Updated image with the player's position.
        """
        position = k3d[:2]
        if position[-1] == 0 or np.isnan(position).any():
            return background_image
        x, y = _court_module.project_point_to_image(position[:2])
        cv2.circle(background_image, (int(x), int(y)), 15, color, -1)
        cv2.putText(
            background_image,
            name,
            (int(x) - 12, int(y) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        return background_image
