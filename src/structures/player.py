from typing import Optional
import numpy as np


class Player:
    """
    Represents a player with basic information for a single frame, including ID, tracking ID, bounding box, pose,
    location, and ball possession status.
    """

    def __init__(self,
                 player_id: Optional[int] = None,
                 tracking_id: Optional[int] = None,
                 team_id: Optional[int] = None,
                 bbox: Optional[np.ndarray] = None,
                 pose: Optional[np.ndarray] = None,
                 location: Optional[np.ndarray] = None,
                 ball_holding: bool = False):
        """
        Initializes a Player instance.

        Args:
            player_id (Optional[int]): Unique player ID, consistent throughout the match.
            tracking_id (Optional[int]): Tracking ID obtained through detection and tracking.
            team_id (Optional[int]): Team ID the player belongs to.
            bbox (Optional[np.ndarray]): Player's 2D bounding box in the image [x1, y1, x2, y2, score].
            pose (Optional[np.ndarray]): Player's pose with 17 keypoints in 2D image coordinates [[x, y, score], ...].
            location (Optional[np.ndarray]): Player's 3D world coordinates [x, y, z] in meters.
            ball_holding (bool): Whether the player is holding the ball.
        """
        self.player_id: Optional[int] = player_id
        self.player_name: str = ""  # Player name, may be obtained after ReID
        self.tracking_id: Optional[int] = tracking_id
        self.team_id: Optional[int] = team_id
        self.reid_confidence: list = []
        self.reid_distances: list = []
        self.bbox: Optional[np.ndarray] = None
        self.pose: Optional[np.ndarray] = None
        self.location: Optional[np.ndarray] = None
        self.ball_on: bool = False
        self.view_id: Optional[int] = None

        self.set(team_id, bbox, pose, location, ball_holding)

    def reid(self, player_id: int, team_id: int) -> None:
        """
        Updates the player ID and team ID after ReID.

        Args:
            player_id (int): New player ID.
            team_id (int): New team ID.
        """
        self.player_id = player_id
        self.team_id = team_id

    def set_bbox(self, bbox: np.ndarray) -> None:
        """
        Sets the player's bounding box.

        Args:
            bbox (np.ndarray): 2D bounding box [x1, y1, x2, y2, score].
        """
        assert isinstance(bbox, np.ndarray), f'bbox should be np.ndarray, but got {type(bbox)}'
        assert bbox.shape == (5,), f'bbox shape should be (5,), but got {bbox.shape}'
        self.bbox = bbox

    def set_pose(self, pose: np.ndarray) -> None:
        """
        Sets the player's pose.

        Args:
            pose (np.ndarray): 17 keypoints in 2D image coordinates [[x, y, score], ...].
        """
        assert isinstance(pose, np.ndarray), f'pose should be np.ndarray, but got {type(pose)}'
        assert pose.shape == (17, 3), f'pose shape should be (17, 3), but got {pose.shape}'
        self.pose = pose

    def set_location(self, location: np.ndarray) -> None:
        """
        Sets the player's 3D location.

        Args:
            location (np.ndarray): 3D world coordinates [x, y, z].
        """
        assert isinstance(location, np.ndarray), f'location should be np.ndarray, but got {type(location)}'
        assert location.shape == (3,), f'location shape should be (3,), but got {location.shape}'
        self.location = location

    def get_ball(self) -> None:
        """Marks the player as holding the ball."""
        self.ball_on = True

    def lose_ball(self) -> None:
        """Marks the player as not holding the ball."""
        self.ball_on = False

    def set(self,
            team_id: Optional[int] = None,
            bbox: Optional[np.ndarray] = None,
            pose: Optional[np.ndarray] = None,
            location: Optional[np.ndarray] = None,
            ball_holding: bool = False) -> None:
        """
        Sets the player's information.

        Args:
            team_id (Optional[int]): Team ID the player belongs to.
            bbox (Optional[np.ndarray]): 2D bounding box [x1, y1, x2, y2, score].
            pose (Optional[np.ndarray]): 17 keypoints in 2D image coordinates [[x, y, score], ...].
            location (Optional[np.ndarray]): 3D world coordinates [x, y, z].
            ball_holding (bool): Whether the player is holding the ball.
        """
        if team_id is not None:
            self.team_id = team_id
        if bbox is not None:
            self.set_bbox(bbox)
        if pose is not None:
            self.set_pose(pose)
        if location is not None:
            self.set_location(location)
        if ball_holding:
            self.get_ball()
        else:
            self.lose_ball()

    def update(self,
               bbox: np.ndarray,
               pose: np.ndarray,
               location: np.ndarray,
               ball_holding: bool = False) -> None:
        """
        Updates the player's information.

        Args:
            bbox (np.ndarray): 2D bounding box [x1, y1, x2, y2, score].
            pose (np.ndarray): 17 keypoints in 2D image coordinates [[x, y, score], ...].
            location (np.ndarray): 3D world coordinates [x, y, z].
            ball_holding (bool): Whether the player is holding the ball.
        """
        self.set_bbox(bbox)
        self.set_pose(pose)
        self.set_location(location)
        if ball_holding:
            self.get_ball()
        else:
            self.lose_ball()