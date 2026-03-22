import os
import logging
import pickle
from typing import List, Union, Optional
import cv2
from src.structures.frame import Frame
from src.structures.multiview_frame import MvFrame
import queue
import threading
from src.utils.camera import load_camera


def frame_data_iter(file_path):
    """Generator to iterate over frame data."""
    with open(file_path, 'rb') as f:
        while True:
            try:
                frame_data = pickle.load(f)
                yield frame_data
            except EOFError:
                break


class FrameInput:
    """
    Handles input frames from video or pickle files, with optional camera parameters.
    """

    def __init__(self,
                 view_id: str,
                 pkl_data_path: Optional[str] = None,
                 video_path: Optional[str] = None,
                 camera_path: Optional[str] = None,
                 frame_downsample_rate: int = 1,
                 frame_offset: int = 0,
                 batch_size: int = -1
                 ):
        """
        Initializes the FrameInput instance.

        Args:
            view_id (str): View ID for the input.
            pkl_data_path (Optional[str]): Path to the pickle file containing frames.
            video_path (Optional[str]): Path to the video file.
            camera_path (Optional[str]): Path to the camera parameters.
            frame_downsample_rate (int): Downsample rate for video frames.
            frame_offset (int): Offset for starting frame.
            batch_size (int): Number of frames per batch.
        """
        self.logger = logging.getLogger("FrameInput-{}".format(view_id))
        assert pkl_data_path or video_path, "Either pkl_file or video_path must be provided."

        self.view_id = view_id
        self.pkl_data_path = pkl_data_path
        self.video_path = video_path
        self.camera_path = camera_path
        # self.dynamic_camera = dynamic_camera
        self.frame_downsample_rate = frame_downsample_rate
        self.frame_offset = frame_offset
        self.batch_size = batch_size

        self.camera_params = self._load_static_camera()
        self.iter_index = 0

        self.video_cap = self._init_video_capture()
        self.queue = queue.Queue(maxsize=32)
        self.video_thread = self._start_video_thread() if self.video_cap else None

        self.frame_iter = self._frame_data_iter()

        if self.pkl_data_path:

            if self.frame_offset > 0:
                # ignore the first few frames if frame_offset is set
                for _ in range(self.frame_offset):
                    try:
                        next(self.frame_iter)
                    except StopIteration:
                        raise ValueError("Frame offset exceeds the number of frames in the pickle file.")

    def _frame_data_iter(self):
        if self.pkl_data_path:
            return frame_data_iter(self.pkl_data_path)
        else:
            def empty_iter():
                while True:
                    yield Frame(image=None, frame_id=self.iter_index)

            return empty_iter()

    def _load_static_camera(self) -> Optional[dict]:
        """Loads static camera parameters."""
        return load_camera(self.camera_path, self.view_id, None) if self.camera_path else None

    def _init_video_capture(self) -> Optional[cv2.VideoCapture]:
        """Initializes video capture if a video path is provided."""
        if not self.video_path:
            return None
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_offset)
        return cap

    def _start_video_thread(self) -> threading.Thread:
        """Starts a thread to read video frames."""
        thread = threading.Thread(target=self._video_read_thread, daemon=True)
        thread.start()
        return thread

    def _video_read_thread(self) -> None:
        """Reads video frames in a separate thread."""
        frame_id = self.frame_offset
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                break
            if frame_id % self.frame_downsample_rate == 0:
                self.queue.put((frame, frame_id))
            frame_id += 1
        self.queue.put((None, -1))
        self.video_cap.release()

    @property
    def video_fps(self) -> float:
        """Returns the video FPS adjusted by the downsample rate."""
        return self.video_cap.get(cv2.CAP_PROP_FPS) / self.frame_downsample_rate if self.video_cap else 0

    def __len__(self) -> int:
        """Returns the total number of frames."""
        if self.pkl_data_path:
            return get_pickle_length(self.pkl_data_path)
        return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_cap else 0

    def __iter__(self):
        return self

    def __next__(self) -> Union[Frame, MvFrame, None]:
        """Returns the next frame."""
        frame = self._get_frame()
        self.iter_index += 1

        if self.video_path:
            frame_image, frame_id = self.queue.get()
            if frame_id == -1:
                raise StopIteration
            assert frame_id == frame.frame_id, f"Frame ID mismatch: video {frame_id}, pkl {frame.frame_id}"
            frame.image = frame_image

        return frame

    def _get_frame(self) -> Union[Frame, MvFrame]:
        """Retrieves a frame from the frame list or creates a new one."""
        frame_data = next(self.frame_iter)
        frame_data.camera_params = self.camera_params
        return frame_data

    def next_batch(self) -> List[Union[Frame, MvFrame]]:
        """Returns the next batch of frames."""
        frames = []
        for _ in range(self.batch_size):
            try:
                frames.append(self.__next__())
            except StopIteration:
                break
        return frames if len(frames) == self.batch_size else []

    def __del__(self):
        """Releases resources on deletion."""
        if self.video_cap:
            self.video_cap.release()
        if self.video_thread:
            self.video_thread.join()


class MvFrameInput:
    """iter load mv frame data from a pickle file."""

    def __init__(self, mv_data_path: str):

        self.mv_data_path = mv_data_path

        self.mv_frame_iter = self._mv_frame_data_iter()

    def _mv_frame_data_iter(self):
        """Generator to iterate over multiview frame data."""
        with open(self.mv_data_path, 'rb') as f:
            while True:
                try:
                    mv_frame_data = pickle.load(f)
                    yield mv_frame_data
                except EOFError:
                    break
        return None

    def __next__(self) -> MvFrame:
        """Returns the next multiview frame."""
        mv_frame_data = next(self.mv_frame_iter)
        return mv_frame_data

    def __iter__(self):
        return self

    def __len__(self):
        """Returns the total number of multiview frames."""
        return get_pickle_length(self.mv_data_path)


def get_pickle_length(pkl_path):
    def iter_pkl():
        with open(pkl_path, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    return sum(1 for _ in iter_pkl())
