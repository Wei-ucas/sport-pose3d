import json
import logging
import pickle as pkl
import numpy as np
import os
from typing import List, Union

from src.structures.multiview_frame import MvFrame
from src.structures.frame import Frame


class FrameSaver:
    """
    Frame Saver
    save frames to pickle file one by one
    """

    def __init__(self, save_path: str, save_type: str = "pickle"):
        """
        Args:
            save_path (str): path to save frames
            save_type (str): type of saving, currently only support "pickle"
        """
        self.save_path = save_path
        self.tmp_save_path = self.save_path + '.tmp'
        if os.path.exists(self.tmp_save_path):
            logging.getLogger("FrameSaver").warning(f"Temporary file {self.tmp_save_path} already exists, removing it.")
            os.remove(self.tmp_save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.save_handler = open(self.tmp_save_path, 'wb')
        logging.getLogger("FrameSaver").info(f"Saving frames to {self.tmp_save_path}")

    def save_frame(self, frame: Union[Frame, MvFrame]):
        """
        Save a single frame to file
        Args:
            frame (Union[Frame, MvFrame]): Frame or MvFrame object to save
        """
        if isinstance(frame, Frame):
            frame.image = None
        pkl.dump(frame, self.save_handler)

    def close(self):
        """ Close the save handler and move the temporary file to the final destination
        """
        self.save_handler.close()
        os.rename(self.tmp_save_path, self.save_path)
        self.save_handler = None
        logging.getLogger("FrameSaver").info(f"Saved frames to {self.save_path}")

    def __del__(self):
        """ Destructor to ensure the save handler is closed
        """
        if self.save_handler is None:
            return
        try:
            # not properly closed, try to close and remove the temporary file
            self.save_handler.close()
            os.remove(self.tmp_save_path)
        except Exception as e:
            logging.getLogger("FrameSaver").error(f"Error closing FrameSaver: {e}")
            if os.path.exists(self.tmp_save_path):
                os.remove(self.tmp_save_path)


class VideoWriter:
    def __init__(self, save_path, fps, resolution=None):
        self.fps = fps
        self.output_path = save_path
        self.resolution = resolution

        if resolution is not None:
            self.video_writer = self.init_writer()
        else:
            self.video_writer = None

    def init_writer(self):
        try:
            import imageio.v2 as iio
            video_writer = iio.get_writer(self.output_path, format='FFMPEG', fps=self.fps,
                                          codec='h264',
                                          ffmpeg_params=['-vf', 'format=yuv420p', '-b:v', '3M'],
                                          # quality=8,
                                          pixelformat='yuv420p',
                                          )
            self.imageio = True
        except ImportError:
            print("imageio not installed, use cv2 instead")
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)
            self.imageio = False
        return video_writer

    def write(self, frame: np.ndarray):
        if self.video_writer is None:  # need to initialize the writer
            self.resolution = (frame.shape[1], frame.shape[0])  # width, height
            self.video_writer = self.init_writer()
        else:
            if self.resolution != (frame.shape[1], frame.shape[0]):
                raise ValueError("Resolution of the frame does not match the initialized resolution")
        if self.imageio:
            self.video_writer.append_data(frame[:, :, ::-1])
        else:
            self.video_writer.write(frame)

    def close(self):
        if self.imageio:
            self.video_writer.close()
        else:
            self.video_writer.release()
        self.video_writer = None

    def __del__(self):
        """ Destructor to ensure the video writer is closed
        """
        if self.video_writer is None:
            return
        try:
            self.close()
        except Exception as e:
            logging.getLogger("VideoWriter").error(f"Error closing VideoWriter: {e}")
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
