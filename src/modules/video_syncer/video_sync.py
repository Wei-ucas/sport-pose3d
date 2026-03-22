import os
import warnings

import numpy as np
import torch
from .audio_matcher import AudioMatcher
from .video_tools import audio_from_video
from ..base_module import BaseModule


class VideoSync(BaseModule):

    def __init__(self, view_ids, video_fps=25):
        super().__init__("video_sync")
        self.view_ids = view_ids
        # self.tmp_video_root = None
        AUDIO_MATCH_CFGS = {
            "sample_rate": 44100,  # the audio data sampling rate
            "nfft": 1024,
            "bin_stride": 128,  # the audio feature bin stride, bin_stride/sample_rate is the audio matching resolution
            "bin_length": 256,  # the audio feature bin size
            "n_bins_for_matching": torch.inf,  # number of bins used for matching
            "audio_trim_length": 60 * 10  # the audio will be trimed no longer then this value (seconds)
        }

        self.AUDIO_MATCH_CFGS = AUDIO_MATCH_CFGS
        self.video_fps = video_fps

    def time_offset_to_frame_offset(self, time_offset):
        frame_offset = time_offset * self.video_fps
        frame_offset = np.round(frame_offset).astype(int)
        return frame_offset

    def get_video_time_offset(self, videos, camera_names, audio_match_cfgs, audio_path):
        audios = []
        offsets = [0, ]
        # videos = list(videos_dict.values())
        for i, video in enumerate(videos):
            audio_name = os.path.basename(video).split('.')[0] + '.wav'
            audio_file = os.path.join(audio_path, audio_name)
            # os.system('ffmpeg -i {} -vn -acodec copy {}'.format(video, audio_file))
            if os.path.exists(audio_file):
                audios.append(audio_file)
                continue
            audio_from_video(video, audio_match_cfgs['sample_rate'], audio_match_cfgs['audio_trim_length'], audio_file)
            # videos[i] = audio_path
            audios.append(audio_file)
        matcher = AudioMatcher(audios[0], audio_match_cfgs)
        for i in range(1, len(audios)):
            match_res = matcher.find_offset(audios[i])
            time_offset = match_res['time_offset'].item()
            stand_score = match_res["standard_score"].item()
            if stand_score < 0.01:
                warnings.warn("video {} and video {} not match".format(videos[0], videos[i]))
                # raise ValueError("video {} and video {} not match".format(videos[0], videos[i]))
            print(camera_names[i], time_offset)
            offsets.append(time_offset)

        offsets = np.array(offsets)
        ss_time = offsets.max() - offsets
        return ss_time

    def process(self, video_dict):
        video_list = [video_dict[view_id] for view_id in self.view_ids]

        video_dir = os.path.dirname(video_list[0])
        tmp_audio_dir = video_dir
        video_names = [os.path.basename(video) for video in video_list]
        time_offset = self.get_video_time_offset(video_list, video_names, self.AUDIO_MATCH_CFGS, tmp_audio_dir)

        for i, view_id in enumerate(self.view_ids):
            print(f"View {view_id}: {time_offset[i]}")

        # return time_offset
        frame_offset = self.time_offset_to_frame_offset(time_offset)
        return frame_offset, time_offset
