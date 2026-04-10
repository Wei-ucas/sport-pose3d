import numpy as np
import torchaudio as ta
import torch.nn.functional as f
import torch

AUDIO_MATCH_CFGS = {
    "sample_rate": 16000,  # the audio data sampling rate
    "nfft": 1024,
    "bin_stride": 128,  # the audio feature bin stride, bin_stride/sample_rate is the audio matching resolution
    "bin_length": 256,  # the audio feature bin size
    "n_bins_for_matching": torch.inf,  # number of bins used for matching
    "audio_trim_length": 60
    * 20,  # the audio will be trimed no longer then this value (seconds)
}


class AudioMatcher:

    def __init__(self, base_audio_file, cfg):
        # self.base_audio, sample_rate = ta.load(base_audio_file, normalize=True)
        self.sample_rate = cfg["sample_rate"]
        self.nfft = cfg["nfft"]
        self.bin_stride = cfg["bin_stride"]
        self.bin_length = cfg["bin_length"]
        self.max_frames = cfg["n_bins_for_matching"]
        self.audio_trim_length = cfg["audio_trim_length"]
        # assert sample_rate == self.sample_rate

        self.base_audio_wave = self.load_audio(base_audio_file)

        self.MFCC = ta.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=26,
            melkwargs={
                "n_fft": self.nfft,
                "hop_length": self.bin_stride,
                "n_mels": 26,
                "center": False,
            },
        )  # CPU: torch.stft has cuFFT issues on some CUDA versions
        self.base_audio_mfcc = self.mfcc_normalize(self.MFCC(self.base_audio_wave))

    def cross_correlation_cuda(self, mfcc1, mfcc2, nframes):
        n1, mdim1 = mfcc1.shape
        n2, mdim2 = mfcc2.shape
        o_min = nframes - n2
        o_max = n1 - nframes + 1
        mfcc1_torch = mfcc1.cuda()
        mfcc2_torch = mfcc2.cuda()
        c_right = f.conv1d(
            mfcc2_torch.transpose(0, 1)[None],
            mfcc1_torch[:nframes].transpose(0, 1)[:, None],
            None,
            stride=1,
            groups=26,
        )[0].norm(dim=0)
        c_right = c_right[1:].__reversed__()
        c_left = f.conv1d(
            mfcc1_torch.transpose(0, 1)[None],
            mfcc2_torch[:nframes].transpose(0, 1)[:, None],
            None,
            stride=1,
            groups=26,
        )[0].norm(dim=0)
        c = torch.cat([c_left, c_right])
        return c, o_min, o_max

    def load_audio(self, audio_f):
        assert audio_f.endswith(".wav")
        audio_wave, sample_rate = ta.load(audio_f, normalize=True)
        assert sample_rate == self.sample_rate
        return audio_wave  # keep on CPU for MFCC computation

    def mfcc_normalize(self, mfcc):
        mfcc = mfcc[0].transpose(1, 0)
        return (mfcc - mfcc.mean(dim=0)) / mfcc.std(dim=0)

    def find_offset(self, ref_audio_f):
        ref_audio_wave = self.load_audio(ref_audio_f)
        ref_audio_mfcc = self.mfcc_normalize(self.MFCC(ref_audio_wave))

        mfcc1 = self.base_audio_mfcc
        mfcc2 = ref_audio_mfcc

        # Derive correl_nframes from the length of audio supplied, to avoid buffer overruns
        correl_nframes = min(len(mfcc1) // 2, len(mfcc2) // 2, self.max_frames)
        if correl_nframes < 10:
            raise RuntimeError(
                "Not enough audio to analyse - try longer clips, less trimming, or higher resolution."
            )

        c, earliest_frame_offset, latest_frame_offset = self.cross_correlation_cuda(
            mfcc1, mfcc2, nframes=correl_nframes
        )
        # c = c/np.sqrt(correl_nframes)
        # max_k_index = np.argmax(c)
        max_k_index = torch.argmax(c)
        max_k_frame_offset = max_k_index
        if max_k_index > len(c) / 2:
            max_k_frame_offset -= len(c)
        time_scale = self.bin_stride / self.sample_rate
        time_offset = (max_k_frame_offset) * time_scale

        # std = np.std(c) / np.sqrt(correl_nframes)
        c = f.softmax(c / np.sqrt(correl_nframes))
        # if std < 1e-10:
        #     score = float('inf')
        # else:
        score = c[max_k_index]  # standard score of peak
        return {
            "time_offset": time_offset,
            "frame_offset": int(max_k_index),
            "standard_score": score,
            "correlation": c,
            "time_scale": time_scale,
            "earliest_frame_offset": int(earliest_frame_offset),
            "latest_frame_offset": int(latest_frame_offset),
        }
