import glob
import os.path
import tempfile
import cv2
import tqdm
from subprocess import Popen, PIPE
import ffmpeg


def get_audio_file(video_name=None, audio_file_root=None):
    if video_name is None or audio_file_root is None:
        tmp = tempfile.NamedTemporaryFile(mode="r+b", prefix="offset_", suffix=".wav")
        tmp_name = tmp.name
        tmp.close()
    else:
        # tmp_name = audio_file
        audio_file = os.path.join(audio_file_root, video_name[:-4] + '.wav')
        tmp_name = audio_file
    return tmp_name


def audio_from_video(video_file, fs, trim=None, audio_file=None):
    """Converts the input video to a temporary 16-bit WAV file and trims it to length.

        Parameters
        ----------
        afile: string
            The input media file to process.  It must contain at least one audio track.
        fs: int
            The sample rate that the audio should be converted to during the conversion
        trim: float
            The length to which the output audio should be trimmed, in seconds.  (Audio beyond this point will be discarded.)

        Returns
        -------
        A string containing the path of the processed media file.  You should delete this file after use.
        """
    if audio_file is None:
        tmp = tempfile.NamedTemporaryFile(mode="r+b", prefix="offset_", suffix=".wav")
        tmp_name = tmp.name
        tmp.close()
    else:
        tmp_name = audio_file

    ffmpeg_cmd = [
        "ffmpeg",
        # "-loglevel",
        # "panic",
        "-i",
        video_file,
        "-ac",
        "1",
        "-ar",
        str(fs),
        "-ss",
        "0"]

    if trim is not None:
        ffmpeg_cmd.extend(["-t", str(trim)])

    # keep original order
    ffmpeg_cmd.extend(
        [
            "-acodec",
            "pcm_s16le",
            "-vn",
            # "-c:a",
            # "copy",
            tmp_name,
        ]
    )

    psox = Popen(ffmpeg_cmd,
                 stderr=PIPE, )
    out, err = psox.communicate()
    if not psox.returncode == 0:
        raise Exception("FFMpeg failed")

    return tmp_name


def get_frame_time(video_path):
    # open the video file
    cap = cv2.VideoCapture(video_path)

    # get the frame rate and total number of frames in the video
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps = []

    # iterate over the frames in the video
    for i in tqdm.tqdm(range(total_frames)):
        # read the frame
        # ret, frame = cap.read()
        ret = cap.grab()

        # get the timestamp for this frame
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

        # add the timestamp to the list
        timestamps.append(timestamp)

    # release the video capture object
    cap.release()

    return timestamps


def video_cut(video_file, start_delay=None, start_time=None, duration=None, stop_padding=None, out_file=None,
              resolution=None, bitrate=None, ffmpeg_path='ffmpeg', cuda=True, fps=None):
    # print("Trim video {}, start_delay {}".format(video_file, start_delay))
    print("++++ Process Video {}".format(video_file))
    print("start padding:{} stop padding:{}".format(start_delay, stop_padding))
    print("start shift:{} duration:{}".format(start_time, duration))
    print("resolution:{} bitrate:{}".format(resolution, bitrate))
    print("==> Output: {}".format(out_file))
    '''
    Cut video with ffmpeg
    :param video_file: input video file path
    :param start_time: start time
    :param durations: cut time length
    :param out_file: output file path
    :param resolution: the output video resolution
    :return:
    '''
    # if start_delay is not None:
    #     assert start_time is None
    # else:
    #     assert start_time is not None

    base_path = os.path.split(out_file)[0]
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    scale_option = "scale_cuda" if cuda else "scale"
    format_option = "hwdownload,format=nv12"

    filter = []
    if resolution is not None:
        assert isinstance(resolution, tuple)
        resolution = "{}:{}".format(resolution[0], resolution[1])
        filter.append("{}={}".format(scale_option, resolution))
    if cuda:
        filter.append(format_option)
    if start_delay is not None:
        filter.append("tpad=start_duration={}:stop_duration={}".format(start_delay, stop_padding))
    filter = ",".join(filter)

    args = [
        ffmpeg_path,
        # "-hwaccel cuda -hwaccel_output_format cuda" if cuda else "",
        '-hwaccel auto',
        # "-loglevel",
        # "panic",
        "-ss {}".format(str(start_time)) if start_time is not None else "",
        "-t {}".format(str(duration)) if duration is not None else "",
        "-i",
        video_file,
        # "-filter_complex \"[0:0]{}=720:540,{} tpad=start_duration={}:stop_duration={}[v0]\" ".format(scale_option,format_option,
        #     start_delay, stop_padding),
        # '-filter_complex \"[0:0]{}[v0]\" '.format(filter) if filter != "" else "",
        '-filter:v "{}"'.format(filter) if filter != "" else "",
        # "-map \"[v0]\"" if filter != "" else "",
        "-r {}".format(fps) if fps is not None else "",
        "-c:v",
        "h264" if bitrate is not None else "copy",
        # "h264",
        "-c:a",
        "aac",
        '-b:v {}'.format(bitrate) if bitrate is not None else "",
        # "-vsync 2",
        '-movflags +faststart',
        "-max_muxing_queue_size 9999",
        out_file,
        "-y"
    ]
    # if resolution != "-1":
    #     args.insert(7, "-vf")
    #     args.insert(8, "scale="+str(resolution))
    print(" ".join(args))
    psox = Popen(
        " ".join(args),
        shell=True,
        stderr=PIPE,
    )
    out, err = psox.communicate()
    if not psox.returncode == 0:
        raise Exception(err)
    print(out_file + "  finished")


def extract_cover_image(video_path, image_path):
    # take the first frame of the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite(image_path, frame)



def get_video_info(video_file):
    return ffmpeg.probe(video_file)


def video_concat(video_list, target_path, target_name, rm_list=False):
    file_list_path = os.path.join(target_path, target_name + '_file_list.txt')
    output_video = os.path.join(target_path, target_name + '.mp4')
    if os.path.exists(output_video):
        print(output_video + "   has processed")
        return
    with open(file_list_path, 'w', encoding='utf-8') as f:
        f.writelines(["file '%s'\n" % item for item in video_list])
    print("==============================================")
    print("Concating videos \n{} \n To \n{}".format("\n".join(video_list), output_video))
    # os.system(f'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64/:/usr/local/lib/x86_64-linux-gnu ffmpeg -f concat -safe 0 -i {file_list_path} -c copy {output_video}.mp4')
    # os.system(f'rm {file_list_path}')
    # input_f = " ".join([" -i {}".format(f) for f in video_list])
    # input_f = "|".join(video_list)
    args = 'ffmpeg -f concat -safe 0 -i {}  -c copy  -movflags faststart {} -y '.format(file_list_path, output_video)
    # args = 'ffmpeg -f concat -safe 0 {}  -c copy -movflags faststart {} -y '.format(input_f, output_video)
    # args = 'ffmpeg -i "concat:{}"  -c copy -movflags faststart {} -y '.format(input_f, output_video)
    psox = Popen(
        args,
        stderr=PIPE,
        shell=True,
    )
    out, err = psox.communicate()
    if not psox.returncode == 0:
        # raise Exception("FFMpeg failed")
        print("!!!Failed " + args + '\n')
        raise ValueError(err)
    else:
        print(output_video + " finished\n")
    print("==============================================")
    if rm_list:
        os.system(f'rm {file_list_path}')


def video_frame_merge_delay(video_file_lists, start_durations, camera_ids, outpath, cut_first=False, cuda=True):
    if cuda:
        inputs = ["-hwaccel cuda -hwaccel_output_format cuda -i {}".format(v) for v in video_file_lists]
    else:
        inputs = [" -i {}".format(v) for v in video_file_lists]
    v_id = ["[{}:v]".format(i) for i in range(len(inputs))]
    rescale = True
    if cuda:
        scale_option = "scale_cuda"
        format_option = "hwdownload,format=nv12,"
    else:
        scale_option = "scale"
        format_option = " "
    if cut_first:
        scale = [
            '[{}:v]{}[s{}]'.format(i, format_option, i)
            for i in range(len(camera_ids))
        ]
        if not cuda:
            rescale = False
        else:
            rescale = True
            v_id = ["[s{}]".format(i) for i in range(len(inputs))]
    else:
        scale = [
            '[{}:v]{}=720:540,{} tpad=start_duration={}[s{}]'.format(i, scale_option, format_option, start_durations[i],
                                                                     i)
            for i in range(len(camera_ids))
        ]
        v_id = ["[s{}]".format(i) for i in range(len(inputs))]
    # else:
    #     scale = []

    drawtext = [
        "{} drawtext=text=\"{}\":fontsize=20:x=(w-text_w)/2:y=text_h/2[t{}]".format(v_id[i], camera_ids[i], i) for i in
        range(len(camera_ids))
    ]
    o_id = ["[t{}]".format(i) for i in range(len(inputs))]

    preset_layout = ['0_0', 'w0_0', 'w0+w1_0', 'w0+w1+w2_0', '0_h0', 'w4_h0', 'w4+w5_h0', 'w4+w5+w6_h0', '0_h0+h4',
                     'w8_h0+h4',
                     'w8+w9_h0+h4', 'w8+w9+w10_h0+h4', '0_h0+h4+h8', 'w12_h0+h4+h8', 'w12+w13_h0+h4+h8',
                     'w12+w13+w14_h0+h4+h8']
    layout = preset_layout[:len(inputs)]
    if rescale:
        filter_complex_arg = ";".join(scale) + ";" + ";".join(drawtext) + ";" + "".join(
            o_id) + "xstack=inputs={}:".format(len(inputs)) + "layout=" + "|".join(layout)
    # filter_complex_arg = ";".join(scale) + ";" + "".join(
    #     v_id) + "xstack=inputs={}:".format(len(inputs)) + "layout=" + "|".join(layout)
    else:
        filter_complex_arg = ";".join(drawtext) + ";" + "".join(
            o_id) + "xstack=inputs={}:".format(len(inputs)) + "layout=" + "|".join(layout)
    args = [
        "ffmpeg",
        # "-hwaccel cuda -hwaccel_output_format cuda ",
        " ".join(inputs),
        "-filter_complex",
        '"' + filter_complex_arg + '"',
        '-r 60',
        '-b:v 5M',
        "-c:v h264_nvenc",
        # "-vsync 2",
        outpath,
        "-y"
    ]
    psox = Popen(
        " ".join(args),
        stderr=PIPE,
        shell=True,
    )
    out, err = psox.communicate()
    if not psox.returncode == 0:
        raise Exception(err)
    print(outpath + " finished")


def video_frame_merge(video_file_lists, video_names, outpath, rescale=False, ffmpeg_path='/usr/bin/ffmpeg'):
    inputs = ["-i {}".format(v) for v in video_file_lists]
    v_id = ["{}:v".format(i) for i in range(len(inputs))]
    if rescale:
        scale = [
            '[{}:v]scale=720:-1[s{}]'.format(i, i) for i in range(len(video_names))
        ]
        v_id = ["s{}".format(i) for i in range(len(inputs))]
    else:
        scale = []
    drawtext = [
        "[{}]drawtext=text='{}':fontsize=20:x=(w-text_w)/2:y=text_h/2[t{}]".format(v_id[i], video_names[i], i) for i in
        range(len(video_names))
    ]
    o_id = ["[t{}]".format(i) for i in range(len(inputs))]
    preset_layout = ['0_0', 'w0_0', 'w0+w1_0', 'w0+w1+w2_0', '0_h0', 'w4_h0', 'w4+w5_h0', 'w4+w5+w6_h0', '0_h0+h4',
                     'w8_h0+h4',
                     'w8+w9_h0+h4', 'w8+w9+w10_h0+h4', '0_h0+h4+h8', 'w12_h0+h4+h8', 'w12+w13_h0+h4+h8',
                     'w12+w13+w14_h0+h4+h8']
    layout = preset_layout[:len(inputs)]
    if rescale:
        filter_complex_arg = ";".join(scale) + ";" + ";".join(drawtext) + ";" + "".join(
            o_id) + "xstack=inputs={}:".format(len(inputs)) + "layout=" + "|".join(layout)
    else:
        filter_complex_arg = ";".join(drawtext) + ";" + "".join(
            o_id) + "xstack=inputs={}:".format(len(inputs)) + "layout=" + "|".join(layout)
    args = [
        ffmpeg_path,
        " ".join(inputs),
        "-filter_complex",
        '"' + filter_complex_arg + '"',
        outpath,
        "-y"
    ]
    psox = Popen(
        " ".join(args),
        stderr=PIPE,
        shell=True,
    )
    psox.communicate()
    if not psox.returncode == 0:
        raise Exception("FFMpeg failed")
    print(outpath + " finished")


if __name__ == '__main__':
    # timestampes = get_frame_time("/mnt/workspace/wangwei/oss_beijing/donglinfeng/shangti_data/floor/9/20230413/1/20230413_1.mp4")
    # video_info = get_video_info("/mnt/workspace/wangwei/oss_beijing/donglinfeng/shangti_data/floor/0/20230413/1/20230413_1.mp4")
    # print(video_info)
    video_lists = glob.glob("/mnt/workspace/wangwei/tmp/aligned/*.mp4")
    video_names = [v[35:] for v in video_lists]
    video_frame_merge(video_lists, video_names, "/mnt/workspace/wangwei/tmp/all_merge-720p-text-1.mp4", rescale=False)
