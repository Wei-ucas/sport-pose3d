import os
import logging
import tqdm
from src.data_io.path_config import GamePath
from src.structures.frame import Frame
from src.data_io.loader import FrameInput
from src.data_io.saver import FrameSaver, VideoWriter
from src.modules.player_matcher.reid_match import PlayerReIDMatch

MAX_FRAME_BATCH = 10000  # Maximum number of frames to process in one batch, avoid memory overflow


def reid_processor(
        view_id: str,
        data_path_cfg: GamePath,
        vis: bool = False,
):

    if os.path.exists(data_path_cfg.get_prediction_save_path("reid", view_id)):
        logging.info(f"View {view_id} already processed, skipping.")
        return

    frame_reader = FrameInput(
        view_id=view_id,
        pkl_data_path=data_path_cfg.get_detection_path(view_id),
        video_path=data_path_cfg.get_video_path(view_id),
        frame_downsample_rate=data_path_cfg.frame_step
    )

    frame_saver = FrameSaver(
        save_path=data_path_cfg.get_prediction_save_path("reid", view_id),
    )

    if vis:
        video_writer = VideoWriter(
            save_path=data_path_cfg.get_vis_path("reid", view_id),
            fps=data_path_cfg.fps,
        )
    else:
        video_writer = None

    player_matcher = PlayerReIDMatch(
        profile_file_path=data_path_cfg.get_profile_path,
        reid_match_threshold=500,
        max_gallery_profile_num=20,
        track_link_overlap_threshold=0.5,
        valid_photo_threshold=0.7,
        min_query_photo_num=2,
        max_query_photo_num=10,
        video_fps=data_path_cfg.fps
    )

    logging.info(f"Start processing {view_id} with reid processor")

    max_frame = len(
        frame_reader) // data_path_cfg.frame_step if data_path_cfg.max_frame_num == -1 else data_path_cfg.max_frame_num
    frame_list = []

    def dump_batch():
        player_matcher.match_rest()
        for j in tqdm.tqdm(range(len(frame_list))):
            frame = frame_list[j]
            player_matcher.reid_frame(frame)
            if vis:
                img = frame.visualize()
                video_writer.write(img)
            frame.image = None
            frame_saver.save_frame(frame)
        frame_list.clear()
        player_matcher.init()

    for i in tqdm.tqdm(range(0, max_frame)):
        frame = next(frame_reader)
        if frame is None:
            break

        if frame.frame_id % data_path_cfg.frame_step != 0:
            continue

        player_matcher.match_frame(frame)

        if not vis or i > MAX_FRAME_BATCH:
            frame.image = None

        frame_list.append(frame)

        if len(frame_list) > MAX_FRAME_BATCH:
            logging.info(f"Dumping batch of {len(frame_list)} frames")
            dump_batch()

    if len(frame_list) > 0:
        logging.info(f"Dumping last batch of {len(frame_list)} frames")
        dump_batch()

    if vis:
        video_writer.close()
    frame_saver.close()
    logging.info(f"Finished processing {view_id} with reid processor")
