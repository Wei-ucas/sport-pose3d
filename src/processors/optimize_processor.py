import os.path
import logging
import tqdm

from src.data_io.path_config import GamePath
from src.data_io.loader import MvFrameInput
from src.data_io.saver import FrameSaver, VideoWriter
from src.modules.player_optimizer.trajectory_optimizer import PlayerOptimizer

MAX_BATCH_SIZE = 10000  # Maximum number of frames to process in a single batch


def optimize_processor(
    data_path_cfg: GamePath,
    use_reid: bool = True,
    kalman_process_noise: float = 1e-3,
    kalman_measurement_noise: float = 5e-2,
    one_player: bool = False,
):
    if os.path.exists(data_path_cfg.get_prediction_save_path("3d-opt")):
        # If the optimized path already exists, skip the optimization
        logging.info(
            f"Optimized path {data_path_cfg.get_prediction_save_path('3d-opt')} already exists, skipping optimization."
        )
        return

    logger = logging.getLogger("optimize_processor")
    logger.info("Starting optimization processor...")

    opt_3d_path = data_path_cfg.get_prediction_save_path("3d-opt")
    if not os.path.exists(opt_3d_path):
        os.makedirs(os.path.dirname(opt_3d_path), exist_ok=True)
    else:
        logger.info(f"{opt_3d_path} already exists, skipping optimization.")
        return

    logger.info(f"Optimizing 3D data to {opt_3d_path}...")

    matched_3d_result_path = data_path_cfg.get_prediction_save_path("3d")
    if not os.path.exists(matched_3d_result_path):
        logger.error(f"Matched 3D result path {matched_3d_result_path} does not exist.")
        return

    mv_frame_reader = MvFrameInput(matched_3d_result_path)

    frame_saver = FrameSaver(opt_3d_path, save_type="pickle")
    # video_writer = VideoWriter(
    #     save_path=data_path_cfg.get_vis_path("3d-opt"),
    #     fps=int(0.2 * data_path_cfg.fps),
    # )
    # vis_step = int(5 / data_path_cfg.frame_step)

    player_id2name = data_path_cfg.get_player_info()

    optimizer = PlayerOptimizer(
        trajectory_range=5,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
        one_player=one_player,
    )

    frame_list = []

    player_ids = None

    def dump_batch(appeared_player_ids=None):
        if len(frame_list) == 0:
            return
        logger.info(f"Processing batch of {len(frame_list)} frames...")
        optimized_frames, _player_ids = optimizer.optimize(
            frame_list, appeared_player_ids, use_reid=use_reid
        )
        for frame in optimized_frames:
            frame_saver.save_frame(frame)
            # if frame.frame_id % vis_step == 0:
            #     vis_image = frame.visualize(player_names=player_id2name)
            #     video_writer.write(vis_image)
        frame_list.clear()
        return _player_ids

    for mv_frame in tqdm.tqdm(mv_frame_reader):
        if mv_frame is None:
            break
        frame_list.append(mv_frame)
        if len(frame_list) >= MAX_BATCH_SIZE:
            player_ids = dump_batch(player_ids)

    dump_batch(player_ids)
    frame_saver.close()
    # video_writer.close()
    logger.info(f"Optimization completed. Results saved to {opt_3d_path}")
