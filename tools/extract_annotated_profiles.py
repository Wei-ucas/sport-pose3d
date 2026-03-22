import glob
import json
import os
import cv2
import numpy as np
import pickle


def get_frame(file_path, time):
    # get frame from file_path according to time
    cap = cv2.VideoCapture(file_path)
    print(file_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, int(time * 1000))
    ret, frame = cap.read()
    return frame


def extract_profile_from_video(video_path, annotation_file, output_folder, view):
    # Read annotation file
    annot_file = json.load(open(annotation_file))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    player_profiles = {}  # {team+player_number:{tmp_id: 0,"team_id": team_id, "player_number": player_number, "profile_image":
    # profile_image_path}}
    player_profiles_path = os.path.join(output_folder, "profiles.pkl")
    if os.path.exists(player_profiles_path):
        player_profiles = pickle.load(open(player_profiles_path, "rb"))

    team_player_dict = {}

    attribute = annot_file["attribute"]
    tmp_player_id = 0

    team_attribute = {
        "3": "Team A",
        "4": "Team B",
    }

    for team_id in team_attribute.keys():
        team = attribute[team_id]
        team_name = team["aname"]
        player_number_dict = team['options']
        team_player_dict[team_id] = {
            "team_name": team_name,
            "player_number_dict": player_number_dict
        }
        for player_number in player_number_dict.values():
            if player_number == '-1':
                continue
            if "{}#{}".format(team_name, player_number) not in player_profiles.keys():
                # continue
                player_profiles["{}#{}".format(team_name, player_number)] = {
                    "team_id": team_id,
                    "tmp_player_id": tmp_player_id,
                    "profile_image": []
                }
            tmp_player_id += 1

    # create folders for each player
    for player in player_profiles.keys():
        player_folder = os.path.join(output_folder, player)
        if not os.path.exists(player_folder):
            os.makedirs(player_folder)

    annotation_list = annot_file['metadata']
    for ann in annotation_list.keys():
        z = annotation_list[ann]['z']
        frame = get_frame(video_path, z[0])
        if frame is None:
            continue

        av = annotation_list[ann]['av']
        for team_id in team_attribute.keys():
            if team_id in av.keys() and av[team_id] != '0':
                team_name = team_player_dict[team_id]['team_name']
                player_number = team_player_dict[team_id]['player_number_dict'][av[team_id]]
                break

        bbox = annotation_list[ann]['xy']
        if len(bbox) != 5:
            continue
        x1, y1, w, h = bbox[1:]
        if w * h < 20:
            continue
        cut_profile = frame[int(y1):int(y1 + h), int(x1):int(x1 + w)]

        if cut_profile.shape[0] == 0 or cut_profile.shape[1] == 0:
            continue

        player_id = "{}#{}".format(team_name, player_number)
        player_profiles[player_id]["profile_image"].append(cut_profile)

        profile_image_path = os.path.join(output_folder,
                                          player_id,
                                          "{}_{}.png".format(len(player_profiles[player_id]["profile_image"]), view))
        cv2.imwrite(profile_image_path, cut_profile)

    # save player profiles to a pkl file
    player_profiles_path = os.path.join(output_folder, "profiles.pkl")
    with open(player_profiles_path, "wb") as f:
        pickle.dump(player_profiles, f)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract player profile image from the video according to annotations by via")
    parser.add_argument("game_name", type=str, help="Name of the video")
    parser.add_argument("--workdir", type=str, help="Path to the work directory", default='games/')

    args = parser.parse_args()

    video_folder = os.path.join(args.workdir, "videos", args.game_name)
    annotation_folder = os.path.join(args.workdir, "prepare/profiles", args.game_name)
    profile_folder = annotation_folder

    # Read annotation file
    annotation_files = glob.glob(os.path.join(annotation_folder, f"*.json"))

    if len(annotation_files) == 0:
        raise ValueError(f"No annotation file found for video {args.game_name}")
    for annotation_file in annotation_files:
        annotation_file = annotation_file.replace("\\", "/")
        print("Processing annotation file:", annotation_file)
        view_name = os.path.basename(annotation_file).split('.')[0]
        video_file = os.path.join(video_folder, f'{view_name}.mp4')

        if not os.path.exists(video_file):
            raise ValueError(f"Video file {video_file} does not exist for view {view_name}")

        annot_file = json.load(open(annotation_file))
        extract_profile_from_video(video_file, annotation_file, profile_folder, view_name)
