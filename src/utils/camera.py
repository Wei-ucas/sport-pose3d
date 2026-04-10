from __future__ import annotations
import os
from typing import Dict, List
import cv2
import numpy as np


from . import constant


def load_camera_params(intri_path, extri_path, camera_ids):
    camera_params = {
        "K": [],
        "dist": [],
        "R": [],
        "T": [],
        "rvec": [],
        "RT": [],
        "invK": [],
        "P": [],
    }
    assert os.path.exists(intri_path), f"{intri_path} not exists"
    intri_f = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    if extri_path is not None:
        extri_f = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    # if camera_ids[0]  not in intri_f.getNone("names")
    camera_names = []
    name_node = intri_f.getNode("names")
    for i in range(name_node.size()):
        val = name_node.at(i).string()
        if val == "":
            val = str(int(name_node.at(i).real()))
        if val != "none":
            camera_names.append(val)
    for camera_id in camera_ids:
        K = intri_f.getNode(f"K_{camera_id}").mat()
        dist = intri_f.getNode(f"dist_{camera_id}").mat()
        camera_params["K"].append(K)
        camera_params["invK"].append(np.linalg.inv(K))
        camera_params["dist"].append(dist)
        if extri_path is not None:
            camera_params["rvec"].append(
                extri_f.getNode("R_{}".format(camera_id)).mat()
            )
            camera_params["R"].append(cv2.Rodrigues(camera_params["rvec"][-1])[0])
            camera_params["T"].append(extri_f.getNode("T_{}".format(camera_id)).mat())
            camera_params["RT"].append(
                cv2.hconcat([camera_params["R"][-1], camera_params["T"][-1]])
            )
            camera_params["P"].append(
                camera_params["K"][-1]
                @ np.hstack((camera_params["R"][-1], camera_params["T"][-1]))
            )
    for key in camera_params:
        if key != "dist":
            camera_params[key] = np.array(camera_params[key])
    camera_params["view_names"] = camera_ids
    camera_params["name2index"] = {name: i for i, name in enumerate(camera_ids)}
    return camera_params


def save_camera_params(camera_params, save_dir):
    # save K, dist to intri.yml, R, T to extri.yml
    intri_path = f"{save_dir}/intri.yml"
    extri_path = f"{save_dir}/extri.yml"
    fs_intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_WRITE)
    fs_extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_WRITE)
    camera_ids = list(camera_params.keys())
    fs_intri.write("names", camera_ids)
    fs_extri.write("names", camera_ids)
    for camera_id in camera_ids:
        fs_intri.write(
            f"K_{camera_id}",
            (camera_params[camera_id]["K"]),  # type: ignore
        )
        fs_intri.write(
            f"dist_{camera_id}",
            (
                #
                camera_params[camera_id]["dist"]  # type: ignore
            ),
        )
        fs_extri.write(
            f"R_{camera_id}",
            (camera_params[camera_id]["Rvec"]),  # type: ignore
        )
        fs_extri.write(
            f"T_{camera_id}",
            (camera_params[camera_id]["T"]),  # type: ignore
        )
    fs_intri.release()
    fs_extri.release()


def undistort_points(keypoints, K, dist):
    # keypoints: (N, 3)
    assert len(keypoints.shape) == 2, keypoints.shape
    kpts = keypoints[:, None, :2]
    kpts = kpts.astype(float)
    kpts = np.ascontiguousarray(kpts)
    # kpts = cv2.undistortPoints(kpts, K, dist, P=K)
    if len(dist.reshape(-1)) == 4:
        kpts = cv2.fisheye.undistortPoints(kpts, K, dist, P=K)
    else:
        kpts = cv2.undistortPoints(kpts, K, dist, P=K)
    keypoints = np.hstack([kpts[:, 0], keypoints[:, 2:]])
    return keypoints


def load_camera(path_prefix: str, camera_ids: str | List[str], index: int = None):
    assert camera_ids is not None, "camera ids should be provided"
    if not isinstance(camera_ids, list):
        camera_id_list = [camera_ids]
    else:
        camera_id_list = camera_ids
    if index is None:
        intri_path = f"{path_prefix}/intri.yml"
        extri_path = f"{path_prefix}/extri.yml"
        camera_params = load_camera_params(intri_path, extri_path, camera_id_list)
    else:
        raise NotImplementedError("Not support read camera with index")

    if not isinstance(camera_ids, list):
        return get_camera_by_id(camera_params, camera_ids)

    return camera_params


def get_camera_by_id(camera_params: Dict, view_id: str):
    """
    get specific cam by id
    Args:
        camera_params: Dict
        view_id: str. target view id
    Returns:
    """
    camera_index = camera_params.get("name2index", {}).get(view_id)
    if camera_index == -1:
        raise ValueError(f"camera {view_id} not in {camera_params}")
    selected_camera = {}
    for key, v in camera_params.items():
        if isinstance(v, np.ndarray) or isinstance(v, list):
            selected_camera[key] = v[camera_index]
    return selected_camera


def image2ground(
    points: np.ndarray, camera_params: Dict[str, np.ndarray] = None, view_id=0
):
    """
    unproject points from image coordinate to world coordinate on ground plane (z=0)
    Args:
        view_id:
        points: (N, 2)
        camera_params: {
            "K": [],
            "dist": [],
            "R": [],
            "T": [],
            "rvec": [],
            "RT": [],
            }
    """
    assert camera_params is not None
    points = np.array(points, dtype=np.float32)
    K = camera_params["K"]
    dist = camera_params["dist"]
    # points = cv2.undistortPoints(points[None, :, :], K, dist, P=K)
    points = undistort_points(points, K, dist)

    points = cv2.convertPointsToHomogeneous(points)
    points = points[:, 0, :, None]
    if "inv_p" in camera_params:
        points = camera_params["inv_p"] @ points
    else:
        # K = camera_params['K'][view_id]
        # dist = camera_params['dist'][view_id]
        # R = camera_params['R'][view_id]
        # T = camera_params['T'][view_id]
        RT = camera_params["RT"]  # (3, 4)
        # points = np.array(points, dtype=np.float32)
        # points = cv2.undistortPoints(points[None, :, :], K, dist, P=K)
        # points = cv2.convertPointsToHomogeneous(points)
        # points = points[:, 0, :, None]
        points = np.linalg.inv(K) @ points
        points = np.linalg.inv(RT[:, [0, 1, 3]]) @ points
    points = points[..., 0]
    points = points[:, :2] / points[:, 2, None]
    return points


def camera_project(points: np.ndarray, camera: dict) -> np.ndarray:
    """
    project 3d points to 2d points
    Args:
        points: np.ndarray, [n_points, 3]
        camera: {"R":np.ndarray [3,3], "T":np.ndarray [3,1], "K":np.ndarray [3,3], "dist":np.ndarray [1,5]}

    Returns:
        points2d: np.ndarray, [n_points, 2]
    """
    r_mat = camera["R"].astype("float64")
    t_vec = camera["T"].astype("float64")
    K = camera["K"].astype("float64")
    dist = np.asarray(camera["dist"]).astype("float64")
    points2d, _ = cv2.projectPoints(points.astype("float64"), r_mat, t_vec, K, dist)

    return points2d.reshape(-1, 2).astype("float64")


def get_basketball_outer_range(camera_params: Dict) -> List:
    all_court_corners = np.concatenate(
        [constant.court_top_corners, constant.court_ground_corners], axis=0
    )
    # court_image_range = {}
    # for key, param in camera_params:
    # project points to image
    image_court_corners = camera_project(all_court_corners, camera_params)
    # get the min max
    court_range = [
        image_court_corners[:, 0].min(),
        image_court_corners[:, 1].min(),
        image_court_corners[:, 0].max(),
        image_court_corners[:, 1].max(),
    ]
    # court_image_range[key] = court_range
    return court_range


if __name__ == "__main__":
    test_camera_params = load_camera_params(
        "D:\\workspace\\gits\\BasketballNet\\workdir\\prepare\\camera\\0303_201122\\intri.yml",
        "D:\\workspace\\gits\\BasketballNet\\workdir\\prepare\\camera\\0303_201122\\extri.yml",
    )

    test_points = np.array([[713, 415], [905, 638]], dtype=np.float32)
    test_ground_points = image2ground(test_points, test_camera_params)
