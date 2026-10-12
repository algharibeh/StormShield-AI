"""Shared helpers: nuScenes loading, radar points per sample, box geometry.

Everything works on the official nuScenes trainval metadata plus whatever
blob(s) are present on disk. Samples whose sensor files are missing are
simply not used.
"""
import json
import os
from functools import lru_cache

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name

DETECTION_NAMES = [
    "car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
    "motorcycle", "bicycle", "traffic_cone", "barrier",
]
RADAR_CHANNELS = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]


def load_nusc(dataroot, version="v1.0-trainval"):
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def split_scene_tokens(nusc, split):
    """Scene tokens of the official split ('train' or 'val', or mini variants)."""
    names = set(create_splits_scenes()[split])
    return {s["token"] for s in nusc.scene if s["name"] in names}


def sample_has_files(nusc, sample):
    """True when the LiDAR keyframe and all five radars exist on disk."""
    chans = ["LIDAR_TOP"] + RADAR_CHANNELS
    for ch in chans:
        sd = nusc.get("sample_data", sample["data"][ch])
        if not os.path.exists(os.path.join(nusc.dataroot, sd["filename"])):
            return False
    return True


def available_samples(nusc, split):
    """Sample tokens of a split whose files are present locally."""
    scenes = split_scene_tokens(nusc, split)
    out = []
    for s in nusc.sample:
        if s["scene_token"] in scenes and sample_has_files(nusc, s):
            out.append(s["token"])
    return out


def scene_description(nusc, sample_token):
    s = nusc.get("sample", sample_token)
    return nusc.get("scene", s["scene_token"])["description"].lower()


def ego_pose_of_sample(nusc, sample_token):
    s = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", sd["ego_pose_token"])


def radar_points_global(nusc, sample_token):
    """All radar returns of a keyframe in the global frame.

    Returns an (N, 5) array: x, y (global), vx, vy (ego-motion compensated
    radial velocity, global frame), rcs. Uses the devkit's default validity
    filters.
    """
    s = nusc.get("sample", sample_token)
    rows = []
    for ch in RADAR_CHANNELS:
        sd = nusc.get("sample_data", s["data"][ch])
        path = os.path.join(nusc.dataroot, sd["filename"])
        if not os.path.exists(path):
            continue
        pc = RadarPointCloud.from_file(path)
        if pc.nbr_points() == 0:
            continue
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        r_cs = Quaternion(cs["rotation"]).rotation_matrix
        r_ep = Quaternion(ep["rotation"]).rotation_matrix
        xyz = pc.points[:3, :]
        xyz = r_cs @ xyz + np.array(cs["translation"])[:, None]
        xyz = r_ep @ xyz + np.array(ep["translation"])[:, None]
        v = np.vstack([pc.points[8, :], pc.points[9, :], np.zeros(pc.nbr_points())])
        v = r_ep @ (r_cs @ v)
        rcs = pc.points[5, :]
        rows.append(np.stack([xyz[0], xyz[1], v[0], v[1], rcs], axis=1))
    if not rows:
        return np.zeros((0, 5))
    return np.concatenate(rows, axis=0)


def box_yaw(rotation):
    q = Quaternion(rotation)
    return q.yaw_pitch_roll[0]


def points_in_box_bev(points_xy, center_xy, size_wl, yaw, margin=1.0):
    """Boolean mask of BEV points inside a box footprint enlarged by margin (m).

    size_wl is (width, length) as in nuScenes size[0], size[1].
    """
    if points_xy.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    d = points_xy - np.asarray(center_xy)[None, :]
    c, s = np.cos(-yaw), np.sin(-yaw)
    local_x = c * d[:, 0] - s * d[:, 1]
    local_y = s * d[:, 0] + c * d[:, 1]
    half_l = size_wl[1] / 2 + margin
    half_w = size_wl[0] / 2 + margin
    return (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)


def load_results(path):
    with open(path) as f:
        d = json.load(f)
    return d["results"], d.get("meta", {})


def save_results(path, results, use_camera, use_lidar, use_radar):
    meta = {
        "use_camera": use_camera, "use_lidar": use_lidar, "use_radar": use_radar,
        "use_map": False, "use_external": False,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f)


def gt_boxes_of_sample(nusc, sample_token):
    """Ground-truth boxes as dicts with detection_name, translation, size, rotation, velocity."""
    s = nusc.get("sample", sample_token)
    out = []
    for t in s["anns"]:
        a = nusc.get("sample_annotation", t)
        name = category_to_detection_name(a["category_name"])
        if name is None:
            continue
        vel = nusc.box_velocity(t)
        if np.any(np.isnan(vel)):
            vel = np.zeros(3)
        out.append({
            "detection_name": name,
            "translation": a["translation"],
            "size": a["size"],
            "rotation": a["rotation"],
            "velocity": [float(vel[0]), float(vel[1])],
        })
    return out
