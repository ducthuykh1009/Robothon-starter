from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObjectAffordance:
    object_name: str
    object_type: str
    object_center: list[float]
    object_orientation: list[float]
    object_size: list[float]
    face_normals: list[list[float]]
    radius: float | None
    long_axis: list[float] | None
    centerline: list[list[float]] | None
    grasp_affordance_regions: dict[str, str]
    recommended_grasp_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OBJECT_TYPE_BY_NAME = {
    "sphere_object": "sphere",
    "cube_object": "cube",
    "cylinder_object": "cylinder_horizontal",
    "capsule_object": "capsule",
    "stylus_tool": "stylus",
    "button": "button",
    "cap_knob": "cap_knob",
    "combination_lock_dial": "combination_lock",
    "combination_lock_station": "combination_lock",
    "assembly_plug": "assembly_plug",
    "vial_body": "vial",
    "vial_cap": "vial_cap",
    "micro_sample": "micro_sample",
}

RECOMMENDED_GRASP = {
    "sphere": "SPHERICAL_ENCLOSURE_GRASP",
    "cube": "OPPOSING_FACE_CUBE_GRASP",
    "box": "OPPOSING_FACE_CUBE_GRASP",
    "cylinder_vertical": "LATERAL_CYLINDER_BODY_GRASP",
    "cylinder_horizontal": "LATERAL_CYLINDER_BODY_GRASP",
    "capsule": "CAPSULE_CENTER_SUPPORT_GRASP",
    "stylus": "TRIPOD_PRECISION_GRASP",
    "button": "INDEX_FINGERTIP_PRESS",
    "cap_knob": "CAP_KNOB_ROTATION_224",
    "combination_lock": "TACTILE_COMBINATION_LOCK",
    "assembly_plug": "TACTILE_PRECISION_ASSEMBLY",
    "vial": "VIAL_UNCAP_AND_DELIVER",
    "vial_cap": "VIAL_UNCAP_AND_DELIVER",
    "micro_sample": "CONTROLLED_RELEASE",
}


def _body_pose(model, data, mujoco, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.xmat[body_id])
    return data.xpos[body_id].copy(), quat.copy()


def classify_object(model, data, mujoco, object_name: str) -> ObjectAffordance:
    object_type = OBJECT_TYPE_BY_NAME.get(object_name, "unknown")
    body_name = "button_plunger" if object_name == "button" else object_name
    center, quat = _body_pose(model, data, mujoco, body_name)

    if object_type == "sphere":
        radius = 0.052
        size = [radius]
        long_axis = None
        centerline = None
        face_normals: list[list[float]] = []
        regions = {
            "thumb": "lateral sphere side",
            "index_middle": "opposite/front side",
            "ring_little": "lower support cage",
        }
    elif object_type == "cube":
        half = 0.045
        radius = None
        size = [half, half, half]
        long_axis = None
        centerline = None
        face_normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        regions = {
            "thumb": "center of selected face",
            "index_middle": "center of opposing face",
            "ring": "lower edge support",
        }
    elif object_type.startswith("cylinder"):
        radius = 0.043
        half_length = 0.060
        size = [radius, half_length]
        long_axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        long_axis = long_axis_vec.round(5).tolist()
        centerline = [
            (center - long_axis_vec * half_length).round(5).tolist(),
            (center + long_axis_vec * half_length).round(5).tolist(),
        ]
        face_normals = []
        regions = {
            "thumb": "one side of cylinder body",
            "index_middle": "opposite side around midpoint",
            "ring_little": "lower body support",
        }
    elif object_type == "stylus":
        radius = 0.008
        half_length = 0.055
        size = [radius, half_length]
        long_axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        long_axis = long_axis_vec.round(5).tolist()
        centerline = [
            (center - long_axis_vec * half_length).round(5).tolist(),
            (center + long_axis_vec * half_length).round(5).tolist(),
        ]
        face_normals = []
        regions = {
            "thumb": "handle side",
            "index": "upper/front control side",
            "middle": "lower support side",
        }
    elif object_type == "button":
        radius = 0.032
        size = [radius, 0.012]
        long_axis = [0.0, 0.0, 1.0]
        centerline = None
        face_normals = [[0, 0, 1]]
        regions = {"index": "button center"}
    elif object_type == "cap_knob":
        radius = 0.038
        half_height = 0.026
        size = [radius, half_height]
        long_axis = [0.0, 0.0, 1.0]
        centerline = [
            (center - np.array([0.0, 0.0, half_height])).round(5).tolist(),
            (center + np.array([0.0, 0.0, half_height])).round(5).tolist(),
        ]
        face_normals = []
        regions = {
            "thumb": "cap side counterhold",
            "index": "tangential marker side push",
            "middle_ring": "opposing support arc",
            "little": "optional lower support",
        }
    elif object_type == "combination_lock":
        radius = 0.034
        half_height = 0.015
        size = [radius, half_height]
        long_axis = [0.0, 1.0, 0.0]
        centerline = [
            (center - np.array([0.0, half_height, 0.0])).round(5).tolist(),
            (center + np.array([0.0, half_height, 0.0])).round(5).tolist(),
        ]
        face_normals = []
        regions = {
            "index": "dial rim detent ridge",
            "thumb": "dial rim counterhold",
            "middle": "opposing dial support",
            "ring": "latch pinch support",
            "little": "micro-door stabilizer",
        }
    elif object_type == "assembly_plug":
        half_length = 0.055
        half_width = 0.015
        half_height = 0.012
        radius = None
        size = [half_length, half_width, half_height]
        long_axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        long_axis = long_axis_vec.round(5).tolist()
        centerline = [
            (center - long_axis_vec * half_length).round(5).tolist(),
            (center + long_axis_vec * half_length).round(5).tolist(),
        ]
        face_normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        regions = {
            "thumb": "plug side face for precision opposition",
            "index": "upper long edge tactile pose probe",
            "middle": "opposing lower side support",
            "ring": "anti-roll stabilizer during socket alignment",
        }
    elif object_type == "vial":
        radius = 0.030
        half_height = 0.058
        size = [radius, half_height]
        long_axis = [0.0, 0.0, 1.0]
        centerline = [
            (center - np.array([0.0, 0.0, half_height])).round(5).tolist(),
            (center + np.array([0.0, 0.0, half_height])).round(5).tolist(),
        ]
        face_normals = []
        regions = {
            "thumb": "vial body side no-crush counterhold",
            "ring_little": "lower body stabilizers",
            "index_middle": "cap twist and mouth clearance support",
        }
    elif object_type == "vial_cap":
        radius = 0.033
        half_height = 0.018
        size = [radius, half_height]
        long_axis = [0.0, 0.0, 1.0]
        centerline = [
            (center - np.array([0.0, 0.0, half_height])).round(5).tolist(),
            (center + np.array([0.0, 0.0, half_height])).round(5).tolist(),
        ]
        face_normals = []
        regions = {
            "thumb": "cap side counterhold",
            "index": "cap stripe tangential twist region",
            "middle": "opposite cap side support",
        }
    elif object_type == "micro_sample":
        radius = 0.012
        size = [radius]
        long_axis = None
        centerline = None
        face_normals = []
        regions = {"delivery": "sample should land inside delivery tray after vial tilt"}
    else:
        radius = None
        size = []
        long_axis = None
        centerline = None
        face_normals = []
        regions = {}

    return ObjectAffordance(
        object_name=object_name,
        object_type=object_type,
        object_center=center.round(5).tolist(),
        object_orientation=quat.round(5).tolist(),
        object_size=size,
        face_normals=face_normals,
        radius=radius,
        long_axis=long_axis,
        centerline=centerline,
        grasp_affordance_regions=regions,
        recommended_grasp_type=RECOMMENDED_GRASP.get(object_type, "SPHERICAL_ENCLOSURE_GRASP"),
    )


def classify_scene_objects(model, data, mujoco, object_names: list[str]) -> list[dict[str, Any]]:
    return [classify_object(model, data, mujoco, name).to_dict() for name in object_names]
