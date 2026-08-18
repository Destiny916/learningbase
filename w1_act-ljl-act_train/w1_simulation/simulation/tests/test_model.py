from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from w1_simulation.simulation.camera import EyeCameraConfig
from w1_simulation.simulation.config import SELF_COLLISION_EXCLUDES, SimulationConfig
from w1_simulation.simulation.model import build_runtime_model


def test_runtime_model_contains_only_validated_internal_collision_excludes(tmp_path) -> None:
    runtime_model = build_runtime_model(tmp_path, config=SimulationConfig())
    root = ET.parse(runtime_model).getroot()
    actual = {(element.get("body1"), element.get("body2")) for element in root.findall("./contact/exclude")}

    assert actual == set(SELF_COLLISION_EXCLUDES)
    assert mujoco.MjModel.from_xml_path(str(runtime_model)).nexclude == len(SELF_COLLISION_EXCLUDES)


def test_runtime_model_attaches_visual_only_eye_camera_to_eyes(tmp_path) -> None:
    camera = EyeCameraConfig(width=640, height=360, scene="grid")
    runtime_model = build_runtime_model(
        tmp_path,
        config=SimulationConfig(),
        eye_camera=camera,
    )
    root = ET.parse(runtime_model).getroot()
    model = mujoco.MjModel.from_xml_path(str(runtime_model))
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
    eyes_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, camera.parent_body)
    grid = root.find("./worldbody/geom[@name='eye_camera_grid_floor']")
    headlight = root.find("./visual/headlight")
    indoor_lights = root.findall("./worldbody/light")
    skybox = root.find("./asset/texture[@name='eye_camera_skybox']")
    grid_texture = root.find("./asset/texture[@name='eye_camera_grid_texture']")

    assert camera_id >= 0
    assert int(model.cam_bodyid[camera_id]) == eyes_id
    np.testing.assert_allclose(model.cam_pos[camera_id], [0.0, 0.0, 0.0])
    assert model.vis.global_.offwidth == 640
    assert model.vis.global_.offheight == 360
    assert grid is not None
    assert grid.get("contype") == "0"
    assert grid.get("conaffinity") == "0"
    assert skybox is not None
    assert skybox.get("rgb1") == "0.16 0.20 0.28"
    assert skybox.get("rgb2") == "0.015 0.020 0.030"
    assert grid_texture is not None
    assert grid_texture.get("rgb1") == "0.16 0.18 0.22"
    assert grid_texture.get("rgb2") == "0.32 0.35 0.40"
    assert headlight is not None
    assert headlight.get("ambient") == "0.22 0.22 0.22"
    assert headlight.get("diffuse") == "0.45 0.45 0.45"
    assert headlight.get("specular") == "0.07 0.07 0.07"
    assert [dict(light.attrib) for light in indoor_lights] == [
        {
            "name": "eye_camera_key",
            "pos": "-2.4 -1.6 3.8",
            "dir": "0.5963 0.3975 -0.6957",
            "directional": "false",
            "castshadow": "false",
            "ambient": "0.05 0.042 0.032",
            "diffuse": "1.0 0.85 0.65",
            "specular": "0.20 0.16 0.12",
            "attenuation": "1 0.03 0.02",
            "cutoff": "55",
            "exponent": "6",
        },
        {
            "name": "eye_camera_fill",
            "pos": "2.0 -1.8 2.8",
            "dir": "-0.6283 0.5654 -0.5341",
            "directional": "false",
            "castshadow": "false",
            "ambient": "0.03 0.04 0.055",
            "diffuse": "0.40 0.52 0.72",
            "specular": "0.07 0.10 0.14",
            "attenuation": "1 0.04 0.03",
            "cutoff": "75",
            "exponent": "2",
        },
        {
            "name": "eye_camera_rim",
            "pos": "0 2.4 3.2",
            "dir": "0 -0.7682 -0.6402",
            "directional": "false",
            "castshadow": "false",
            "ambient": "0.01 0.013 0.02",
            "diffuse": "0.55 0.70 0.95",
            "specular": "0.14 0.18 0.24",
            "attenuation": "1 0.02 0.025",
            "cutoff": "45",
            "exponent": "8",
        },
    ]
    assert model.nlight == 3
    visual_rgba = model.geom_rgba[model.geom_group == 1]
    assert int((model.geom_group == 0).sum()) == 45
    assert len(visual_rgba) == 68
    assert len(np.unique(np.round(visual_rgba, decimals=6), axis=0)) >= 8
    assert np.any(np.all(np.isclose(visual_rgba, [0.6862745, 0.4666667, 0.0, 1.0]), axis=1))
    assert model.nexclude == len(SELF_COLLISION_EXCLUDES)
