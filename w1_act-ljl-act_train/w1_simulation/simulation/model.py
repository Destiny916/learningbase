from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from w1_simulation.simulation.camera import EyeCameraConfig
from w1_simulation.simulation.config import (
    ACTIVE_JOINTS,
    BODY_JOINTS,
    HAND_MIMIC_JOINTS,
    SELF_COLLISION_EXCLUDES,
    SOURCE_URDF,
    SimulationConfig,
)
from w1_simulation.simulation.urdf import create_mujoco_compatible_urdf, write_urdf_report


def _joint_limits(source: Path) -> dict[str, tuple[float, float, float]]:
    root = ET.parse(source).getroot()
    limits: dict[str, tuple[float, float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[joint.get("name", "")] = (
            float(limit.get("lower", "0")),
            float(limit.get("upper", "0")),
            max(float(limit.get("effort", "0")), 8.0),
        )
    return limits


def _eye_camera_parent(root: ET.Element, config: EyeCameraConfig) -> ET.Element:
    parents = [body for body in root.iter("body") if body.get("name") == config.parent_body]
    if len(parents) != 1:
        raise ValueError(
            f"expected exactly one eye camera parent body {config.parent_body!r}, got {len(parents)}"
        )
    return parents[0]


def _configure_eye_camera(root: ET.Element, config: EyeCameraConfig) -> None:
    if not config.enabled:
        return
    if any(camera.get("name") == config.name for camera in root.iter("camera")):
        raise ValueError(f"runtime model already contains camera {config.name!r}")
    ET.SubElement(
        _eye_camera_parent(root, config),
        "camera",
        {
            "name": config.name,
            "mode": "fixed",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "fovy": str(config.fovy_degrees),
        },
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        worldbody = root.find("worldbody")
        root.insert(list(root).index(worldbody) if worldbody is not None else 0, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(config.width))
    global_visual.set("offheight", str(config.height))
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.22 0.22 0.22")
    headlight.set("diffuse", "0.45 0.45 0.45")
    headlight.set("specular", "0.07 0.07 0.07")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("runtime model must contain a worldbody section")
    indoor_lights = (
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
    )
    for light in indoor_lights:
        ET.SubElement(worldbody, "light", light)
    asset = root.find("asset")
    if asset is None:
        raise ValueError("runtime model must contain an asset section")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "eye_camera_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.16 0.20 0.28",
            "rgb2": "0.015 0.020 0.030",
            "width": "512",
            "height": "3072",
        },
    )
    if config.scene != "grid":
        return
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "eye_camera_grid_texture",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.16 0.18 0.22",
            "rgb2": "0.32 0.35 0.40",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "eye_camera_grid_material",
            "texture": "eye_camera_grid_texture",
            "texrepeat": "12 12",
            "texuniform": "true",
            "reflectance": "0.08",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "eye_camera_grid_floor",
            "type": "plane",
            "size": "4 4 0.05",
            "pos": "0 0 0",
            "material": "eye_camera_grid_material",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )


def build_runtime_model(
    generated_dir: Path,
    source: Path = SOURCE_URDF,
    config: SimulationConfig | None = None,
    locked_joint_values: Mapping[str, float] | None = None,
    eye_camera: EyeCameraConfig | None = None,
) -> Path:
    import mujoco

    simulation = SimulationConfig() if config is None else config
    locked = simulation.locked_joint_values if locked_joint_values is None else dict(locked_joint_values)
    generated_dir.mkdir(parents=True, exist_ok=True)
    adapted_urdf = generated_dir / "robot_mujoco.urdf"
    compiled_mjcf = generated_dir / "robot_compiled.xml"
    final_mjcf = generated_dir / "robot_runtime.xml"
    create_mujoco_compatible_urdf(source, adapted_urdf, locked)
    write_urdf_report(generated_dir / "urdf_report.json", source, locked)
    imported_model = mujoco.MjModel.from_xml_path(str(adapted_urdf))
    mujoco.mj_saveLastXML(str(compiled_mjcf), imported_model)
    tree = ET.parse(compiled_mjcf)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(simulation.timestep))
    option.set("integrator", "implicitfast")
    limits = _joint_limits(source)
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name in BODY_JOINTS:
            joint.set("damping", "2")
            joint.set("armature", "0.02")
        elif name in ACTIVE_JOINTS or name in HAND_MIMIC_JOINTS:
            joint.set("damping", "0.08")
            joint.set("armature", "0.002")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuator.clear()
    for name in ACTIVE_JOINTS:
        lower, upper, effort = limits[name]
        kp = simulation.body_kp if name in BODY_JOINTS else simulation.hand_kp
        force = min(max(effort, 8.0), 80.0)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"act_{name}",
                "joint": name,
                "kp": str(kp),
                "kv": str(2.0 * kp**0.5),
                "ctrllimited": "true",
                "ctrlrange": f"{lower} {upper}",
                "forcelimited": "true",
                "forcerange": f"{-force} {force}",
            },
        )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    equality.clear()
    for dependent, (source_joint, multiplier, offset) in HAND_MIMIC_JOINTS.items():
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"mimic_{dependent}",
                "joint1": dependent,
                "joint2": source_joint,
                "polycoef": f"{offset} {multiplier} 0 0 0",
                "solref": "0.004 1",
            },
        )
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing_excludes = {
        (element.get("body1"), element.get("body2")) for element in contact.findall("exclude")
    }
    for body1, body2 in SELF_COLLISION_EXCLUDES:
        if (body1, body2) not in existing_excludes and (body2, body1) not in existing_excludes:
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
    actual_excludes = {(element.get("body1"), element.get("body2")) for element in contact.findall("exclude")}
    if actual_excludes != set(SELF_COLLISION_EXCLUDES):
        raise ValueError(f"Unexpected runtime contact excludes: {sorted(actual_excludes)}")
    if eye_camera is not None:
        _configure_eye_camera(root, eye_camera)
    ET.indent(tree, space="  ")
    tree.write(final_mjcf, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(final_mjcf))
    missing = [
        name for name in ACTIVE_JOINTS if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
    ]
    if missing:
        raise ValueError(f"compiled model missing active joints: {missing}")
    if model.nu != len(ACTIVE_JOINTS):
        raise ValueError(f"expected {len(ACTIVE_JOINTS)} actuators, got {model.nu}")
    if model.nexclude != len(SELF_COLLISION_EXCLUDES):
        raise ValueError(f"expected {len(SELF_COLLISION_EXCLUDES)} contact excludes, got {model.nexclude}")
    if eye_camera is not None and eye_camera.enabled:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, eye_camera.name)
        parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eye_camera.parent_body)
        if camera_id < 0 or parent_id < 0 or int(model.cam_bodyid[camera_id]) != parent_id:
            raise ValueError("compiled eye camera is missing or attached to the wrong body")
    return final_mjcf
