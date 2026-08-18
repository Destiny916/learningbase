from __future__ import annotations

import copy
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from w1_simulation.simulation.config import ACTIVE_JOINTS, HAND_MIMIC_JOINTS, SOURCE_URDF

_GLTF_COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_GLTF_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass(frozen=True)
class UrdfReport:
    links: int
    joints: int
    movable_joints: int
    active_joints: int
    mimic_joints: int
    locked_joints: int
    mesh_references: int
    missing_mesh_references: tuple[str, ...]

    @property
    def controlled_joints(self) -> int:
        return self.active_joints

    def as_dict(self) -> dict[str, int | list[str]]:
        return {
            "links": self.links,
            "joints": self.joints,
            "movable_joints": self.movable_joints,
            "active_joints": self.active_joints,
            "controlled_joints": self.controlled_joints,
            "mimic_joints": self.mimic_joints,
            "locked_joints": self.locked_joints,
            "mesh_references": self.mesh_references,
            "missing_mesh_references": list(self.missing_mesh_references),
        }


@dataclass(frozen=True)
class _GlbVisualPrimitive:
    path: Path
    rgba: tuple[float, float, float, float]


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rpy_to_quaternion(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _axis_angle_to_quaternion(
    axis: tuple[float, float, float], position: float
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in axis))
    if norm == 0.0:
        raise ValueError("joint axis must be non-zero")
    scale = math.sin(position / 2.0) / norm
    return (math.cos(position / 2.0), *(component * scale for component in axis))


def _quaternion_to_rpy(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    w, x, y, z = (component / norm for component in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(0.0 if abs(value) < 1e-15 else value for value in (roll, pitch, yaw))


def _bake_joint_position(joint: ET.Element, position: float) -> None:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
    axis_element = joint.find("axis")
    axis = tuple(
        float(value)
        for value in (axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0").split()
    )
    baked = _quaternion_multiply(_rpy_to_quaternion(rpy), _axis_angle_to_quaternion(axis, position))
    origin.set("rpy", " ".join(f"{value:.17g}" for value in _quaternion_to_rpy(baked)))


def _read_glb(source: Path) -> tuple[dict[str, Any], bytes]:
    payload = source.read_bytes()
    if len(payload) < 20:
        raise ValueError(f"GLB file is truncated: {source}")
    magic, version, total_length = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise ValueError(f"Invalid glTF 2.0 binary header: {source}")
    document: dict[str, Any] | None = None
    binary = b""
    offset = 12
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise ValueError(f"GLB chunk is truncated: {source}")
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += chunk_length
    if document is None or not binary:
        raise ValueError(f"GLB must contain JSON and binary chunks: {source}")
    unsupported = set(document.get("extensionsRequired", ()))
    if unsupported:
        raise ValueError(f"GLB requires unsupported extensions {sorted(unsupported)}: {source}")
    return document, binary


def _glb_accessor(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: int,
) -> npt.NDArray[np.generic]:
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("Sparse glTF accessors are not supported")
    component_type = int(accessor["componentType"])
    accessor_type = accessor["type"]
    if component_type not in _GLTF_COMPONENT_DTYPES or accessor_type not in _GLTF_COMPONENT_COUNTS:
        raise ValueError(f"Unsupported glTF accessor type: component={component_type}, type={accessor_type}")
    dtype = _GLTF_COMPONENT_DTYPES[component_type]
    components = _GLTF_COMPONENT_COUNTS[accessor_type]
    count = int(accessor["count"])
    view = document["bufferViews"][accessor["bufferView"]]
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("GLB accessor references an external buffer")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    item_bytes = dtype.itemsize * components
    stride = int(view.get("byteStride", item_bytes))
    if stride < item_bytes:
        raise ValueError("GLB accessor byteStride is smaller than its element size")
    end = start + (count - 1) * stride + item_bytes if count else start
    if start < 0 or end > len(binary):
        raise ValueError("GLB accessor points outside the binary chunk")
    values = np.ndarray(
        shape=(count, components),
        dtype=dtype,
        buffer=binary,
        offset=start,
        strides=(stride, dtype.itemsize),
    ).copy()
    if accessor.get("normalized", False):
        if dtype.kind == "u":
            values = values.astype(np.float64) / np.iinfo(dtype).max
        elif dtype.kind == "i":
            values = np.maximum(values.astype(np.float64) / np.iinfo(dtype).max, -1.0)
    return values


def _glb_node_transform(node: Mapping[str, Any]) -> npt.NDArray[np.float64]:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    x, y, z, w = (float(value) for value in node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation @ np.diag(scale)
    transform[:3, 3] = translation
    return transform


def _glb_mesh_instances(
    document: Mapping[str, Any],
) -> list[tuple[int, npt.NDArray[np.float64], str]]:
    nodes = document.get("nodes", ())
    scenes = document.get("scenes", ())
    if scenes:
        scene_index = int(document.get("scene", 0))
        roots = scenes[scene_index].get("nodes", ())
    else:
        child_nodes = {int(child) for node in nodes for child in node.get("children", ())}
        roots = [index for index in range(len(nodes)) if index not in child_nodes]
    instances: list[tuple[int, npt.NDArray[np.float64], str]] = []

    def visit(node_index: int, parent_transform: npt.NDArray[np.float64], ancestors: set[int]) -> None:
        if node_index in ancestors:
            raise ValueError("glTF node graph contains a cycle")
        node = nodes[node_index]
        transform = parent_transform @ _glb_node_transform(node)
        if "mesh" in node:
            instances.append((int(node["mesh"]), transform, str(node.get("name", node_index))))
        next_ancestors = ancestors | {node_index}
        for child in node.get("children", ()):
            visit(int(child), transform, next_ancestors)

    for root in roots:
        visit(int(root), np.eye(4, dtype=np.float64), set())
    return instances


def _glb_primitive_rgba(
    document: Mapping[str, Any],
    primitive: Mapping[str, Any],
    source: Path,
) -> tuple[float, float, float, float]:
    if "material" not in primitive:
        return (1.0, 1.0, 1.0, 1.0)
    material_index = int(primitive["material"])
    materials = document.get("materials", ())
    if material_index < 0 or material_index >= len(materials):
        raise ValueError(f"glTF primitive material index is out of range: {source}")
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    if "baseColorTexture" in pbr:
        raise ValueError(f"Textured glTF materials are not supported: {source}")
    values = pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))
    if len(values) != 4:
        raise ValueError(f"glTF baseColorFactor must contain four values: {source}")
    rgba = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in rgba):
        raise ValueError(f"glTF baseColorFactor must be finite and within [0, 1]: {source}")
    return rgba


def _write_msh(
    destination: Path,
    vertices: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    texcoords: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<4i", len(vertices), len(normals), len(texcoords), len(faces))
    with destination.open("wb") as stream:
        stream.write(header)
        stream.write(vertices.tobytes(order="C"))
        stream.write(normals.tobytes(order="C"))
        stream.write(texcoords.tobytes(order="C"))
        stream.write(faces.tobytes(order="C"))


def _write_glb_visual_msh(source: Path, destination: Path) -> tuple[_GlbVisualPrimitive, ...]:
    document, binary = _read_glb(source)
    converted: list[_GlbVisualPrimitive] = []
    for mesh_index, transform, _node_name in _glb_mesh_instances(document):
        mesh = document["meshes"][mesh_index]
        linear = transform[:3, :3]
        normal_transform = np.linalg.inv(linear).T
        for primitive in mesh.get("primitives", ()):
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError(f"Only triangular glTF primitives are supported: {source}")
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                raise ValueError(f"glTF primitive has no POSITION attribute: {source}")
            positions = _glb_accessor(document, binary, int(attributes["POSITION"])).astype(np.float64)
            positions = positions[:, :3] @ linear.T + transform[:3, 3]
            normals = None
            if "NORMAL" in attributes:
                normals = _glb_accessor(document, binary, int(attributes["NORMAL"])).astype(np.float64)
                normals = normals[:, :3] @ normal_transform.T
                lengths = np.linalg.norm(normals, axis=1)
                if np.any(lengths == 0.0):
                    raise ValueError(f"glTF primitive contains a zero normal: {source}")
                normals /= lengths[:, None]
            texcoords = None
            if "TEXCOORD_0" in attributes:
                texcoords = _glb_accessor(document, binary, int(attributes["TEXCOORD_0"])).astype(np.float64)
                texcoords = texcoords[:, :2]
            if "indices" in primitive:
                indices = _glb_accessor(document, binary, int(primitive["indices"])).reshape(-1)
            else:
                indices = np.arange(len(positions), dtype=np.int64)
            if len(indices) % 3:
                raise ValueError(f"glTF triangle index count is not divisible by three: {source}")
            if normals is not None and len(normals) != len(positions):
                raise ValueError(f"glTF POSITION and NORMAL counts differ: {source}")
            if texcoords is not None and len(texcoords) != len(positions):
                raise ValueError(f"glTF POSITION and TEXCOORD_0 counts differ: {source}")
            if len(indices) and (int(np.min(indices)) < 0 or int(np.max(indices)) >= len(positions)):
                raise ValueError(f"glTF primitive index is out of range: {source}")
            if normals is None:
                raise ValueError(f"glTF primitive has no NORMAL attribute: {source}")
            primitive_path = destination.with_name(
                f"{destination.stem}_primitive_{len(converted):03d}{destination.suffix}"
            )
            _write_msh(
                primitive_path,
                np.asarray(positions, dtype="<f4"),
                np.asarray(normals, dtype="<f4"),
                (
                    np.asarray(texcoords, dtype="<f4")
                    if texcoords is not None
                    else np.empty((0, 2), dtype="<f4")
                ),
                np.asarray(indices.reshape(-1, 3), dtype="<i4"),
            )
            converted.append(
                _GlbVisualPrimitive(
                    path=primitive_path,
                    rgba=_glb_primitive_rgba(document, primitive, source),
                )
            )
    if not converted:
        raise ValueError(f"GLB contains no scene mesh instances: {source}")
    return tuple(converted)


def inspect_urdf(
    source: Path = SOURCE_URDF,
    locked_joint_values: Mapping[str, float] | None = None,
) -> UrdfReport:
    locked = {} if locked_joint_values is None else dict(locked_joint_values)
    root = ET.parse(source).getroot()
    joints = root.findall("joint")
    joints_by_name = {joint.get("name", ""): joint for joint in joints}
    missing_active = sorted(set(ACTIVE_JOINTS) - joints_by_name.keys())
    missing_mimic = sorted(set(HAND_MIMIC_JOINTS) - joints_by_name.keys())
    missing_locked = sorted(set(locked) - joints_by_name.keys())
    overlap = sorted((set(ACTIVE_JOINTS) | set(HAND_MIMIC_JOINTS)) & set(locked))
    if missing_active or missing_mimic or missing_locked or overlap:
        raise ValueError(
            "URDF joint contract mismatch: "
            f"active={missing_active}, mimic={missing_mimic}, locked={missing_locked}, overlap={overlap}"
        )
    for name, value in locked.items():
        joint = joints_by_name[name]
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            raise ValueError(f"locked joint {name!r} is not movable")
        limit = joint.find("limit")
        if limit is not None and joint.get("type") != "continuous":
            lower = float(limit.get("lower", "-inf"))
            upper = float(limit.get("upper", "inf"))
            if not lower <= float(value) <= upper:
                raise ValueError(f"locked joint {name!r} value {value} is outside [{lower}, {upper}]")
    mesh_references = [mesh.get("filename", "") for mesh in root.iter("mesh")]
    missing_meshes = tuple(sorted(ref for ref in mesh_references if not (source.parent / ref).is_file()))
    movable = [joint for joint in joints if joint.get("type") in {"revolute", "continuous", "prismatic"}]
    return UrdfReport(
        links=len(root.findall("link")),
        joints=len(joints),
        movable_joints=len(movable),
        active_joints=len(ACTIVE_JOINTS),
        mimic_joints=len(HAND_MIMIC_JOINTS),
        locked_joints=len(locked),
        mesh_references=len(mesh_references),
        missing_mesh_references=missing_meshes,
    )


def create_mujoco_compatible_urdf(
    source: Path,
    destination: Path,
    locked_joint_values: Mapping[str, float] | None = None,
) -> UrdfReport:
    locked = {} if locked_joint_values is None else dict(locked_joint_values)
    report = inspect_urdf(source, locked)
    if report.missing_mesh_references:
        raise FileNotFoundError(report.missing_mesh_references)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", "dexforce_w1_mujoco")
    for link in root.findall("link"):
        collision_mesh = link.find("collision/geometry/mesh")
        if collision_mesh is not None:
            collision_path = (source.parent / collision_mesh.get("filename", "")).resolve()
            collision_mesh.set("filename", str(collision_path))
        for visual_index, visual in enumerate(list(link.findall("visual"))):
            visual_mesh = visual.find("geometry/mesh")
            if visual_mesh is None:
                continue
            visual_reference = visual_mesh.get("filename", "")
            visual_path = (source.parent / visual_reference).resolve()
            if visual_path.suffix.lower() == ".glb":
                relative_path = Path(visual_reference)
                converted_path = destination.parent / "urdf_visual_meshes" / relative_path.with_suffix(".msh")
                converted = _write_glb_visual_msh(visual_path, converted_path)
                insertion_index = list(link).index(visual)
                link.remove(visual)
                for primitive_index, primitive in enumerate(converted):
                    converted_visual = copy.deepcopy(visual)
                    converted_mesh = converted_visual.find("geometry/mesh")
                    converted_mesh.set("filename", str(primitive.path.resolve()))
                    for material in converted_visual.findall("material"):
                        converted_visual.remove(material)
                    material = ET.SubElement(
                        converted_visual,
                        "material",
                        {
                            "name": (
                                f"{link.get('name', 'link')}_visual_{visual_index}_"
                                f"primitive_{primitive_index}"
                            )
                        },
                    )
                    ET.SubElement(
                        material,
                        "color",
                        {"rgba": " ".join(f"{value:.9g}" for value in primitive.rgba)},
                    )
                    link.insert(insertion_index + primitive_index, converted_visual)
            elif visual_path.suffix.lower() in {".obj", ".stl"}:
                visual_mesh.set("filename", str(visual_path))
            elif collision_mesh is not None:
                visual_mesh.set("filename", collision_mesh.get("filename", ""))
            else:
                raise ValueError(f"Unsupported URDF visual mesh format: {visual_path}")
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name in locked:
            _bake_joint_position(joint, locked[name])
            joint.set("type", "fixed")
            for element_name in ("limit", "dynamics"):
                element = joint.find(element_name)
                if element is not None:
                    joint.remove(element)
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)
        limit = joint.find("limit")
        if limit is not None and name in ACTIVE_JOINTS:
            if float(limit.get("effort", "0")) <= 0.0:
                limit.set("effort", "8")
            if float(limit.get("velocity", "0")) <= 0.0:
                limit.set("velocity", "2")
    mujoco_extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mujoco_extension,
        "compiler",
        {"discardvisual": "false", "fusestatic": "false", "strippath": "false"},
    )
    ET.indent(tree, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return report


def write_urdf_report(
    destination: Path,
    source: Path = SOURCE_URDF,
    locked_joint_values: Mapping[str, float] | None = None,
) -> UrdfReport:
    report = inspect_urdf(source, locked_joint_values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return report
