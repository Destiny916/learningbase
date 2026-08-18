from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

W1_ACTION_JOINTS = (
    "WAIST",
    "LEFT_J1",
    "LEFT_J2",
    "LEFT_J3",
    "LEFT_J4",
    "LEFT_J5",
    "LEFT_J6",
    "LEFT_J7",
    "NECK1",
    "NECK2",
    "RIGHT_J1",
    "RIGHT_J2",
    "RIGHT_J3",
    "RIGHT_J4",
    "RIGHT_J5",
    "RIGHT_J6",
    "RIGHT_J7",
    "LEFT_GRIPPER",
    "RIGHT_GRIPPER",
)
W1_ACTION_INDEX = {name: index for index, name in enumerate(W1_ACTION_JOINTS)}


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Tensor
    axis: Tensor


def _parse_vector(value: str, length: int = 3) -> Tensor:
    result = torch.tensor([float(item) for item in value.split()], dtype=torch.float64)
    if result.shape != (length,):
        raise ValueError(f"Expected {length} values, got {value!r}")
    return result


def _rpy_matrix(rpy: Tensor) -> Tensor:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = torch.tensor(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=torch.float64)
    rotation_y = torch.tensor(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=torch.float64)
    rotation_z = torch.tensor(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=torch.float64)
    return rotation_z @ rotation_y @ rotation_x


def _homogeneous(translation: Tensor, rotation: Tensor) -> Tensor:
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _parse_urdf(path: Path) -> dict[str, _Joint]:
    root = ET.parse(path).getroot()
    by_child: dict[str, _Joint] = {}
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {element.get('name')} lacks parent or child")
        origin = element.find("origin")
        axis = element.find("axis")
        xyz = _parse_vector(origin.get("xyz", "0 0 0")) if origin is not None else torch.zeros(3)
        rpy = _parse_vector(origin.get("rpy", "0 0 0")) if origin is not None else torch.zeros(3)
        joint_axis = _parse_vector(axis.get("xyz", "1 0 0")) if axis is not None else torch.ones(3)
        joint = _Joint(
            name=element.get("name", ""),
            joint_type=element.get("type", "fixed"),
            parent=parent.get("link", ""),
            child=child.get("link", ""),
            origin=_homogeneous(xyz, _rpy_matrix(rpy)),
            axis=joint_axis,
        )
        if joint.child in by_child:
            raise ValueError(f"Link {joint.child} has multiple parent joints")
        by_child[joint.child] = joint
    return by_child


def _path(by_child: dict[str, _Joint], reference: str, target: str) -> tuple[_Joint, ...]:
    reverse_path: list[_Joint] = []
    current = target
    while current != reference:
        try:
            joint = by_child[current]
        except KeyError as exc:
            raise ValueError(f"{target!r} is not a descendant of {reference!r}") from exc
        reverse_path.append(joint)
        current = joint.parent
    return tuple(reversed(reverse_path))


def _axis_angle_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    axis = axis.to(device=angle.device, dtype=angle.dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = torch.zeros((), device=angle.device, dtype=angle.dtype)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
    return (
        identity
        + torch.sin(angle)[..., None, None] * skew
        + (1.0 - torch.cos(angle))[..., None, None] * (skew @ skew)
    )


class _DifferentiableChain(nn.Module):
    def __init__(self, joints: tuple[_Joint, ...]) -> None:
        super().__init__()
        self.joint_names = tuple(joint.name for joint in joints)
        self.joint_types = tuple(joint.joint_type for joint in joints)
        self.action_indices = tuple(
            W1_ACTION_INDEX.get(joint.name, -1) if joint.joint_type != "fixed" else -1 for joint in joints
        )
        unsupported = [
            joint.name
            for joint, action_index in zip(joints, self.action_indices, strict=True)
            if joint.joint_type not in {"fixed", "revolute", "continuous"}
            or (joint.joint_type != "fixed" and action_index < 0)
        ]
        if unsupported:
            raise ValueError(f"Unsupported or unobserved joints in FK chain: {unsupported}")
        self.register_buffer(
            "origins",
            torch.stack([joint.origin for joint in joints]),
            persistent=False,
        )
        self.register_buffer(
            "axes",
            torch.stack([joint.axis for joint in joints]),
            persistent=False,
        )

    def forward(self, actions: Tensor) -> Tensor:
        batch_shape = actions.shape[:-1]
        result = torch.eye(4, device=actions.device, dtype=actions.dtype)
        result = result.expand(*batch_shape, 4, 4)
        for index, (joint_type, action_index) in enumerate(
            zip(self.joint_types, self.action_indices, strict=True)
        ):
            origin = self.origins[index].to(device=actions.device, dtype=actions.dtype)
            result = result @ origin
            if joint_type in {"revolute", "continuous"}:
                rotation = _axis_angle_matrix(self.axes[index], actions[..., action_index])
                motion = torch.nn.functional.pad(rotation, (0, 1, 0, 1))
                motion = motion.clone()
                motion[..., 3, 3] = 1.0
                result = result @ motion
        return result


class W1BimanualKinematics(nn.Module):
    def __init__(
        self,
        urdf_path: str | Path,
        reference_link: str = "buttock",
        left_end_effector_link: str = "left_hand_base_link",
        right_end_effector_link: str = "right_hand_base_link",
    ) -> None:
        super().__init__()
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        by_child = _parse_urdf(path)
        left_joints = _path(by_child, reference_link, left_end_effector_link)
        right_joints = _path(by_child, reference_link, right_end_effector_link)
        for side, joints in (("left", left_joints), ("right", right_joints)):
            names = tuple(joint.name for joint in joints)
            required = "LEFT_J1" if side == "left" else "RIGHT_J1"
            if "WAIST" not in names or required not in names:
                raise ValueError(f"Unexpected {side} kinematic chain: {names}")
            if any(name in names for name in ("ANKLE", "KNEE", "BUTTOCK")):
                raise ValueError(f"Lower-body joints leaked into the {side} chain: {names}")
        self.urdf_path = path
        self.urdf_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.reference_link = reference_link
        self.end_effector_links = (left_end_effector_link, right_end_effector_link)
        self.left_chain = _DifferentiableChain(left_joints)
        self.right_chain = _DifferentiableChain(right_joints)

    def forward(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        if actions.shape[-1] != len(W1_ACTION_JOINTS):
            raise ValueError(f"Expected {len(W1_ACTION_JOINTS)}D W1 actions, got {tuple(actions.shape)}")
        if not torch.is_floating_point(actions):
            raise TypeError("W1 actions must be floating point")
        compute_actions = actions.float() if actions.dtype in {torch.float16, torch.bfloat16} else actions
        transforms = torch.stack(
            (self.left_chain(compute_actions), self.right_chain(compute_actions)),
            dim=-3,
        )
        return transforms[..., :3, 3], transforms[..., :3, :3]


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: float = 1e-6) -> Tensor:
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotations, got {tuple(rotation_6d.shape)}")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    first = torch.nn.functional.normalize(first, dim=-1, eps=eps)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1, eps=eps)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)
