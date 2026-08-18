from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from w1_simulation.artifacts import sha256_bytes, sha256_file
from w1_simulation.w1_profile import DEFAULT_CAMERA_SOURCES


@dataclass(frozen=True)
class CameraRecord:
    camera_type: str
    frame_id: int
    timestamp: float
    path: Path


@dataclass(frozen=True)
class OriginFrame:
    frame_id: int
    timestamp: float
    records: dict[str, CameraRecord]
    camera_skew_ms: float

    def load_images(self) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
        images: dict[str, np.ndarray] = {}
        source_hashes: dict[str, str] = {}
        input_hashes: dict[str, str] = {}
        for key, record in self.records.items():
            bgr = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not decode image: {record.path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Unexpected image contract for {record.path}: {rgb.shape} {rgb.dtype}")
            images[key] = rgb
            source_hashes[key] = sha256_file(record.path)
            input_hashes[key] = sha256_bytes(rgb.tobytes())
        return images, source_hashes, input_hashes


class OriginReplay:
    def __init__(
        self,
        root: Path,
        max_camera_skew_ms: float = 50.0,
        camera_sources: dict[str, str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.camera_sources = dict(camera_sources or DEFAULT_CAMERA_SOURCES)
        if not self.camera_sources:
            raise ValueError("At least one model image input source is required")
        self.metadata_path = self.root / "metadata.jsonl"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        records_by_key: dict[str, list[CameraRecord]] = {key: [] for key in self.camera_sources}
        with self.metadata_path.open(encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                camera = str(payload.get("camera_type", ""))
                frame_id = int(payload["frame_id"])
                relative_path = Path(payload["image_path"])
                path = (self.root / relative_path).resolve()
                if not path.is_file() or self.root not in path.parents:
                    raise FileNotFoundError(path)
                for key, source in self.camera_sources.items():
                    source_path = Path(source)
                    source_matches = (
                        source == camera
                        or source_path.as_posix().rstrip("/") == relative_path.parent.as_posix()
                    )
                    if source_path.is_absolute():
                        source_matches = source_path.resolve() == path.parent
                    if source_matches:
                        records_by_key[key].append(
                            CameraRecord(
                                camera_type=camera,
                                frame_id=frame_id,
                                timestamp=float(payload["timestamp"]),
                                path=path,
                            )
                        )
        ordered = {
            key: sorted(records, key=lambda record: record.timestamp)
            for key, records in records_by_key.items()
        }
        if any(not records for records in ordered.values()):
            missing = [key for key, records in ordered.items() if not records]
            raise ValueError(f"Configured image sources have no metadata records: {missing}")
        for key, records in ordered.items():
            frame_ids = [record.frame_id for record in records]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError(f"Duplicate frame ids in configured image source {key!r}")
        stream_timestamps = {
            key: [record.timestamp for record in records] for key, records in ordered.items()
        }

        def nearest(key: str, timestamp: float) -> CameraRecord:
            candidates = ordered[key]
            values = stream_timestamps[key]
            position = bisect.bisect_left(values, timestamp)
            indices = [index for index in (position - 1, position) if 0 <= index < len(candidates)]
            index = min(indices, key=lambda item: abs(values[item] - timestamp))
            return candidates[index]

        self.frames: list[OriginFrame] = []
        dropped = 0
        anchor_key = next(iter(self.camera_sources))
        for anchor in ordered[anchor_key]:
            records = {anchor_key: anchor}
            for key in self.camera_sources:
                if key != anchor_key:
                    records[key] = nearest(key, anchor.timestamp)
            triplet_timestamps = [record.timestamp for record in records.values()]
            skew_ms = (max(triplet_timestamps) - min(triplet_timestamps)) * 1000.0
            if skew_ms > max_camera_skew_ms:
                dropped += 1
                continue
            self.frames.append(
                OriginFrame(
                    frame_id=anchor.frame_id,
                    timestamp=anchor.timestamp,
                    records=records,
                    camera_skew_ms=skew_ms,
                )
            )
        if not self.frames:
            raise ValueError(f"All camera triplets exceeded {max_camera_skew_ms} ms skew")
        self.dropped_for_skew = dropped
        self.metadata_sha256 = sha256_file(self.metadata_path)

    def select(self, start_frame: int, max_frames: int) -> list[OriginFrame]:
        if start_frame < 0 or start_frame >= len(self.frames):
            raise ValueError(f"start_frame {start_frame} is outside [0, {len(self.frames) - 1}]")
        stop = len(self.frames) if max_frames <= 0 else min(len(self.frames), start_frame + max_frames)
        return self.frames[start_frame:stop]


@dataclass(frozen=True)
class StateAlignedSelection:
    frame: OriginFrame
    index: int
    match_distance: float
    frozen: bool


class StateAlignedFrameSelector:
    def __init__(
        self,
        frames: list[OriginFrame],
        reference_states: np.ndarray,
        state_span: np.ndarray,
        *,
        search_ahead_frames: int = 15,
        max_advance_frames: int = 2,
        match_threshold: float = 0.18,
        similarity_slack: float = 0.005,
        waist_weight: float = 4.0,
        gripper_weight: float = 2.0,
    ) -> None:
        references = np.asarray(reference_states, dtype=np.float64)
        span = np.asarray(state_span, dtype=np.float64)
        if not frames or references.ndim != 2 or len(references) != len(frames):
            raise ValueError("State-aligned replay requires one reference state per synchronized frame")
        if span.shape != (references.shape[1],) or not np.isfinite(span).all() or np.any(span <= 0.0):
            raise ValueError("State-aligned replay requires a finite positive state span")
        if not np.isfinite(references).all():
            raise ValueError("State-aligned replay reference states must be finite")
        if search_ahead_frames < 1 or max_advance_frames < 1:
            raise ValueError("State-aligned replay search and advance windows must be positive")
        if match_threshold <= 0.0 or similarity_slack < 0.0:
            raise ValueError("State-aligned replay thresholds are invalid")
        if waist_weight <= 0.0 or gripper_weight <= 0.0:
            raise ValueError("State-aligned replay weights must be positive")

        weights = np.ones(references.shape[1], dtype=np.float64)
        weights[0] = waist_weight
        if references.shape[1] >= 2:
            weights[-2:] = gripper_weight
        self.frames = frames
        self.reference_states = references
        self.state_span = span
        self.weights = weights
        self.search_ahead_frames = search_ahead_frames
        self.max_advance_frames = max_advance_frames
        self.match_threshold = match_threshold
        self.similarity_slack = similarity_slack
        self.index = 0
        self.freeze_count = 0
        self.repeat_count = 0
        self.match_distances: list[float] = []

    def _distances(self, state: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != self.state_span.shape or not np.isfinite(values).all():
            raise ValueError(f"State-aligned replay expected a finite {self.state_span.shape} state")
        normalized = (self.reference_states[candidates] - values) / self.state_span
        return np.sqrt(np.sum(self.weights * np.square(normalized), axis=1) / np.sum(self.weights))

    def current(self, state: np.ndarray) -> StateAlignedSelection:
        distance = float(self._distances(state, np.asarray([self.index]))[0])
        self.match_distances.append(distance)
        return StateAlignedSelection(self.frames[self.index], self.index, distance, False)

    def select(self, state: np.ndarray) -> StateAlignedSelection:
        stop = min(len(self.frames), self.index + self.search_ahead_frames + 1)
        candidates = np.arange(self.index, stop, dtype=np.int32)
        distances = self._distances(state, candidates)
        best_distance = float(np.min(distances))
        frozen = best_distance > self.match_threshold
        previous = self.index
        if frozen:
            self.freeze_count += 1
        else:
            near_best = candidates[distances <= best_distance + self.similarity_slack]
            self.index = min(int(near_best[-1]), self.index + self.max_advance_frames)
        if self.index == previous:
            self.repeat_count += 1
        self.match_distances.append(best_distance)
        return StateAlignedSelection(self.frames[self.index], self.index, best_distance, frozen)

    def summary(self) -> dict[str, float | int]:
        distances = np.asarray(self.match_distances, dtype=np.float64)
        return {
            "final_frame_index": self.index,
            "freeze_count": self.freeze_count,
            "repeat_count": self.repeat_count,
            "mean_match_distance": float(np.mean(distances)) if len(distances) else 0.0,
            "p95_match_distance": float(np.percentile(distances, 95)) if len(distances) else 0.0,
            "max_match_distance": float(np.max(distances)) if len(distances) else 0.0,
        }


class InitialPoseLoader:
    def __init__(self, origin_root: Path) -> None:
        matches = sorted(origin_root.resolve().glob("pose_record_*.json"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one pose_record JSON under {origin_root}, found {len(matches)}"
            )
        self.path = matches[0]
        self.read_count = 0
        self.file_sha256 = sha256_file(self.path)

    def read_nearest(self, timestamp: float) -> tuple[dict[str, float], dict[str, float | int]]:
        if self.read_count != 0:
            raise RuntimeError("Origin pose data may only be read once during simulator reset")
        self.read_count += 1
        with self.path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        frames = payload.get("frames", [])
        if not frames:
            raise ValueError(f"No pose frames in {self.path}")
        timestamps = [float(frame["timestamp"]) for frame in frames]
        position = bisect.bisect_left(timestamps, timestamp)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(frames)]
        index = min(candidates, key=lambda item: abs(timestamps[item] - timestamp))
        frame = frames[index]
        pose = {
            name: float(value) for name, value in frame["data"].items() if isinstance(value, (int, float))
        }
        return pose, {
            "frame_id": int(frame["frame_id"]),
            "timestamp": float(frame["timestamp"]),
            "delta_ms": abs(float(frame["timestamp"]) - timestamp) * 1000.0,
        }
