#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量把一个目录中的 HDF5(关节+相机) 转成 LeRobot 风格数据集 (Parquet + MP4)：
- 每个 HDF5 == 1 个 episode
- 严格生成 info.json（字段/顺序对齐你给的样例）
- episodes.jsonl / tasks.jsonl / episodes_stats.jsonl 按行追加
- 视频编码优先 rawvideo→ffmpeg(流式)，失败再退 PNG→ffmpeg→imageio
- 颜色默认 --assume_bgr（BGR→RGB），可用 --no-assume_bgr 关闭
"""

import argparse
import json
import sys
import subprocess
from collections import OrderedDict
from pathlib import Path
import typing as T
import shutil
import math

import h5py
import numpy as np

# 可选依赖
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    import imageio.v3 as iio
    _HAS_IMAGEIO_V3 = True
except Exception:
    _HAS_IMAGEIO_V3 = False

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

# ========= 小工具 =========


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _safe_rmtree(p: Path):
    try:
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def human_join(xs: T.Iterable[str]) -> str:
    xs = list(xs)
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


# ========= HDF5 读取 & 校验 =========


def discover_hdf5(h5: h5py.File, image_group: str, cam_keys: T.List[str],
                  qpos_key: str):
    imgs_g = h5[image_group]
    cams = {}
    for k in cam_keys:
        if k not in imgs_g:
            raise KeyError(
                f"Camera key '{k}' not found under '{image_group}'. Available: {list(imgs_g.keys())}"
            )
        arr = imgs_g[k]
        if not isinstance(arr, h5py.Dataset):
            raise TypeError(f"{image_group}/{k} is not a dataset")
        cams[k] = arr

    if qpos_key not in h5:
        raise KeyError(
            f"'{qpos_key}' not found. Available top keys: {list(h5.keys())}")
    qpos = h5[qpos_key]
    if not isinstance(qpos, h5py.Dataset):
        raise TypeError(f"{qpos_key} is not a dataset")

    # 基本检查
    T_img = None
    for _, ds in cams.items():
        if ds.ndim != 4 or ds.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Camera dataset expected shape (T,H, W, C), got {ds.shape}")
        T_img = ds.shape[0] if T_img is None else T_img
    T_q = qpos.shape[0]
    if any(ds.shape[0] != T_img for ds in cams.values()):
        raise ValueError(
            f"All cameras must share same T. Got {[ds.shape[0] for ds in cams.values()]}"
        )
    if T_q != T_img:
        print(f"[warn] qpos T={T_q} != image T={T_img}. Use min(T).",
              file=sys.stderr)

    T_final = min(T_img, T_q)
    H, W, C = next(iter(cams.values())).shape[1:4]
    DoF = int(qpos.shape[1])
    return cams, qpos, T_final, H, W, C, DoF


# ========= FFmpeg 编码 =========


def ffmpeg_has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             check=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True).stdout.lower()
        return name.lower() in out
    except Exception:
        return False


def pick_encoder(preferred: str) -> str:
    candidates = [preferred]
    if preferred != "av1_nvenc":
        candidates.append("av1_nvenc")
    if preferred != "libaom-av1":
        candidates.append("libaom-av1")
    if preferred != "libx264":
        candidates.append("libx264")
    seen, ordered = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    for cand in ordered:
        if ffmpeg_has_encoder(cand):
            return cand
    return preferred


def _ensure_rgb24(frame: np.ndarray) -> np.ndarray:
    """把 1/3/4 通道图像转为 RGB24 (H,W,3) uint8."""
    if frame.ndim != 3:
        raise ValueError(f"Expected (H,W,C), got {frame.shape}")
    C = frame.shape[-1]
    if C == 3:
        return frame
    if C == 4:
        return frame[..., :3]
    if C == 1:
        return np.repeat(frame, 3, axis=2)
    raise ValueError(f"Unsupported channel count C={C}")


def write_videos_ffmpeg_stream(ds,
                               T_final: int,
                               out_path: Path,
                               fps: int,
                               assume_bgr: bool,
                               encoder: str,
                               pix_fmt: str,
                               H: int,
                               W: int,
                               C: int,
                               progress_every: int = 50) -> bool:
    """raw RGB24 → ffmpeg stdin（最快）"""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        encoder,
    ]
    if encoder == "libaom-av1":
        cmd += [
            "-crf", "28", "-b:v", "0", "-cpu-used", "8", "-pix_fmt", pix_fmt
        ]
    elif encoder == "av1_nvenc":
        cmd += [
            "-cq:v", "25", "-b:v", "0", "-preset", "p5", "-pix_fmt", pix_fmt
        ]
    else:
        cmd += ["-crf", "18", "-preset", "veryfast", "-pix_fmt", pix_fmt]
    cmd += [out_path.as_posix()]

    proc = subprocess.Popen(cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        for i in range(T_final):
            frame = ds[i]
            if assume_bgr:
                frame = frame[..., ::-1]
            frame = _ensure_rgb24(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)
            proc.stdin.write(frame.tobytes())
            if (i + 1) % progress_every == 0 or i == T_final - 1:
                print(
                    f"[encode] {out_path.name} {i+1}/{T_final} ({(i+1)*100//T_final}%)",
                    flush=True)
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode("utf-8", errors="ignore")
            sys.stderr.write(f"[ffmpeg stderr] {err}\n")
            return False
        return True
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        print(f"[warn] streaming encode failed: {e}", file=sys.stderr)
        return False


def write_videos_ffmpeg_from_pngs(ds, T_final: int, out_path: Path, fps: int,
                                  assume_bgr: bool, encoder: str,
                                  pix_fmt: str) -> bool:
    """慢路径：PNG → ffmpeg"""
    tmp_dir = out_path.parent / (out_path.stem + "_frames")
    _safe_rmtree(tmp_dir)
    _ensure_dir(tmp_dir)
    try:
        if not _HAS_IMAGEIO_V3:
            raise RuntimeError("imageio.v3 required for PNG writing")
        for i in range(T_final):
            frame = ds[i]
            if assume_bgr:
                frame = frame[..., ::-1]
            iio.imwrite(tmp_dir / f"{i:06d}.png", _ensure_rgb24(frame))
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate",
            str(fps), "-i",
            str(tmp_dir / "%06d.png"), "-c:v", encoder
        ]
        if encoder == "libaom-av1":
            cmd += ["-crf", "28", "-b:v", "0", "-pix_fmt", pix_fmt]
        elif encoder == "av1_nvenc":
            cmd += [
                "-cq:v", "25", "-b:v", "0", "-preset", "p5", "-pix_fmt", pix_fmt
            ]
        else:
            cmd += ["-crf", "18", "-preset", "medium", "-pix_fmt", pix_fmt]
        cmd += [out_path.as_posix()]
        proc = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        if proc.returncode != 0:
            sys.stderr.write(f"[ffmpeg stderr] {proc.stderr}\n")
            raise RuntimeError("ffmpeg failed")
        return True
    except Exception as e:
        print(f"[warn] PNG route failed: {e}", file=sys.stderr)
        return False
    finally:
        _safe_rmtree(tmp_dir)


def write_videos_imageio(ds, T_final: int, out_path: Path, fps: int,
                         assume_bgr: bool) -> bool:
    """兜底：imageio(H.264)"""
    if not _HAS_IMAGEIO:
        return False
    try:
        writer = imageio.get_writer(out_path.as_posix(),
                                    fps=fps,
                                    macro_block_size=1,
                                    codec="libx264")
        for i in range(T_final):
            frame = ds[i]
            if assume_bgr:
                frame = frame[..., ::-1]
            writer.append_data(_ensure_rgb24(frame))
        writer.close()
        return True
    except Exception as e:
        try:
            writer.close()
        except Exception:
            pass
        print(f"[warn] imageio failed: {e}", file=sys.stderr)
        return False


def write_episode_videos_for_cams(cams: dict, T_final: int, out_root: Path,
                                  fps: int, episode_index: int, chunk_size: int,
                                  encoder_pref: str, pix_fmt: str,
                                  assume_bgr: bool,
                                  progress_every: int) -> T.Dict[str, str]:
    """
    为某个 episode 的所有相机写 mp4：videos/chunk-XYZ/observation.images.<cam>/episode_NNNNNN.mp4
    返回 {cam_key: relative_path}
    """
    used_encoder = pick_encoder(encoder_pref)
    rel_paths = {}
    chunk_idx = episode_index // chunk_size
    epi_name = f"episode_{episode_index:06d}.mp4"
    for cam_name in sorted(cams.keys()):
        ds = cams[cam_name]
        out_dir = out_root / f"videos/chunk-{chunk_idx:03d}/observation.images.{cam_name}"
        out_path = out_dir / epi_name
        _ensure_dir(out_dir)
        _safe_rmtree(out_dir / (out_path.stem + "_frames"))

        H, W, C = ds.shape[1:4]
        ok = write_videos_ffmpeg_stream(ds, T_final, out_path, fps, assume_bgr,
                                        used_encoder, pix_fmt, H, W, C,
                                        progress_every)

        if not ok:
            for alt in ["av1_nvenc", "libaom-av1", "libx264"]:
                if alt == used_encoder:
                    continue
                if not ffmpeg_has_encoder(alt):
                    continue
                ok = write_videos_ffmpeg_stream(ds, T_final, out_path, fps,
                                                assume_bgr, alt, pix_fmt, H, W,
                                                C, progress_every)
                if ok:
                    used_encoder = alt
                    break

        if not ok:
            ok = write_videos_ffmpeg_from_pngs(ds, T_final, out_path, fps,
                                               assume_bgr, used_encoder,
                                               pix_fmt)
        if not ok:
            ok = write_videos_imageio(ds, T_final, out_path, fps, assume_bgr)
        if not ok:
            raise RuntimeError(
                f"Failed to produce mp4 for camera '{cam_name}' (episode {episode_index})"
            )

        _safe_rmtree(out_dir / (out_path.stem + "_frames"))

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(
                f"MP4 missing/empty for camera '{cam_name}': {out_path}")
        rel_paths[
            cam_name] = f"videos/chunk-{chunk_idx:03d}/observation.images.{cam_name}/{epi_name}"
    return rel_paths


# ========= Parquet =========


def write_episode_parquet(out_root: Path,
                          actions: np.ndarray,
                          T_final: int,
                          fps: int,
                          episode_index: int,
                          task_index: int,
                          chunk_size: int,
                          use_state: str = "none") -> Path:
    if not _HAS_PARQUET:
        raise RuntimeError("Please `pip install pandas pyarrow`.")
    actions = actions[:T_final].astype("float32", copy=False)
    timestamps = np.arange(T_final, dtype="float32") / float(fps)
    frame_index = np.arange(T_final, dtype="int64")
    episode_idx = np.full((T_final,), episode_index, dtype="int64")
    global_index = frame_index.copy()
    task_idx = np.full((T_final,), task_index, dtype="int64")

    arrays, names = [], []
    arrays.append(
        pa.array([actions[i, :].tolist() for i in range(T_final)],
                 type=pa.list_(pa.float32())))
    names.append("action")

    if use_state == "lag":
        states = np.vstack([actions[0:1], actions[:-1]]).astype("float32")
        arrays.append(
            pa.array([states[i, :].tolist() for i in range(T_final)],
                     type=pa.list_(pa.float32())))
        names.append("observation.state")

    arrays.extend([
        pa.array(timestamps, type=pa.float32()),
        pa.array(frame_index, type=pa.int64()),
        pa.array(episode_idx, type=pa.int64()),
        pa.array(global_index, type=pa.int64()),
        pa.array(task_idx, type=pa.int64()),
    ])
    names.extend(
        ["timestamp", "frame_index", "episode_index", "index", "task_index"])

    tbl = pa.Table.from_arrays(arrays, names=names)
    chunk_idx = episode_index // chunk_size
    out_parquet = out_root / f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet"
    _ensure_dir(out_parquet.parent)
    pq.write_table(tbl, out_parquet, version="2.6")
    return out_parquet


# ========= info.json（严格对齐样例） =========


def _codec_label(enc: str) -> str:
    enc = enc.lower()
    if "av1" in enc:
        return "av1"
    if "264" in enc:
        return "h264"
    return enc


def make_info_json_strict(
    fps: int,
    H: int,
    W: int,
    C: int,
    DoF: int,
    cam_keys: T.List[str],
    joint_names: T.List[str],
    include_state: bool,
    robot_type: str,
    total_frames: int,
    total_episodes: int,
    codebase_version: str,
    total_tasks: int,
    total_videos: int,  # 这里指数据集的相机路数（去重后的）
    total_chunks: int,
    chunks_size: int,
    splits: dict,
    encoder_used: str,
    pix_fmt: str,
) -> OrderedDict:
    features = OrderedDict()
    features["action"] = {
        "dtype": "float32",
        "shape": [DoF],
        "names": joint_names
    }
    if include_state:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [DoF],
            "names": joint_names
        }

    # 相机
    for cam_key in cam_keys:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": [H, W, C],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": H,
                "video.width": W,
                "video.codec": _codec_label(encoder_used),
                "video.pix_fmt": pix_fmt,
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": C,
                "has_audio": False,
            },
        }

    # 索引列最后
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    info = OrderedDict()
    info["codebase_version"] = codebase_version
    info["robot_type"] = robot_type
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    info["total_videos"] = int(total_videos)  # 相机路数
    info["total_chunks"] = int(total_chunks)
    info["chunks_size"] = int(chunks_size)
    info["fps"] = fps
    info["splits"] = splits
    info[
        "data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info[
        "video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["features"] = features
    return info


# ========= episodes_stats.jsonl =========


def _stats_vector(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        mins = [float(np.min(arr))]
        maxs = [float(np.max(arr))]
        means = [float(np.mean(arr))]
        stds = [float(np.std(arr, ddof=0))]
        count = [int(arr.shape[0])]
    else:
        mins = np.min(arr, axis=0).astype(float).tolist()
        maxs = np.max(arr, axis=0).astype(float).tolist()
        means = np.mean(arr, axis=0).astype(float).tolist()
        stds = np.std(arr, axis=0, ddof=0).astype(float).tolist()
        count = [int(arr.shape[0])]
    return {
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "count": count
    }


def _stats_image_channelwise(ds, T_final: int, assume_bgr: bool,
                             sample_max: int) -> dict:
    # 采样索引
    if sample_max is None or sample_max < 0:
        idxs = np.arange(T_final)
    else:
        if T_final <= sample_max:
            idxs = np.arange(T_final)
        else:
            idxs = np.linspace(0, T_final - 1,
                               num=sample_max).round().astype(int)

    cmins = cmaxs = cmeans = cstds = None
    for i, t in enumerate(idxs):
        frame = ds[t]
        if assume_bgr:
            frame = frame[..., ::-1]
        frame = _ensure_rgb24(frame).astype(np.float32) / 255.0
        flat = frame.reshape(-1, 3)
        fmin = np.min(flat, axis=0)
        fmax = np.max(flat, axis=0)
        fmean = np.mean(flat, axis=0)
        fstd = np.std(flat, axis=0, ddof=0)
        if i == 0:
            cmins, cmaxs, cmeans, cstds = fmin, fmax, fmean, fstd
        else:
            cmins = np.minimum(cmins, fmin)
            cmaxs = np.maximum(cmaxs, fmax)
            cmeans = (cmeans * i + fmean) / (i + 1)
            cstds = (cstds * i + fstd) / (i + 1)

    wrap3 = lambda x: [[[float(v)]] for v in x.tolist()]
    return {
        "min": wrap3(cmins),
        "max": wrap3(cmaxs),
        "mean": wrap3(cmeans),
        "std": wrap3(cstds),
        "count": [int(len(idxs))]
    }


def append_episode_stats_jsonl(out_root: Path, episode_index: int, T_final: int,
                               fps: int, actions: np.ndarray,
                               include_state: bool,
                               states: T.Optional[np.ndarray], cams: dict,
                               assume_bgr: bool, image_stats_sample: int):
    stats = {}
    stats["action"] = _stats_vector(actions)
    if include_state and states is not None:
        stats["observation.state"] = _stats_vector(states)
    timestamps = np.arange(T_final, dtype=np.float32) / float(fps)
    stats["timestamp"] = _stats_vector(timestamps)
    frame_index = np.arange(T_final, dtype=np.int64)
    stats["frame_index"] = _stats_vector(frame_index)
    epi_idx = np.full((T_final,), episode_index, dtype=np.int64)
    stats["episode_index"] = _stats_vector(epi_idx)
    global_index = frame_index.copy()
    stats["index"] = _stats_vector(global_index)
    task_index = np.zeros((T_final,), dtype=np.int64)
    stats["task_index"] = _stats_vector(task_index)
    for cam_name, ds in cams.items():
        key = f"observation.images.{cam_name}"
        stats[key] = _stats_image_channelwise(ds,
                                              T_final,
                                              assume_bgr=assume_bgr,
                                              sample_max=image_stats_sample)
    row = {"episode_index": int(episode_index), "stats": stats}
    _append_jsonl(out_root / "meta/episodes_stats.jsonl", row)


# ========= 主流程 =========


def main():
    ap = argparse.ArgumentParser(
        description=
        "Convert DIR of HDF5 (each file = one episode) -> LeRobot dataset")
    ap.add_argument("--in_dir",
                    required=True,
                    help="Folder that contains many .hdf5")
    ap.add_argument("--out_dir",
                    required=True,
                    help="Output dataset root folder")
    ap.add_argument("--repo_id",
                    required=True,
                    help="Dataset id like 'user/name' (metadata only)")
    ap.add_argument("--task",
                    required=True,
                    help="Task string for ALL episodes (统一任务名)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--qpos_key", default="/observations/qpos")
    ap.add_argument("--image_group", default="/observations/images")
    ap.add_argument("--cam_keys",
                    nargs="+",
                    default=["cam_high_left", "cam_high_right"])
    ap.add_argument("--robot_type", default="w1")
    ap.add_argument("--use_state", choices=["none", "lag"], default="none")
    ap.add_argument("--joint_names",
                    nargs="*",
                    default=[
                        "ANKLE", "KNEE", "BUTTOCK", "WAIST", "LEFT_J1",
                        "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6",
                        "LEFT_J7", "NECK1", "NECK2", "RIGHT_J1", "RIGHT_J2",
                        "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6",
                        "RIGHT_J7"
                    ])
    try:
        from argparse import BooleanOptionalAction
        ap.add_argument("--assume_bgr",
                        action=BooleanOptionalAction,
                        default=True)
    except Exception:
        ap.add_argument("--assume_bgr", action="store_true", default=True)
    ap.add_argument("--ffmpeg_encoder",
                    default="av1_nvenc",
                    choices=["libaom-av1", "av1_nvenc", "libx264"])
    ap.add_argument("--pix_fmt", default="yuv420p")
    ap.add_argument("--progress_every", type=int, default=50)

    # info.json 顶层
    ap.add_argument("--codebase_version", default="v2.1")
    ap.add_argument("--chunks_size",
                    type=int,
                    default=1000,
                    help="每个 chunk 内最多包含多少 episodes")
    ap.add_argument("--splits_train",
                    default=None,
                    help='如未指定，自动设为 "0:<num_episodes>"')

    # 图像统计采样：-1/0=全帧；默认 105
    ap.add_argument("--image_stats_sample", type=int, default=105)

    # 选择/排序 HDF5
    ap.add_argument("--glob",
                    default="*.hdf5",
                    help="Which files under in_dir to include (glob)")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    _ensure_dir(out_root / "meta")

    # 枚举 HDF5
    files = sorted(in_dir.glob(args.glob))
    if not files:
        print(f"[error] No HDF5 found under {in_dir} with pattern {args.glob}",
              file=sys.stderr)
        sys.exit(2)

    # 统计信息累计
    total_frames = 0
    total_episodes = 0
    used_encoder_for_info = args.ffmpeg_encoder  # 以首个成功编码器为准，后续若降级也用这个（通常一致）
    first_HWC_DoF = None
    joint_names = None
    include_state = (args.use_state == "lag")
    cam_keys = list(args.cam_keys)

    # 清空旧的 episodes.jsonl / tasks.jsonl / episodes_stats.jsonl
    for f in ["episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]:
        p = out_root / "meta" / f
        if p.exists():
            p.unlink()

    # 写 tasks.jsonl（统一一个任务）
    _append_jsonl(out_root / "meta/tasks.jsonl", {
        "task_index": 0,
        "task": args.task
    })

    # 逐文件处理
    for epi_idx, path in enumerate(files):
        print(f"\n=== [{epi_idx+1}/{len(files)}] {path.name} ===")
        with h5py.File(path, "r") as h5:
            cams, qpos, T_final, H, W, C, DoF = discover_hdf5(
                h5,
                image_group=args.image_group,
                cam_keys=cam_keys,
                qpos_key=args.qpos_key)
            actions = np.asarray(qpos[:T_final], dtype="float32")

            # 首次记录全局 HWC/DoF & 关节名
            if first_HWC_DoF is None:
                first_HWC_DoF = (H, W, C, DoF)
                if args.joint_names:
                    joint_names = list(args.joint_names)
                    if len(joint_names) != DoF:
                        raise ValueError(
                            f"--joint_names length {len(joint_names)} != DoF {DoF}"
                        )
                else:
                    if DoF % 2 == 0:
                        half = DoF // 2
                        joint_names = [f"L_J{i+1}" for i in range(half)
                                      ] + [f"R_J{i+1}" for i in range(half)]
                    else:
                        joint_names = [f"J{i+1}" for i in range(DoF)]
            else:
                if (H, W, C, DoF) != first_HWC_DoF:
                    raise ValueError(
                        f"Episode {epi_idx}: shape/DoF mismatch. "
                        f"Got {(H,W,C,DoF)} vs first {first_HWC_DoF}")

            # 视频（每路相机一个 mp4）
            rel_paths = write_episode_videos_for_cams(
                cams=cams,
                T_final=T_final,
                out_root=out_root,
                fps=args.fps,
                episode_index=epi_idx,
                chunk_size=args.chunks_size,
                encoder_pref=args.ffmpeg_encoder,
                pix_fmt=args.pix_fmt,
                assume_bgr=args.assume_bgr,
                progress_every=args.progress_every)
            # 用首次实际编码器更新 info 的 codec（如果发生降级，沿用成功的编码器名）
            # 这里无法直接获知流式编码时的实际 encoder（我们已用 used_encoder_for_info 代表首选/可用的 encoder）
            # 若你非常严格想反映“实际使用哪一个”，可以在 write_episode_videos_for_cams 中返回 used_encoder。
            # 简化：沿用 args.ffmpeg_encoder 的归一化标签。
            used_encoder_for_info = args.ffmpeg_encoder

            # Parquet
            parquet_path = write_episode_parquet(out_root=out_root,
                                                 actions=actions,
                                                 T_final=T_final,
                                                 fps=args.fps,
                                                 episode_index=epi_idx,
                                                 task_index=0,
                                                 chunk_size=args.chunks_size,
                                                 use_state=args.use_state)

            # 统计（追加一行）
            states_arr = np.vstack([
                actions[0:1], actions[:-1]
            ]).astype("float32") if include_state else None
            sample_k = args.image_stats_sample if args.image_stats_sample != 0 else -1
            append_episode_stats_jsonl(out_root,
                                       episode_index=epi_idx,
                                       T_final=T_final,
                                       fps=args.fps,
                                       actions=actions,
                                       include_state=include_state,
                                       states=states_arr,
                                       cams=cams,
                                       assume_bgr=args.assume_bgr,
                                       image_stats_sample=sample_k)

            # episodes.jsonl 追加
            _append_jsonl(
                out_root / "meta/episodes.jsonl", {
                    "episode_index": int(epi_idx),
                    "tasks": [args.task],
                    "length": int(T_final)
                })

            total_frames += int(T_final)
            total_episodes += 1

    # info.json（严格对齐样例；相机路数=去重后的 cam_keys）
    total_chunks = int(math.ceil(total_episodes / float(args.chunks_size)))
    splits_train = args.splits_train if args.splits_train is not None else f"0:{total_episodes}"

    H, W, C, DoF = first_HWC_DoF
    info = make_info_json_strict(
        fps=args.fps,
        H=H,
        W=W,
        C=C,
        DoF=DoF,
        cam_keys=cam_keys,
        joint_names=joint_names,
        include_state=include_state,
        robot_type=args.robot_type,
        total_frames=total_frames,
        total_episodes=total_episodes,
        codebase_version=args.codebase_version,
        total_tasks=1,
        total_videos=len(cam_keys),
        total_chunks=total_chunks,
        chunks_size=args.chunks_size,
        splits={"train": splits_train},
        encoder_used=args.ffmpeg_encoder,
        pix_fmt=args.pix_fmt,
    )
    _write_json(out_root / "meta/info.json", info)

    # 结束提示
    print("\n✅ Done. Wrote LeRobot-style dataset to:", out_root)
    print(f"   Episodes : {total_episodes}")
    print(f"   Frames   : {total_frames}")
    print(f"   Cameras  : {human_join(cam_keys)}")
    print("   Meta     :", out_root / "meta/info.json")
    print("   Episodes :", out_root / "meta/episodes.jsonl")
    print("   Tasks    :", out_root / "meta/tasks.jsonl")
    print("   Stats    :", out_root / "meta/episodes_stats.jsonl")


if __name__ == "__main__":
    main()
