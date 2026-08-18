#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import List, Optional, Tuple


def load_metadata(img_dir: Path):
    """Load metadata JSONL: list of dicts with 'timestamp', 'left_img', 'right_img'."""
    metadata_path = img_dir / 'metadata.jsonl'
    meta = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            meta.append(entry)
    meta.sort(key=lambda x: x['timestamp'])
    return meta


def load_qpos(qpos_path: Path,
              joint_keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load qpos JSON: returns timestamps and qpos array (N×D)."""
    with open(qpos_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = data['frames']
    frames.sort(key=lambda x: x['timestamp'])
    ts = np.array([f['timestamp'] for f in frames], dtype=np.float64)
    qpos = np.stack([
        np.array([f['data'].get(k, 0.0)
                  for k in joint_keys], dtype=np.float32)
        for f in frames
    ],
                    axis=0)
    return ts, qpos


def interp_qpos(qpos_ts: np.ndarray, qpos: np.ndarray,
                target_ts: np.ndarray) -> np.ndarray:
    """Linearly interpolate each joint to the target timestamps."""
    D = qpos.shape[1]
    out = np.zeros((len(target_ts), D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(target_ts, qpos_ts, qpos[:, d])
    return out


def load_images(img_dir: Path, meta: list) -> Tuple[np.ndarray, np.ndarray]:
    """Load left/right images into arrays."""
    sample = cv2.imread(str(img_dir / meta[0]['left_img']))
    if sample is None:
        raise FileNotFoundError(f"Cannot read image {meta[0]['left_img']}")
    H, W, _ = sample.shape
    T = len(meta)
    left = np.zeros((T, H, W, 3), dtype=np.uint8)
    right = np.zeros((T, H, W, 3), dtype=np.uint8)
    for i, m in enumerate(meta):
        li = cv2.imread(str(img_dir / m['left_img']))
        ri = cv2.imread(str(img_dir / m['right_img']))
        if li is None or ri is None:
            raise FileNotFoundError(f"Missing image at index {i}: {m}")
        left[i] = li
        right[i] = ri
    return left, right


def write_hdf5(out_path: Path, left_imgs: np.ndarray, right_imgs: np.ndarray,
               qpos_interp: np.ndarray):
    """Write datasets into HDF5 with the required structure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = left_imgs.shape
    D = qpos_interp.shape[1]
    with h5py.File(str(out_path), 'w') as f:
        obs = f.create_group('observations')
        imgs = obs.create_group('images')
        imgs.create_dataset('cam_high_left',
                            data=left_imgs,
                            shape=(T, H, W, 3),
                            dtype='uint8',
                            chunks=(1, H, W, 3),
                            compression='gzip')
        imgs.create_dataset('cam_high_right',
                            data=right_imgs,
                            shape=(T, H, W, 3),
                            dtype='uint8',
                            chunks=(1, H, W, 3),
                            compression='gzip')
        ds = obs.create_dataset('qpos',
                                shape=(T, D),
                                dtype='float32',
                                chunks=(1, D),
                                compression='gzip')
        ds[...] = qpos_interp
    print(f"✅ Written HDF5 to {out_path}")


def find_qpos_json(session_dir: Path, pattern: str) -> Optional[Path]:
    """Find a teleop qpos JSON in a session directory using a glob pattern."""
    candidates = sorted(session_dir.glob(pattern))
    if not candidates:
        return None
    # Heuristic: prefer the most recent by mtime if multiple
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def convert_one_session(session_dir: Path, qpos_pattern: str,
                        out_root: Optional[Path],
                        overwrite: bool) -> Optional[Path]:
    """
    Convert one session directory.
    Expected structure:
      session_dir/
        img/
          metadata.jsonl
          <left/right images>
        pose_record_*.json   (or other name matched by qpos_pattern)
    """
    img_dir = session_dir / 'img'
    meta_path = img_dir / 'metadata.jsonl'
    if not img_dir.is_dir() or not meta_path.is_file():
        print(f"⚠️  Skip: {session_dir} (missing img/metadata.jsonl)")
        return None

    qpos_json = find_qpos_json(session_dir, qpos_pattern)
    if qpos_json is None:
        print(
            f"⚠️  Skip: {session_dir} (no qpos JSON matches pattern '{qpos_pattern}')"
        )
        return None

    # Output path
    if out_root is None:
        out_h5 = session_dir / (qpos_json.stem + '.hdf5')
    else:
        rel = session_dir.name  # one level
        out_h5 = out_root / (qpos_json.stem + '.hdf5')

    if out_h5.exists() and not overwrite:
        print(f"⏭️  Exists, skip (use --overwrite to force): {out_h5}")
        return out_h5

    # Define the joint keys (left/right J1–J7 + grippers)
    joint_keys = [
        "ANKLE",
        "KNEE",
        "BUTTOCK",
        "WAIST",
        *(f'LEFT_J{i}' for i in range(1, 8)),
        "NECK1",
        "NECK2",
        *(f'RIGHT_J{i}' for i in range(1, 8)),
    ]

    try:
        print(f"📥 [{session_dir.name}] Loading metadata…")
        meta = load_metadata(img_dir)
        ts_meta = np.array([m['timestamp'] for m in meta], dtype=np.float64)

        print(f"📥 [{session_dir.name}] Loading qpos JSON: {qpos_json.name}")
        ts_q, qpos = load_qpos(qpos_json, joint_keys)

        print(f"🔄 [{session_dir.name}] Interpolating qpos to image timestamps…")
        qpos_i = interp_qpos(ts_q, qpos, ts_meta)

        print(f"📥 [{session_dir.name}] Loading images…")
        left_imgs, right_imgs = load_images(img_dir, meta)

        print(f"💾 [{session_dir.name}] Writing HDF5…")
        write_hdf5(out_h5, left_imgs, right_imgs, qpos_i)
        return out_h5
    except Exception as e:
        print(f"❌ [{session_dir.name}] Failed: {e}")
        return None


def discover_sessions(root_dir: Path) -> List[Path]:
    """Return immediate subdirectories under root_dir (each treated as a session)."""
    return [p for p in root_dir.iterdir() if p.is_dir()]


def main():
    parser = argparse.ArgumentParser(
        description=
        "Convert teleop datasets (single folder or batch over all subfolders) → HDF5"
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=None,
        help="If set, batch-convert all subfolders under this directory.")
    parser.add_argument(
        '--session_dir',
        type=str,
        default=None,
        help="Convert a single session folder (contains img/ and qpos JSON).")
    parser.add_argument(
        '--qpos_pattern',
        type=str,
        default='pose_record*.json',
        help=
        "Glob pattern to locate qpos JSON inside each session folder (default: 'pose_record*.json')."
    )
    parser.add_argument(
        '--out_root',
        type=str,
        default=None,
        help=
        "If set in batch mode, write outputs under this root, preserving per-session subfolders."
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help="Overwrite existing HDF5 files.")
    args = parser.parse_args()

    # Mode selection
    if args.root_dir is None and args.session_dir is None:
        parser.error(
            "Please provide either --root_dir for batch mode or --session_dir for single session."
        )

    if args.root_dir is not None and args.session_dir is not None:
        parser.error(
            "Provide only one of --root_dir or --session_dir, not both.")

    out_root = Path(args.out_root) if args.out_root else None

    if args.session_dir:
        out = convert_one_session(Path(args.session_dir), args.qpos_pattern,
                                  out_root, args.overwrite)
        if out:
            print(f"✅ Done: {out}")
        else:
            raise SystemExit(1)

    if args.root_dir:
        root = Path(args.root_dir)
        sessions = discover_sessions(root)
        print(f"🔎 Found {len(sessions)} session folders under {root}")
        success = 0
        for s in sorted(sessions):
            out = convert_one_session(s, args.qpos_pattern, out_root,
                                      args.overwrite)
            if out:
                success += 1
        print(
            f"🎉 Batch finished: {success}/{len(sessions)} converted successfully."
        )


if __name__ == '__main__':
    main()
