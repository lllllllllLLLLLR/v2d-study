"""Small, deterministic helpers used by the executed V2D CPU notebooks."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "video_to_data_challenge"
OUTPUT_ROOT = ROOT / "outputs"


def ensure_outputs() -> dict[str, Path]:
    paths = {
        "root": OUTPUT_ROOT,
        "figs": OUTPUT_ROOT / "figs",
        "frames": OUTPUT_ROOT / "frames",
        "metrics": OUTPUT_ROOT / "metrics",
        "report": OUTPUT_ROOT / "report",
        "logs": OUTPUT_ROOT / "logs",
        "meshes": OUTPUT_ROOT / "meshes",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_parquet(path: str | Path):
    import pandas as pd
    return pd.read_parquet(path)


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "importable"))
        except Exception as exc:
            return f"unavailable: {type(exc).__name__}: {exc}"


def video_info(path: str | Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"path": str(path), "fps": fps, "frame_count": n, "width": w, "height": h, "duration_s": n / fps if fps else None}


def sample_video(path: str | Path, max_frames: int = 16, resize_width: int = 640) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Read evenly spaced BGR frames; bounded memory is important on a 16 GB machine."""
    info = video_info(path)
    n = info["frame_count"]
    indices = np.linspace(0, max(0, n - 1), min(max_frames, max(1, n)), dtype=int).tolist()
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if resize_width and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    info.update({"sample_indices": indices, "sample_count": len(frames), "sample_fps": info["fps"] * len(indices) / max(1, n)})
    return frames, info


def save_contact_sheet(frames: list[np.ndarray], path: str | Path, titles: Iterable[str] | None = None, cols: int = 4, cmap: str | None = None) -> None:
    import matplotlib.pyplot as plt
    titles = list(titles or ["" for _ in frames])
    rows = max(1, math.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(frames):
            img = frames[i] if cmap else cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img, cmap=cmap)
            ax.set_title(titles[i] if i < len(titles) else "")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def track1_paths() -> list[dict[str, Any]]:
    root = DATA_ROOT / "track_1"
    meta = {int(x["episode_index"]): x for x in read_jsonl(root / "meta/episodes_metadata.jsonl")}
    episodes = {int(x["episode_index"]): x for x in read_jsonl(root / "meta/episodes.jsonl")}
    rows = []
    for p in sorted((root / "data/chunk-000").glob("*.parquet")):
        idx = int(p.stem.split("_")[-1])
        v = root / "videos/chunk-000/observation.images.exo_camera" / f"episode_{idx:06d}.mp4"
        row = {**meta.get(idx, {}), **episodes.get(idx, {}), "episode_index": idx, "parquet": str(p), "video": str(v)}
        row.update(video_info(v))
        rows.append(row)
    return rows


def track2_paths(tier: int = 1) -> list[dict[str, Any]]:
    root = DATA_ROOT / "track_2" / ("tier_1_multiview_caption" if tier == 1 else "tier_2_synthetic_noise")
    rows = read_jsonl(root / "meta/episodes_metadata.jsonl")
    for row in rows:
        idx = int(row["episode_index"])
        row.update({"tier": tier, "root": str(root), "parquet": str(root / f"data/chunk-000/episode_{idx:06d}.parquet"), "video": str(root / f"videos/chunk-000/observation.images.exo_camera/episode_{idx:06d}.mp4")})
    return rows


def track3_paths() -> list[dict[str, Any]]:
    root = DATA_ROOT / "track_3/public"
    meta = {int(x["episode_index"]): x for x in read_jsonl(root / "meta/episodes_metadata.jsonl")}
    rows = []
    for p in sorted((root / "data/chunk-000").glob("*.parquet")):
        idx = int(p.stem.split("_")[-1])
        row = {**meta.get(idx, {}), "episode_index": idx, "parquet": str(p)}
        row["videos"] = {cam: str(root / f"videos/chunk-000/observation.images.{cam}/episode_{idx:06d}.mp4") for cam in ("ego_cam_a", "ego_cam_b", "ego_cam_c")}
        rows.append(row)
    return rows


def decode_track3_objects(df) -> dict[str, dict[str, np.ndarray]]:
    first = df.iloc[0]["observation.objects"]
    names = [str(o["name"]) for o in first]
    out = {name: {"pose": [], "visible": []} for name in names}
    for entry in df["observation.objects"]:
        by_name = {str(o["name"]): o for o in entry}
        for name in names:
            item = by_name[name]
            out[name]["pose"].append(np.asarray(item["pose"], dtype=float))
            out[name]["visible"].append(bool(item["visible"]))
    for name in names:
        out[name]["pose"] = np.asarray(out[name]["pose"], dtype=float)
        out[name]["visible"] = np.asarray(out[name]["visible"], dtype=bool)
    return out


def quat_angle_rad(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion geodesic angle for w-first [w,x,y,z] arrays."""
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    dot = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return 2.0 * np.arccos(np.clip(np.abs(dot), -1.0, 1.0))


def finite_difference(x: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if order == 1:
        return np.gradient(x, dt, axis=0, edge_order=1)
    return np.gradient(np.gradient(x, dt, axis=0, edge_order=1), dt, axis=0, edge_order=1)


def smooth_signal(x: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    from scipy.signal import savgol_filter
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x.copy()
    w = min(window, len(x) if len(x) % 2 else len(x) - 1)
    w = max(5, w)
    if w <= polyorder:
        return x.copy()
    return savgol_filter(x, w, polyorder, axis=0, mode="interp")


def plane_signed_distance(points: np.ndarray, plane: Iterable[float]) -> np.ndarray:
    a, b, c, d = np.asarray(list(plane), dtype=float)
    p = np.asarray(points, dtype=float)
    return (p @ np.array([a, b, c]) + d) / (math.sqrt(a * a + b * b + c * c) + 1e-12)


def peak_rss_mb() -> float | None:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1024**2)
    except Exception:
        return None


def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
