"""CPU-only direct silhouette search for the Track 3 cup sequence.

This run is isolated from the earlier cup PnP attempts.  It uses the public CAD
meshes, ego intrinsics, color components, mask IoU search, and one frame-0 mocap
pose for world alignment.  Frame-wise camera motion is inferred from the static
blue cup CAD track.
"""

from pathlib import Path
import json
import resource
import time

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from v2_cad_tracking_dustpan import search_pose, render_score
from v2_cad_tracking_v2 import gt_transform, load_calibration, pose_matrix, projected_hull


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/v2/m4/cup_mask_search"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["blue_cup", "beige_cup"]


def read_frames():
    path = ROOT / "video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4"
    cap = cv2.VideoCapture(str(path))
    k, dist = load_calibration()
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        frames.append(cv2.undistort(frame, k, dist))
    cap.release()
    return frames


def mask_for(frame, name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if name == "blue_cup":
        raw = ((h >= 92) & (h <= 140) & (s >= 55) & (v >= 35)).astype(np.uint8) * 255
    else:
        raw = ((h >= 14) & (h <= 40) & (s <= 105) & (v > 95)).astype(np.uint8) * 255
    return cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def start_state(name):
    if name == "blue_cup":
        return {"center": np.array([292.0, 350.0]), "bbox": np.array([250.0, 285.0, 90.0, 125.0]), "area": 6000.0}
    return {"center": np.array([447.0, 283.0]), "bbox": np.array([391.0, 235.0, 112.0, 95.0]), "area": 4030.0}


def component(frame, name, state):
    full = mask_for(frame, name)
    x, y, bw, bh = state["bbox"]
    margin = max(48.0, 1.25 * max(bw, bh))
    h, w = full.shape
    x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
    x1, y1 = min(w, int(x + bw + margin)), min(h, int(y + bh + margin))
    roi = np.zeros_like(full)
    roi[y0:y1, x0:x1] = full[y0:y1, x0:x1]
    n, labels, stats, centers = cv2.connectedComponentsWithStats(roi)
    choices = []
    prior_area = max(150.0, float(state["area"]))
    prior_center = np.asarray(state["center"], dtype=float)
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < max(90.0, 0.035 * prior_area):
            continue
        box = stats[idx, :4].astype(float)
        aspect = box[2] / max(1.0, box[3])
        if name == "blue_cup" and not 0.25 <= aspect <= 1.25:
            continue
        if name == "beige_cup" and not 0.30 <= aspect <= 1.80:
            continue
        center = centers[idx]
        distance = np.linalg.norm(center - prior_center) / max(25.0, np.hypot(bw, bh))
        ratio = abs(np.log(max(area, 1.0) / prior_area))
        shape = abs(np.log(max(aspect, 1e-3) / (0.62 if name == "beige_cup" else 0.62)))
        score = 2.0 * distance + 0.65 * ratio + 0.18 * shape
        choices.append((score, idx, area, center, box))
    if not choices:
        return None, state, {"failure": "no_color_component", "roi": [x0, y0, x1, y1]}
    _, idx, area, center, box = min(choices, key=lambda row: row[0])
    mask = (labels == idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, state, {"failure": "no_contour", "roi": [x0, y0, x1, y1]}
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    state_next = {"center": center.astype(float), "bbox": box.astype(float), "area": area}
    return mask, state_next, {"contour": contour, "area_ratio": area / prior_area, "roi": [x0, y0, x1, y1]}


def main():
    t0 = time.perf_counter()
    frames = read_frames()
    k, _ = load_calibration()
    vertices = {}
    states = {name: start_state(name) for name in NAMES}
    for name in NAMES:
        v = np.load(ROOT / "outputs/v2/m4" / f"{name}_mesh.npz")["vertices"].astype(np.float64)
        vertices[name] = v[np.linspace(0, len(v) - 1, 2800, dtype=int)]
    poses = {}
    observations = []
    view_frames = {0, 20, 33, 35, 50, 70, 100, 120, 136}
    for i, frame in enumerate(frames):
        overlay = frame.copy()
        for name in NAMES:
            mask, next_state, details = component(frame, name, states[name])
            if mask is None:
                observations.append({"frame": i, "object": name, "success": False, "failure": details["failure"], "occlusion_bucket": "hand_or_segmentation_failure"})
                continue
            states[name] = next_state
            contour = details["contour"]
            pose, score, hull, iou = search_pose(vertices[name], contour, mask, k, poses.get(name))
            poses[name] = pose
            x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
            observed_area = float(np.count_nonzero(mask))
            rendered = np.zeros_like(mask)
            if len(hull):
                cv2.fillConvexPoly(rendered, hull.astype(np.int32).reshape(-1, 1, 2), 255)
            projected_area = float(np.count_nonzero(rendered))
            bucket = "hand_or_partial" if observed_area < 0.55 * max(1.0, projected_area) else "mask_reliable"
            observations.append({
                "frame": i,
                "object": name,
                "success": True,
                "camera_T_object": pose_matrix(pose[:3], pose[3:]).tolist(),
                "silhouette_loss": float(score),
                "silhouette_iou": float(iou),
                "mask_pixels": int(observed_area),
                "projected_hull_area_px": projected_area,
                "occlusion_bucket": bucket,
                "mask_bbox": [int(x), int(y), int(w), int(h)],
                "component_area_ratio": float(details["area_ratio"]),
            })
            if i in view_frames:
                cv2.imwrite(str(OUT / f"mask_{name}_{i:04d}.png"), mask)
                cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
                if len(hull):
                    cv2.polylines(overlay, [hull.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 1)
                cv2.putText(overlay, f"{name} IoU {iou:.2f}", (8, 18 + (18 if name == "beige_cup" else 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if i in view_frames:
            cv2.imwrite(str(OUT / f"overlay_{i:04d}.jpg"), overlay)
        if i % 25 == 0:
            print(f"frame {i}/{len(frames)}", flush=True)

    gt = pd.read_parquet(ROOT / "video_to_data_challenge/track_3/public/data/chunk-000/episode_000011.parquet")
    gt_objects = {name: [] for name in NAMES}
    for entries in gt["observation.objects"]:
        lookup = {obj["name"]: obj for obj in entries}
        for name in NAMES:
            gt_objects[name].append(lookup[name])
    by_key = {(row["frame"], row["object"]): row for row in observations}
    gt_blue0 = gt_transform(gt_objects["blue_cup"][0])
    rows = []
    for i in range(len(frames)):
        blue = by_key.get((i, "blue_cup"))
        if blue is None or not blue.get("success"):
            continue
        world_T_cam = gt_blue0 @ np.linalg.inv(np.asarray(blue["camera_T_object"], dtype=float))
        for name in NAMES:
            image_row = by_key.get((i, name))
            gt_row = gt_objects[name][i]
            if not gt_row["visible"]:
                continue
            if image_row is None or not image_row.get("success"):
                rows.append({"frame": i, "object": name, "visible": True, "success": False, "occlusion_bucket": "failure" if image_row is None else image_row.get("occlusion_bucket", "failure")})
                continue
            pred = world_T_cam @ np.asarray(image_row["camera_T_object"], dtype=float)
            actual = gt_transform(gt_row)
            baseline = gt_transform(gt_objects[name][0])
            rows.append({
                "frame": i,
                "object": name,
                "visible": True,
                "success": True,
                "occlusion_bucket": image_row["occlusion_bucket"],
                "translation_error_m": float(np.linalg.norm(pred[:3, 3] - actual[:3, 3])),
                "rotation_error_deg": float(np.degrees(Rotation.from_matrix(pred[:3, :3].T @ actual[:3, :3]).magnitude())),
                "baseline_translation_error_m": float(np.linalg.norm(baseline[:3, 3] - actual[:3, 3])),
                "baseline_rotation_error_deg": float(np.degrees(Rotation.from_matrix(baseline[:3, :3].T @ actual[:3, :3]).magnitude())),
                "silhouette_loss": image_row["silhouette_loss"],
                "silhouette_iou": image_row["silhouette_iou"],
                "calibration_frame": i == 0,
            })
    errors = pd.DataFrame(rows)
    errors.to_csv(OUT / "errors.csv", index=False)
    (OUT / "observations.json").write_text(json.dumps(observations, indent=2))
    metrics = {}
    for name in NAMES:
        metrics[name] = {}
        for label, bucket in [("all_successful", None), ("mask_reliable", "mask_reliable"), ("hand_or_partial", "hand_or_partial")]:
            sub = errors[(errors["object"] == name) & (errors["success"] == True) & (errors["calibration_frame"] != True)]
            if bucket is not None:
                sub = sub[sub["occlusion_bucket"] == bucket]
            metrics[name][label] = {"frames": int(len(sub))}
            for col in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "silhouette_loss", "silhouette_iou"]:
                metrics[name][label][col] = float(sub[col].mean()) if len(sub) else None
    summary = {
        "method": "M4 known CAD direct silhouette IoU search + PnP refinement",
        "status": "real CPU camera-space pose tracking; world alignment uses frame-0 blue_cup mocap pose",
        "episode": "track_3 episode_000011, ego_cam_a, 137 frames",
        "frames": len(frames),
        "mesh_vertices_sampled": {name: int(len(vertices[name])) for name in NAMES},
        "metrics": metrics,
        "wall_time_s": time.perf_counter() - t0,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "camera_alignment": "each-frame blue cup CAD camera pose is aligned to blue cup GT at frame 0; this estimates full camera translation and rotation",
        "baseline": "constant first-frame mocap pose; frame 0 excluded from means",
        "limitations": [
            "color masks are not manual or learned segmentation and can include table or hand pixels",
            "the blue cup is assumed static; small mocap drift is ignored",
            "the beige and blue cup surfaces are approximately axisymmetric; axial orientation is weakly observable from silhouette",
            "mask quality and silhouette IoU are image-space evidence, not a guarantee of accurate 6DoF pose",
            "GT is used for one frame-0 blue cup world alignment only; all per-frame image pose estimates are completed before GT evaluation",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
