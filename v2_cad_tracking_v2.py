"""CPU-only known-CAD tracking with constrained color masks and silhouette PnP.

This is an independent v2 run.  The earlier cup_pnp directory is intentionally
left untouched so that the failure mode remains auditable.
"""

from pathlib import Path
import json
import resource
import time

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/v2/m4/cup_pnp_v3"
OUT.mkdir(parents=True, exist_ok=True)


def pose_matrix(rvec, tvec):
    out = np.eye(4, dtype=float)
    out[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    out[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return out


def load_calibration():
    raw = json.loads(
        (ROOT / "video_to_data_challenge/track_3/public/meta/camera_calibration.json").read_text()
    )["cameras"]["ego_cam_a"]
    k0 = np.asarray(raw["intrinsic_matrix"], dtype=float)
    # The stored ego frames are rotated by 180 degrees relative to factory calibration.
    k = k0.copy()
    k[0, 2] = 2027.0 - k0[0, 2]
    k[1, 2] = 1519.0 - k0[1, 2]
    k[0, :] *= 640.0 / 2028.0
    k[1, :] *= 480.0 / 1520.0
    dist = np.asarray(raw["distortion_coefficients"], dtype=float)
    dist[2:4] *= -1.0
    return k, dist


def read_frames():
    path = ROOT / "video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        frames.append(cv2.undistort(frame, *load_calibration()))
    cap.release()
    return frames


def mesh_vertices(name):
    path = ROOT / "outputs/v2/m4" / f"{name}_mesh.npz"
    vertices = np.load(path)["vertices"].astype(np.float64)
    # Deterministic sampling keeps all runs below the 16 GB CPU limit.
    return vertices[np.linspace(0, len(vertices) - 1, 2400, dtype=int)]


def initial_state(name):
    # Approximate frame-0 locations are used only to bootstrap the image tracker.
    if name == "blue_cup":
        return {"center": np.array([292.0, 350.0]), "bbox": np.array([250.0, 285.0, 90.0, 125.0]), "area": 6000.0}
    return {"center": np.array([438.0, 335.0]), "bbox": np.array([390.0, 275.0, 96.0, 125.0]), "area": 8750.0}


def color_mask(frame, name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if name == "blue_cup":
        raw = ((h >= 92) & (h <= 140) & (s >= 55) & (v >= 35)).astype(np.uint8) * 255
    else:
        raw = ((h >= 14) & (h <= 40) & (s <= 105) & (v > 95)).astype(np.uint8) * 255
    return cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def component_mask(frame, name, state, predicted_bbox=None):
    full = color_mask(frame, name)
    h, w = full.shape
    if predicted_bbox is None:
        x, y, bw, bh = state["bbox"]
    else:
        x, y, bw, bh = predicted_bbox
    margin = max(20.0, 0.30 * max(bw, bh))
    x0 = max(0, int(x - margin))
    y0 = max(0, int(y - margin))
    x1 = min(w, int(x + bw + margin))
    y1 = min(h, int(y + bh + margin))
    roi = np.zeros_like(full)
    roi[y0:y1, x0:x1] = full[y0:y1, x0:x1]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
    choices = []
    prior_area = max(100.0, float(state["area"]))
    prior_center = np.asarray(state["center"], dtype=float)
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < max(80.0, 0.035 * prior_area):
            continue
        center = centroids[idx]
        bbox = stats[idx, :4].astype(float)
        aspect = bbox[2] / max(1.0, bbox[3])
        if aspect < 0.30 or aspect > 1.50:
            continue
        area_ratio = abs(np.log(max(area, 1.0) / prior_area))
        dist = np.linalg.norm(center - prior_center) / max(20.0, np.hypot(bw, bh))
        # Favor a component with a plausible area and temporal location.
        score = 2.2 * dist + 0.9 * area_ratio + 0.35 * abs(np.log(max(aspect, 1e-3) / 0.65))
        if name == "beige_cup" and bbox[1] < 100:
            score += 0.5
        choices.append((score, idx, area, center, bbox))
    if not choices:
        return None, state, {"reason": "no_component", "roi": [x0, y0, x1, y1]}
    _, idx, area, center, bbox = min(choices, key=lambda row: row[0])
    selected = (labels == idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, state, {"reason": "no_contour", "roi": [x0, y0, x1, y1]}
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    next_state = {"center": center.astype(float), "bbox": bbox.astype(float), "area": area}
    reliability = {
        "reason": "component",
        "roi": [x0, y0, x1, y1],
        "component_area": area,
        "area_ratio": area / prior_area,
    }
    return selected, next_state, {"contour": contour, **reliability}


def projected_hull(vertices, pose, k):
    projected = cv2.projectPoints(vertices, pose[:3], pose[3:], k, None)[0].reshape(-1, 2)
    ids = cv2.convexHull(projected.astype(np.float32), returnPoints=False).ravel()
    return projected, ids


def silhouette_score(vertices, pose, contour, mask, k):
    projected, ids = projected_hull(vertices, pose, k)
    if len(ids) < 6:
        return 1e6, np.empty((0, 2), dtype=float), 0.0
    polygon = projected[ids].astype(np.int32).reshape(-1, 1, 2)
    rendered = np.zeros_like(mask)
    cv2.fillConvexPoly(rendered, polygon, 255)
    union = np.count_nonzero((rendered > 0) | (mask > 0))
    inter = np.count_nonzero((rendered > 0) & (mask > 0))
    iou = float(inter / max(1, union))
    tree = cKDTree(contour)
    distances = tree.query(projected[ids])[0]
    # IoU dominates; contour distance breaks ties between similarly sized poses.
    score = (1.0 - iou) + 0.004 * float(np.median(distances))
    return score, projected[ids], iou


def valid_pose(pose, previous=None):
    z = float(pose[5])
    if not np.isfinite(pose).all() or not 0.20 <= z <= 2.5:
        return False
    if previous is not None:
        if np.linalg.norm(pose[3:] - previous[3:]) > 0.35:
            return False
    return True


def fit_pose(vertices, contour, mask, k, initial, previous=None):
    current = initial.copy()
    best_score, hull, best_iou = silhouette_score(vertices, current, contour, mask, k)
    tree = cKDTree(contour)
    for _ in range(8):
        projected, ids = projected_hull(vertices, current, k)
        if len(ids) < 8:
            break
        distances, nearest = tree.query(projected[ids])
        threshold = max(7.0, float(np.percentile(distances, 72)))
        keep = distances <= threshold
        if np.count_nonzero(keep) < 8:
            break
        try:
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                vertices[ids[keep]],
                contour[nearest[keep]],
                k,
                None,
                current[:3].copy().reshape(3, 1),
                current[3:].copy().reshape(3, 1),
                useExtrinsicGuess=True,
                iterationsCount=80,
                reprojectionError=8.0,
                confidence=0.90,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            break
        if not ok:
            break
        candidate = np.r_[rv.ravel(), tv.ravel()]
        if not valid_pose(candidate, previous):
            break
        score, candidate_hull, candidate_iou = silhouette_score(vertices, candidate, contour, mask, k)
        if score + 1e-4 >= best_score:
            break
        current, best_score, hull, best_iou = candidate, score, candidate_hull, candidate_iou
    return current, best_score, hull, best_iou


def initial_pose(vertices, contour, k):
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
    extent = float(vertices[:, 2].max() - vertices[:, 2].min())
    z = float(k[0, 0] * extent / max(10.0, h))
    center = np.array([x + w / 2.0, y + h / 2.0])
    t = np.array([(center[0] - k[0, 2]) * z / k[0, 0], (center[1] - k[1, 2]) * z / k[1, 1], z])
    candidates = []
    for deg in [0.0, 45.0, 65.0, 90.0, 135.0, 180.0]:
        rv = Rotation.from_euler("x", deg, degrees=True).as_rotvec()
        candidates.append(np.r_[rv, t])
    # A small model-scale perturbation handles partial top rims at bootstrap.
    for scale in [0.82, 1.18]:
        candidates.append(np.r_[Rotation.from_euler("x", 65, degrees=True).as_rotvec(), t * scale])
    best = None
    for candidate in candidates:
        score, hull, iou = silhouette_score(vertices, candidate, contour, np.zeros((480, 640), np.uint8), k)
        # Bootstrap scoring is contour-only because the first color component may be partial.
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1]


def gt_transform(obj):
    p = np.asarray(obj["pose"], dtype=float)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    out[:3, 3] = p[:3]
    return out


def main():
    t0 = time.perf_counter()
    frames = read_frames()
    k, _ = load_calibration()
    vertices = {name: mesh_vertices(name) for name in ["blue_cup", "beige_cup"]}
    states = {name: initial_state(name) for name in vertices}
    poses = {}
    observations = []
    overlays = {0, 35, 70, 100, 136}
    for frame_idx, frame in enumerate(frames):
        overlay = frame.copy()
        for name in ["blue_cup", "beige_cup"]:
            previous = poses.get(name)
            predicted_bbox = None
            if previous is not None:
                _, ids = projected_hull(vertices[name], previous, k)
                pp = cv2.projectPoints(vertices[name], previous[:3], previous[3:], k, None)[0].reshape(-1, 2)[ids]
                x0, y0 = pp.min(axis=0)
                x1, y1 = pp.max(axis=0)
                predicted_bbox = np.array([x0, y0, x1 - x0, y1 - y0])
            mask, next_state, details = component_mask(frame, name, states[name], predicted_bbox)
            if mask is None:
                observations.append({"frame": frame_idx, "object": name, "success": False, "failure": details["reason"], "occlusion_bucket": "hand_or_segmentation_failure"})
                continue
            states[name] = next_state
            contour = details["contour"]
            if previous is None:
                seed = initial_pose(vertices[name], contour, k)
            else:
                seed = previous
            pose, score, hull, iou = fit_pose(vertices[name], contour, mask, k, seed, previous)
            if previous is not None and score > 0.95:
                # Carrying the last well-constrained pose is safer than accepting a floor/hand jump.
                pose = previous.copy()
            poses[name] = pose
            x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
            projected_area = max(1.0, float(cv2.contourArea(hull.astype(np.float32)))) if len(hull) else 1.0
            observed_area = float(np.count_nonzero(mask))
            occlusion = "hand_or_partial" if observed_area < 0.55 * projected_area else "mask_reliable"
            row = {
                "frame": frame_idx,
                "object": name,
                "success": True,
                "camera_T_object": pose_matrix(pose[:3], pose[3:]).tolist(),
                "contour_median_error_px": float(score),
                "silhouette_iou": float(iou),
                "mask_pixels": int(observed_area),
                "projected_hull_area_px": projected_area,
                "occlusion_bucket": occlusion,
                "mask_bbox": [int(x), int(y), int(w), int(h)],
                "component_area_ratio": float(details["area_ratio"]),
            }
            observations.append(row)
            cv2.imwrite(str(OUT / f"mask_{name}_{frame_idx:04d}.png"), mask)
            if frame_idx in overlays:
                cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
                if len(hull):
                    cv2.polylines(overlay, [hull.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 1)
                cv2.putText(overlay, f"{name}: IoU {iou:.2f}", (8, 20 + (20 if name == "beige_cup" else 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        if frame_idx in overlays:
            cv2.imwrite(str(OUT / f"overlay_{frame_idx:04d}.jpg"), overlay)
        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}/{len(frames)}", flush=True)

    gt = pd.read_parquet(ROOT / "video_to_data_challenge/track_3/public/data/chunk-000/episode_000011.parquet")
    gt_obj = {}
    for entries in gt["observation.objects"]:
        for obj in entries:
            gt_obj.setdefault(obj["name"], []).append(obj)
    by_key = {(row["frame"], row["object"]): row for row in observations}
    blue0 = gt_transform(gt_obj["blue_cup"][0])
    rows = []
    for frame_idx in range(len(frames)):
        blue = by_key.get((frame_idx, "blue_cup"))
        if blue is None or not blue.get("success"):
            continue
        cam_T_blue = np.asarray(blue["camera_T_object"], dtype=float)
        world_T_cam = blue0 @ np.linalg.inv(cam_T_blue)
        for name in ["blue_cup", "beige_cup"]:
            row = by_key.get((frame_idx, name))
            if row is None:
                continue
            gt_row = gt_obj[name][frame_idx]
            actual = gt_transform(gt_row)
            if not row.get("success") or not gt_row["visible"]:
                rows.append({"frame": frame_idx, "object": name, "visible": bool(gt_row["visible"]), "success": False, "occlusion_bucket": row.get("occlusion_bucket", "failure")})
                continue
            pred = world_T_cam @ np.asarray(row["camera_T_object"], dtype=float)
            baseline = gt_transform(gt_obj[name][0])
            te = np.linalg.norm(pred[:3, 3] - actual[:3, 3])
            re = np.degrees(Rotation.from_matrix(pred[:3, :3].T @ actual[:3, :3]).magnitude())
            bt = np.linalg.norm(baseline[:3, 3] - actual[:3, 3])
            br = np.degrees(Rotation.from_matrix(baseline[:3, :3].T @ actual[:3, :3]).magnitude())
            rows.append({
                "frame": frame_idx,
                "object": name,
                "visible": bool(gt_row["visible"]),
                "success": True,
                "occlusion_bucket": row["occlusion_bucket"],
                "translation_error_m": float(te),
                "rotation_error_deg": float(re),
                "baseline_translation_error_m": float(bt),
                "baseline_rotation_error_deg": float(br),
                "contour_error_px": float(row["contour_median_error_px"]),
                "silhouette_iou": float(row["silhouette_iou"]),
                "calibration_frame": frame_idx == 0,
            })
    errors = pd.DataFrame(rows)
    errors.to_csv(OUT / "errors.csv", index=False)
    (OUT / "observations.json").write_text(json.dumps(observations, indent=2))

    metrics = {}
    for name in ["blue_cup", "beige_cup"]:
        metrics[name] = {}
        for bucket in ["all_reliable", "hand_or_partial"]:
            subset = errors[(errors["object"] == name) & errors["success"] & (~errors["calibration_frame"]) ]
            if bucket == "all_reliable":
                subset = subset[subset["occlusion_bucket"] == "mask_reliable"]
            else:
                subset = subset[subset["occlusion_bucket"] == "hand_or_partial"]
            metrics[name][bucket] = {"frames": int(len(subset))}
            for col in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "contour_error_px", "silhouette_iou"]:
                metrics[name][bucket][col] = float(subset[col].mean()) if len(subset) else None
        subset = errors[(errors["object"] == name) & errors["success"] & (~errors["calibration_frame"]) ]
        metrics[name]["all_successful"] = {"frames": int(len(subset))}
        for col in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "contour_error_px", "silhouette_iou"]:
            metrics[name]["all_successful"][col] = float(subset[col].mean()) if len(subset) else None
    summary = {
        "method": "M4 known CAD constrained silhouette PnP v2",
        "status": "real CPU image tracking; metric world evaluation after frame-0 GT alignment",
        "episode": "track_3 episode_000011, ego_cam_a",
        "frames": len(frames),
        "mesh_vertices_sampled": {name: int(len(vertices[name])) for name in vertices},
        "metrics": metrics,
        "wall_time_s": time.perf_counter() - t0,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "baseline": "constant first-frame mocap pose; first frame excluded from means",
        "camera_alignment": "static blue_cup frame-0 mocap pose plus each-frame blue CAD pose estimates; this estimates camera translation and rotation",
        "occlusion_bucket": "heuristic: observed mask area below 55% of projected CAD hull is hand_or_partial",
        "limitations": [
            "segmentation is classical color plus temporal ROI, not an annotated mask",
            "cup symmetry makes rotation around the cup axis weakly observable",
            "the global ego-to-mocap transform is calibrated from frame 0 only",
            "blue cup is treated as a static camera-motion reference; its small mocap drift is ignored",
            "visible=true in the parquet does not mean the full silhouette is unoccluded; the heuristic bucket is reported separately",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
