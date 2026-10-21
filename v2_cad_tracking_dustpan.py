"""Track 3 mini_dust_pan known-CAD silhouette PnP on CPU.

The cup sequence is retained as a failure case.  This second Track 3 sequence
has a textured, asymmetric yellow object and tests the same classical pipeline
on a less degenerate silhouette.
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

from v2_cad_tracking_v2 import (
    gt_transform,
    initial_pose,
    load_calibration,
    pose_matrix,
    projected_hull,
    fit_pose,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/v2/m4/dustpan_mask_search"
OUT.mkdir(parents=True, exist_ok=True)


def read_frames():
    path = ROOT / "video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000039.mp4"
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


def component(frame, state, predicted_bbox=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Yellow-green plastic of the dust pan and handle.
    raw = ((h >= 15) & (h <= 48) & (s >= 75) & (v >= 70)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if predicted_bbox is None:
        x, y, bw, bh = state["bbox"]
    else:
        x, y, bw, bh = predicted_bbox
    margin = max(25.0, 0.45 * max(bw, bh))
    x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
    x1, y1 = min(raw.shape[1], int(x + bw + margin)), min(raw.shape[0], int(y + bh + margin))
    roi = np.zeros_like(raw)
    roi[y0:y1, x0:x1] = raw[y0:y1, x0:x1]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(roi)
    prior_area = max(100.0, state["area"])
    prior_center = state["center"]
    choices = []
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < max(100.0, 0.06 * prior_area):
            continue
        bbox = stats[idx, :4].astype(float)
        aspect = bbox[2] / max(1.0, bbox[3])
        if aspect < 0.20 or aspect > 4.0:
            continue
        center = cents[idx]
        score = 2.0 * np.linalg.norm(center - prior_center) / max(25.0, np.hypot(bw, bh))
        score += 0.8 * abs(np.log(max(area, 1.0) / prior_area))
        choices.append((score, idx, area, center, bbox))
    if not choices:
        return None, state, {"reason": "no_yellow_component", "roi": [x0, y0, x1, y1]}
    _, idx, area, center, bbox = min(choices, key=lambda row: row[0])
    mask = (labels == idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, state, {"reason": "no_contour", "roi": [x0, y0, x1, y1]}
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    next_state = {"center": center.astype(float), "bbox": bbox.astype(float), "area": area}
    return mask, next_state, {"contour": contour, "area_ratio": area / prior_area, "roi": [x0, y0, x1, y1]}


def camera_rotations(frames, k):
    sift = cv2.SIFT_create(nfeatures=1800)
    background = np.zeros((480, 640), np.uint8)
    background[:235] = 255
    kp0, des0 = sift.detectAndCompute(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), background)
    matcher = cv2.BFMatcher()
    rotations = [np.eye(3)]
    for frame in frames[1:]:
        kp, des = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), background)
        good = []
        if des is not None and des0 is not None:
            for pair in matcher.knnMatch(des0, des, k=2):
                if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                    good.append(pair[0])
        rcam = np.eye(3)
        if len(good) >= 10:
            p0 = np.float32([kp0[m.queryIdx].pt for m in good])
            p1 = np.float32([kp[m.trainIdx].pt for m in good])
            h, inliers = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            if h is not None:
                candidate = np.linalg.inv(k) @ h @ k
                candidate /= np.cbrt(max(1e-9, np.linalg.det(candidate)))
                u, _, vt = np.linalg.svd(candidate)
                rcam = u @ vt
                if np.linalg.det(rcam) < 0:
                    rcam *= -1.0
        rotations.append(rcam)
    return rotations


def render_score(vertices, pose, mask, k):
    if not np.isfinite(pose).all() or not 0.20 <= float(pose[5]) <= 2.5:
        return 1e6, np.empty((0, 2), dtype=float), 0.0
    projected = cv2.projectPoints(vertices, pose[:3], pose[3:], k, None)[0].reshape(-1, 2)
    ids = cv2.convexHull(projected.astype(np.float32), returnPoints=False).ravel()
    if len(ids) < 6 or not np.isfinite(projected[ids]).all():
        return 1e6, np.empty((0, 2), dtype=float), 0.0
    rendered = np.zeros_like(mask)
    cv2.fillConvexPoly(rendered, projected[ids].astype(np.int32).reshape(-1, 1, 2), 255)
    union = np.count_nonzero((rendered > 0) | (mask > 0))
    inter = np.count_nonzero((rendered > 0) & (mask > 0))
    iou = float(inter / max(1, union))
    yy, xx = np.nonzero(mask > 0)
    if len(xx) == 0:
        return 1e6, projected[ids], iou
    pred = projected[ids]
    target_center = np.array([xx.mean(), yy.mean()])
    pred_center = pred.mean(axis=0)
    center_penalty = np.linalg.norm(pred_center - target_center) / 640.0
    target_area = float(len(xx))
    pred_area = float(max(1, cv2.contourArea(pred.astype(np.float32))))
    area_penalty = abs(np.log(pred_area / target_area))
    return (1.0 - iou) + 0.01 * center_penalty + 0.015 * area_penalty, projected[ids], iou


def perturb(pose, axis, delta):
    candidate = pose.copy()
    if axis < 3:
        rot = Rotation.from_rotvec(pose[:3]) * Rotation.from_euler("xyz", [delta if j == axis else 0.0 for j in range(3)], degrees=True)
        candidate[:3] = rot.as_rotvec()
    else:
        candidate[axis] += delta
    return candidate


def search_pose(vertices, contour, mask, k, previous=None):
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
    extent_z = float(vertices[:, 2].max() - vertices[:, 2].min())
    z0 = float(k[0, 0] * extent_z / max(12.0, h))
    center = np.array([x + w / 2.0, y + h / 2.0])
    t0 = np.array([(center[0] - k[0, 2]) * z0 / k[0, 0], (center[1] - k[1, 2]) * z0 / k[1, 1], z0])
    if previous is None:
        seeds = []
        for rx in [0, 45, 90, 135, 180, 225, 270, 315]:
            for ry in [0, 90, 180, 270]:
                for rz in [0, 90]:
                    seeds.append(np.r_[Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_rotvec(), t0])
        candidates = seeds
    else:
        candidates = [previous.copy()]
    best = None
    for candidate in candidates:
        score, hull, iou = render_score(vertices, candidate, mask, k)
        if best is None or score < best[0]:
            best = [score, candidate, hull, iou]
    # A coarse-to-fine local search uses the real mask and keeps per-frame motion bounded.
    stages = [(10.0, 0.035, 0.10), (4.0, 0.012, 0.035), (1.5, 0.004, 0.012)]
    for rot_step, xy_step, z_step in stages:
        improved = True
        rounds = 0
        while improved and rounds < 3:
            improved = False
            rounds += 1
            for axis in range(6):
                delta = xy_step if axis in [3, 4] else z_step if axis == 5 else rot_step
                for sign in [-1.0, 1.0]:
                    candidate = perturb(best[1], axis, sign * delta)
                    if previous is not None and np.linalg.norm(candidate[3:] - previous[3:]) > 0.30:
                        continue
                    score, hull, iou = render_score(vertices, candidate, mask, k)
                    if score + 1e-5 < best[0]:
                        best = [score, candidate, hull, iou]
                        improved = True
    # PnP is a final local refinement from the mask-optimized hypothesis.
    refined, pnp_score, pnp_hull, pnp_iou = fit_pose(vertices, contour, mask, k, best[1], previous)
    if pnp_score < best[0] and np.isfinite(refined).all() and 0.20 <= refined[5] <= 2.5:
        return refined, pnp_score, pnp_hull, pnp_iou
    return best[1], best[0], best[2], best[3]


def main():
    t0 = time.perf_counter()
    frames = read_frames()
    k, _ = load_calibration()
    vertices = np.load(ROOT / "outputs/v2/m4/mini_dust_pan_mesh.npz")["vertices"].astype(np.float64)
    vertices = vertices[np.linspace(0, len(vertices) - 1, 3200, dtype=int)]
    state = {"center": np.array([301.0, 289.0]), "bbox": np.array([263.0, 241.0, 99.0, 97.0]), "area": 4569.0}
    rotations = camera_rotations(frames, k)
    pose = None
    observations = []
    overlays = {0, 80, 160, 240, 318}
    for i, frame in enumerate(frames):
        predicted_bbox = None
        if pose is not None:
            projected, ids = projected_hull(vertices, pose, k)
            xy = projected[ids]
            predicted_bbox = np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].ptp(), xy[:, 1].ptp()])
        mask, state_next, details = component(frame, state, predicted_bbox)
        overlay = frame.copy()
        if mask is None:
            observations.append({"frame": i, "success": False, "failure": details["reason"], "occlusion_bucket": "hand_or_segmentation_failure"})
            if i in overlays:
                cv2.imwrite(str(OUT / f"overlay_{i:04d}.jpg"), overlay)
            continue
        state = state_next
        contour = details["contour"]
        pose, score, hull, iou = search_pose(vertices, contour, mask, k, pose)
        x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
        projected_area = max(1.0, float(cv2.contourArea(hull.astype(np.float32)))) if len(hull) else 1.0
        observed_area = float(np.count_nonzero(mask))
        bucket = "hand_or_partial" if observed_area < 0.55 * projected_area else "mask_reliable"
        observations.append({
            "frame": i,
            "success": True,
            "camera_T_object": pose_matrix(pose[:3], pose[3:]).tolist(),
            "contour_median_error_px": float(score),
            "silhouette_iou": float(iou),
            "mask_pixels": int(observed_area),
            "projected_hull_area_px": projected_area,
            "occlusion_bucket": bucket,
            "mask_bbox": [int(x), int(y), int(w), int(h)],
        })
        cv2.imwrite(str(OUT / f"mask_mini_dust_pan_{i:04d}.png"), mask)
        if i in overlays:
            cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
            if len(hull):
                cv2.polylines(overlay, [hull.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 1)
            cv2.putText(overlay, f"dustpan: IoU {iou:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(OUT / f"overlay_{i:04d}.jpg"), overlay)
        if i % 50 == 0:
            print(f"frame {i}/{len(frames)}", flush=True)

    gt = pd.read_parquet(ROOT / "video_to_data_challenge/track_3/public/data/chunk-000/episode_000039.parquet")
    gt_obj = [entry[0] for entry in gt["observation.objects"]]
    by_frame = {row["frame"]: row for row in observations}
    first = gt_transform(gt_obj[0])
    first_row = by_frame.get(0)
    if first_row is None or not first_row.get("success"):
        raise RuntimeError("dustpan frame-0 segmentation did not initialize")
    world_T_cam0 = first @ np.linalg.inv(np.asarray(first_row["camera_T_object"], dtype=float))
    errors = []
    for i, obj in enumerate(gt_obj):
        row = by_frame.get(i)
        actual = gt_transform(obj)
        if row is None or not row.get("success"):
            errors.append({"frame": i, "success": False, "visible": bool(obj["visible"]), "occlusion_bucket": "failure"})
            continue
        cam_T_obj = np.asarray(row["camera_T_object"], dtype=float)
        cam_motion = np.eye(4)
        cam_motion[:3, :3] = np.linalg.inv(rotations[i])
        world_T_cam = world_T_cam0 @ cam_motion
        pred = world_T_cam @ cam_T_obj
        baseline = first
        errors.append({
            "frame": i,
            "success": True,
            "visible": bool(obj["visible"]),
            "occlusion_bucket": row["occlusion_bucket"],
            "translation_error_m": float(np.linalg.norm(pred[:3, 3] - actual[:3, 3])),
            "rotation_error_deg": float(np.degrees(Rotation.from_matrix(pred[:3, :3].T @ actual[:3, :3]).magnitude())),
            "baseline_translation_error_m": float(np.linalg.norm(baseline[:3, 3] - actual[:3, 3])),
            "baseline_rotation_error_deg": float(np.degrees(Rotation.from_matrix(baseline[:3, :3].T @ actual[:3, :3]).magnitude())),
            "contour_error_px": row["contour_median_error_px"],
            "silhouette_iou": row["silhouette_iou"],
            "calibration_frame": i == 0,
        })
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(OUT / "errors.csv", index=False)
    (OUT / "observations.json").write_text(json.dumps(observations, indent=2))
    eval_df = errors_df[(errors_df["success"] == True) & (errors_df["calibration_frame"] != True)]
    metrics = {"all_successful": {"frames": int(len(eval_df))}}
    for col in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "contour_error_px", "silhouette_iou"]:
        metrics["all_successful"][col] = float(eval_df[col].mean()) if len(eval_df) else None
    for bucket in ["mask_reliable", "hand_or_partial"]:
        sub = eval_df[eval_df["occlusion_bucket"] == bucket]
        metrics[bucket] = {"frames": int(len(sub))}
        for col in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "contour_error_px", "silhouette_iou"]:
            metrics[bucket][col] = float(sub[col].mean()) if len(sub) else None
    summary = {
        "method": "M4 known CAD constrained silhouette PnP, mini_dust_pan",
        "status": "real CPU image tracking; metric world evaluation after frame-0 GT alignment",
        "episode": "track_3 episode_000039, ego_cam_a",
        "frames": len(frames),
        "mesh_vertices_sampled": int(len(vertices)),
        "metrics": metrics,
        "wall_time_s": time.perf_counter() - t0,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "camera_alignment": "frame-0 GT alignment plus background SIFT homography rotation; translation is unobserved",
        "baseline": "constant frame-0 mocap pose; frame 0 excluded from means",
        "occlusion_bucket": "heuristic: observed mask area below 55% of projected CAD hull is hand_or_partial",
        "limitations": [
            "yellow color mask is classical and can include table reflections or the hand",
            "camera translation is not recovered because no global ego-to-mocap extrinsic is public",
            "background homography is used only for rotation, so world translation errors include camera-motion bias",
            "first-frame alignment uses public GT only for the one permitted calibration frame",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
