"""Track known cup CAD models using CSRT boxes, GrabCut masks, and CPU PnP.

This experiment uses the OpenCV-contrib build in the v2d environment.  It is a
new run and writes to cup_grabcut_tracking without replacing prior attempts.
"""

from pathlib import Path
import json
import resource
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/v2/m4/cup_grabcut_tracking"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["blue_cup", "beige_cup"]


def calibration():
    import json as json_module
    raw = json_module.loads((ROOT / "video_to_data_challenge/track_3/public/meta/camera_calibration.json").read_text())["cameras"]["ego_cam_a"]
    k0 = np.asarray(raw["intrinsic_matrix"], dtype=float)
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
    k, dist = calibration()
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        frames.append(cv2.undistort(frame, k, dist))
    cap.release()
    return frames


def load_mesh(name):
    path = ROOT / "outputs/v2/m4" / f"{name}_mesh.npz"
    vertices = np.load(path)["vertices"].astype(np.float64)
    return vertices[np.linspace(0, len(vertices) - 1, 2600, dtype=int)]


def color_seed(crop, name):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if name == "blue_cup":
        return (h >= 92) & (h <= 140) & (s >= 45) & (v >= 30)
    return (h >= 8) & (h <= 35) & (s <= 145) & (v >= 100)


def grabcut_mask(frame, box, name):
    x, y, w, h = [int(round(v)) for v in box]
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    crop = frame[y0:y1, x0:x1]
    mask = np.full(crop.shape[:2], cv2.GC_PR_FGD, dtype=np.uint8)
    mask[:3, :] = cv2.GC_BGD
    mask[-3:, :] = cv2.GC_BGD
    mask[:, :3] = cv2.GC_BGD
    mask[:, -3:] = cv2.GC_BGD
    seed = color_seed(crop, name)
    if int(seed.sum()) >= 24:
        mask[seed] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(crop, mask, None, bg_model, fg_model, 3, cv2.GC_INIT_WITH_MASK)
    local = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    selected = np.zeros_like(local)
    cv2.drawContours(selected, [largest], -1, 255, thickness=cv2.FILLED)
    full = np.zeros(frame.shape[:2], dtype=np.uint8)
    full[y0:y1, x0:x1] = selected
    contour = largest.reshape(-1, 2).astype(np.float64) + np.array([x0, y0], dtype=np.float64)
    return full, contour


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
    return 1.0 - iou, projected[ids], iou


def perturb(pose, axis, delta):
    candidate = pose.copy()
    if axis < 3:
        rot = cv2.Rodrigues(pose[:3])[0]
        step = np.eye(3)
        angle = np.radians(delta)
        if axis == 0:
            step = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        elif axis == 1:
            step = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        else:
            step = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        candidate[:3] = cv2.Rodrigues(rot @ step)[0].ravel()
    else:
        candidate[axis] += delta
    return candidate


def solve_pose(vertices, contour, mask, k, previous=None):
    x, y, w, h = cv2.boundingRect(contour.astype(np.float32))
    extent = float(vertices[:, 2].max() - vertices[:, 2].min())
    z0 = float(k[0, 0] * extent / max(10.0, h))
    center = np.array([x + w / 2.0, y + h / 2.0])
    t0 = np.array([(center[0] - k[0, 2]) * z0 / k[0, 0], (center[1] - k[1, 2]) * z0 / k[1, 1], z0])
    if previous is None:
        seeds = []
        for rx in [0, 45, 90, 135, 180, 225, 270, 315]:
            for ry in [0, 90, 180, 270]:
                for rz in [0, 90]:
                    r = np.eye(3)
                    for axis, angle in [(0, rx), (1, ry), (2, rz)]:
                        d = np.radians(angle)
                        if axis == 0:
                            r = r @ np.array([[1,0,0],[0,np.cos(d),-np.sin(d)],[0,np.sin(d),np.cos(d)]])
                        elif axis == 1:
                            r = r @ np.array([[np.cos(d),0,np.sin(d)],[0,1,0],[-np.sin(d),0,np.cos(d)]])
                        else:
                            r = r @ np.array([[np.cos(d),-np.sin(d),0],[np.sin(d),np.cos(d),0],[0,0,1]])
                    seeds.append(np.r_[cv2.Rodrigues(r)[0].ravel(), t0])
    else:
        seeds = [previous.copy()]
    best = None
    for seed in seeds:
        score, hull, iou = render_score(vertices, seed, mask, k)
        if best is None or score < best[0]:
            best = [score, seed, hull, iou]
    for rot_step, xy_step, z_step in [(8.0, 0.03, 0.08), (3.0, 0.01, 0.025), (1.0, 0.003, 0.008)]:
        improved = True
        rounds = 0
        while improved and rounds < 2:
            improved = False
            rounds += 1
            for axis in range(6):
                delta = xy_step if axis in [3, 4] else z_step if axis == 5 else rot_step
                for sign in [-1.0, 1.0]:
                    candidate = perturb(best[1], axis, sign * delta)
                    if previous is not None and np.linalg.norm(candidate[3:] - previous[3:]) > 0.25:
                        continue
                    score, hull, iou = render_score(vertices, candidate, mask, k)
                    if score + 1e-5 < best[0]:
                        best = [score, candidate, hull, iou]
                        improved = True
    projected = cv2.projectPoints(vertices, best[1][:3], best[1][3:], k, None)[0].reshape(-1, 2)
    hull_ids = cv2.convexHull(projected.astype(np.float32), returnPoints=False).ravel()
    target_tree = cv2.flann.Index(contour.astype(np.float32), dict(algorithm=1, trees=2)) if len(contour) >= 6 else None
    if target_tree is not None and len(hull_ids) >= 6:
        try:
            nearest, _ = target_tree.knnSearch(projected[hull_ids].astype(np.float32), 1, params={"checks": 32})
            image_points = contour[nearest.ravel().astype(int)]
            ok, rv, tv = cv2.solvePnP(vertices[hull_ids], image_points, k, None, best[1][:3].reshape(3,1), best[1][3:].reshape(3,1), True, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                candidate = np.r_[rv.ravel(), tv.ravel()]
                score, hull, iou = render_score(vertices, candidate, mask, k)
                if score < best[0] and 0.20 <= candidate[5] <= 2.5:
                    best = [score, candidate, hull, iou]
        except cv2.error:
            pass
    return best[1], best[0], best[2], best[3]


def main():
    t0 = time.perf_counter()
    frames = read_frames()
    k, _ = calibration()
    vertices = {name: load_mesh(name) for name in NAMES}
    boxes = {"blue_cup": (250, 285, 90, 125), "beige_cup": (390, 270, 100, 130)}
    trackers = {name: cv2.legacy.TrackerCSRT_create() for name in NAMES}
    for name in NAMES:
        trackers[name].init(frames[0], boxes[name])
    poses = {}
    observations = []
    overlay_frames = {0, 20, 33, 35, 50, 70, 100, 120, 136}
    tracker_failures = {name: 0 for name in NAMES}
    for i, frame in enumerate(frames):
        overlay = frame.copy()
        for name in NAMES:
            if i == 0:
                box = boxes[name]
                ok = True
            else:
                ok, box = trackers[name].update(frame)
            if not ok:
                tracker_failures[name] += 1
                observations.append({"frame": i, "object": name, "success": False, "failure": "CSRT lost target", "occlusion_bucket": "tracking_failure"})
                continue
            segmented = grabcut_mask(frame, box, name)
            if segmented is None:
                observations.append({"frame": i, "object": name, "success": False, "failure": "GrabCut empty mask", "occlusion_bucket": "segmentation_failure"})
                continue
            mask, contour = segmented
            pose, loss, hull, iou = solve_pose(vertices[name], contour, mask, k, poses.get(name))
            poses[name] = pose
            bx, by, bw, bh = [int(round(v)) for v in box]
            observed = float(np.count_nonzero(mask))
            projected = float(max(1, cv2.contourArea(hull.astype(np.float32)))) if len(hull) else 1.0
            bucket = "hand_or_partial" if observed < 0.55 * projected else "mask_reliable"
            observations.append({
                "frame": i,
                "object": name,
                "success": True,
                "camera_T_object": np.block([[cv2.Rodrigues(pose[:3])[0], pose[3:].reshape(3,1)], [np.zeros((1,3)), np.ones((1,1))]]).tolist(),
                "silhouette_loss": float(loss),
                "silhouette_iou": float(iou),
                "mask_pixels": int(observed),
                "projected_hull_area_px": projected,
                "occlusion_bucket": bucket,
                "tracker_bbox": [float(v) for v in box],
            })
            if i in overlay_frames:
                cv2.rectangle(overlay, (bx, by), (bx + int(bw), by + int(bh)), (255, 255, 0), 1)
                cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
                if len(hull):
                    cv2.polylines(overlay, [hull.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 1)
                cv2.putText(overlay, f"{name} IoU {iou:.2f}", (8, 18 + (18 if name == "beige_cup" else 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imwrite(str(OUT / f"mask_{name}_{i:04d}.png"), mask)
        if i in overlay_frames:
            cv2.imwrite(str(OUT / f"overlay_{i:04d}.jpg"), overlay)
        if i % 25 == 0:
            print(f"frame {i}/{len(frames)}", flush=True)
    summary = {
        "method": "M4 CSRT tracking + GrabCut masks + direct CAD silhouette IoU search",
        "status": "real CPU camera-space CAD pose estimates; GT world alignment/evaluation in separate step",
        "episode": "track_3 episode_000011, ego_cam_a",
        "frames": len(frames),
        "mesh_vertices_sampled": {name: int(len(vertices[name])) for name in NAMES},
        "tracker_failures": tracker_failures,
        "wall_time_s": time.perf_counter() - t0,
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "device": "CPU",
        "segmentation": "per-frame CSRT bounding box and OpenCV GrabCut initialized by color foreground seeds",
        "pose_estimator": "known CAD projected silhouette IoU coordinate search with local PnP refinement",
        "limitations": ["cup axial rotation remains weakly observable from silhouette", "CSRT and GrabCut can include hands during contact", "metric camera pose is scale anchored by the known Track 3 mesh"],
    }
    (OUT / "observations.json").write_text(json.dumps(observations, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
