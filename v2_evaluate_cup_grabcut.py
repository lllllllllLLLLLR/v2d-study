"""Align and score saved cup camera-space estimates against visible Track 3 GT."""

from pathlib import Path
import json
import resource
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from v2_cad_tracking_v2 import gt_transform


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/v2/m4/cup_grabcut_tracking"


def main():
    t0 = time.perf_counter()
    observations = json.loads((OUT / "observations.json").read_text())
    gt = pd.read_parquet(ROOT / "video_to_data_challenge/track_3/public/data/chunk-000/episode_000011.parquet")
    names = ["blue_cup", "beige_cup"]
    gt_objects = {name: [] for name in names}
    for entries in gt["observation.objects"]:
        lookup = {obj["name"]: obj for obj in entries}
        for name in names:
            gt_objects[name].append(lookup[name])
    by_key = {(row["frame"], row["object"]): row for row in observations}
    blue0 = gt_transform(gt_objects["blue_cup"][0])
    rows = []
    for i in range(len(gt)):
        blue = by_key.get((i, "blue_cup"))
        if blue is None or not blue.get("success"):
            continue
        world_T_cam = blue0 @ np.linalg.inv(np.asarray(blue["camera_T_object"], dtype=float))
        for name in names:
            gt_row = gt_objects[name][i]
            if not gt_row["visible"]:
                continue
            image = by_key.get((i, name))
            if image is None or not image.get("success"):
                rows.append({"frame": i, "object": name, "visible": True, "success": False})
                continue
            pred = world_T_cam @ np.asarray(image["camera_T_object"], dtype=float)
            actual = gt_transform(gt_row)
            baseline = gt_transform(gt_objects[name][0])
            rows.append({
                "frame": i,
                "object": name,
                "visible": True,
                "success": True,
                "occlusion_bucket": image["occlusion_bucket"],
                "translation_error_m": float(np.linalg.norm(pred[:3, 3] - actual[:3, 3])),
                "rotation_error_deg": float(np.degrees(Rotation.from_matrix(pred[:3, :3].T @ actual[:3, :3]).magnitude())),
                "baseline_translation_error_m": float(np.linalg.norm(baseline[:3, 3] - actual[:3, 3])),
                "baseline_rotation_error_deg": float(np.degrees(Rotation.from_matrix(baseline[:3, :3].T @ actual[:3, :3]).magnitude())),
                "silhouette_loss": image["silhouette_loss"],
                "silhouette_iou": image["silhouette_iou"],
                "calibration_frame": i == 0,
            })
    errors = pd.DataFrame(rows)
    errors.to_csv(OUT / "errors.csv", index=False)
    metrics = {}
    for name in names:
        metrics[name] = {}
        for label, bucket in [("all_successful", None), ("mask_reliable", "mask_reliable"), ("hand_or_partial", "hand_or_partial")]:
            sub = errors[(errors["object"] == name) & (errors["success"] == True) & (errors["calibration_frame"] != True)]
            if bucket is not None:
                sub = sub[sub["occlusion_bucket"] == bucket]
            metrics[name][label] = {"frames": int(len(sub))}
            for col in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "silhouette_loss", "silhouette_iou"]:
                metrics[name][label][col] = float(sub[col].mean()) if len(sub) else None
    summary = json.loads((OUT / "summary.json").read_text())
    summary["world_evaluation"] = {
        "alignment": "per-frame blue cup image pose aligned to frame-0 blue cup mocap GT; blue cup is assumed static",
        "visible_rows_only": True,
        "frame0_excluded_from_reported_means": True,
        "baseline": "constant first-frame mocap pose",
        "metrics": metrics,
        "world_evaluation_runtime_s": time.perf_counter() - t0,
    }
    summary["wall_time_s"] += summary["world_evaluation"]["world_evaluation_runtime_s"]
    summary["process_peak_rss_mb"] = max(summary["process_peak_rss_mb"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "world_predictions.csv").write_text(errors.to_csv(index=False))
    print(json.dumps(summary["world_evaluation"], indent=2))


if __name__ == "__main__":
    main()
