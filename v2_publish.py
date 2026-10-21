"""Create additive v2 reports, plots, executed notebook, and manifests."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "outputs/v2"
REPORT = V2 / "report"
FIGS = V2 / "figs"
NOTEBOOKS = V2 / "notebooks"
for path in [REPORT, FIGS, NOTEBOOKS, V2 / "m3/attempt", V2 / "cache/matplotlib"]:
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(V2 / "cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from IPython.display import Image, display
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_m3_log():
    paths = {
        "triposr_model_ckpt": V2 / "m3/triposr/model.ckpt",
        "instantmesh_large_ckpt": V2 / "m3/instantmesh/instant_mesh_large.ckpt",
        "instantmesh_diffusion_weights": V2 / "m3/instantmesh/diffusion_pytorch_model.bin",
    }
    modules = ["torch", "transformers", "diffusers", "omegaconf", "einops", "pytorch_lightning", "rembg", "kiui", "trimesh", "torchmcubes", "nvdiffrast", "xformers"]
    deps = {name: importlib.util.find_spec(name) is not None for name in modules}
    checkpoints = {name: {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None} for name, path in paths.items()}
    log = {
        "method": "M3 TripoSR CPU inference attempt",
        "status": "blocked_before_inference; no generated mesh claimed",
        "device": "CPU only",
        "local_checkpoints": checkpoints,
        "required_dependency_probe": deps,
        "inference_runtime_s": None,
        "generated_mesh": None,
        "token_required": False,
        "failure_reasons": [
            "TripoSR runtime dependencies are missing in the installed Python interpreters, including omegaconf, einops, pytorch_lightning, rembg, diffusers, and the model source package.",
            "Hugging Face and GitHub DNS resolution failed during public metadata and source probes.",
            "The user-authorized pip installation request was rejected before execution because the automatic approval service returned HTTP 503 (no available approval model channel). This is an execution-channel failure, not a token requirement or safety finding.",
            "A checkpoint alone is not an inference runtime; no compatible offline source checkout or dependency wheel cache was found.",
        ],
        "cpu_alternative": "Use Track 3 known-CAD silhouette PnP from M4, or retry TripoSR on CPU after the authorized package/network channel is restored. InstantMesh has a larger runtime and checkpoint footprint.",
        "attempted_at": "2026-10-09",
    }
    write_json(V2 / "m3/attempt/triposr_cpu_attempt.json", log)
    missing = [name for name, present in deps.items() if not present]
    text = [
        "M3 CPU inference attempt",
        "Status: blocked before inference; no generated mesh claimed.",
        f"Missing runtime packages: {', '.join(missing)}",
        "Network probe: huggingface.co and github.com DNS resolution failed.",
        "Authorized pip request: automatic approval service returned HTTP 503 before execution.",
        "No API token was requested or required for the public repositories.",
    ]
    for name, value in checkpoints.items():
        text.append(f"{name}: exists={value['exists']}; bytes={value['bytes']}")
    (V2 / "m3/attempt/triposr_cpu_attempt.log").write_text("\n".join(text) + "\n", encoding="utf-8")
    return log


def make_buckets():
    path = V2 / "m4/cup_grabcut_tracking/errors.csv"
    data = pd.read_csv(path)
    # Frames 28-66 were reviewed directly in the ego video; this is not a parquet label.
    data["video_review_bucket"] = np.where(
        data["object"].eq("beige_cup") & data["frame"].between(28, 66),
        "visible_hand_contact_reviewed",
        "visible_no_direct_hand_contact_reviewed",
    )
    data["bucket_source"] = "manual video review; parquet visible flag is separate"
    data.to_csv(V2 / "m4/cup_grabcut_tracking/interaction_buckets.csv", index=False)
    evaluated = data[(data["success"] == True) & (data["calibration_frame"] != True) & (data["object"] == "beige_cup")]
    groups = []
    for label, subset in evaluated.groupby("video_review_bucket"):
        row = {"video_review_bucket": label, "frames": int(len(subset)), "visible_gt_frames": int(subset["visible"].sum())}
        for metric in ["translation_error_m", "rotation_error_deg", "baseline_translation_error_m", "baseline_rotation_error_deg", "silhouette_iou"]:
            row[f"mean_{metric}"] = float(subset[metric].mean())
        groups.append(row)
    out = {
        "ground_truth_visible_frames": int(data[data["object"] == "beige_cup"]["visible"].sum()),
        "ground_truth_frames": int((data["object"] == "beige_cup").sum()),
        "manual_hand_contact_interval_inclusive": [28, 66],
        "bucket_note": "Manual video review only. Track 3 parquet marks all rows visible here and does not annotate hand occlusion.",
        "groups": groups,
    }
    write_json(V2 / "m4/cup_grabcut_tracking/interaction_bucket_summary.json", out)
    return out


def make_comparison(m3):
    t1 = read_json(V2 / "m1/track1/summary.json")
    t3 = read_json(V2 / "m1/track3/summary.json")
    sfm = read_json(V2 / "m1/track3_ep0/export_summary.json")
    depth = read_json(V2 / "m2/relative_depth_corrected/summary.json")
    old_m4 = read_json(V2 / "m4/cup_pnp/summary.json")
    direct_cup = read_json(V2 / "m4/cup_mask_search/summary.json")
    cup = read_json(V2 / "m4/cup_grabcut_tracking/summary.json")
    dust = read_json(V2 / "m4/dustpan_mask_search/summary.json")
    cup_beige = cup["world_evaluation"]["metrics"]["beige_cup"]["all_successful"]
    dust_all = dust["metrics"]["all_successful"]
    rows = [
        {"method": "M1 pycolmap SIFT SfM", "track": "Track 1 episode 16; 18 images", "status": "failed_real_matching_no_sparse_model", "runtime_s": t1["wall_time_s"], "peak_rss_mb": None, "geometry_count": 0, "geometry_unit": "3D points", "pose_metric": "No registered camera poses", "mean_translation_error_m": None, "mean_rotation_error_deg": None, "scale_status": "Mapper did not initialize", "failure_mode": "No good initial image pair found"},
        {"method": "M1 pycolmap SIFT SfM", "track": "Track 3 episode 11; 18 images", "status": "failed_real_matching_no_sparse_model", "runtime_s": t3["wall_time_s"], "peak_rss_mb": None, "geometry_count": 0, "geometry_unit": "3D points", "pose_metric": "No registered camera poses", "mean_translation_error_m": None, "mean_rotation_error_deg": None, "scale_status": "Mapper did not initialize", "failure_mode": "No good initial image pair found"},
        {"method": "M1 pycolmap SIFT SfM", "track": "Track 3 episode 0; 30 images", "status": "real_sparse_reconstruction", "runtime_s": sfm["wall_time_s"], "peak_rss_mb": None, "geometry_count": sfm["model_2"]["num_points3D"], "geometry_unit": "3D points", "pose_metric": f"30 registered cameras; {sfm['mean_reprojection_error_px']:.3f} px mean reprojection", "mean_translation_error_m": None, "mean_rotation_error_deg": None, "scale_status": "Arbitrary SfM units; not metric", "failure_mode": "Successful; original mapper RSS was not instrumented; export RSS was 76.1 MiB"},
        {"method": "M2 Depth Anything V2 Small", "track": "Track 3 episode 0; 3 frames", "status": "real_cpu_inference_relative_depth", "runtime_s": depth["wall_time_s"], "peak_rss_mb": depth["process_peak_rss_mb"], "geometry_count": sum(row["points"] for row in depth["frames"]), "geometry_unit": "back-projected points", "pose_metric": "19,200 points/frame; no object pose", "mean_translation_error_m": None, "mean_rotation_error_deg": None, "scale_status": "Relative depth; not metric", "failure_mode": "Inference completed; monocular metric scale unavailable"},
        {"method": "M3 TripoSR CPU", "track": "Track 3 representative frame", "status": "failed_before_inference_dependencies_and_network", "runtime_s": None, "peak_rss_mb": None, "geometry_count": None, "geometry_unit": "generated mesh faces", "pose_metric": "No generated mesh", "mean_translation_error_m": None, "mean_rotation_error_deg": None, "scale_status": "N/A", "failure_mode": "Checkpoint exists; dependencies missing and authorized network/install channel failed"},
        {"method": "M4 cup contour ICP/PnP initial", "track": "Track 3 episode 11; 137 frames", "status": "real_estimator_failed_world_pose", "runtime_s": old_m4["wall_time_s"], "peak_rss_mb": old_m4["process_peak_rss_mb"], "geometry_count": 1800, "geometry_unit": "sampled CAD vertices/object", "pose_metric": "Image contour fit; unstable world pose", "mean_translation_error_m": old_m4["metrics"]["beige_cup"]["translation_error_m"], "mean_rotation_error_deg": old_m4["metrics"]["beige_cup"]["rotation_error_deg"], "scale_status": "CAD metric scale; camera model incomplete", "failure_mode": "Loose color mask and unstable correspondence; z diverged"},
        {"method": "M4 cup direct silhouette search", "track": "Track 3 episode 11; 137 frames", "status": "real_camera_pose_estimates_but_no_baseline_gain", "runtime_s": direct_cup["wall_time_s"], "peak_rss_mb": direct_cup["process_peak_rss_mb"], "geometry_count": 2800, "geometry_unit": "sampled CAD vertices/object", "pose_metric": f"Silhouette IoU {direct_cup['metrics']['beige_cup']['all_successful']['silhouette_iou']:.3f}", "mean_translation_error_m": direct_cup["metrics"]["beige_cup"]["all_successful"]["translation_error_m"], "mean_rotation_error_deg": direct_cup["metrics"]["beige_cup"]["all_successful"]["rotation_error_deg"], "scale_status": "CAD metric camera scale; world frame aligned on frame-0 blue cup", "failure_mode": "Cup symmetry leaves 6DoF pose ambiguous; worse than constant baseline"},
        {"method": "M4 cup CSRT + GrabCut + CAD IoU/PnP", "track": "Track 3 episode 11; 137 frames", "status": "real_camera_pose_estimates_but_worse_than_constant_baseline", "runtime_s": cup["wall_time_s"], "peak_rss_mb": cup["process_peak_rss_mb"], "geometry_count": 2600, "geometry_unit": "sampled CAD vertices/object", "pose_metric": f"Silhouette IoU {cup_beige['silhouette_iou']:.3f}; 136 visible frames scored", "mean_translation_error_m": cup_beige["translation_error_m"], "mean_rotation_error_deg": cup_beige["rotation_error_deg"], "scale_status": "CAD metric scale; frame-0 blue-cup world alignment", "failure_mode": f"Mean error {cup_beige['translation_error_m']:.3f} m / {cup_beige['rotation_error_deg']:.1f} deg exceeds constant baseline {cup_beige['baseline_translation_error_m']:.3f} m / {cup_beige['baseline_rotation_error_deg']:.1f} deg"},
        {"method": "M4 dustpan CAD mask search + PnP", "track": "Track 3 episode 39; 319 frames", "status": "real_camera_pose_estimates_but_no_baseline_gain", "runtime_s": dust["wall_time_s"], "peak_rss_mb": dust["process_peak_rss_mb"], "geometry_count": dust["mesh_vertices_sampled"], "geometry_unit": "sampled CAD vertices", "pose_metric": f"Silhouette IoU {dust_all['silhouette_iou']:.3f}; {dust_all['frames']} frames", "mean_translation_error_m": dust_all["translation_error_m"], "mean_rotation_error_deg": dust_all["rotation_error_deg"], "scale_status": "CAD metric scale; camera translation unobserved", "failure_mode": f"World pose {dust_all['translation_error_m']:.3f} m / {dust_all['rotation_error_deg']:.1f} deg; camera motion and orientation remain ambiguous"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "method_comparison_v2.csv", index=False)
    return df


def make_figures():
    from PIL import Image as PILImage

    ply = V2 / "m1/track3_ep0/sparse_points.ply"
    with open(ply, "r", encoding="ascii") as f:
        while f.readline().strip() != "end_header":
            pass
        cloud = np.loadtxt(f)
    camera_poses = read_json(V2 / "m1/track3_ep0/camera_poses.json")
    centers = np.asarray([item["camera_center_sfm_units"] for item in camera_poses])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 2], cmap="viridis", s=7, alpha=0.75)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="#c44e52", marker="o", markersize=3, label="Camera centers")
    ax.set(xlabel="X (SfM units)", ylabel="Y (SfM units)", zlabel="Z (SfM units)", title="M1 sparse cloud and registered cameras; arbitrary scale")
    ax.legend()
    fig.colorbar(p, ax=ax, shrink=0.7, label="Z (SfM units)")
    fig.tight_layout()
    fig.savefig(FIGS / "m1_track3_sparse_cloud.png", dpi=160)
    plt.close(fig)

    rgb = np.asarray(PILImage.open(V2 / "inputs/track3_ep0_ego_a/0000.jpg").convert("RGB"))
    rel_depth = np.load(V2 / "m2/relative_depth_corrected/0000_relative_depth.npy")
    pc = np.load(V2 / "m2/relative_depth_corrected/0000_point_cloud.npz")
    points, colors = pc["points"], pc["colors"] / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].imshow(rgb); axes[0].set_title("Input ego frame"); axes[0].axis("off")
    image = axes[1].imshow(rel_depth, cmap="magma"); axes[1].set_title("Relative depth; median = 1"); axes[1].axis("off")
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="Relative units")
    stride = max(1, len(points) // 5000)
    axes[2].scatter(points[::stride, 0], points[::stride, 2], c=colors[::stride], s=1)
    axes[2].set(title="Back-projected points", xlabel="X (relative units)", ylabel="Z (relative units)")
    fig.suptitle("M2 Depth Anything V2 Small on CPU; scale is not metric")
    fig.tight_layout()
    fig.savefig(FIGS / "m2_depth_and_point_cloud.png", dpi=160)
    plt.close(fig)

    errors = pd.read_csv(V2 / "m4/cup_grabcut_tracking/errors.csv")
    beige = errors[(errors["object"] == "beige_cup") & (errors["visible"] == True)].sort_values("frame")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(beige.frame, beige.translation_error_m, label="CAD PnP", color="#1d6996")
    axes[0].plot(beige.frame, beige.baseline_translation_error_m, label="Constant baseline", color="#d1495b")
    axes[0].set_ylabel("Translation error (m)"); axes[0].grid(alpha=0.25); axes[0].legend(ncol=2)
    axes[1].plot(beige.frame, beige.rotation_error_deg, label="CAD PnP", color="#1d6996")
    axes[1].plot(beige.frame, beige.baseline_rotation_error_deg, label="Constant baseline", color="#d1495b")
    axes[1].axvspan(28, 66, color="#edc948", alpha=0.2, label="Reviewed hand-contact interval")
    axes[1].set(xlabel="Frame index", ylabel="Rotation error (deg)"); axes[1].grid(alpha=0.25); axes[1].legend(ncol=2)
    fig.suptitle("M4 visible-frame errors; frame 0 is alignment only")
    fig.tight_layout()
    fig.savefig(FIGS / "m4_cup_pose_error_timeseries.png", dpi=160)
    plt.close(fig)

    rows = pd.read_csv(REPORT / "method_comparison_v2.csv")
    rows = rows[rows["method"].str.startswith("M4") & rows["mean_translation_error_m"].notna()]
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = rows["mean_translation_error_m"].to_numpy()
    ax.bar(np.arange(len(vals)), vals, color=["#1d6996", "#38a6a5", "#e17c05", "#cc503e"][:len(vals)])
    ax.set_xticks(np.arange(len(vals)), [f"{row.method}\n{row.track}" for row in rows.itertuples()], rotation=14, ha="right")
    ax.set_ylabel("Mean translation error (m)"); ax.set_title("M4 pose error across real CAD estimates"); ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(vals):
        ax.text(i, value, f"{value:.3f} m", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "m4_translation_error_comparison.png", dpi=160)
    plt.close(fig)


def make_summary(interaction):
    sfm = read_json(V2 / "m1/track3_ep0/export_summary.json")
    depth = read_json(V2 / "m2/relative_depth_corrected/summary.json")
    cup = read_json(V2 / "m4/cup_grabcut_tracking/summary.json")
    dust = read_json(V2 / "m4/dustpan_mask_search/summary.json")
    cup_metrics = cup["world_evaluation"]["metrics"]["beige_cup"]["all_successful"]
    dust_metrics = dust["metrics"]["all_successful"]
    groups = {row["video_review_bucket"]: row for row in interaction["groups"]}
    hand = groups["visible_hand_contact_reviewed"]
    clear = groups["visible_no_direct_hand_contact_reviewed"]
    m3 = read_json(V2 / "m3/attempt/triposr_cpu_attempt.json")
    m3_size = m3["local_checkpoints"]["triposr_model_ckpt"]["bytes"] / 1e9
    instant_size = sum(m3["local_checkpoints"][key]["bytes"] or 0 for key in ["instantmesh_large_ckpt", "instantmesh_diffusion_weights"]) / 1e9
    lines = [
        "# V2D CPU Reconstruction Experiments: Execution Summary (2026-10-09)",
        "",
        "## Scope and environment",
        "",
        "All new geometry and inference steps ran on Apple Silicon CPU. No CUDA, GPU, Isaac Lab, simulation, or policy training was used. `/opt/anaconda3/bin/python` supplied pandas, pyarrow, Transformers, and notebook execution; `/opt/anaconda3/envs/v2d/bin/python` supplied pycolmap, trimesh, Open3D, Pinocchio, and OpenCV-contrib CSRT. The original four notebooks and all prior outputs remain intact.",
        "",
        "## Measured results",
        "",
        "| Method / sequence | Runtime | Peak RSS | Geometry and pose evidence | Result and scale |",
        "|---|---:|---:|---|---|",
        f"| M1 pycolmap, Track 1 episode 16, 18 images | {read_json(V2 / 'm1/track1/summary.json')['wall_time_s']:.2f} s | Not instrumented | 0 points, no registered cameras | Failed: no good initial image pair |",
        f"| M1 pycolmap, Track 3 episode 11, 18 images | {read_json(V2 / 'm1/track3/summary.json')['wall_time_s']:.2f} s | Not instrumented | 0 points, no registered cameras | Failed: no good initial image pair |",
        f"| M1 pycolmap, Track 3 episode 0, 30 images | {sfm['wall_time_s']:.2f} s | Original run not instrumented; export process {sfm['process_peak_rss_mb']:.1f} MiB | {sfm['model_2']['num_images']} registered cameras, {sfm['model_2']['num_points3D']} points, {sfm['model_2']['num_observations']} observations, {sfm['mean_reprojection_error_px']:.3f} px mean reprojection | Real sparse SfM; arbitrary units, not metres |",
        f"| M2 Depth Anything V2 Small, Track 3 episode 0, 3 frames | {depth['wall_time_s']:.2f} s total; {np.mean([row['inference_s'] for row in depth['frames']]):.2f} s/frame | {depth['process_peak_rss_mb']:.1f} MiB | {sum(row['points'] for row in depth['frames']):,} points, 19,200/frame | Real CPU depth; relative scale, not metric |",
        f"| M3 TripoSR / InstantMesh | No inference | No inference | Local TripoSR checkpoint {m3_size:.2f} GB; InstantMesh weights {instant_size:.2f} GB | Blocked before inference: dependencies and authorized network/install channel; no mesh claimed |",
        f"| M4 cup CSRT + GrabCut + CAD IoU/PnP, beige cup | {cup['wall_time_s']:.2f} s including {cup['world_evaluation']['world_evaluation_runtime_s']:.2f} s evaluation | {cup['process_peak_rss_mb']:.1f} MiB | 2,600 sampled CAD vertices/object; 137/137 parquet-visible frames; silhouette IoU {cup_metrics['silhouette_iou']:.3f} | {cup_metrics['translation_error_m']:.3f} m / {cup_metrics['rotation_error_deg']:.1f} deg vs baseline {cup_metrics['baseline_translation_error_m']:.3f} m / {cup_metrics['baseline_rotation_error_deg']:.1f} deg; no improvement |",
        f"| M4 mini dustpan CAD mask/PnP, Track 3 episode 39 | {dust['wall_time_s']:.2f} s | {dust['process_peak_rss_mb']:.1f} MiB | {dust_metrics['frames']} frames; IoU {dust_metrics['silhouette_iou']:.3f}; 3,200 CAD vertices | {dust_metrics['translation_error_m']:.3f} m / {dust_metrics['rotation_error_deg']:.1f} deg vs baseline {dust_metrics['baseline_translation_error_m']:.3f} m / {dust_metrics['baseline_rotation_error_deg']:.1f} deg; no improvement |",
        "",
        "M4 produced real CAD camera-space pose estimates, but neither evaluated sequence beat the constant-pose baseline in world-pose accuracy. A strong silhouette overlap is image-space evidence, not a 6DoF accuracy metric. The cups are near-axisymmetric, and the public dataset has no ego-camera-to-mocap transform. Per-frame blue-cup alignment is the motion estimate used by the experiment; its own error is a calibration diagnostic rather than independent validation.",
        "",
        f"The beige-cup parquet marks {interaction['ground_truth_visible_frames']}/{interaction['ground_truth_frames']} frames visible. Frame 0 is used for alignment and excluded from means. The parquet has no hand-occlusion field; visual review labels frames 28–66 as hand contact. Those {hand['frames']} visible frames average {hand['mean_translation_error_m']:.3f} m / {hand['mean_rotation_error_deg']:.1f} deg, versus {clear['mean_translation_error_m']:.3f} m / {clear['mean_rotation_error_deg']:.1f} deg on the other {clear['frames']} visible frames. This is a manual video-review bucket, not dataset ground truth.",
        "",
        "## M3 failure and next inputs",
        "",
        "The local TripoSR and InstantMesh checkpoints exist. The available interpreters lack the TripoSR runtime packages; DNS lookup for Hugging Face and GitHub failed. An explicitly authorized pip install request was rejected before execution because the automatic approval service returned HTTP 503 (no approval channel). The repositories are public; no token or API key is required. Exact probes are saved in `outputs/v2/m3/attempt/triposr_cpu_attempt.json` and `.log`. The previous ellipsoid proxy remains separate under `outputs/meshes/`.",
        "",
        "To resume M3, the authorized dependency/network execution channel must work. For improved M4 world accuracy, the most useful input is a calibrated ego-to-mocap transform or a distinctive static reference, plus segmentation masks for hand-contact frames.",
        "",
        "## Artifacts",
        "",
        "- Executed notebook: `outputs/v2/notebooks/04_v2_real_m1_m4_experiments.ipynb`",
        "- Comparison table: `outputs/v2/report/method_comparison_v2.csv`",
        "- M1 geometry and cameras: `outputs/v2/m1/track3_ep0/`",
        "- M2 depth and points: `outputs/v2/m2/relative_depth_corrected/`",
        "- M4 cup estimates and visible-frame errors: `outputs/v2/m4/cup_grabcut_tracking/`",
        "- M4 alternate object: `outputs/v2/m4/dustpan_mask_search/`",
        "- Figures: `outputs/v2/figs/`",
        "- Additive manifests: `outputs/v2/report/artifact_manifest_v2.json` and `outputs/report/artifact_manifest.json`",
    ]
    (REPORT / "executive_summary_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_notebook():
    cells = [new_markdown_cell("""# V2D CPU Reconstruction Results: M1–M4

This notebook reviews the real CPU experiment outputs for the V2D Challenge. It distinguishes completed inference, failed initialization, and pose estimates that did not beat the constant-pose baseline. Long inference is not rerun here; each code block reads saved results and produces visible output.""")]

    def block(title, what, input_text, output_text, refs, source):
        cells.append(new_markdown_cell(f"### {title}\n\n**What it does:** {what}\n\n**Input:** {input_text}\n\n**Output:** {output_text}\n\n**References:** {refs}"))
        cells.append(new_code_cell(source))

    block(
        "Load Experiment Summaries",
        "Loads the measured M1–M4 comparison table and prints the key M1, M2, and M4 quantities.",
        "Saved v2 comparison CSV and JSON summaries.",
        "A table with runtime in seconds, memory in MiB, geometry counts, pose errors, and scale status.",
        "https://github.com/colmap/pycolmap and https://huggingface.co/docs/transformers/tasks/monocular_depth_estimation",
        """from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
V2 = ROOT / 'outputs/v2'
comparison = pd.read_csv(V2 / 'report/method_comparison_v2.csv')
display(comparison[['method', 'track', 'status', 'runtime_s', 'peak_rss_mb', 'geometry_count', 'geometry_unit', 'mean_translation_error_m', 'mean_rotation_error_deg', 'scale_status']].round(4))
sfm = json.loads((V2 / 'm1/track3_ep0/export_summary.json').read_text())
depth = json.loads((V2 / 'm2/relative_depth_corrected/summary.json').read_text())
cup = json.loads((V2 / 'm4/cup_grabcut_tracking/summary.json').read_text())
print(f"M1: {sfm['model_2']['num_images']} registered images, {sfm['model_2']['num_points3D']} points, {sfm['mean_reprojection_error_px']:.3f} px reprojection error; arbitrary SfM scale.")
print(f"M2: {sum(row['points'] for row in depth['frames']):,} points from {len(depth['frames'])} frames in {depth['wall_time_s']:.2f} s; relative scale.")
print(f"M4 beige cup: {cup['world_evaluation']['metrics']['beige_cup']['all_successful']['translation_error_m']:.3f} m translation and {cup['world_evaluation']['metrics']['beige_cup']['all_successful']['rotation_error_deg']:.1f} deg rotation mean error.")""",
    )
    block(
        "M1 Sparse Reconstruction",
        "Displays the exported pycolmap sparse point cloud and camera centers, then reports the initialization failures.",
        "The Track 3 episode 0 ASCII PLY, camera pose JSON, and M1 attempt summaries.",
        "A 3D sparse cloud and registered cameras in arbitrary SfM units, plus point count and failure messages.",
        "https://colmap.github.io/ and https://github.com/colmap/pycolmap",
        """from IPython.display import Image, display
import json

display(Image(filename=str(V2 / 'figs/m1_track3_sparse_cloud.png')))
for label, path in [('Track 1 episode 16', V2 / 'm1/track1/summary.json'), ('Track 3 episode 11', V2 / 'm1/track3/summary.json')]:
    result = json.loads(path.read_text())
    print(f"{label}: {result['models']} sparse models, {result['wall_time_s']:.2f} s, no good initial image pair found.")
print('Successful Track 3 episode 0: 30 registered images and arbitrary SfM scale. Export RSS was 76.1 MiB; original mapper RSS was not instrumented.')""",
    )
    block(
        "M2 Depth And Metric Scale",
        "Shows actual Depth Anything V2 Small CPU output and the point cloud back-projected with ego camera intrinsics.",
        "Saved frame 0000 RGB, relative-depth array, point cloud, and inference summary.",
        "A three-panel image/depth/point-cloud plot with latency in seconds, point count, and process peak RSS in MiB.",
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
        """from IPython.display import Image, display
display(Image(filename=str(V2 / 'figs/m2_depth_and_point_cloud.png')))
for row in depth['frames']:
    print(f"{row['frame']}: {row['inference_s']:.3f} s, {row['points']:,} points, median depth {row['depth_median_relative_units']:.3f} relative units.")
print(f"CPU process peak RSS: {depth['process_peak_rss_mb']:.1f} MiB. Intrinsics do not make monocular relative depth metric.")""",
    )
    block(
        "M3 CPU Attempt",
        "Reads the local TripoSR and InstantMesh checkpoint inventory and the dependency/network failure log without claiming a proxy mesh as inference.",
        "The probe JSON saved under `outputs/v2/m3/attempt/`.",
        "Checkpoint sizes in bytes, missing dependency names, whether a token is required, and the exact stop reason.",
        "https://huggingface.co/stabilityai/TripoSR and https://github.com/VAST-AI-Research/TripoSR",
        """m3 = json.loads((V2 / 'm3/attempt/triposr_cpu_attempt.json').read_text())
print('Status:', m3['status'])
for name, item in m3['local_checkpoints'].items():
    print(f"{name}: exists={item['exists']}, size={item['bytes']:,} bytes")
print('Missing packages:', ', '.join(name for name, present in m3['required_dependency_probe'].items() if not present))
print('Token required:', m3['token_required'])
for reason in m3['failure_reasons']:
    print('-', reason)
print('No TripoSR inference or generated mesh is claimed.')""",
    )
    block(
        "M4 Visible-Frame Errors",
        "Compares the cup CAD estimates with mocap GT, filters the parquet visible flag, excludes the alignment frame from means, and separates the video-reviewed contact interval.",
        "Saved visible-frame errors and manually reviewed frame interval 28–66.",
        "Mean and median translation errors in metres, rotation errors in degrees, baseline values, frame counts, and silhouette IoU.",
        "Track 3 schema: `video_to_data_challenge/track_3/README.md`; OpenCV solvePnP: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html",
        """errors = pd.read_csv(V2 / 'm4/cup_grabcut_tracking/errors.csv')
visible = errors[(errors['visible'] == True) & (errors['success'] == True) & (errors['calibration_frame'] != True)].copy()
metrics = ['translation_error_m', 'rotation_error_deg', 'baseline_translation_error_m', 'baseline_rotation_error_deg', 'silhouette_iou']
display(visible.groupby('object')[metrics].agg(['mean', 'median']).round(4))
bucket = pd.read_csv(V2 / 'm4/cup_grabcut_tracking/interaction_buckets.csv')
contact = bucket[(bucket['object'] == 'beige_cup') & (bucket['visible'] == True) & (bucket['success'] == True) & (bucket['calibration_frame'] != True)]
display(contact.groupby('video_review_bucket')[metrics].mean().round(4))
print(f"Parquet visible flags: {int(errors[errors.object == 'beige_cup'].visible.sum())}/{int((errors.object == 'beige_cup').sum())} beige-cup frames.")
print('The contact interval is manual video review, not a parquet annotation. Frame 0 was used for alignment and is excluded from means.')""",
    )
    block(
        "M4 Pose Curves And Overlays",
        "Displays world-pose errors over time against the constant baseline and shows representative segmentation/CAD silhouette overlays.",
        "Visible beige-cup errors and saved M4 overlays at selected frame indices.",
        "A time curve with units and the five executed overlay images; the manual contact interval is shaded.",
        "https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html",
        """display(Image(filename=str(V2 / 'figs/m4_cup_pose_error_timeseries.png')))
for frame in [0, 35, 70, 100, 136]:
    display(Image(filename=str(V2 / f'm4/cup_grabcut_tracking/overlay_{frame:04d}.jpg')))
beige = visible[visible.object == 'beige_cup']
print(f"CAD estimate: {beige.translation_error_m.mean():.3f} m / {beige.rotation_error_deg.mean():.1f} deg mean.")
print(f"Constant baseline: {beige.baseline_translation_error_m.mean():.3f} m / {beige.baseline_rotation_error_deg.mean():.1f} deg mean.")
print('This estimate is worse on this sequence; image-space overlap is not 6DoF accuracy.')""",
    )
    block(
        "Cross-Method Conclusion",
        "Displays all real, failed, and non-metric outcomes alongside the M4 translation-error comparison.",
        "The v2 method comparison CSV and plot.",
        "The comparison table, chart, and concise limitations for M1 through M4.",
        "https://github.com/nvidia-isaac/video_to_data and https://huggingface.co/datasets/nvidia/video_to_data_challenge",
        """display(comparison)
display(Image(filename=str(V2 / 'figs/m4_translation_error_comparison.png')))
print('M1: one successful Track 3 sparse reconstruction; two failed initializations; arbitrary scale.')
print('M2: real CPU depth and points; relative rather than metric depth.')
print('M3: blocked before inference by missing packages and failed authorized network/install execution.')
print('M4: real known-CAD image estimates, but neither cup nor dust-pan world-pose means beat the constant baseline.')
print('No GPU, CUDA, Isaac Lab, simulation, or policy training was used.')""",
    )
    notebook = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    path = NOTEBOOKS / "04_v2_real_m1_m4_experiments.ipynb"
    shell = InteractiveShell.instance()
    shell.user_ns.clear()
    for count, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        cell.execution_count = count
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True, silent=False)
        if result.error_before_exec is not None:
            raise result.error_before_exec
        if result.error_in_exec is not None:
            raise result.error_in_exec
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output(
                "display_data",
                data=item.data,
                metadata=item.metadata or {},
            ))
        cell.outputs = outputs
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return path


def update_manifests(notebook):
    paths = [path for path in V2.rglob("*") if path.is_file() and path.name != "artifact_manifest_v2.json"]
    paths += [ROOT / "v2_publish.py", ROOT / "v2_cad_tracking_v2.py", ROOT / "v2_cad_tracking_dustpan.py", ROOT / "v2_cad_tracking_cup_search.py", ROOT / "v2_cad_tracking_cup_grabcut.py", ROOT / "v2_evaluate_cup_grabcut.py", notebook]
    entries = []
    seen = set()
    for path in sorted(paths):
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)})
    v2_manifest = {"generated_at": "2026-10-09", "manifest_note": "This manifest is excluded from its own file list to avoid a self-referential hash.", "files": entries}
    write_json(REPORT / "artifact_manifest_v2.json", v2_manifest)

    old_path = ROOT / "outputs/report/artifact_manifest.json"
    old = read_json(old_path)
    old["generated_at"] = "2026-10-09"
    old["manifest_note"] = "All legacy entries are retained. The self-hash entry is omitted to avoid a self-referential checksum; new v2 artifact entries are appended."
    old_entries = [entry for entry in old.get("files", []) if entry.get("path") != "outputs/report/artifact_manifest.json"]
    known = {entry.get("path") for entry in old_entries}
    old_entries.extend(entry for entry in entries if entry["path"] not in known)
    old["files"] = old_entries
    write_json(old_path, old)


def main():
    make_m3_log()
    buckets = make_buckets()
    make_comparison(read_json(V2 / "m3/attempt/triposr_cpu_attempt.json"))
    make_figures()
    make_summary(buckets)
    notebook = make_notebook()
    update_manifests(notebook)
    executed = nbformat.read(notebook, as_version=4)
    print(json.dumps({
        "notebook": str(notebook.relative_to(ROOT)),
        "notebook_code_cells": sum(cell.cell_type == "code" for cell in executed.cells),
        "code_cells_with_outputs": sum(cell.cell_type == "code" and bool(cell.outputs) for cell in executed.cells),
        "comparison_rows": len(pd.read_csv(REPORT / "method_comparison_v2.csv")),
        "figures": [str(path.relative_to(ROOT)) for path in sorted(FIGS.glob("*.png"))],
        "summary": str((REPORT / "executive_summary_v2.md").relative_to(ROOT)),
        "m3_status": read_json(V2 / "m3/attempt/triposr_cpu_attempt.json")["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
