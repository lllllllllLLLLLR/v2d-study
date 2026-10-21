# NVIDIA V2D Challenge — My Work

My participation work for the **NVIDIA V2D (Video-to-Data) Challenge** (CoRL 2026): turning human demonstration videos into robot-ready data. This repo contains my own analysis notebooks and CPU experiments — the official starter toolkit lives in [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data) and is not included here.

## Highlights

- **Dataset audit** (`01_dataset_audit.ipynb`) — audits the V2D dataset: file inventory, formats, platform checks.
- **CPU reconstruction comparison** (`02_cpu_reconstruction_comparison.ipynb`) — compares reconstruction approaches under CPU constraints.
- **Track 2 noise & retarget** (`03_track2_noise_and_retarget.ipynb`) — noise robustness and retargeting analysis for robotic grounding.
- **Track 1 reconstruction** (`V2D_Track1_reconstruction.ipynb`) — monocular third-person video → 4D human-object interaction reconstruction.
- **CAD tracking experiments** — `v2_cad_tracking*.py` series (cup grabcut / cup search / dustpan / v2), CPU-based object tracking pipelines.
- **Report** — `V2D_Challenge_调研报告.md` (Chinese): challenge analysis from organizer / participant / judge perspectives.

## Layout

```
├── 01_dataset_audit.ipynb
├── 02_cpu_reconstruction_comparison.ipynb
├── 03_track2_noise_and_retarget.ipynb
├── V2D_Track1_reconstruction.ipynb
├── V2D_Challenge_调研报告.md
├── v2_cad_tracking*.py          # CAD object tracking experiments (CPU)
├── v2_cpu_experiments.py
├── v2_evaluate_cup_grabcut.py
├── v2_publish.py
└── v2d_cpu_utils.py
```

## Notes

- Depends on the official [video_to_data](https://github.com/nvidia-isaac/video_to_data) starter toolkit and dataset (not included).
- Heavy outputs, videos and rendered data are excluded (see `.gitignore`).
