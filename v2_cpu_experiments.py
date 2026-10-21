"""CPU-only v2 experiments. Legacy notebooks and outputs are read-only inputs."""
from pathlib import Path
import json
import time
import resource
import os

ROOT = Path(__file__).resolve().parent
V2 = ROOT / 'outputs/v2'


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def depth():
    import numpy as np
    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(4)
    out = V2 / 'm2/relative_depth_corrected'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model_dir = V2 / 'm2/model'
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), local_files_only=True).cpu().eval()
    cal = json.loads((ROOT / 'video_to_data_challenge/track_3/public/meta/camera_calibration.json').read_text())['cameras']['ego_cam_a']
    k0 = np.asarray(cal['intrinsic_matrix'], dtype=float)
    rows = []
    for fname in ['0000.jpg', '0015.jpg', '0029.jpg']:
        from PIL import Image
        ipath = V2 / 'inputs/track3_ep0_ego_a' / fname
        image = Image.open(ipath).convert('RGB')
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        inputs = processor(images=image, return_tensors='pt')
        start = time.perf_counter()
        with torch.inference_mode():
            pred = model(**inputs).predicted_depth
            inverse = torch.nn.functional.interpolate(pred[:, None], size=(h,w), mode='bicubic', align_corners=False).squeeze().numpy()
        latency = time.perf_counter() - start
        # The released small model predicts relative inverse depth, not metres.
        floor = max(float(np.percentile(inverse, 1)), 1e-3)
        z = 1.0 / np.maximum(inverse, floor)
        z /= np.median(z)
        # Ego images have already been rotated by 180 degrees; use rotated intrinsics.
        k = k0.copy()
        k[0,2] = 2027 - k0[0,2]
        k[1,2] = 1519 - k0[1,2]
        k[0,:] *= w / 2028
        k[1,:] *= h / 1520
        dist = np.asarray(cal['distortion_coefficients'], dtype=float)
        dist[2:4] *= -1
        yy,xx = np.mgrid[0:h:4,0:w:4]
        pixels = np.stack([xx, yy], axis=-1).astype(np.float32)
        rays = cv2.undistortPoints(pixels.reshape(-1,1,2), k, dist).reshape(-1,2)
        zz = z[yy,xx].ravel()
        points = np.c_[rays * zz[:,None], zz]
        np.save(out / (Path(fname).stem + '_inverse_depth.npy'), inverse)
        np.save(out / (Path(fname).stem + '_relative_depth.npy'), z)
        np.savez(out / (Path(fname).stem + '_point_cloud.npz'), points=points, colors=rgb[yy,xx].reshape(-1,3), K=k)
        write_ply(out / (Path(fname).stem + '_points.ply'), points, rgb[yy,xx].reshape(-1,3))
        rows.append({'frame':fname,'inference_s':latency,'points':len(points),'inverse_min':float(inverse.min()),'inverse_max':float(inverse.max()),'depth_median_relative_units':float(np.median(z))})
    summary = {'method':'M2 Depth Anything V2 Small', 'status':'real CPU inference', 'device':'cpu', 'parameters':sum(p.numel() for p in model.parameters()), 'frames':rows, 'wall_time_s':time.perf_counter()-t0, 'process_peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 'scale':'relative inverse depth inverted and median-normalized; NOT metric; intrinsics alone do not establish scale', 'intrinsics_note':'180-degree rotated principal point and tangential distortion signs; OpenCV undistortPoints before back-projection', 'preliminary_output_note':'m2/track3_ep0 contains a superseded preliminary normalization; use relative_depth_corrected only'}
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


def write_ply(path, points, colors):
    with open(path, 'w', encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(points, colors):
            f.write(' '.join(str(float(x)) for x in p) + ' ' + ' '.join(str(int(x)) for x in c) + '\n')


def export_sfm():
    import pycolmap
    import numpy as np
    out = V2 / 'm1/track3_ep0'
    rec = pycolmap.Reconstruction(str(out/'model_2'))
    points = np.array([p.xyz for p in rec.points3D.values()])
    colors = np.array([p.color for p in rec.points3D.values()])
    write_ply(out/'sparse_points.ply',points,colors)
    rows=[]
    for img in rec.images.values():
        pose=img.cam_from_world()
        rows.append({'image':img.name,'camera_id':int(img.camera_id),'cam_from_sfm_world':pose.matrix().tolist(),'camera_center_sfm_units':img.projection_center().tolist()})
    save_json(out/'camera_poses.json',rows)
    summary=json.loads((out/'summary.json').read_text())
    summary['camera_mode']='SINGLE shared SIMPLE_RADIAL; focal length refined by bundle adjustment'
    summary['cameras']=[{'id':int(k),'model':c.model.name,'parameters':c.params.tolist()} for k,c in rec.cameras.items()]
    summary['mean_reprojection_error_px']=float(rec.compute_mean_reprojection_error())
    summary['process_peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    summary['memory_note']='export process peak; original SfM run peak not instrumented; resource cannot recover historical peak'
    save_json(out/'export_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    import sys
    {'depth':depth,'export_sfm':export_sfm}[sys.argv[1]]()
