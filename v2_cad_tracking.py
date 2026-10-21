"""Observed-mask to known-CAD contour ICP/PnP; no per-frame mocap pose input."""
from pathlib import Path
import json
import time
import resource
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/v2/m4/cup_pnp'
OUT.mkdir(parents=True, exist_ok=True)


def segment(frame, name, previous):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    if name=='beige_cup': mask=((h>=14)&(h<=40)&(s<=105)&(v>95)).astype(np.uint8)*255
    else: mask=((h>=95)&(h<=140)&(s>65)&(v>45)).astype(np.uint8)*255
    mask[:int(frame.shape[0]*.48)]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,cent=cv2.connectedComponentsWithStats(mask)
    choices=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>150]
    if not choices: return None,previous,None
    i=min(choices,key=lambda k:np.linalg.norm(cent[k]-previous))
    result=(labels==i).astype(np.uint8)*255
    contours,_=cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea).reshape(-1,2).astype(float)
    return result,cent[i],contour


def pose_matrix(rv,tv):
    t=np.eye(4);t[:3,:3]=cv2.Rodrigues(rv)[0];t[:3,3]=np.asarray(tv).ravel();return t


def fit_pose(vertices,contour,k,initial):
    # Closest-point image contour ICP with CAD hull points, followed by PnP updates.
    rv,tv=initial[:3].copy(),initial[3:].copy()
    tree=cKDTree(contour)
    for _ in range(7):
        proj=cv2.projectPoints(vertices,rv,tv,k,None)[0].reshape(-1,2)
        hull=cv2.convexHull(proj.astype(np.float32),returnPoints=False).ravel()
        ids=hull
        if len(ids)<6: break
        dd,ii=tree.query(proj[ids])
        keep=dd<max(12,np.percentile(dd,85))
        if keep.sum()<6: break
        ok,rn,tn=cv2.solvePnP(vertices[ids[keep]],contour[ii[keep]],k,None,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not .15<float(tn.ravel()[2])<3:break
        rv,tv=rn.ravel(),tn.ravel()
    projected=cv2.projectPoints(vertices,rv,tv,k,None)[0].reshape(-1,2)
    hull=cv2.convexHull(projected.astype(np.float32),returnPoints=False).ravel()
    distance=tree.query(projected[hull])[0]
    return np.r_[rv,tv],float(np.median(distance)),projected[hull]


def main():
    t0=time.perf_counter()
    cap=cv2.VideoCapture(str(ROOT/'video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4'))
    frames=[]
    while True:
        ok,fr=cap.read()
        if not ok:break
        frames.append(cv2.resize(fr,(640,480),interpolation=cv2.INTER_AREA))
    cap.release()
    cal=json.loads((ROOT/'video_to_data_challenge/track_3/public/meta/camera_calibration.json').read_text())['cameras']['ego_cam_a']
    k=np.array(cal['intrinsic_matrix'],float); k[0,2]=2027-k[0,2];k[1,2]=1519-k[1,2];k[0]*=640/2028;k[1]*=480/1520
    dist=np.array(cal['distortion_coefficients'],float);dist[2:4]*=-1
    frames=[cv2.undistort(f,k,dist) for f in frames]
    vertices={}
    for name in ['blue_cup','beige_cup']:
        v=np.load(ROOT/f'outputs/v2/m4/{name}_mesh.npz')['vertices']
        vertices[name]=v[np.linspace(0,len(v)-1,1800,dtype=int)].astype(float)
    previous={'blue_cup':np.array([289.,348.]),'beige_cup':np.array([425.,332.])}
    poses={}; first_poses={}; observations=[]; camera_rotations=[]; overlays=[]
    sift=cv2.SIFT_create(nfeatures=1200)
    bgmask=np.zeros((480,640),np.uint8);bgmask[:245]=255
    kp0,des0=sift.detectAndCompute(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),bgmask)
    matcher=cv2.BFMatcher(); kinv=np.linalg.inv(k)
    for i,fr in enumerate(frames):
        kp,des=sift.detectAndCompute(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY),bgmask)
        good=[]
        if des is not None:
            for pair in matcher.knnMatch(des0,des,k=2):
                if len(pair)==2 and pair[0].distance<.72*pair[1].distance:good.append(pair[0])
        rcam=np.eye(3)
        if len(good)>=10:
            h,inliers=cv2.findHomography(np.float32([kp0[m.queryIdx].pt for m in good]),np.float32([kp[m.trainIdx].pt for m in good]),cv2.RANSAC,3)
            if h is not None:
                candidate=kinv@h@k;candidate/=np.cbrt(np.linalg.det(candidate));u,_,vt=np.linalg.svd(candidate);rcam=u@vt
        camera_rotations.append(rcam)
        overlay=fr.copy()
        for name in ['blue_cup','beige_cup']:
            mask,previous[name],contour=segment(fr,name,previous[name])
            if contour is None:
                observations.append({'frame':i,'object':name,'success':False,'failure':'no segmented component'})
                continue
            x,y,w,h=cv2.boundingRect(contour.astype(np.float32))
            if name not in poses:
                # CAD size supplies metric camera-space scale; z-axis points toward cup bottom.
                z=.10*k[0,0]/max(1,w)
                trans=np.array([(x+w/2-k[0,2])*z/k[0,0],(y+h/2-k[1,2])*z/k[1,1],z])
                rv=Rotation.from_euler('x',65,degrees=True).as_rotvec()
                poses[name]=np.r_[rv,trans]
            pose,err,hull=fit_pose(vertices[name],contour,k,poses[name])
            poses[name]=pose
            if i==0:first_poses[name]=pose.copy()
            observations.append({'frame':i,'object':name,'success':True,'contour_median_error_px':err,'mask_pixels':int((mask>0).sum()),'camera_T_object':pose_matrix(pose[:3],pose[3:]).tolist(),'background_matches':len(good),'mask_bbox':[x,y,w,h]})
            if i in [0,35,70,100,136]:
                cv2.drawContours(overlay,[contour.astype(np.int32).reshape(-1,1,2)],-1,(0,255,0),1)
                cv2.polylines(overlay,[hull.astype(np.int32).reshape(-1,1,2)],True,(0,0,255),1)
                cv2.putText(overlay,f'{name}: {err:.1f} px',(10,20+(20 if name=='beige_cup' else 0)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imwrite(str(OUT/f'mask_{name}_{i:04d}.png'),mask)
        if i in [0,35,70,100,136]:
            cv2.imwrite(str(OUT/f'overlay_{i:04d}.jpg'),overlay)
        if i%30==0:print(f'frame {i}/{len(frames)}',flush=True)
    # Ground truth enters only after every image estimate is finished.
    gt=pd.read_parquet(ROOT/'video_to_data_challenge/track_3/public/data/chunk-000/episode_000011.parquet')
    gt_obj={}
    for entry in gt['observation.objects']:
        for obj in entry:
            gt_obj.setdefault(obj['name'],[]).append(obj)
    def gt_transform(obj):
        p=np.array(obj['pose']);t=np.eye(4);t[:3,:3]=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();t[:3,3]=p[:3];return t
    # One first-frame GT calibration. Relative camera rotation comes from background only.
    world_t_cam0=gt_transform(gt_obj['beige_cup'][0])@np.linalg.inv(pose_matrix(first_poses['beige_cup'][:3],first_poses['beige_cup'][3:]))
    results=[]
    for row in observations:
        i=row['frame'];name=row['object'];obj=gt_obj[name][i]
        if not obj['visible'] or not row['success']:continue
        cami_t_cam0=np.eye(4);cami_t_cam0[:3,:3]=camera_rotations[i]
        pred=world_t_cam0@np.linalg.inv(cami_t_cam0)@np.array(row['camera_T_object'])
        actual=gt_transform(obj);base=gt_transform(gt_obj[name][0])
        te=np.linalg.norm(pred[:3,3]-actual[:3,3]);re=Rotation.from_matrix(pred[:3,:3].T@actual[:3,:3]).magnitude()
        bt=np.linalg.norm(base[:3,3]-actual[:3,3]);br=Rotation.from_matrix(base[:3,:3].T@actual[:3,:3]).magnitude()
        results.append({'frame':i,'object':name,'translation_error_m':float(te),'rotation_error_deg':float(np.degrees(re)),'baseline_translation_error_m':float(bt),'baseline_rotation_error_deg':float(np.degrees(br)),'contour_error_px':row['contour_median_error_px'],'calibration_frame':i==0,'world_T_object':pred.tolist()})
    (OUT/'observations.json').write_text(json.dumps(observations,indent=2))
    (OUT/'world_predictions.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k!='world_T_object'} for r in results]).to_csv(OUT/'errors.csv',index=False)
    metrics={}
    for name in gt_obj:
        rows=[r for r in results if r['object']==name and not r['calibration_frame']]
        metrics[name]={k:float(np.mean([r[k] for r in rows])) for k in ['translation_error_m','rotation_error_deg','baseline_translation_error_m','baseline_rotation_error_deg','contour_error_px']}
        metrics[name]['evaluated_frames']=len(rows)
    summary={'method':'M4 CAD silhouette ICP + solvePnP','status':'real observed-mask image tracking; metric camera-space poses; world evaluation uses one GT calibration frame and image-only rotation compensation','frames':len(frames),'metrics':metrics,'wall_time_s':time.perf_counter()-t0,'process_peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'limitations':['camera translation not estimated; background homography rotation-only compensation','near-axisymmetric CAD leaves rotation around cup axis ambiguous','color segmentation can truncate cups under hand occlusion','first-frame alignment excludes calibration frame from reported means','no guaranteed improvement versus baseline; see measured errors']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
