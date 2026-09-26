"""Shared calibrated image contract: OpenCV camera axes, world-to-camera poses."""
from pathlib import Path
import json
import numpy as np
import cv2
from common import read, resolve, digest


def load_scene(path, max_side=None):
    path=Path(path).resolve();cfg=read(path);frames=[];seen={}
    if cfg.get('schema')!='calibrated-scene-v1':raise ValueError('Expected calibrated-scene-v1')
    if cfg.get('camera_convention')!='opencv_world_to_camera':raise ValueError('Explicit OpenCV world-to-camera convention required')
    for row in cfg['frames']:
        p=resolve(path.parent,row['image']);im=cv2.imread(str(p))
        if im is None:raise ValueError(f'Missing image: {p.name}')
        h=digest(p)
        if h in seen:raise ValueError('Repeated image content; heldout views must be independent')
        seen[h]=row['split']
        if row['split'] not in ('fit','development','test'):raise ValueError('Invalid image split')
        K=np.asarray(row['K'],float);T=np.asarray(row['world_to_camera'],float)
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():raise ValueError('Invalid calibration')
        if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3]@T[:3,:3].T,np.eye(3),atol=1e-4):raise ValueError('Non-rigid camera transform')
        if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]):raise ValueError('Invalid intrinsic matrix')
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(str(resolve(path.parent,row['mask'])),0) if row.get('mask') else np.full(im.shape[:2],255,np.uint8)
        if mask is None or mask.shape!=im.shape[:2]:raise ValueError('Mask/image shape mismatch')
        dist=np.asarray(row.get('distortion',[]),float)
        if dist.size and np.any(dist!=0):
            mapx,mapy=cv2.initUndistortRectifyMap(K,dist,None,K,im.shape[1::-1],cv2.CV_32FC1)
            im=cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR);mask=cv2.remap(mask,mapx,mapy,cv2.INTER_NEAREST)
        if max_side and max(im.shape[:2])>max_side:
            factor=max_side/max(im.shape[:2]);new=(round(im.shape[1]*factor),round(im.shape[0]*factor))
            sx,sy=new[0]/im.shape[1],new[1]/im.shape[0]
            K=np.diag([sx,sy,1])@K;im=cv2.resize(im,new,interpolation=cv2.INTER_AREA);mask=cv2.resize(mask,new,interpolation=cv2.INTER_NEAREST)
        frames.append(dict(row,rgb=im.astype(np.float32)/255,mask=mask>127,K=K,T=T,image_sha256=h))
    if len([f for f in frames if f['split']=='fit'])<2:raise ValueError('Need multiple fit views')
    return cfg,frames,path.parent


def scene_fingerprint(path):
    p=Path(path).resolve();cfg=read(p);records=[digest(p)]
    for f in cfg['frames']:
        records.append(digest(resolve(p.parent,f['image'])))
        if f.get('mask'):records.append(digest(resolve(p.parent,f['mask'])))
    if cfg.get('seed_points'):records.append(digest(resolve(p.parent,cfg['seed_points'])))
    import hashlib
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()
