"""Optional image routes: chessboard calibration, known marker tracking, sparse SfM.

Marker tracking requires calibrated intrinsics and a known marker size. SfM is
an arbitrary-scale sparse reconstruction, never automatically a collision mesh.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from common import read, dump, fresh, resolve, digest


def calibrate(cfg, base, out):
    grid = tuple(cfg['inner_corners']); square = float(cfg['square_m'])
    if square <= 0: raise ValueError('Known positive square size is required')
    points = np.zeros((grid[0]*grid[1], 3), np.float32)
    points[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)*square
    detected = []; size = None; rejected = []; split_hashes={}
    for f in cfg['frames']:
        path = resolve(base, f['file']); im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None: raise ValueError(f'Unreadable image {path}')
        if f['split'] not in ('fit','test'):raise ValueError('Calibration split must be fit or test')
        h=digest(path)
        if h in split_hashes and split_hashes[h]!=f['split']:raise ValueError('Same image reused for fit and heldout calibration')
        split_hashes[h]=f['split']
        wh = im.shape[::-1]
        if size is not None and size != wh: raise ValueError('Image sizes differ')
        size = wh
        ok, corners = cv2.findChessboardCornersSB(im, grid)
        if ok: detected.append((f, corners))
        else: rejected.append(f['file'])
    fit = [(f,c) for f,c in detected if f['split'] == 'fit']
    if len(fit) < 6: raise ValueError('Need >=6 usable calibration views, with varied tilts and positions')
    rms, K, dist, _, _ = cv2.calibrateCamera([points]*len(fit), [c for f,c in fit], size, None, None)
    test = []
    for f,c in detected:
        if f['split'] != 'test': continue
        ok, r, t = cv2.solvePnP(points, c, K, dist)
        if not ok: continue
        projected = cv2.projectPoints(points, r, t, K, dist)[0].reshape(-1,2)
        # OpenCV versions return SB corners as (N,2) or (N,1,2). Explicitly
        # flatten both to prevent broadcasting into all-to-all corner errors.
        difference=projected-c.reshape(-1,2)
        test.append({'file': f['file'], 'reprojection_rmse_px': float(np.sqrt(np.mean(np.sum(difference**2,axis=1))))})
    result = {'K': K.tolist(), 'distortion': dist.ravel().tolist(), 'image_size': list(size),
              'fit_rms_px': float(rms), 'heldout_views': test, 'rejected_frames': rejected,
              'metric_reference': {'square_m': square},
              'scope': 'intrinsic calibration; heldout board pose still estimated from that image',
              'contact_geometry_verified': False, 'source_config_sha256': cfg['_hash']}
    dump(out/'camera.json', result)


def track(cfg, base, out):
    cam = read(resolve(base, cfg['camera'])); K = np.array(cam['K']); dist = np.array(cam['distortion'])
    length = float(cfg['marker_size_m'])
    if length <= 0: raise ValueError('Known marker size required')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.get('dictionary', 'DICT_4X4_50')))
    detector = cv2.aruco.ArucoDetector(dictionary)
    s = length/2
    corners3 = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], float)
    records = []; last_time = -np.inf
    for f in cfg['frames']:
        t = float(f['t_s'])
        if t <= last_time: raise ValueError('Frame times must strictly increase; no FPS-derived timestamps')
        last_time = t; im = cv2.imread(str(resolve(base, f['file'])))
        if im is None: raise ValueError(f"Unreadable image {f['file']}")
        if list(im.shape[1::-1]) != cam['image_size']: raise ValueError('Calibration/image resolution mismatch')
        corners, ids, _ = detector.detectMarkers(im); matches = []
        if ids is not None: matches = [i for i,v in enumerate(ids.ravel()) if v == cfg['marker_id']]
        if len(matches) != 1:
            records.append({'t_s':t, 'valid':False, 'reason':'marker_missing_or_duplicated'}); continue
        observed = corners[matches[0]].reshape(4,2).astype(float)
        solved = cv2.solvePnPGeneric(corners3, observed, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        for r,v in zip(solved[1], solved[2]):
            R = cv2.Rodrigues(r)[0]
            if np.any((corners3 @ R.T + v.reshape(3))[:,2] <= 0): continue
            projected = cv2.projectPoints(corners3,r,v,K,dist)[0].reshape(4,2)
            error = float(np.sqrt(np.mean(np.sum((projected-observed)**2,axis=1))))
            candidates.append((error,r,v))
        candidates.sort(key=lambda x:x[0])
        if not candidates:
            records.append({'t_s':t,'valid':False,'reason':'no_positive_depth_solution'}); continue
        err,r,v = candidates[0]
        gap = candidates[1][0]-err if len(candidates)>1 else None
        ambiguous = gap is not None and gap < cfg.get('min_solution_gap_px', .1)
        valid = err <= cfg['max_reprojection_px'] and not ambiguous
        records.append({'t_s':t, 'valid':valid, 'translation_camera_m':v.ravel().tolist(),
                        'rotation_vector':r.ravel().tolist(), 'reprojection_px':err,
                        'second_solution_gap_px':gap, 'planar_pose_ambiguous':ambiguous})
    dump(out/'marker_tracks.json', {'records':records, 'marker_size_m':length,
         'camera_sha256':digest(resolve(base,cfg['camera'])),
         'scope':'camera-frame marker poses only; marker-to-object, camera-to-world and clock alignment still required',
         'missing_frames_interpolated':False, 'reprojection_is_not_metric_accuracy_certificate':True})


def sfm(cfg, base, out):
    import pycolmap
    import shutil
    if not cfg.get('static_scene_verified'):
        raise ValueError('SfM requires a static scene; moving objects/turntable background need explicit masking')
    images = out/'images'; images.mkdir(); names = []
    for i,f in enumerate(cfg['frames']):
        if f['split'] != 'fit': continue
        path = resolve(base,f['file']); name = f'{i:06d}{path.suffix}'
        shutil.copy2(path,images/name); names.append(name)
    if len(names)<3: raise ValueError('Need >=3 static overlapping views')
    db = out/'database.db'
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = 'SIMPLE_RADIAL'
    eo = pycolmap.FeatureExtractionOptions(); eo.num_threads=4; eo.sift.max_num_features=4000
    pycolmap.extract_features(db,images,camera_mode=pycolmap.CameraMode.SINGLE,
                             reader_options=ro,extraction_options=eo,device=pycolmap.Device.cpu)
    mo=pycolmap.FeatureMatchingOptions(); mo.num_threads=4
    pycolmap.match_exhaustive(db,matching_options=mo,device=pycolmap.Device.cpu)
    op=pycolmap.IncrementalPipelineOptions(); op.num_threads=4
    maps=pycolmap.incremental_mapping(db,images,out/'sparse',options=op)
    reports=[]
    for k,r in maps.items():
        r.export_PLY(out/f'sparse_{k}.ply')
        reports.append({'registered_images':r.num_reg_images(),'points':r.num_points3D(),
                        'fit_reprojection_px':r.compute_mean_reprojection_error()})
    dump(out/'sfm.json',{'models':reports,'scale':'arbitrary','dense_mesh_generated':False,
                       'gaussian_training_executed':False,'physics_eligible':False,
                       'next':'metric scale, dense/GS backend, collision geometry and independent surface checks'})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['calibrate','track','sfm'])
    p.add_argument('--config',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    path=Path(a.config).resolve(); cfg=read(path); cfg['_hash']=digest(path); out=fresh(a.out)
    globals()[a.mode](cfg,path.parent,out)


if __name__=='__main__': main()
