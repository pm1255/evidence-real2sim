"""Convert calibrated camera marker poses to world-frame planar object/pusher tracks.

No nominal FPS timestamps or automatic camera/base assumptions. Requires explicit
marker-to-body and camera-to-world transforms and a measured clock mapping.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import read,dump,digest,fresh,resolve,bundle


def transform(value):
    T=np.asarray(value,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('Expected finite homogeneous transform')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-5):raise ValueError('Transform must be rigid, with no guessed scale')
    return T


def convert(config,base):
    path=resolve(base,config['tracks']);tracks=read(path)['records']
    world_camera=transform(config['T_world_camera']);marker_body=transform(config['T_marker_body'])
    clock=config['clock_camera_to_robot'];a=clock['a'];b=clock['b'];origin=clock['origin_s']
    if not np.isfinite([a,b,origin]).all() or a<=0:raise ValueError('Invalid affine clock map')
    times=[];poses=[];discarded=[]
    for row in tracks:
        if not row['valid']:discarded.append({'t_s':row['t_s'],'reason':'invalid_image_pose'});continue
        camera_marker=np.eye(4);camera_marker[:3,:3]=Rotation.from_rotvec(row['rotation_vector']).as_matrix()
        camera_marker[:3,3]=row['translation_camera_m'];T=world_camera@camera_marker@marker_body
        tilt=float(np.arccos(np.clip(T[2,2],-1,1)))
        if tilt>config['max_tilt_rad']:
            discarded.append({'t_s':row['t_s'],'reason':'outside_planar_assumption'});continue
        times.append(a*(row['t_s']-origin)+b);poses.append([T[0,3],T[1,3],np.arctan2(T[1,0],T[0,0])])
    if len(times)<2 or not np.all(np.diff(times)>0):raise ValueError('Insufficient valid monotonic tracks')
    return np.array(times),np.array(poses),{'track_sha256':digest(path),'discarded':discarded,
           'max_gap_s':float(np.max(np.diff(times))),'clock_uncertainty_s':clock['uncertainty_s'],
           'world_pose_accuracy_m':config['world_pose_accuracy_m']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--manifest')
    p.add_argument('--out',required=True);a=p.parse_args();cp=Path(a.config).resolve();cfg=read(cp);out=fresh(a.out)
    originals={}
    if a.manifest:
        source,episodes=bundle(a.manifest);originals={e['id']:e for e in episodes}
    rows=[];diagnostics=[]
    for entry in cfg['episodes']:
        eid=entry['id']
        if Path(eid).name!=eid or eid in ('','.','..'):raise ValueError('Plain episode IDs required')
        if originals and eid not in originals:raise ValueError('Image episode does not match robot episode')
        old=originals.get(eid);arrays=dict(old['arrays']) if old else {};metadata=dict(old.get('metadata',{})) if old else {}
        if old and entry['split']!=old['split']:raise ValueError('Do not change episode split while adding images')
        diag={'id':eid}
        for kind in ('object','pusher'):
            if kind in entry:
                times,poses,report=convert(entry[kind],cp.parent)
                arrays[kind+'_t']=times;arrays[kind+'_pose']=poses;diag[kind]=report
        metadata.update(entry.get('metadata',{}))
        metadata['metric_geometry_and_sync_verified']=bool(entry.get('independent_geometry_and_sync_verified',False))
        if old:metadata['parent_episode_sha256']=old['sha256']
        path=out/(eid+'.npz');np.savez_compressed(path,**arrays)
        rows.append({'id':eid,'split':entry['split'],'file':path.name,'sha256':digest(path),'metadata':metadata})
        diagnostics.append(diag)
    dump(out/'manifest.json',{'schema':'adaptive-real2sim-v1','data_origin':cfg['data_origin'],'episodes':rows,
                             'conversion_config_sha256':digest(cp)})
    dump(out/'track_quality.json',{'episodes':diagnostics,'missing_data_interpolated':False,
          'note':'Planar conversion preserves only measured frames; large gaps are rejected later by contact fitting.'})


if __name__=='__main__':main()
