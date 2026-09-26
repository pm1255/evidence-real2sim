"""General calibrated RGB plane-sweep stereo + measured-depth TSDF fusion.

Depth is estimated from fit RGB only or explicitly supplied by sensors. Low
texture, ambiguous matching and inconsistent cross-view depth are rejected.
Unknown TSDF voxels remain unknown; no automatic hole filling/convexification.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import trimesh
from skimage.measure import marching_cubes
from common import dump,fresh,resolve
from vision_data import load_scene,scene_fingerprint


def project(points,frame):
    pc=points@frame['T'][:3,:3].T+frame['T'][:3,3];p=pc@frame['K'].T
    return p[:,:2]/np.maximum(p[:,2:],1e-8),pc[:,2]


def sample_pixels(image,uv,interpolation=cv2.INTER_NEAREST):
    """OpenCV remap uses signed-short image dimensions; batch large point sets."""
    uv=np.asarray(uv,np.float32);parts=[]
    for begin in range(0,len(uv),16000):
        batch=uv[begin:begin+16000]
        sampled=cv2.remap(image,batch[:,0,None],batch[:,1,None],interpolation,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        parts.append(sampled.reshape((len(batch),)+image.shape[2:]))
    if not parts:return np.empty((0,)+image.shape[2:],dtype=image.dtype)
    return np.concatenate(parts)


def backproject(frame,depth):
    h,w=depth.shape;y,x=np.mgrid[:h,:w];rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(frame['K']).T
    camera=rays*depth[:,:,None];return (camera-frame['T'][:3,3])@frame['T'][:3,:3]


def stereo(ref,sources,bounds,planes=72,patch=5):
    im=ref['rgb'];h,w=im.shape[:2];gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    variance=cv2.boxFilter(gray*gray,-1,(patch,patch))-cv2.boxFilter(gray,-1,(patch,patch))**2
    corners=np.array(np.meshgrid(*zip(*bounds))).T.reshape(-1,3)
    depths=(corners@ref['T'][:3,:3].T+ref['T'][:3,3])[:,2]
    lo=max(.01,float(depths.min()));hi=float(depths.max())
    if hi<=lo:raise ValueError('Bounding volume behind camera')
    hypotheses=np.linspace(lo,hi,planes);costs=[]
    for z in hypotheses:
        xyz=backproject(ref,np.full((h,w),z)).reshape(-1,3);view_costs=[]
        for src in sources:
            uv,depth=project(xyz,src);uv=uv.reshape(h,w,2).astype(np.float32)
            warped=cv2.remap(src['rgb'],uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            valid=cv2.remap(src['mask'].astype(np.float32),uv[:,:,0],uv[:,:,1],cv2.INTER_NEAREST)>0
            valid&=(depth.reshape(h,w)>0)
            error=cv2.boxFilter(np.abs(warped-im).mean(-1),-1,(patch,patch))
            fraction=cv2.boxFilter(valid.astype(np.float32),-1,(patch,patch))
            view_costs.append(np.where(fraction>.8,error,10.))
        # Two-source agreement; one accidental matching source is insufficient.
        ordered=np.sort(np.stack(view_costs),axis=0);costs.append(ordered[:min(2,len(sources))].mean(0))
    costs=np.stack(costs);best=np.argmin(costs,axis=0);minimum=np.take_along_axis(costs,best[None],axis=0)[0]
    separated=costs.copy()
    for k in range(planes):separated[k][np.abs(best-k)<=2]=10.
    margin=separated.min(0)-minimum
    valid=ref['mask']&(variance>.00015)&(minimum<.18)&(margin>.0005)&(best>0)&(best<planes-1)
    depth=np.where(valid,hypotheses[best],0).astype(np.float32)
    return depth,{'photometric_valid_pixels':int(valid.sum()),'object_pixels':int(ref['mask'].sum()),
                  'depth_range_m':[lo,hi],'median_valid_cost':float(np.median(minimum[valid])) if valid.any() else None}


def consistency(frames,depths,tolerance=.006,min_views=1):
    filtered=[];counts=[]
    for i,(ref,depth) in enumerate(zip(frames,depths)):
        xyz=backproject(ref,depth).reshape(-1,3);votes=np.zeros(len(xyz),int)
        for j,(src,other) in enumerate(zip(frames,depths)):
            if i==j:continue
            uv,z=project(xyz,src);uv=uv.astype(np.float32)
            sampled=sample_pixels(other,uv)
            votes+=(sampled>0)&(z>0)&(np.abs(sampled-z)<tolerance)
        good=(depth.ravel()>0)&(votes>=min_views)
        filtered.append(np.where(good.reshape(depth.shape),depth,0));counts.append(int(good.sum()))
    return filtered,counts


def fuse(frames,depths,bounds,resolution=64,truncation=.01,min_observations=2):
    lo,hi=np.asarray(bounds);shape=np.maximum(8,np.round((hi-lo)/max(hi-lo)*resolution).astype(int))
    axes=[np.linspace(lo[i],hi[i],shape[i]) for i in range(3)]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    values=np.zeros(len(points));weights=np.zeros(len(points));colors=np.zeros((len(points),3))
    for f,depth in zip(frames,depths):
        uv,z=project(points,f);uv=uv.astype(np.float32)
        sampled=sample_pixels(depth,uv)
        sdf=sampled-z;valid=(sampled>0)&(z>0)&(sdf>=-truncation)
        values[valid]+=np.clip(sdf[valid]/truncation,-1,1);weights[valid]+=1
        rgb=sample_pixels(f['rgb'],uv,cv2.INTER_LINEAR)
        colors[valid]+=rgb[valid]
    known=weights>=min_observations;field=np.ones(len(points));field[known]=values[known]/weights[known]
    vol=field.reshape(shape);mask=known.reshape(shape)
    if not np.any(vol[mask]<0) or not np.any(vol[mask]>0):return None,{'status':'insufficient_consistent_depth','known_voxels':int(known.sum())}
    vertices,faces,_,_=marching_cubes(vol.astype(np.float32),0,spacing=(hi-lo)/(shape-1),mask=mask)
    vertices+=lo
    # Retain faces only when all 8 corners of the surrounding cells are measured.
    from scipy.ndimage import minimum_filter
    interior=minimum_filter(mask.astype(np.uint8),size=3,mode='constant')>0
    index=np.clip(np.rint((vertices-lo)/(hi-lo)*(shape-1)).astype(int),0,shape-1)
    supported=interior[tuple(index.T)];faces=faces[np.all(supported[faces],axis=1)]
    mesh=trimesh.Trimesh(vertices,faces,process=False);mesh.remove_unreferenced_vertices()
    if len(mesh.faces)==0:return None,{'status':'no_fully_supported_faces','known_voxels':int(known.sum())}
    indices=np.clip(np.rint((mesh.vertices-lo)/(hi-lo)*(shape-1)).astype(int),0,shape-1)
    flat=np.ravel_multi_index(indices.T,shape);rgb=colors[flat]/np.maximum(weights[flat,None],1)
    mesh.visual.vertex_colors=np.uint8(np.clip(rgb,0,1)*255)
    return mesh,{'status':'partial_observed_surface','known_voxels':int(known.sum()),'total_voxels':len(points),
                 'voxel_spacing_m':((hi-lo)/(shape-1)).tolist(),'vertices':len(mesh.vertices),'faces':len(mesh.faces),
                 'watertight':bool(mesh.is_watertight),'holes_filled':False}


def reconstruct(scene,out,mode='rgb',max_side=96,planes=72,max_views=16,resolution=64):
    out=fresh(out);cfg,all_frames,base=load_scene(scene,max_side);fit=[f for f in all_frames if f['split']=='fit']
    if len(fit)>max_views:fit=[fit[i] for i in np.linspace(0,len(fit)-1,max_views).astype(int)]
    bounds=cfg['bounds_m'];depths=[];diagnostics=[]
    centers=np.array([-f['T'][:3,:3].T@f['T'][:3,3] for f in fit])
    for i,f in enumerate(fit):
        if mode=='rgb':
            neighbours=np.argsort(np.linalg.norm(centers-centers[i],axis=1))[1:5]
            if len(neighbours)<2:raise ValueError('Need at least three fit views for RGB stereo')
            depth,diag=stereo(f,[fit[j] for j in neighbours],bounds,planes)
        else:
            if not f.get('depth'):raise ValueError('Sensor mode requires explicit registered depth NPY per view')
            raw=np.load(resolve(base,f['depth']),allow_pickle=False).astype(np.float32)*cfg['depth_meters_per_unit']
            if f.get('distortion') and np.any(f['distortion']):raise ValueError('Sensor depths must be pre-registered to undistorted RGB')
            depth=cv2.resize(raw,f['mask'].shape[::-1],interpolation=cv2.INTER_NEAREST)
            depth=np.where(f['mask']&np.isfinite(depth)&(depth>0),depth,0)
            diag={'sensor_valid_pixels':int((depth>0).sum())}
        depths.append(depth);diagnostics.append(dict(view_id=f['id'],**diag));print('Depth',i+1,'/',len(fit),flush=True)
    filtered,counts=consistency(fit,depths,cfg.get('depth_consistency_tolerance_m',.006))
    clouds=[];colours=[]
    for i,(f,d) in enumerate(zip(fit,filtered)):
        np.save(out/f'depth_{i:03d}.npy',d);xyz=backproject(f,d);valid=d>0
        clouds.append(xyz[valid]);colours.append(f['rgb'][valid]);diagnostics[i]['cross_view_valid_pixels']=counts[i]
    xyz=np.concatenate(clouds);rgb=np.concatenate(colours)
    np.savez_compressed(out/'seed_points.npz',xyz=xyz,rgb=rgb)
    if len(xyz):trimesh.points.PointCloud(xyz,colors=np.uint8(rgb*255)).export(out/'observed_points.ply')
    mesh,fusion=fuse(fit,filtered,bounds,resolution,cfg.get('tsdf_truncation_m',.01))
    if mesh is not None:mesh.export(out/'surface.ply')
    result={'mode':'RGB_plane_sweep_multiview' if mode=='rgb' else 'registered_sensor_depth_TSDF',
            'data_origin':cfg['data_origin'],'scene_sha256':scene_fingerprint(scene),'views':diagnostics,
            'fit_views_only':True,'observed_points':len(xyz),'fusion':fusion,
            'metric_scale_source':cfg.get('metric_scale_source'),'contact_surface_accuracy_certified':False,
            'limitations':['Dense surface is partial where texture/depth observations are insufficient.',
                           'Cross-view agreement is not independent metrology.','No automatic watertight filling or collision certification.']}
    dump(out/'dense_report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['rgb','sensor-depth'],default='rgb');p.add_argument('--max-side',type=int,default=96)
    p.add_argument('--planes',type=int,default=72);p.add_argument('--max-views',type=int,default=16);p.add_argument('--resolution',type=int,default=64)
    a=p.parse_args();reconstruct(a.scene,a.out,a.mode,a.max_side,a.planes,a.max_views,a.resolution)


if __name__=='__main__':main()
