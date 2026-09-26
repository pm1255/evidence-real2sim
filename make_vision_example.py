"""Generate a redistributable textured two-object RGB/RGB-D fixture by ray tracing.

Every output is marked SYNTHETIC. Seeds are backprojected from fit-view depth,
not heldout views. Used to test a second scene, camera handling and TSDF geometry.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from common import dump,fresh


def look_at(eye,target):
    z=target-eye;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    R=np.stack([x,y,z]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@eye;return T


def create(out,size=160,views=24):
    out=fresh(out)
    for name in ('images','masks','depth'):(out/name).mkdir()
    K=np.array([[size*1.4,0,size/2],[0,size*1.4,size/2],[0,0,1.]])
    yy,xx=np.mgrid[:size,:size];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    lo=np.array([-.06,-.04,0]);hi=np.array([.06,.04,.10]);sphere=np.array([.105,.005,.04]);radius=.035
    rng=np.random.default_rng(2026);frames=[];seeds=[];colors=[]
    for i in range(views):
        angle=2*np.pi*i/views;eye=np.array([.03+.4*np.cos(angle),.4*np.sin(angle),.24+.045*np.sin(angle*2)])
        T=look_at(eye,np.array([.03,0,.045]));direction=rays@T[:3,:3];d=direction.reshape(-1,3)
        safe=np.where(abs(d)>1e-10,d,1e-10);a=(lo-eye)/safe;b=(hi-eye)/safe
        enter=np.minimum(a,b).max(1);leave=np.maximum(a,b).min(1)
        zb=np.where((leave>=enter)&(leave>0)&(enter>0),enter,np.inf)
        oc=eye-sphere;A=np.sum(d*d,1);B=2*d@oc;C=oc@oc-radius**2;disc=B*B-4*A*C
        zs=np.where(disc>=0,(-B-np.sqrt(np.maximum(disc,0)))/(2*A),np.inf)
        zs=np.where(zs>0,zs,np.inf);depth=np.minimum(zb,zs);valid=np.isfinite(depth)
        points=eye+d*np.where(valid,depth,0)[:,None]
        # Fixed world-space texture, identical across viewpoints.
        texture=.5+.5*np.sin(points*310+np.sin(points[:,[1,2,0]]*170)*2)
        rgb=.15+.75*texture
        rgb[zs<zb]=rgb[zs<zb][:,[1,2,0]]*.7+np.array([.2,.05,.1])
        rgb=np.clip(rgb,0,1);rgb[~valid]=1;image=np.uint8(rgb.reshape(size,size,3)*255)
        depth=np.where(valid,depth,0).reshape(size,size).astype(np.float32);mask=valid.reshape(size,size)
        name=f'{i:03d}';cv2.imwrite(str(out/'images'/(name+'.png')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out/'masks'/(name+'.png')),mask.astype(np.uint8)*255);np.save(out/'depth'/(name+'.npy'),depth)
        split='test' if i%6==0 else 'fit'
        if split=='fit':seeds.append(points[valid]);colors.append(rgb[valid])
        frames.append({'id':'synthetic_'+name,'image':f'images/{name}.png','mask':f'masks/{name}.png','depth':f'depth/{name}.npy',
                       'split':split,'K':K.tolist(),'world_to_camera':T.tolist()})
    xyz=np.concatenate(seeds);rgb=np.concatenate(colors);choice=rng.choice(len(xyz),min(12000,len(xyz)),False)
    np.savez_compressed(out/'seed.npz',xyz=xyz[choice],rgb=rgb[choice])
    dump(out/'scene.json',{'schema':'calibrated-scene-v1','data_origin':'SYNTHETIC_RAY_TRACED_RGBD_SOFTWARE_VALIDATION',
      'camera_convention':'opencv_world_to_camera','metric_scale_source':'synthetic known coordinates',
      'seed_points':'seed.npz','seed_points_source_split':'fit','seed_provenance':'backprojected fit-view depths only',
      'depth_meters_per_unit':1.,'depth_consistency_tolerance_m':.004,'tsdf_truncation_m':.008,
      'bounds_m':[[-.07,-.06,-.01],[.15,.06,.11]],'frames':frames})
    dump(out/'truth.json',{'data_origin':'SYNTHETIC','box_bounds_m':[lo.tolist(),hi.tolist()],
                          'sphere_center_m':sphere.tolist(),'sphere_radius_m':radius})
    dump(out/'sfm_config.json',{'static_scene_verified':True,'frames':[{'file':f['image'],'split':f['split']} for f in frames]})
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=160)
    a=p.parse_args();create(a.out,a.size)
