"""Local YCB adapter; source photographs are NOT redistributed by this repository.

Consumes the existing calibrated image/mask/visual-hull experiment. Seed geometry
was carved from fit views only. This is not an unposed-phone-photo experiment.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2,h5py,trimesh
from common import read,dump,fresh,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source=Path(a.source).resolve();out=fresh(a.out);(out/'images').mkdir();(out/'masks').mkdir()
    rows=read(source/'input_manifest.json');report=read(source/'photo_asset_report.json');frames=[]
    calibration=h5py.File(source/'dataset_calibration/calibration.h5')
    for i,row in enumerate(rows):
        filename=row['file'];cam,angle=Path(filename).stem.split('_')
        with h5py.File(source/'dataset_calibration'/f'NP5_{angle}_pose.h5') as f:Htable=np.array(f['H_table_from_reference_camera'])
        T=np.array(calibration[f'H_{cam}_from_NP5'])@np.linalg.inv(Htable)
        K=np.array(calibration[f'{cam}_rgb_K']);dist=np.array(calibration[f'{cam}_rgb_d'])
        image=cv2.imread(str(source/'images'/filename));mask=cv2.imread(str(source/'calibrated_masks'/(filename+'.png')),0)
        if image is None or mask is None:raise ValueError('Missing input image/mask')
        mapx,mapy=cv2.initUndistortRectifyMap(K,dist,None,K,image.shape[1::-1],cv2.CV_32FC1)
        image=cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR);mask=cv2.remap(mask,mapx,mapy,cv2.INTER_NEAREST)
        y,x=np.where(mask>127);cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2
        half=int(max(x.max()-x.min(),y.max()-y.min())*.65);x0=max(0,int(cx-half));y0=max(0,int(cy-half));x1=min(image.shape[1],int(cx+half));y1=min(image.shape[0],int(cy+half))
        image=image[y0:y1,x0:x1];mask=mask[y0:y1,x0:x1];K[0,2]-=x0;K[1,2]-=y0
        sx,sy=160/image.shape[1],160/image.shape[0];K=np.diag([sx,sy,1])@K
        image=cv2.resize(image,(160,160),interpolation=cv2.INTER_AREA);mask=cv2.resize(mask,(160,160),interpolation=cv2.INTER_NEAREST)
        name=f'{i:03d}.png';cv2.imwrite(str(out/'images'/name),image);cv2.imwrite(str(out/'masks'/name),mask)
        frames.append({'id':f'ycb_{i:03d}','image':'images/'+name,'mask':'masks/'+name,'split':'test' if i%6==0 else 'fit',
                       'K':K.tolist(),'world_to_camera':T.tolist(),'source_image_sha256':digest(source/'images'/filename)})
    calibration.close();mesh=trimesh.load(source/'photo_asset.ply',process=False)
    xyz=mesh.vertices+np.array(report['table_frame_offset_m']);rgb=np.asarray(mesh.visual.vertex_colors[:,:3],float)/255
    np.savez_compressed(out/'seed.npz',xyz=xyz,rgb=rgb)
    lo=xyz.min(0)-.01;hi=xyz.max(0)+.01
    dump(out/'scene.json',{'schema':'calibrated-scene-v1','data_origin':'REAL_YCB_RGB_WITH_PUBLISHED_METRIC_CALIBRATION',
         'camera_convention':'opencv_world_to_camera','metric_scale_source':'published YCB metric calibration',
         'seed_points':'seed.npz','seed_points_source_split':'fit','seed_provenance':'visual hull carved from 80 fit silhouettes, no scan vertices used',
         'bounds_m':[lo.tolist(),hi.tolist()],'frames':frames,
         'limitations':['Object masks and metric camera poses supplied; this is not unposed generic reconstruction.','Evaluation uses segmented object crops.']})
    print('Prepared',len(frames),'real views; source images remain local.')


if __name__=='__main__':main()
