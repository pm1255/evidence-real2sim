"""Non-destructive mesh QA/registry and conservative tabletop candidate layouts.

No hole filling, automatic convex hull replacement, guessed metric scale or
automatic promotion to a real-validated benchmark.
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from common import read,dump,digest,fresh,resolve,stats


def asset(cfg,base,out):
    path=resolve(base,cfg['mesh']);raw=trimesh.load(path,force='mesh',process=False)
    if not isinstance(raw,trimesh.Trimesh) or len(raw.faces)==0:raise ValueError('Triangle mesh required, Gaussian centres are not a mesh')
    if not np.isfinite(raw.vertices).all():raise ValueError('Nonfinite geometry')
    scale=cfg.get('meters_per_unit')
    if scale is None or scale<=0:raise ValueError('Explicit metric scale required; never guessed from category')
    raw.apply_scale(scale);clean=raw.copy();clean.process(validate=True)
    clean.export(out/'collision_candidate.ply')
    result={'asset_id':cfg['asset_id'],'source_sha256':digest(path),
            'metric_scale_source':cfg['metric_scale_source'],'meters_per_unit':scale,
            'source_bounds_m':raw.bounds.tolist(),'bounds_m':clean.bounds.tolist(),
            'vertices_before':len(raw.vertices),'vertices_after':len(clean.vertices),
            'faces_before':len(raw.faces),'faces_after':len(clean.faces),
            'watertight':bool(clean.is_watertight),'winding_consistent':bool(clean.is_winding_consistent),
            'volume_m3':float(clean.volume) if clean.is_volume else None,
            'cleanup_extent_change_m':(clean.extents-raw.extents).tolist(),
            'collision_sha256':digest(out/'collision_candidate.ply'),
            'surface_error':None,'unobserved_surfaces_verified':False,
            'quality_state':'candidate','contact_dynamics_validated':False,'benchmark_eligible':False,
            'notes':['Topology validity is not surface accuracy.','No holes were filled and no convex hull substitution was made.',
                     'Cleaned collision candidate requires contact-region and task-specific review.']}
    if cfg.get('independent_points'):
        points_path=resolve(base,cfg['independent_points'])
        points=np.loadtxt(points_path,delimiter=',',ndmin=2)
        if points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Reference points must be finite xyz in the same metre frame')
        distances=[]
        for start in range(0,len(points),8):
            _,dist,_=trimesh.proximity.closest_point_naive(clean,points[start:start+8]);distances.extend(dist)
        result['surface_error']={'reference_sha256':digest(points_path),'point_to_mesh_m':stats(distances),
                                 'scope':'only supplied sampled points; not a maximum surface-error guarantee',
                                 'reference_source':cfg['reference_source']}
    dump(out/'asset.json',result)


def layouts(cfg,base,out):
    rng=np.random.default_rng(cfg['seed']);table=np.asarray(cfg['table_xy_bounds_m'],float)
    if table.shape!=(2,2) or np.any(table[1]<=table[0]):raise ValueError('Invalid tabletop bounds')
    assets=[]
    for a in cfg['assets']:
        path=resolve(base,a['report']);report=read(path)
        if report.get('collision_sha256')!=digest(path.parent/'collision_candidate.ply'):raise ValueError('Asset revision changed')
        assets.append((a['instance_id'],report,np.asarray(report['bounds_m']),digest(path)))
    generated=[];attempts=0;pad=cfg.get('clearance_m',.01)
    while len(generated)<cfg['count'] and attempts<cfg.get('max_proposals',1000):
        attempts+=1;placed=[];boxes=[]
        for name,report,bounds,h in assets:
            yaw=rng.uniform(-np.pi,np.pi);c,s=np.cos(yaw),np.sin(yaw)
            ext=bounds[1]-bounds[0];half=np.array([[abs(c),abs(s)],[abs(s),abs(c)]])@ext[:2]/2
            lo=table[0]+half+pad;hi=table[1]-half-pad
            if np.any(hi<=lo):break
            xy=rng.uniform(lo,hi);box=np.array([xy-half-pad/2,xy+half+pad/2])
            if any(np.all(box[0]<b[1]) and np.all(b[0]<box[1]) for b in boxes):break
            centre=(bounds[0,:2]+bounds[1,:2])/2;R=np.array([[c,-s],[s,c]])
            translation=np.r_[xy-R@centre,cfg.get('table_z_m',0)-bounds[0,2]]
            placed.append({'instance_id':name,'asset_id':report['asset_id'],'asset_report_sha256':h,
                           'translation_m':translation.tolist(),'yaw_rad':float(yaw)})
            boxes.append(box)
        if len(placed)==len(assets):
            generated.append({'scene_id':f'candidate_{len(generated):04d}','instances':placed,
                              'support_and_AABB_nonoverlap_checked':True,
                              'gravity_stability_checked':False,'robot_IK_and_path_checked':False,
                              'contact_model_scope_checked':False,'real_reset_verified':False,
                              'benchmark_eligible':False})
    dump(out/'layouts.json',{'seed':cfg['seed'],'proposals':attempts,'accepted_candidates':len(generated),
                            'requested_count':cfg['count'],'scenes':generated,
                            'next':'gravity settling, robot/path checks, task reachability, physical reset and paired validation'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['asset','layouts'])
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    path=Path(a.config).resolve();cfg=read(path);globals()[a.mode](cfg,path.parent,fresh(a.out))


if __name__=='__main__':main()
