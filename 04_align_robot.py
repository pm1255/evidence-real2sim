"""Fit camera/base transform and selected joint zero offsets against independent TCP points.

Input NPZ: q[N,J], observed_xyz[N,3], split[N] (0 fit, 1 test), optionally
clock_robot_s and clock_camera_s with clock_split. Matched samples must already
refer to the same physical instant. This script does not silently resample them.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco
from common import read,dump,digest,fresh,resolve,stats


def fit_rigid(source,target):
    x=source-source.mean(0); y=target-target.mean(0)
    if np.linalg.matrix_rank(x)<2: raise ValueError('Need noncollinear positions to determine rigid transform')
    U,_,Vt=np.linalg.svd(x.T@y); correction=np.diag([1,1,np.linalg.det(Vt.T@U.T)])
    R=Vt.T@correction@U.T
    return R,target.mean(0)-R@source.mean(0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();cp=Path(a.config).resolve();cfg=read(cp);out=fresh(a.out)
    if not cfg.get('observations_independent_of_robot_fk'):
        raise ValueError('FK-derived Cartesian logs cannot calibrate the same robot geometry')
    with np.load(resolve(cp.parent,cfg['data']),allow_pickle=False) as f: data=dict(f)
    q=data['q']; observed=data['observed_xyz']; split=data['split']; fit=split==0; test=split==1
    if q.ndim!=2 or observed.shape!=(len(q),3) or split.shape!=(len(q),): raise ValueError('Invalid calibration shapes')
    if not np.isin(split,[0,1]).all() or not np.isfinite(q).all() or not np.isfinite(observed).all(): raise ValueError('Invalid calibration samples')
    if fit.sum()<8 or test.sum()<3: raise ValueError('Need >=8 fit and >=3 heldout pose observations')
    model_path=resolve(cp.parent,cfg['model']); m=mujoco.MjModel.from_xml_path(str(model_path)); d=mujoco.MjData(m)
    names=cfg['joint_names']; indices=[]
    for name in names:
        j=m.joint(name)
        if int(j.type[0]) not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE): raise ValueError('Only scalar joints supported')
        indices.append(int(j.qposadr[0]))
    if q.shape[1]!=len(indices): raise ValueError('Joint order/dimension mismatch')
    site=m.site(cfg['tcp_site']).id; selected=cfg.get('fit_zero_joint_indices',[])
    if len(set(selected))!=len(selected) or any(i<0 or i>=len(indices) for i in selected): raise ValueError('Invalid zero-offset indices')
    if any(int(m.joint(names[i]).type[0])!=mujoco.mjtJoint.mjJNT_HINGE for i in selected):raise ValueError('This radian-offset implementation fits revolute joints only')
    def fk(values,offset):
        points=[]
        for row in values:
            mujoco.mj_resetData(m,d); d.qpos[indices]=row+offset
            mujoco.mj_forward(m,d);points.append(d.site_xpos[site].copy())
        return np.asarray(points)
    nominal=fk(q,np.zeros(len(indices)));R,t=fit_rigid(nominal[fit],observed[fit])
    initial=np.r_[Rotation.from_matrix(R).as_rotvec(),t,np.zeros(len(selected))]
    def prediction(x,values):
        off=np.zeros(len(indices));off[selected]=x[6:]
        pts=fk(values,off)
        return pts@Rotation.from_rotvec(x[:3]).as_matrix().T+x[3:6]
    bound=float(cfg.get('max_zero_offset_rad',.05))
    low=np.r_[np.full(6,-np.inf),np.full(len(selected),-bound)]
    high=np.r_[np.full(6,np.inf),np.full(len(selected),bound)]
    fit_result=least_squares(lambda x:(prediction(x,q[fit])-observed[fit]).ravel(),initial,
                             bounds=(low,high),loss='soft_l1',f_scale=cfg['observation_sigma_m'],max_nfev=250)
    # Dimensionless Jacobian for a useful rank/conditioning diagnostic.
    scales=np.r_[np.ones(3),np.full(3,cfg.get('workspace_scale_m',1.)),np.ones(len(selected))]
    J=fit_result.jac*scales[None,:];s=np.linalg.svd(J,compute_uv=False)
    rank=int(np.sum(s>s[0]*1e-6));identifiable=rank==len(initial)
    nominal_test=nominal[test]@R.T+t-observed[test]
    corrected_test=prediction(fit_result.x,q[test])-observed[test]
    result={'status':'estimated' if identifiable else 'degenerate_do_not_apply_offsets',
            'model_sha256':digest(model_path),'input_sha256':digest(resolve(cp.parent,cfg['data'])),
            'transform_robot_to_observation':{'rotvec':fit_result.x[:3].tolist(),'translation_m':fit_result.x[3:6].tolist()},
            'selected_joint_zero_offsets_rad':dict(zip([names[i] for i in selected],fit_result.x[6:].tolist())),
            'jacobian_rank':rank,'parameter_count':len(initial),'jacobian_singular_values':s.tolist(),
            'fit_xyz':stats(prediction(fit_result.x,q[fit])-observed[fit]),
            'heldout_nominal_xyz':stats(nominal_test),'heldout_corrected_xyz':stats(corrected_test),
            'independent_position_measurements_declared':True,
            'scope':'selected offsets and rigid transform only; no link length, TCP or full orientation calibration',
            'physical_accuracy_certified':False,'note':'Rank is local; uncertainty, calibration target error and threshold review still required.'}
    if 'clock_robot_s' in data:
        x=data['clock_robot_s'];y=data['clock_camera_s'];cs=data['clock_split'];tr=cs==0;te=cs==1
        if tr.sum()<3 or te.sum()<2: raise ValueError('Need fit and heldout matched clock events')
        origin=float(x[tr].mean());coef=np.polyfit(x[tr]-origin,y[tr],1)
        result['clock_mapping']={'camera_t_equals_a_times_robot_t_minus_origin_plus_b':True,
                                 'origin_s':origin,'a':float(coef[0]),'b':float(coef[1]),
                                 'heldout_residual_s':stats(coef[0]*(x[te]-origin)+coef[1]-y[te]),
                                 'is_controller_delay':False}
    dump(out/'alignment.json',result);print(result['status'])


if __name__=='__main__':main()
