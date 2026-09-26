"""Measured torque inventory and optional inverse-dynamics residual diagnostics.

Residuals include model, transmission, payload and differentiation errors. They
are not automatically contact forces or identified friction coefficients.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from common import bundle,read,dump,digest,fresh,resolve,usable,stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cp=Path(a.config).resolve();cfg=read(cp);spec,episodes=bundle(a.manifest);out=fresh(a.out);rows=[]
    model_path=resolve(cp.parent,cfg['model']) if cfg.get('model') else None
    m=None
    if model_path:
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(model_path));d=mujoco.MjData(m)
        # No artificial scene contacts, joint stops or dry-friction constraints.
        # Declared model damping remains included; unmodelled dry friction stays in residual.
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        qadr=[];vadr=[]
        for name in cfg['joint_names']:
            j=m.joint(name)
            if int(j.type[0])!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('Nm torque implementation requires revolute joints; prismatic force channels need a separate unit-aware adapter')
            qadr.append(int(j.qposadr[0]));vadr.append(int(j.dofadr[0]))
    for e in episodes:
        if not usable(e,('q','tau','t')):
            rows.append({'id':e['id'],'status':'skipped_missing_measured_torque'});continue
        ar=e['arrays'];meta=e.get('metadata',{})
        row={'id':e['id'],'source':meta.get('tau_source','unspecified'),
             'semantics_verified':meta.get('tau_semantics_verified',False),
             'recorded_torque_Nm':stats(ar['tau'])}
        if m is None:
            row['status']='inventory_only_missing_usable_dynamics_model';rows.append(row);continue
        if not row['semantics_verified'] and not cfg.get('allow_unverified_diagnostic',False):
            row['status']='skipped_unverified_torque_semantics';rows.append(row);continue
        if ar['q'].shape[1]!=len(qadr):raise ValueError('Joint order/dimension mismatch')
        t=ar['t'];dt=float(np.median(np.diff(t)))
        if np.diff(t).max()>cfg.get('max_gap_s',.15):
            row['status']='skipped_large_time_gap';rows.append(row);continue
        grid=np.linspace(t[0],t[-1],int(round((t[-1]-t[0])/dt))+1);dt=float(grid[1]-grid[0])
        q=np.stack([np.interp(grid,t,ar['q'][:,j]) for j in range(len(qadr))],axis=1)
        measured=np.stack([np.interp(grid,t,ar['tau'][:,j]) for j in range(len(qadr))],axis=1)
        windows=cfg.get('smoothing_windows_samples',[7,11]);runs=[]
        for window in windows:
            if window%2!=1 or window<5 or window>=len(grid):raise ValueError('Derivative window must be odd, >=5 and shorter than episode')
            smooth=savgol_filter(q,window,3,axis=0);dq=savgol_filter(q,window,3,deriv=1,delta=dt,axis=0)
            ddq=savgol_filter(q,window,3,deriv=2,delta=dt,axis=0)
            inverse=[];wrenches=[];conditions=[];unexplained=[]
            edge=window//2
            for i in range(edge,len(grid)-edge):
                mujoco.mj_resetData(m,d);d.qpos[qadr]=smooth[i];d.qvel[vadr]=dq[i];d.qacc[vadr]=ddq[i]
                mujoco.mj_inverse(m,d);required=d.qfrc_inverse[vadr].copy();inverse.append(required)
                residual=required-measured[i]
                if cfg.get('single_contact_site'):
                    sid=m.site(cfg['single_contact_site']).id;Jp=np.zeros((3,m.nv));Jr=np.zeros_like(Jp)
                    mujoco.mj_jacSite(m,d,Jp,Jr,sid);A=np.vstack([Jp[:,vadr],Jr[:,vadr]]).T
                    # Scale moment by characteristic length to compare singular values.
                    length=cfg['wrench_characteristic_length_m'];B=A@np.diag([1,1,1,length,length,length])
                    singular=np.linalg.svd(B,compute_uv=False);rank=np.linalg.matrix_rank(B,tol=1e-6)
                    if rank<6: wrenches.append([np.nan]*6);conditions.append(np.inf);unexplained.append(np.nan)
                    else:
                        scaled=np.linalg.lstsq(B,residual,rcond=1e-6)[0]
                        wrench=scaled*np.array([1,1,1,length,length,length]);wrenches.append(wrench)
                        conditions.append(singular[0]/singular[-1]);unexplained.append(np.linalg.norm(A@wrench-residual))
            inverse=np.array(inverse);residual=inverse-measured[edge:-edge]
            save={'t':grid[edge:-edge],'measured_tau':measured[edge:-edge],'inverse_tau':inverse,'unmodelled_external_residual':residual}
            run={'window_samples':window,'window_duration_s':window*dt,'residual_Nm':stats(residual)}
            if wrenches:
                save['candidate_wrench_world_N_Nm']=np.array(wrenches)
                run['full_rank_sample_count']=int(np.isfinite(conditions).sum())
                run['total_samples']=len(conditions)
                run['wrench_is_conditional_on_single_known_contact']=True
            np.savez_compressed(out/f"{e['id']}_window{window}.npz",**save);runs.append(run)
        row.update(status='inverse_dynamics_diagnostic',derivative_sensitivity=runs,
                   interpretation='required model torque minus recorded motor torque; external force plus model/sensor errors')
        rows.append(row)
    dump(out/'torque_report.json',{'data_origin':spec['data_origin'],'episodes':rows,
         'model_sha256':digest(model_path) if model_path else None,
         'contact_force_validated':False,'contact_parameters_identified':False,
         'limitations':['URDF/MJCF inertias and payload are not automatically calibrated.',
                        'Constraint friction, transmission losses, offsets and derivative error remain in residual.',
                        'No peak-impact force claim from low-rate smoothed trajectories.',
                        'Single-site wrench inference requires one known contact and full Jacobian rank.']})
    print(f'Processed {len(rows)} episodes; force validation remains separate.')


if __name__=='__main__':main()
