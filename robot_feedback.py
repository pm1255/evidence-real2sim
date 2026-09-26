"""Actuator-driven robot/object dynamics with two-way contact feedback.

Supports issued target replay and closed-loop policies. Robot/object qpos are
written only during initial reset. Every subsequent state is integrated.
"""
import argparse,importlib.util
from pathlib import Path
import numpy as np
import mujoco
from common import read,dump,fresh,resolve,model_digest,digest,stats


def run(config,out,commands=None,record=False):
    cp=Path(config).resolve();cfg=read(cp);out=fresh(out)
    model=resolve(cp.parent,cfg['model']);m=mujoco.MjModel.from_xml_path(str(model));d=mujoco.MjData(m)
    if m.nmocap:raise ValueError('Full feedback validation requires actuator-driven model, no prescribed mocap bodies')
    joints=[m.joint(n) for n in cfg['joint_names']]
    if any(int(j.type[0]) not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) for j in joints):raise ValueError('Scalar robot joints required')
    qa=[int(j.qposadr[0]) for j in joints];va=[int(j.dofadr[0]) for j in joints];act=[m.actuator(n).id for n in cfg['actuator_names']]
    if len(qa)!=len(act):raise ValueError('Explicit one target per configured actuator is required')
    robot_roots={int(m.jnt_bodyid[j.id]) for j in joints};robot_bodies=set()
    for body in range(1,m.nbody):
        cursor=body
        while cursor:
            if cursor in robot_roots:robot_bodies.add(body);break
            cursor=int(m.body_parentid[cursor])
    if not cfg.get('control_semantics_verified'):raise ValueError('Actuator semantics need explicit verification')
    if cfg['control_mode']!='joint_position_target':raise ValueError('This adapter currently supports joint-position target commands')
    obj=m.body(cfg['object_body']).id;jid=m.joint(cfg['object_free_joint']);oqa=int(jid.qposadr[0])
    if int(jid.type[0])!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('Free object required')
    mujoco.mj_resetData(m,d);d.qpos[qa]=cfg['initial_q'];d.qpos[oqa:oqa+7]=cfg['initial_object_qpos'];mujoco.mj_forward(m,d)
    home=np.array(cfg['initial_q']);target=home.copy()
    command_arrays=None;policy=None;policy_hash=None
    if commands:
        with np.load(commands,allow_pickle=False) as f:command_arrays=dict(f)
        ts=command_arrays['action_t'];us=command_arrays['action'];valid=command_arrays['action_valid'].astype(bool)
        if not np.all(np.diff(ts)>0) or us.shape!=(len(ts),len(act)) or not np.isfinite(us).all():raise ValueError('Bad command stream')
        ts=ts[valid];us=us[valid]
    else:
        path=resolve(cp.parent,cfg['policy_file']);spec=importlib.util.spec_from_file_location('user_policy',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);policy=module.Policy(cfg,m)
        policy_hash=digest(path)
    duration=float(cfg['duration_s']);period=float(cfg['control_period_s']);next_control=0.;rows=[];actions=[];contacts=[]
    force=np.zeros(6);peak=0.;contact_steps=0;trace=[];renderer=None;render_error=None;frames=[];steps=int(np.ceil(duration/m.opt.timestep))
    if record:
        try:
            renderer=mujoco.Renderer(m,height=320,width=480)
            camera=mujoco.MjvCamera();camera.lookat[:]=cfg.get('camera_lookat',[0,0,.05]);camera.distance=cfg.get('camera_distance',.7);camera.azimuth=135;camera.elevation=-30
        except Exception as exc:render_error=str(exc)
    for step in range(steps):
        if command_arrays is not None:
            index=np.searchsorted(ts,d.time,side='right')-1;target=us[index].copy() if index>=0 else home.copy()
        if d.time+1e-9>=next_control:
            observation={'t_s':float(d.time),'q':d.qpos[qa].copy(),'dq':d.qvel[va].copy(),
                         'object_position':d.xpos[obj].copy(),'object_rotation':d.xmat[obj].reshape(3,3).copy()}
            if command_arrays is not None:
                target=target.copy()
            else:target=np.asarray(policy.act(observation),float)
            if target.shape!=(len(act),) or not np.isfinite(target).all():raise ValueError('Invalid policy command')
            for i,aid in enumerate(act):
                if m.actuator_ctrllimited[aid]:target[i]=np.clip(target[i],*m.actuator_ctrlrange[aid])
            actions.append([float(d.time),*target]);next_control+=period
        d.ctrl[act]=target
        if cfg.get('assumed_gravity_compensation',False):d.qfrc_applied[va]=d.qfrc_bias[va]
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all():raise ValueError('Simulation diverged')
        for k in range(d.ncon):
            c=d.contact[k];bodies=m.geom_bodyid[c.geom]
            if obj in bodies:
                other=int(bodies[0] if bodies[1]==obj else bodies[1])
                if other in robot_bodies:
                    mujoco.mj_contactForce(m,d,k,force);peak=max(peak,float(np.linalg.norm(force[:3])));contact_steps+=1
        if step%max(1,round(.02/m.opt.timestep))==0:
            mujoco.mj_forward(m,d)
            rows.append([float(d.time),*d.qpos[qa],*d.qvel[va],*d.xpos[obj],*d.qfrc_constraint[va]])
            trace.append({'t':float(d.time),'p':d.xpos.tolist(),'q':d.xquat.tolist()})
        if renderer and step%max(1,round(.05/m.opt.timestep))==0:
            renderer.update_scene(d,camera=camera);frames.append(renderer.render())
    if renderer:
        import imageio.v2 as imageio
        imageio.mimsave(out/'rollout.mp4',frames,fps=20);renderer.close()
    array=np.array(rows);n=len(qa);action=np.array(actions)
    np.savez_compressed(out/'episode.npz',t=array[:,0],q=array[:,1:1+n],dq=array[:,1+n:1+2*n],
         object_xyz=array[:,1+2*n:4+2*n],generalized_contact_force=array[:,4+2*n:],
         action_t=action[:,0],action=action[:,1:],action_valid=np.ones(len(action),bool))
    goal=np.asarray(cfg['goal_xy_m']);error=float(np.linalg.norm(d.xpos[obj,:2]-goal));tolerance=cfg['success_position_tolerance_m']
    result={'mode':'closed_loop_policy' if policy else 'recorded_command_open_loop_prediction',
       'data_origin':'SIMULATION_NOT_REAL','model_and_assets_sha256':model_digest(model),'config_sha256':digest(cp),
       'policy_hash':policy_hash,'state_overwrite_after_reset':False,'robot_mocap_bodies':m.nmocap,
       'robot_joint_count':n,'contact_samples':contact_steps,'peak_robot_object_contact_force_N':peak,
       'final_object_position_m':d.xpos[obj].tolist(),'goal_position_error_m':error,'success':bool(error<=tolerance),
       'max_abs_generalized_contact_force':np.abs(array[:,4+2*n:]).max(0).tolist(),
       'generalized_force_units':['Nm' if int(j.type[0])==mujoco.mjtJoint.mjJNT_HINGE else 'N' for j in joints],
       'gravity_compensation_is_assumed':cfg.get('assumed_gravity_compensation',False),
       'real_contact_feedback_validated':False,'real_sim_policy_equivalence_validated':False,
       'video_render_error':render_error,'video_rendered':bool(frames)}
    if command_arrays is not None and 'q' in command_arrays:
        if not np.all(np.diff(command_arrays['t'])>0):raise ValueError('Nonmonotonic observation time')
        times=command_arrays['t'];mask=(times>=array[0,0])&(times<=array[-1,0])
        pred=np.stack([np.interp(times[mask],array[:,0],array[:,1+j]) for j in range(n)],1)
        result['measured_robot_prediction_error']=stats(pred-command_arrays['q'][mask])
    dump(out/'feedback_report.json',result);dump(out/'trace.json',trace)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--commands');p.add_argument('--record',action='store_true');a=p.parse_args();run(a.config,a.out,a.commands,a.record)
