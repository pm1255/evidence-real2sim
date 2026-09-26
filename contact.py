"""Contact predictors. Only initialise object state once; never teacher-force it."""
from pathlib import Path
import numpy as np
from common import usable,resolve


def angle_error(x):return np.arctan2(np.sin(x),np.cos(x))


def observed(e,cfg):
    a=e['arrays']
    if not usable(e,('object_t','object_pose')):raise ValueError('Missing observed object response')
    if cfg['backend']=='free_slide':
        interval=e['metadata'].get('free_slide_interval_s')
        if interval is None:raise ValueError('Free sliding needs an explicitly annotated interval')
        mask=(a['object_t']>=interval[0])&(a['object_t']<=interval[1])
    else:mask=np.ones(len(a['object_t']),bool)
    if mask.sum()<6:raise ValueError('Too few object observations')
    if 'object_confidence' in a and (a['object_confidence'][mask]<cfg.get('min_confidence',.8)).any():
        raise ValueError('Low-confidence object observations; split reliable segments explicitly')
    t=a['object_t'][mask];pose=a['object_pose'][mask]
    if np.max(np.diff(t))>cfg.get('max_observation_gap_s',.1):raise ValueError('Observation gap too large')
    return t,pose


def free_slide(e,cfg,theta):
    t,pose=observed(e,cfg);meta=e['metadata']
    for key in ('table_horizontal_verified','rotation_negligible_verified','no_pusher_contact_verified'):
        if not meta.get(key):raise ValueError(f'Free-slide assumption unverified: {key}')
    v=np.array(meta['initial_velocity_xy_m_s'],float)
    if v.shape!=(2,) or not np.isfinite(v).all():raise ValueError('Initial velocity required, with documented source')
    speed=float(np.linalg.norm(v))
    if speed<1e-4:raise ValueError('No measurable sliding excitation')
    dt=t-t[0];mu=float(theta[0]);stop=speed/(mu*9.81)
    active=np.minimum(dt,stop);distance=speed*active-.5*mu*9.81*active**2
    pred=np.tile(pose[0],(len(t),1));pred[:,:2]+=distance[:,None]*v/speed
    return pred


class MujocoPredictor:
    def __init__(self,cfg,base):
        import mujoco
        self.mj=mujoco;self.cfg=cfg
        self.m=mujoco.MjModel.from_xml_path(str(resolve(base,cfg['model'])))
        self.d=mujoco.MjData(self.m)
        joint=self.m.joint(cfg['object_free_joint'])
        if int(joint.type[0])!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('Object needs a free joint')
        self.qa=int(joint.qposadr[0]);self.va=int(joint.dofadr[0])
        self.bid=int(self.m.jnt_bodyid[joint.id]);self.mid=int(self.m.body(cfg['pusher_mocap_body']).mocapid[0])
        if self.mid<0:raise ValueError('Pusher must be a prescribed mocap boundary in this conditional stage')
        self.pairs=[self.m.pair(p['pair_name']).id for p in cfg['parameters']]
        self.dt=float(self.m.opt.timestep)
    def __call__(self,e,theta,dt_factor=1.):
        a=e['arrays'];meta=e.get('metadata',{});cfg=self.cfg
        t,pose=observed(e,cfg)
        if not usable(e,('pusher_t','pusher_pose')):raise ValueError('Missing actual pusher trajectory')
        if not meta.get('metric_geometry_and_sync_verified'):raise ValueError('Metric geometry and clock alignment must be verified first')
        pt=a['pusher_t'];pp=a['pusher_pose']
        if pt[0]>t[0] or pt[-1]<t[-1] or np.max(np.diff(pt))>cfg.get('max_pusher_gap_s',.1):raise ValueError('Pusher motion does not cover observations without large gaps')
        mj=self.mj;m=self.m;d=self.d;mj.mj_resetData(m,d)
        for pair,value in zip(self.pairs,theta):m.pair_friction[pair,:2]=value
        yaw=pose[0,2];qa=self.qa;va=self.va
        d.qpos[qa:qa+7]=[pose[0,0],pose[0,1],meta['initial_object_z_m'],np.cos(yaw/2),0,0,np.sin(yaw/2)]
        twist=np.asarray(meta['initial_twist_xy_yaw'],float)
        if twist.shape!=(3,):raise ValueError('Initial twist must be independently specified (including declared rest)')
        d.qvel[va:va+6]=[twist[0],twist[1],0,0,0,twist[2]]
        pyaw=np.unwrap(pp[:,2]);pred=[];d.time=0
        def boundary(time):
            absolute=t[0]+time
            xy=[np.interp(absolute,pt,pp[:,j]) for j in range(2)]
            y=np.interp(absolute,pt,pyaw)
            d.mocap_pos[self.mid]=[*xy,cfg['pusher_z_m']]
            d.mocap_quat[self.mid]=[np.cos(y/2),0,0,np.sin(y/2)]
        boundary(0);mj.mj_forward(m,d)
        for target in t-t[0]:
            while d.time<target-1e-10:
                step=min(self.dt*dt_factor,target-d.time);m.opt.timestep=step
                boundary(d.time+step);mj.mj_step(m,d)
                if not np.isfinite(d.qpos).all():raise ValueError('Nonfinite simulation')
            mj.mj_forward(m,d);R=d.xmat[self.bid].reshape(3,3)
            pred.append([d.xpos[self.bid,0],d.xpos[self.bid,1],np.arctan2(R[1,0],R[0,0])])
        m.opt.timestep=self.dt
        return np.array(pred)


def errors(pred,truth,cfg):
    pos=(pred[:,:2]-truth[:,:2])/cfg['position_scale_m']
    yaw=angle_error(pred[:,2]-truth[:,2])/cfg['yaw_scale_rad']
    return np.c_[pos,yaw].ravel()
