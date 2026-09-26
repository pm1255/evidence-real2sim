"""Independent simulated state-feedback Panda pushing adapter; requires a matching model.

No copied robot assets. IK computes targets on a separate kinematic MjData;
simulation states are never overwritten after reset.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class Policy:
    def __init__(self,cfg,model):
        self.cfg=cfg;self.m=model;self.d=mujoco.MjData(model);self.site=model.site(cfg['tcp_site']).id
        self.qa=[int(model.joint(n).qposadr[0]) for n in cfg['joint_names']]
        self.va=[int(model.joint(n).dofadr[0]) for n in cfg['joint_names']]
        self.d.qpos[self.qa]=cfg['initial_q'];mujoco.mj_forward(model,self.d)
        self.home=self.d.site_xpos[self.site].copy();self.start=np.asarray(cfg['initial_object_qpos'][:2])
        self.goal=np.asarray(cfg['goal_xy_m']);delta=self.goal-self.start;self.direction=delta/np.linalg.norm(delta)
        self.behind=self.start-self.direction*cfg['approach_offset_m'];self.stopped=False
    def act(self,obs):
        t=obs['t_s'];q=obs['q'];pos=obs['object_position'];safe=np.r_[self.behind,.38];low=np.r_[self.behind,self.cfg['push_height_m']]
        if t<1.5:target=self.home+(safe-self.home)*t/1.5
        elif t<3.:target=safe+(low-safe)*(t-1.5)/1.5
        else:
            if np.linalg.norm(pos[:2]-self.goal)<self.cfg['success_position_tolerance_m']*.75:self.stopped=True
            target=low.copy();target[:2]+=self.direction*min((t-3)*self.cfg.get('speed_m_s',.05),np.linalg.norm(self.goal-self.start)+self.cfg['approach_offset_m'])
            lateral=pos[:2]-self.start-self.direction*np.dot(pos[:2]-self.start,self.direction)
            target[:2]+=.7*lateral
            if self.stopped:target[:2]=pos[:2]-self.direction*self.cfg['approach_offset_m'];target[2]=.3
        self.d.qpos[self.qa]=q;jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp);desired=np.diag([1.,-1.,-1.])
        for _ in range(25):
            mujoco.mj_forward(self.m,self.d);ep=target-self.d.site_xpos[self.site]
            er=Rotation.from_matrix(desired@self.d.site_xmat[self.site].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(ep)<.001 and np.linalg.norm(er)<.015:break
            mujoco.mj_jacSite(self.m,self.d,jp,jr,self.site);J=np.vstack([jp[:,self.va],.25*jr[:,self.va]])
            dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-4,np.r_[ep,.25*er]);new=self.d.qpos[self.qa]+np.clip(dq,-.1,.1)
            for i,name in enumerate(self.cfg['joint_names']):
                j=self.m.joint(name)
                if self.m.jnt_limited[j.id]:new[i]=np.clip(new[i],self.m.jnt_range[j.id,0]+.003,self.m.jnt_range[j.id,1]-.003)
            self.d.qpos[self.qa]=new
        return self.d.qpos[self.qa].copy()
