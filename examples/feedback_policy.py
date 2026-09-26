"""State-feedback pushing policy for the redistributable Cartesian robot fixture."""
import numpy as np


class Policy:
    def __init__(self,config,model):
        self.cfg=config;self.goal=np.array(config['goal_xy_m']);self.speed=config.get('push_speed_m_s',.06)
    def act(self,observation):
        t=observation['t_s'];obj=observation['object_position'];q=observation['q']
        error=self.goal-obj[:2]
        if np.linalg.norm(error)<self.cfg['success_position_tolerance_m']*.7:
            return np.array([q[0]-.03,q[1],0.])
        desired=np.array([-.095+self.speed*t,obj[1]*.8,0.])
        desired[0]=min(desired[0],self.goal[0]-.018)
        return desired
