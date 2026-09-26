"""Generate explicitly SYNTHETIC fixtures; these never count as real validation."""
import argparse
from pathlib import Path
import importlib.util
import numpy as np
from scipy.spatial.transform import Rotation
from common import fresh,dump,digest
from contact import MujocoPredictor


PUSH_XML='''<mujoco model="synthetic_push"><option timestep="0.002" iterations="80" gravity="0 0 -9.81"/>
<worldbody><geom name="table" type="plane" size="1 1 .1"/>
<body name="object" pos="0 0 .025"><freejoint name="object_joint"/><geom name="object_geom" type="box" size=".03 .025 .025" mass=".2"/></body>
<body name="pusher" mocap="true" pos="-.15 0 .022"><geom name="pusher_geom" type="sphere" size=".012"/></body></worldbody>
<contact><pair name="object_table" geom1="object_geom" geom2="table" condim="3" friction=".27 .27 .001 .0001 .0001" solref=".005 1"/>
<pair name="object_pusher" geom1="object_geom" geom2="pusher_geom" condim="3" friction=".6 .6 .001 .0001 .0001" solref=".005 1"/></contact></mujoco>'''

ARM_XML='''<mujoco model="synthetic_arm"><option gravity="0 -9.81 0"/><worldbody>
<body name="a"><joint name="j0" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .25 0 0" size=".02" mass="1" contype="0" conaffinity="0"/>
<body name="b" pos=".25 0 0"><joint name="j1" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".7" contype="0" conaffinity="0"/>
<body name="c" pos=".2 0 0"><joint name="j2" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .15 0 0" size=".02" mass=".4" contype="0" conaffinity="0"/><site name="tcp" pos=".15 0 0"/></body></body></body>
</worldbody></mujoco>'''


def save_bundle(out,name,episodes):
    folder=out/name;folder.mkdir();rows=[]
    for eid,split,arrays,meta in episodes:
        path=folder/(eid+'.npz');np.savez_compressed(path,**arrays)
        rows.append({'id':eid,'split':split,'file':path.name,'sha256':digest(path),'metadata':meta})
    dump(folder/'manifest.json',{'schema':'adaptive-real2sim-v1','data_origin':'SYNTHETIC_SOFTWARE_FIXTURE_NOT_REAL','episodes':rows})


def generate(out):
    out=fresh(out);rng=np.random.default_rng(2026)
    root=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('controller',root/'05_fit_controller.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    episodes=[]
    for i,split in enumerate(['fit','fit','fit','test','test']):
        t=np.arange(0,4,.025);at=np.arange(0,4,.11);u=np.c_[.4*np.sin(at*(1+i*.1)),.3*np.cos(at*1.3)]
        valid=np.ones(len(at),bool);valid[::7]=False
        q=np.zeros((len(t),2));arrays={'t':t,'q':q,'action_t':at,'action':u,'action_valid':valid}
        arrays['q']=control.rollout({'arrays':arrays},np.array([.18,.3]),.02)
        meta={'action':'joint_position_target','action_source':'issued_command_log','units':{'q':'rad','time':'s'}}
        episodes.append((f'controller_{i}',split,arrays,meta))
    save_bundle(out,'controller',episodes)
    # Missing-action and missing-torque variants must degrade explicitly.
    qonly=[]
    for eid,split,a,m in episodes:qonly.append((eid,split,{k:v for k,v in a.items() if k in ('q','t')},m))
    save_bundle(out,'q_only',qonly)
    cfg={'backend':'free_slide','parameters':[{'name':'effective_table_mu','bounds':[.03,.8]}],
         'position_scale_m':.001,'yaw_scale_rad':.02,'max_observation_gap_s':.1,
         'search_samples':12,'acceptance':{'max_position_rmse_m':None,'max_yaw_rmse_rad':None}}
    dump(out/'slide_config.json',cfg);episodes=[]
    for i,split in enumerate(['fit','fit','fit','test','test']):
        t=np.arange(0,.9,.02);speed=.55+i*.06;direction=np.array([np.cos(i*.4),np.sin(i*.4)])
        active=np.minimum(t,speed/(.23*9.81));distance=speed*active-.5*.23*9.81*active**2
        pose=np.c_[distance[:,None]*direction,np.full(len(t),i*.4)]
        noise=rng.normal(0,.00005,pose[:,:2].shape);noise[0]=0;pose[:,:2]+=noise
        meta={'free_slide_interval_s':[0,.9],'initial_velocity_xy_m_s':(speed*direction).tolist(),
              'initial_velocity_source':'synthetic_known_initial_state','table_horizontal_verified':True,
              'rotation_negligible_verified':True,'no_pusher_contact_verified':True}
        episodes.append((f'slide_{i}',split,{'object_t':t,'object_pose':pose},meta))
    save_bundle(out,'slide',episodes)
    (out/'push.xml').write_text(PUSH_XML)
    cfg={'backend':'mujoco_prescribed_pusher','model':'push.xml','object_free_joint':'object_joint',
         'pusher_mocap_body':'pusher','pusher_z_m':.022,
         'parameters':[{'name':'object_table_mu','pair_name':'object_table','bounds':[.05,.7]}],
         'position_scale_m':.001,'yaw_scale_rad':.02,'search_samples':8,
         'acceptance':{'max_position_rmse_m':None,'max_yaw_rmse_rad':None}}
    dump(out/'push_config.json',cfg);predict=MujocoPredictor(cfg,out);episodes=[]
    for i,split in enumerate(['fit','fit','test']):
        t=np.arange(0,1.52,.02);speed=.25+i*.045;offset=[0,.012,-.008][i]
        px=np.where(t<1,-.1+speed*t,-.1+speed-.6*(t-1))
        a={'object_t':t,'object_pose':np.zeros((len(t),3)),'pusher_t':t,
           'pusher_pose':np.c_[px,np.full(len(t),offset),np.zeros(len(t))]}
        meta={'metric_geometry_and_sync_verified':True,'initial_object_z_m':.025,
              'initial_twist_xy_yaw':[0,0,0],'initial_state_source':'synthetic_known_initial_state'}
        a['object_pose']=predict({'arrays':a,'metadata':meta,'id':f'push_{i}'},[.27])
        episodes.append((f'push_{i}',split,a,meta))
    save_bundle(out,'push',episodes)
    (out/'arm.xml').write_text(ARM_XML)
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(out/'arm.xml'));d=mujoco.MjData(m)
    q=rng.uniform(-1,1,(42,3));points=[]
    for row in q:
        d.qpos[:]=row+[0,.02,0];mujoco.mj_forward(m,d);points.append(d.site_xpos[0].copy())
    R=Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix();obs=np.array(points)@R.T+[.12,-.08,.3]
    np.savez_compressed(out/'alignment.npz',q=q,observed_xyz=obs,split=np.r_[np.zeros(30,int),np.ones(12,int)])
    dump(out/'alignment_config.json',{'model':'arm.xml','data':'alignment.npz','joint_names':['j0','j1','j2'],
         'tcp_site':'tcp','fit_zero_joint_indices':[1],'observation_sigma_m':.001,
         'observations_independent_of_robot_fk':True,
         'data_origin':'SYNTHETIC_known_transform_and_offset_not_real_calibration'})
    t=np.arange(0,4,.025);q=np.c_[.2*np.sin(t),.2*np.cos(t*1.2),.1*np.sin(t*.8)]
    dq=np.c_[.2*np.cos(t),-.24*np.sin(t*1.2),.08*np.cos(t*.8)]
    ddq=np.c_[-.2*np.sin(t),-.288*np.cos(t*1.2),-.064*np.sin(t*.8)];tau=[]
    for qq,v,acc in zip(q,dq,ddq):
        mujoco.mj_resetData(m,d);d.qpos[:]=qq;d.qvel[:]=v;d.qacc[:]=acc;mujoco.mj_inverse(m,d);tau.append(d.qfrc_inverse.copy())
    save_bundle(out,'torque',[('torque_0','fit',{'t':t,'q':q,'dq':dq,'tau':np.array(tau)},
                               {'tau_source':'synthetic_inverse_dynamics','tau_semantics_verified':True})])
    dump(out/'torque_config.json',{'model':'arm.xml','joint_names':['j0','j1','j2'],
                                  'smoothing_windows_samples':[7,11]})
    dump(out/'torque_inventory_config.json',{'model':None})
    # Texture-free metric asset and explicit candidate layout examples.
    import trimesh
    trimesh.creation.box(extents=[.06,.05,.05]).export(out/'box.ply')
    dump(out/'asset_config.json',{'asset_id':'synthetic_box','mesh':'box.ply','meters_per_unit':1,
                                 'metric_scale_source':'synthetic_exact_dimensions'})
    dump(out/'truth.json',{'warning':'All data in this folder is synthetic, for software checks only',
                          'controller_tau_s':[.18,.3],'controller_delay_s':.02,'slide_mu':.23,
                          'push_mu':.27,'joint_1_offset_rad':.02})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();generate(a.out)


if __name__=='__main__':main()
