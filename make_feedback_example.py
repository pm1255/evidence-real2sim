"""Actuated three-DOF robot fixture with real dynamic reaction inside simulation.

No mocap, no forced recorded state. Not a hardware robot or hardware validation.
"""
import argparse,shutil
from pathlib import Path
from common import dump,fresh


XML='''<mujoco model="actuated_cartesian_pusher"><option timestep="0.001" integrator="implicitfast" iterations="100" gravity="0 0 -9.81"/>
<default><joint damping=".4"/><geom condim="3" friction=".35 .005 .0001" solref=".008 1"/></default>
<worldbody><light pos="0 0 1"/><geom name="table" type="plane" size=".5 .5 .02" rgba=".7 .7 .72 1"/>
<body name="robot_x" pos="0 0 .028"><joint name="axis_x" type="slide" axis="1 0 0" range="-.2 .3" limited="true"/><inertial pos="0 0 0" mass=".4" diaginertia=".001 .001 .001"/>
<body name="robot_y"><joint name="axis_y" type="slide" axis="0 1 0" range="-.15 .15" limited="true"/><inertial pos="0 0 0" mass=".3" diaginertia=".001 .001 .001"/>
<body name="tool"><joint name="axis_yaw" axis="0 0 1"/><geom name="tool_geom" type="sphere" size=".014" mass=".15" rgba=".1 .6 .6 1"/><site name="tcp"/></body></body></body>
<body name="object" pos="0 0 .025"><freejoint name="object_joint"/><geom name="object_geom" type="box" size=".03 .025 .025" mass=".25" rgba=".9 .5 .1 1"/></body>
</worldbody><actuator>
<position name="x_target" joint="axis_x" kp="100" kv="12" forcelimited="true" forcerange="-8 8" ctrllimited="true" ctrlrange="-.2 .3"/>
<position name="y_target" joint="axis_y" kp="100" kv="10" forcelimited="true" forcerange="-8 8" ctrllimited="true" ctrlrange="-.15 .15"/>
<position name="yaw_target" joint="axis_yaw" kp="10" kv="1" forcelimited="true" forcerange="-2 2"/>
</actuator></mujoco>'''


def create(out):
    out=fresh(out);(out/'robot.xml').write_text(XML)
    shutil.copy2(Path(__file__).parent/'examples/feedback_policy.py',out/'feedback_policy.py')
    cfg={'model':'robot.xml','joint_names':['axis_x','axis_y','axis_yaw'],'actuator_names':['x_target','y_target','yaw_target'],
         'control_mode':'joint_position_target','control_semantics_verified':True,'object_body':'object','object_free_joint':'object_joint',
         'initial_q':[-.095,0,0],'initial_object_qpos':[0,0,.025,1,0,0,0],
         'duration_s':5.,'control_period_s':.02,'goal_xy_m':[.12,0],
         'success_position_tolerance_m':.018,'policy_file':'feedback_policy.py','camera_lookat':[.04,0,.025],
         'data_origin':'SYNTHETIC_ACTUATED_ROBOT_FIXTURE'}
    dump(out/'config.json',cfg)
    # Counterfactual identical commands without robot-object collision.
    no_contact=XML.replace('name="tool_geom"','name="tool_geom" contype="0" conaffinity="0"')
    (out/'robot_without_object_contact.xml').write_text(no_contact)
    dump(out/'no_contact.json',dict(cfg,model='robot_without_object_contact.xml'))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();create(a.out)
