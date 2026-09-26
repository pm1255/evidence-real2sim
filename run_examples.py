"""Run portable examples covering missing action, torque, images and object motion.

Base suite uses redistributable synthetic signals only. --vision adds actual
CPU Gaussian optimisation and RGB/depth reconstruction on generated images.
"""
import argparse,subprocess,sys
from pathlib import Path
from common import fresh,dump,read
from make_demo_data import generate
from make_feedback_example import create as feedback_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--vision',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;out=fresh(a.out).resolve();data=generate(out/'inputs');rows=[]
    commands=[('A_actions_without_images',['run_pipeline.py','--manifest',str(data/'controller/manifest.json'),'--out',str(out/'A_actions_without_images')]),
      ('B_joint_states_only',['run_pipeline.py','--manifest',str(data/'q_only/manifest.json'),'--contact-config',str(data/'slide_config.json'),'--out',str(out/'B_joint_states_only')]),
      ('C_object_motion_without_robot',['run_pipeline.py','--manifest',str(data/'slide/manifest.json'),'--contact-config',str(data/'slide_config.json'),'--evaluate-test','--out',str(out/'C_object_motion_without_robot')]),
      ('D_motion_without_torque',['run_pipeline.py','--manifest',str(data/'push/manifest.json'),'--contact-config',str(data/'push_config.json'),'--evaluate-test','--out',str(out/'D_motion_without_torque')]),
      ('E_torque_without_images',['07_torque_residual.py','--manifest',str(data/'torque/manifest.json'),'--config',str(data/'torque_config.json'),'--out',str(out/'E_torque_without_images')])]
    ff=feedback_fixture(out/'feedback_input');commands.append(('F_full_robot_feedback',['robot_feedback.py','--config',str(ff/'config.json'),'--out',str(out/'F_full_robot_feedback')]))
    if a.vision:
        from make_vision_example import create
        vf=create(out/'vision_input',size=96)
        commands.extend([('G_rgb_gaussian_training',['train_gaussians.py','--scene',str(vf/'scene.json'),'--steps','80','--max-side','48','--gaussians','128','--out',str(out/'G_rgb_gaussian_training')]),
           ('H_rgb_dense',['dense_reconstruct.py','--scene',str(vf/'scene.json'),'--max-side','64','--planes','48','--max-views','12','--resolution','48','--out',str(out/'H_rgb_dense')]),
           ('I_rgbd_dense',['dense_reconstruct.py','--scene',str(vf/'scene.json'),'--mode','sensor-depth','--max-side','64','--max-views','12','--resolution','48','--out',str(out/'I_rgbd_dense')])])
    for name,args in commands:
        run=subprocess.run([sys.executable,str(root/args[0]),*args[1:]],capture_output=True,text=True)
        (out/(name+'.log')).write_text(run.stdout+run.stderr);rows.append({'case':name,'exit_code':run.returncode})
        print(name,'ok' if run.returncode==0 else 'FAILED',flush=True)
    dump(out/'examples.json',{'data_origin':'SYNTHETIC_SOFTWARE_EXAMPLES','cases':rows,'real_equivalence_validated':False})
    if any(r['exit_code'] for r in rows):raise SystemExit(2)


if __name__=='__main__':main()
