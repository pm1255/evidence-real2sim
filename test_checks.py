"""Scientific-software regression checks, explicitly not real-robot experiments."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from common import bundle,read,dump,digest,model_digest,validate_episode,check_independent
from make_demo_data import generate
from contact import MujocoPredictor,free_slide

ROOT=Path(__file__).resolve().parent


def module(name):
    s=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name);cls.data=generate(cls.base/'data')
        cls.control=module('05_fit_controller');cls.contact=module('06_fit_contact');cls.audit=module('02_audit_inputs')
        cls.policies=module('09_evaluate_policies');cls.images=module('03_images');cls.tracks=module('10_prepare_tracks')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def run_script(self,name,*args):
        r=subprocess.run([sys.executable,str(ROOT/name),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    def test_01_nonmonotonic_times_rejected(self):
        e={'arrays':{'t':np.array([0.,1.,.9]),'q':np.zeros((3,2))}}
        self.assertTrue(validate_episode(e)[0])
    def test_02_nan_not_imputed(self):
        e={'arrays':{'t':np.array([0.,1.]),'q':np.array([[0.],[np.nan]])}}
        self.assertTrue(validate_episode(e)[0])
    def test_03_shape_mismatch_rejected(self):
        e={'arrays':{'t':np.arange(3.),'q':np.zeros((2,2))}}
        self.assertTrue(validate_episode(e)[0])
    def test_04_q_only_degrades(self):
        r=self.audit.audit(self.data/'q_only/manifest.json')
        self.assertEqual(r['episodes'][0]['routes']['contact_from_motion'],'missing_object_response')
        self.assertEqual(r['episodes'][0]['routes']['controller'],'missing_or_unsupported')
    def test_05_missing_contact_skips(self):
        r=self.contact.fit(self.data/'q_only/manifest.json',self.data/'slide_config.json',self.base/'missing')
        self.assertEqual(r['status'],'skipped')
    def test_06_command_skip_is_not_zero_target(self):
        _,e=bundle(self.data/'controller/manifest.json');ep=e[0]
        a=self.control.rollout(ep,np.array([.18,.3]),.02)
        ep['arrays']['action'][~ep['arrays']['action_valid']]=10000
        b=self.control.rollout(ep,np.array([.18,.3]),.02)
        np.testing.assert_allclose(a,b)
    def test_07_mid_interval_command_not_applied_early(self):
        e={'arrays':{'t':np.array([0.,1.]),'q':np.zeros((2,1)),
           'action_t':np.array([.5,2.]),'action':np.array([[1.],[0.]]),'action_valid':np.ones(2,bool)}}
        p=self.control.rollout(e,np.array([1.]),0)
        self.assertAlmostEqual(p[-1,0],1-np.exp(-.5),places=12)
    def test_08_known_controller_recovered(self):
        r=self.control.fit(self.data/'controller/manifest.json',self.base/'control')
        np.testing.assert_allclose(r['tau_s'],[.18,.3],atol=1e-5)
        self.assertAlmostEqual(r['additional_delay_s'],.02)
        self.assertLess(max(e['whole_episode_error_rad']['rmse'] for e in r['heldout']),1e-6)
    def test_09_known_slide_friction_and_holdout(self):
        fitdir=self.base/'slide_fit';r=self.contact.fit(self.data/'slide/manifest.json',self.data/'slide_config.json',fitdir)
        self.assertAlmostEqual(r['theta'][0],.23,places=3)
        evaluation=self.contact.evaluate(self.data/'slide/manifest.json',self.data/'slide_config.json',fitdir/'contact_model.json',self.base/'slide_test','test')
        self.assertIsNone(evaluation['configured_numerical_gate_passed'])
        self.assertLess(max(x['position_m']['rmse'] for x in evaluation['rows']),.0002)
    def test_10_training_overlap_rejected(self):
        with self.assertRaises(ValueError):check_independent({'fit_episodes':[{'id':'a','sha256':'h'}]},[{'id':'b','sha256':'h'}])
    def test_11_changed_model_asset_changes_fingerprint(self):
        p=self.base/'dependency.xml';mesh=self.base/'mesh.obj';mesh.write_text('v 0 0 0\n')
        p.write_text('<mujoco><asset><mesh name="x" file="mesh.obj"/></asset></mujoco>')
        first=model_digest(p);mesh.write_text('v 1 0 0\n');self.assertNotEqual(first,model_digest(p))
    def test_12_mujoco_prediction_not_forced_to_observed_object(self):
        _,es=bundle(self.data/'push/manifest.json');cfg=read(self.data/'push_config.json');p=MujocoPredictor(cfg,self.data)
        e=es[-1];truth=e['arrays']['object_pose'].copy();pred=p(e,[.27]);np.testing.assert_allclose(pred,truth,atol=1e-12)
        e['arrays']['object_pose'][1:,:2]+=10
        np.testing.assert_allclose(p(e,[.27]),pred,atol=1e-12)
    def test_13_unsynchronized_pusher_rejected(self):
        _,es=bundle(self.data/'push/manifest.json');cfg=read(self.data/'push_config.json');p=MujocoPredictor(cfg,self.data)
        es[0]['metadata']['metric_geometry_and_sync_verified']=False
        with self.assertRaises(ValueError):p(es[0],[.27])
    def test_14_slide_requires_no_contact_evidence(self):
        _,es=bundle(self.data/'slide/manifest.json');cfg=read(self.data/'slide_config.json')
        es[0]['metadata']['no_pusher_contact_verified']=False
        with self.assertRaises(ValueError):free_slide(es[0],cfg,[.23])
    def test_15_independent_geometry_offset_recovered(self):
        out=self.base/'alignment';self.run_script('04_align_robot.py','--config',self.data/'alignment_config.json','--out',out)
        r=read(out/'alignment.json');self.assertEqual(r['status'],'estimated')
        self.assertAlmostEqual(r['selected_joint_zero_offsets_rad']['j1'],.02,places=5)
    def test_16_joint_base_gauge_degeneracy_detected(self):
        cfg=read(self.data/'alignment_config.json');cfg['fit_zero_joint_indices']=[0]
        cfg['model']=str(self.data/'arm.xml');cfg['data']=str(self.data/'alignment.npz')
        path=self.base/'degenerate.json';dump(path,cfg);out=self.base/'degenerate'
        self.run_script('04_align_robot.py','--config',path,'--out',out)
        self.assertEqual(read(out/'alignment.json')['status'],'degenerate_do_not_apply_offsets')
    def test_17_known_free_motion_torque_residual_small(self):
        out=self.base/'torque';self.run_script('07_torque_residual.py','--manifest',self.data/'torque/manifest.json','--config',self.data/'torque_config.json','--out',out)
        r=read(out/'torque_report.json');self.assertFalse(r['contact_force_validated'])
        self.assertLess(r['episodes'][0]['derivative_sensitivity'][0]['residual_Nm']['rmse'],.001)
    def test_18_mesh_topology_not_real_validation(self):
        out=self.base/'asset';self.run_script('08_assets_and_layouts.py','asset','--config',self.data/'asset_config.json','--out',out)
        r=read(out/'asset.json');self.assertTrue(r['watertight']);self.assertFalse(r['benchmark_eligible'])
    def policy_data(self):
        rows=[]
        for condition in range(8):
            for policy in ('a','b'):
                for env in ('real','sim'):
                    rows.append({'task_id':'push','condition_id':str(condition),'policy_id':policy,'environment':env,
                                 'success':bool(condition<(6 if policy=='a' else 3)),
                                 'policy_hash':policy,'observation_action_contract_hash':'o','scoring_rule_hash':'s','scope_hash':'g'})
        return {'data_origin':'SYNTHETIC','trials':rows}
    def test_19_same_task_paired_policy_metrics(self):
        path=self.base/'policies.json';dump(path,self.policy_data());r=self.policies.evaluate(path)
        self.assertAlmostEqual(r['tasks'][0]['within_task_spearman'],1)
        self.assertFalse(r['evaluation_equivalence_certified']);self.assertTrue(r['tasks'][0]['small_sample_warning'])
    def test_20_mismatched_policy_pair_rejected(self):
        d=self.policy_data();d['trials'][0]['policy_hash']='different';path=self.base/'bad_policy.json';dump(path,d)
        with self.assertRaises(ValueError):self.policies.evaluate(path)
    def test_21_episode_end_not_accepted_as_success(self):
        d=self.policy_data();del d['trials'][0]['success'];d['trials'][0]['next.done']=True
        path=self.base/'done.json';dump(path,d)
        with self.assertRaises(KeyError):self.policies.evaluate(path)
    def test_22_camera_marker_and_missing_frame(self):
        import cv2
        from scipy.spatial.transform import Rotation
        folder=self.base/'images';folder.mkdir();K=np.array([[600.,0,320],[0,600.,240],[0,0,1.]])
        dump(folder/'camera.json',{'K':K.tolist(),'distortion':[0,0,0,0,0],'image_size':[640,480]})
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),7,200)
        obj=np.array([[-.04,.04,0],[.04,.04,0],[.04,-.04,0],[-.04,-.04,0]])
        rv=Rotation.from_euler('xyz',[2.8,.15,.05]).as_rotvec();tv=np.array([.01,.02,.5])
        uv=cv2.projectPoints(obj,rv,tv,K,np.zeros(5))[0].reshape(4,2).astype(np.float32)
        H=cv2.getPerspectiveTransform(np.array([[0,0],[199,0],[199,199],[0,199]],np.float32),uv)
        im=cv2.warpPerspective(marker,H,(640,480),borderValue=255)
        cv2.imwrite(str(folder/'marker.png'),im);cv2.imwrite(str(folder/'blank.png'),np.full((480,640),255,np.uint8))
        cfg={'camera':'camera.json','marker_id':7,'marker_size_m':.08,'max_reprojection_px':2.,'min_solution_gap_px':.01,
             'frames':[{'file':'marker.png','t_s':0.},{'file':'blank.png','t_s':.02}]}
        out=folder/'out';out.mkdir();self.images.track(cfg,folder,out);records=read(out/'marker_tracks.json')['records']
        self.assertTrue(records[0]['valid']);self.assertFalse(records[1]['valid'])
        np.testing.assert_allclose(records[0]['translation_camera_m'],tv,atol=.006)
    def test_23_track_transform_rejects_fake_metric_scale(self):
        T=np.eye(4);T[0,0]=2
        with self.assertRaises(ValueError):self.tracks.transform(T)
    def test_24_duplicate_content_across_split_rejected(self):
        path=self.base/'duplicates';path.mkdir();arrays={'t':np.arange(3.),'q':np.ones((3,1))}
        rows=[]
        for i,split in enumerate(('fit','test')):
            f=path/f'{i}.npz';np.savez_compressed(f,**arrays)
            rows.append({'id':str(i),'file':f.name,'sha256':digest(f),'split':split})
        dump(path/'manifest.json',{'schema':'adaptive-real2sim-v1','episodes':rows})
        with self.assertRaises(ValueError):bundle(path/'manifest.json')
    def test_25_chessboard_intrinsics_and_heldout_views(self):
        import cv2
        folder=self.base/'chessboard';folder.mkdir();K=np.array([[650.,0,320],[0,650.,240],[0,0,1.]])
        board=np.full((600,700),255,np.uint8)
        for y in range(6):
            for x in range(7):
                if (x+y)%2==0:board[y*100:(y+1)*100,x*100:(x+1)*100]=0
        outer=np.array([[0.,0,0],[.21,0,0],[.21,.18,0],[0,.18,0]])
        frames=[]
        for i in range(10):
            rv=np.array([-.3+.075*i,.25*np.sin(i),.08*np.cos(i)])
            uv=cv2.projectPoints(outer,rv,np.array([-.1,-.08,.5+.016*i]),K,np.zeros(5))[0].reshape(4,2).astype(np.float32)
            H=cv2.getPerspectiveTransform(np.array([[0,0],[699,0],[699,599],[0,599]],np.float32),uv)
            image=cv2.warpPerspective(board,H,(640,480),borderValue=255)
            path=folder/f'{i}.png';cv2.imwrite(str(path),image)
            frames.append({'file':path.name,'split':'fit' if i<8 else 'test'})
        cfg={'inner_corners':[6,5],'square_m':.03,'frames':frames,'_hash':'synthetic'}
        out=folder/'calibration';out.mkdir();self.images.calibrate(cfg,folder,out);r=read(out/'camera.json')
        self.assertEqual(len(r['heldout_views']),2)
        self.assertLess(abs(r['K'][0][0]-650)/650,.08)
        self.assertLess(max(v['reprojection_rmse_px'] for v in r['heldout_views']),1.)
    def test_26_world_track_clock_and_marker_offset(self):
        folder=self.base/'track_conversion';folder.mkdir()
        rows=[{'t_s':t,'valid':True,'rotation_vector':[0,0,0],'translation_camera_m':[t,0,.5]} for t in (10.,10.1,10.2)]
        dump(folder/'tracks.json',{'records':rows});T=np.eye(4);T[0,3]=.1
        cfg={'tracks':'tracks.json','T_world_camera':T.tolist(),'T_marker_body':np.eye(4).tolist(),
             'clock_camera_to_robot':{'a':1.,'b':2.,'origin_s':10.,'uncertainty_s':.001},
             'max_tilt_rad':.1,'world_pose_accuracy_m':.002}
        t,poses,quality=self.tracks.convert(cfg,folder)
        np.testing.assert_allclose(t,[2.,2.1,2.2]);np.testing.assert_allclose(poses[:,0],[10.1,10.2,10.3])
    def test_27_missing_timestamp_audit_does_not_claim_ready(self):
        folder=self.base/'missing_time';folder.mkdir();p=folder/'data.npz'
        np.savez_compressed(p,q=np.zeros((3,1)),action=np.zeros((3,1)),action_t=np.arange(3.),action_valid=np.ones(3,bool))
        dump(folder/'manifest.json',{'schema':'adaptive-real2sim-v1','data_origin':'SYNTHETIC','episodes':[
            {'id':'missing_time','split':'fit','file':p.name,'sha256':digest(p),'metadata':{
                'action':'joint_position_target','action_source':'issued_command_log'}}]})
        r=self.audit.audit(folder/'manifest.json');self.assertEqual(r['valid_episode_count'],0)


if __name__=='__main__':unittest.main(verbosity=2)
