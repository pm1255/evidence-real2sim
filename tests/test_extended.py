"""Independent geometry/physics invariants; no hardware-validation claims."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_gaussians import Gaussians,quat_matrix
from dense_reconstruct import sample_pixels,backproject,project,reconstruct
from make_vision_example import create as vision_fixture
from make_feedback_example import create as feedback_fixture
from robot_feedback import run
from common import read,dump,digest
from paired_protocol import freeze,audit
from vision_data import load_scene


class ExtendedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_01_gaussian_gradients_are_finite(self):
        model=Gaussians(np.array([[0.,0.,1.],[.04,0,1.1]]),np.array([[.8,.2,.1],[.1,.3,.8]]),np.array([.03,.02]))
        K=torch.tensor([[100.,0,16],[0,100,16],[0,0,1.]]);T=torch.eye(4)
        image,alpha=model.render(K,T,32,32);image.mean().backward()
        self.assertTrue(torch.isfinite(image).all());self.assertGreater(float(alpha.max()),0)
        for p in model.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
    def test_02_quaternion_rotation_is_orthogonal(self):
        R=quat_matrix(torch.tensor([[1.,2.,3.,4.]]));np.testing.assert_allclose((R@R.transpose(1,2)).numpy(),np.eye(3)[None],atol=1e-6)
    def test_03_large_pixel_batch_no_opencv_limit(self):
        im=np.arange(100,dtype=np.float32).reshape(10,10);uv=np.tile([[3.,4.]],(40000,1))
        values=sample_pixels(im,uv);np.testing.assert_array_equal(values,np.full(40000,43.))
    def test_04_projection_backprojection_roundtrip(self):
        f={'K':np.array([[60.,0,5],[0,60,4],[0,0,1]]),'T':np.eye(4)}
        depth=np.full((9,11),.6);xyz=backproject(f,depth);uv,z=project(xyz.reshape(-1,3),f)
        y,x=np.mgrid[:9,:11];np.testing.assert_allclose(uv,np.c_[x.ravel(),y.ravel()],atol=1e-10)
    def test_05_tsdf_against_analytic_geometry(self):
        fixture=vision_fixture(self.root/'vision',size=64,views=12)
        result=reconstruct(fixture/'scene.json',self.root/'dense','sensor-depth',64,24,10,48)
        self.assertGreater(result['observed_points'],500)
        import trimesh
        points=trimesh.load(self.root/'dense/surface.ply',process=False).vertices
        q=np.abs(points-np.array([0,0,.05]))-np.array([.06,.04,.05])
        box=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
        sphere=np.linalg.norm(points-np.array([.105,.005,.04]),axis=1)-.035
        self.assertLess(np.quantile(np.abs(np.minimum(box,sphere)),.95),.005)
        self.assertFalse(result['contact_surface_accuracy_certified'])
    def test_06_two_way_feedback_changes_robot_motion(self):
        folder=feedback_fixture(self.root/'feedback')
        a=run(folder/'config.json',self.root/'contact')
        b=run(folder/'no_contact.json',self.root/'no_contact',self.root/'contact/episode.npz')
        qa=np.load(self.root/'contact/episode.npz')['q'];qb=np.load(self.root/'no_contact/episode.npz')['q']
        self.assertGreater(a['contact_samples'],100);self.assertEqual(b['contact_samples'],0)
        self.assertGreater(np.max(np.abs(qa-qb)),.001)
        self.assertTrue(a['success']);self.assertFalse(b['success']);self.assertFalse(a['state_overwrite_after_reset'])
    def test_07_synthetic_trials_cannot_pass_real_gate(self):
        protocol=self.root/'protocol';artifact=self.root/'artifact.txt';artifact.write_text('test fixture')
        spec={'scope':{'hardware_ids':['fixture']},'policies':{'a':'hash'},'scoring_rule':'fixture only',
              'max_success_gap_pp':10,'max_sim_success_real_failure_rate':.1,'min_real_conditions_per_task':30,
              'minimum_within_task_spearman':.8,'artifacts':{'test':str(artifact)}}
        dump(self.root/'spec.json',spec);freeze(self.root/'spec.json',protocol)
        dump(self.root/'trials.json',{'data_origin':'SYNTHETIC','protocol_sha256':digest(protocol/'protocol.json'),'trials':[]})
        audit(protocol/'protocol.json',self.root/'trials.json',self.root/'gate')
        result=read(self.root/'gate/paired_gate.json');self.assertEqual(result['status'],'not_real_paired_evidence')
    def test_08_duplicate_fit_and_test_images_rejected(self):
        f=self.root/'duplicate_vision';f.mkdir();cv2.imwrite(str(f/'im.png'),np.zeros((8,8,3),np.uint8))
        frames=[{'id':str(i),'image':'im.png','K':np.eye(3).tolist(),'world_to_camera':np.eye(4).tolist(),'split':split} for i,split in enumerate(['fit','fit','test'])]
        dump(f/'scene.json',{'schema':'calibrated-scene-v1','camera_convention':'opencv_world_to_camera','frames':frames})
        with self.assertRaises(ValueError):load_scene(f/'scene.json')


if __name__=='__main__':unittest.main(verbosity=2)
