"""Input-scope regressions, not photorealism or real-robot validation."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_workspace import prepare
from vision_data import load_scene


class WorkspaceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(52)
        frames = []
        for i in range(4):
            pixels = rng.integers(0, 256, (53, 79, 3), dtype=np.uint8)
            cv2.imwrite(str(self.root / f'{i}.png'), pixels)
            labels = np.zeros((53, 79), dtype=np.uint8)
            labels[10:20, 15:25] = 1
            np.save(self.root / f'{i}.npy', labels)
            frames.append({'id': f'view_{i}', 'image': f'{i}.png', 'instance_labels': f'{i}.npy',
                           'split': 'test' if i == 3 else 'fit',
                           'K': [[80, 0, 39], [0, 80, 26], [0, 0, 1]],
                           'world_to_camera': np.eye(4).tolist()})
        pose = np.eye(4); pose[:3, 3] = [.2, .3, .8]
        self.cfg = {'schema': 'robot-workspace-capture-v1', 'frame_scope': 'full_workspace',
                    'capture_mode': 'moving_camera_static_workspace', 'scene_static_during_capture': True,
                    'camera_convention': 'opencv_world_to_camera', 'coordinate_units': 'meters',
                    'metric_scale_source': 'SYNTHETIC_TEST_ONLY', 'bounds_m': [[-1, -1, 0], [1, 1, 2]],
                    'data_origin': 'SYNTHETIC_INPUT_CONTRACT_TEST', 'frames': frames,
                    'instances': [{'id': 'table', 'role': 'support_surface'},
                                  {'id': 'bottle', 'role': 'rigid_object', 'label_id': 1,
                                   'support_id': 'table', 'world_from_instance': pose.tolist()}]}

    def tearDown(self):
        self.temp.cleanup()

    def execute(self):
        p = self.root / 'capture.json'; p.write_text(json.dumps(self.cfg))
        return prepare(p, self.root / 'prepared')

    def test_full_frames_and_background_preserved(self):
        result = self.execute()
        self.assertTrue(result['calibrated_scene_exported'])
        self.assertFalse(result['whole_scene_reconstruction_completed'])
        self.assertEqual((self.root / '0.png').read_bytes(), (self.root / 'prepared/images/view_0.png').read_bytes())
        _, frames, _ = load_scene(self.root / 'prepared/scene.json')
        self.assertEqual(frames[0]['rgb'].shape, (53, 79, 3))
        self.assertTrue(frames[0]['mask'].all())  # Object labels must not hide background.

    def test_unknown_poses_are_not_identity_guesses(self):
        for f in self.cfg['frames']:
            f.pop('world_to_camera')
        result = self.execute()
        self.assertFalse(result['calibrated_scene_exported'])
        self.assertFalse((self.root / 'prepared/scene.json').exists())
        rows = json.loads((self.root / 'prepared/capture.json').read_text())['frames']
        self.assertTrue(all('world_to_camera' not in f for f in rows))

    def test_dynamic_sequence_not_silently_fused(self):
        self.cfg['capture_mode'] = 'dynamic_episode'
        result = self.execute()
        self.assertFalse(result['calibrated_scene_exported'])

    def test_single_object_crop_rejected(self):
        self.cfg['frames'][0]['mask'] = '0.npy'
        with self.assertRaisesRegex(ValueError, 'foreground mask'):
            self.execute()
        self.assertFalse((self.root / 'prepared').exists())

    def test_shared_layout_is_preserved_but_not_certified(self):
        self.execute()
        graph = json.loads((self.root / 'prepared/scene_graph.json').read_text())
        self.assertEqual(graph['instances'], self.cfg['instances'])
        self.assertFalse(graph['original_layout_measured'])
        self.assertFalse(graph['interactive_scene_validated'])

    def test_turntable_and_reflection_rejected(self):
        self.cfg['capture_mode'] = 'object_turntable'
        with self.assertRaisesRegex(ValueError, 'turntables'):
            self.execute()
        self.cfg['capture_mode'] = 'moving_camera_static_workspace'
        self.cfg['frames'][0]['world_to_camera'][0][0] = -1
        with self.assertRaisesRegex(ValueError, 'proper rigid'):
            self.execute()

    def test_unknown_instance_label_rejected(self):
        np.save(self.root / '0.npy', np.full((53, 79), 99, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'absent from inventory'):
            self.execute()

    def test_duplicate_capture_rejected(self):
        self.cfg['frames'][3]['image'] = '0.png'
        with self.assertRaisesRegex(ValueError, 'Duplicate decoded'):
            self.execute()


if __name__ == '__main__':
    unittest.main(verbosity=2)
