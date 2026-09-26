"""Preserve full-workspace captures and shared-world instance metadata.

This is an input preparation stage, NOT a reconstruction, segmentation, camera
recovery or physics implementation. No image is cropped, whitened or resized.
Unknown poses/instances remain unknown and cannot certify an interactive scene.
"""
import argparse
import re
import shutil
from pathlib import Path
import hashlib
import numpy as np
import cv2
from common import read, dump, fresh, resolve, digest


ROLES = {'static_environment', 'support_surface', 'rigid_object',
         'articulated_object', 'robot'}


def rigid(value, label):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f'{label}: expected a proper rigid 4x4 transform')
    return t.tolist()


def prepare(config, out):
    path = Path(config).resolve()
    cfg = read(path)
    if cfg.get('schema') != 'robot-workspace-capture-v1':
        raise ValueError('Expected robot-workspace-capture-v1, not an object-crop manifest')
    if cfg.get('frame_scope') != 'full_workspace':
        raise ValueError('Full workspace frames required; object crops do not cover the scene')
    if cfg.get('capture_mode') not in ('moving_camera_static_workspace', 'dynamic_episode'):
        raise ValueError('Specify a full-workspace capture mode; object turntables are not static workspace scans')
    if cfg.get('camera_convention') != 'opencv_world_to_camera':
        raise ValueError('Explicit OpenCV world-to-camera convention required')
    units = cfg.get('coordinate_units', 'unknown')
    if units not in ('meters', 'arbitrary', 'unknown'):
        raise ValueError('coordinate_units must be meters, arbitrary or unknown')
    metric = units == 'meters' and bool(cfg.get('metric_scale_source'))
    if units == 'meters' and not metric:
        raise ValueError('A declared metric scale needs its source')
    bounds = cfg.get('bounds_m')
    if bounds is not None:
        b = np.asarray(bounds, float)
        if not metric or b.shape != (2, 3) or not np.isfinite(b).all() or np.any(b[1] <= b[0]):
            raise ValueError('bounds_m requires finite ordered bounds and a declared metric frame')

    # Validate before creating output; labels never cut anything out of RGB.
    instances = cfg.get('instances', [])
    ids, labels = set(), set()
    for obj in instances:
        name = obj.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in ids:
            raise ValueError('Instance IDs must be unique safe identifiers')
        ids.add(name)
        if obj.get('role') not in ROLES:
            raise ValueError('Unknown instance role')
        label = obj.get('label_id')
        if label is not None:
            if isinstance(label, bool) or not isinstance(label, int) or label < 1 or label in labels:
                raise ValueError('Instance label IDs must be unique positive integers; 0 is unassigned')
            labels.add(label)
        if obj.get('world_from_instance') is not None:
            rigid(obj['world_from_instance'], name)
        # Appearance and collision identities are distinct; mass/friction are never guessed.
        for key in ('appearance_asset_id', 'collision_asset_id'):
            if obj.get(key) is not None and not isinstance(obj[key], str):
                raise ValueError(f'{key} must be an asset identifier or null')
    for obj in instances:
        if obj.get('support_id') is not None and obj['support_id'] not in ids:
            raise ValueError('Unknown support instance')
        visited = {obj['id']}
        current = obj
        by_id = {item['id']: item for item in instances}
        while current.get('support_id') is not None:
            next_id = current['support_id']
            if next_id in visited:
                raise ValueError('Cyclic support relation')
            visited.add(next_id)
            current = by_id[next_id]

    frames = cfg.get('frames', [])
    if len(frames) < 4 or sum(f.get('split') == 'fit' for f in frames) < 3 or not any(f.get('split') == 'test' for f in frames):
        raise ValueError('At least three fit frames and an independent test frame required')
    seen, frame_ids, validated = set(), set(), []
    calibrated = True
    for row in frames:
        fid = row.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', fid) or fid in frame_ids:
            raise ValueError('Frame IDs must be unique safe identifiers')
        frame_ids.add(fid)
        if row.get('split') not in ('fit', 'development', 'test'):
            raise ValueError('Invalid split')
        if any(k in row for k in ('mask', 'crop', 'crop_box')):
            raise ValueError('Object crop/global foreground mask is not allowed; use instance_labels without removing background')
        image = resolve(path.parent, row['image'])
        pixels = cv2.imread(str(image))
        if pixels is None:
            raise ValueError(f'Unreadable frame {fid}')
        h = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if h in seen:
            raise ValueError('Duplicate decoded image content across capture splits')
        seen.add(h)
        entry = {'id': fid, 'split': row['split'], 'image_sha256': digest(image),
                 'width': pixels.shape[1], 'height': pixels.shape[0]}
        if row.get('K') is not None:
            k = np.asarray(row['K'], float)
            if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
                raise ValueError(f'Invalid camera intrinsics: {fid}')
            entry['K'] = k.tolist()
        if row.get('world_to_camera') is not None:
            entry['world_to_camera'] = rigid(row['world_to_camera'], fid)
        calibrated &= 'K' in entry and 'world_to_camera' in entry
        if row.get('distortion') is not None:
            dist = np.asarray(row['distortion'], float)
            if dist.ndim != 1 or not np.isfinite(dist).all() or len(dist) not in (0, 4, 5, 8, 12, 14):
                raise ValueError('Invalid distortion coefficients')
            entry['distortion'] = dist.tolist()
        label_path = depth_path = None
        if row.get('instance_labels'):
            label_path = resolve(path.parent, row['instance_labels'])
            label_map = np.load(label_path, allow_pickle=False)
            if label_map.shape != pixels.shape[:2] or not np.issubdtype(label_map.dtype, np.integer):
                raise ValueError('Instance labels must be an integer array matching full RGB dimensions')
            if not set(np.unique(label_map)).issubset(labels | {0}):
                raise ValueError('Instance label map contains IDs absent from inventory')
            entry['unassigned_pixel_fraction'] = float((label_map == 0).mean())
        if row.get('depth'):
            depth_path = resolve(path.parent, row['depth'])
            depth = np.load(depth_path, allow_pickle=False)
            scale = cfg.get('depth_meters_per_unit')
            if (depth.shape != pixels.shape[:2] or scale is None
                    or not np.isfinite(scale) or scale <= 0
                    or not cfg.get('depth_registered_to_rgb')):
                raise ValueError('Depth requires full-frame registration and explicit positive metre units')
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError('Depth must use 0 for missing measurements, not negative/NaN values')
        validated.append((entry, image, label_path, depth_path))

    dest = fresh(out)
    emitted = []
    for entry, image, label_path, depth_path in validated:
        row = dict(entry)
        for key, source, directory in [('image', image, 'images'), ('instance_labels', label_path, 'labels'), ('depth', depth_path, 'depth')]:
            if source is None:
                continue
            target = Path(directory) / (row['id'] + source.suffix.lower())
            (dest / directory).mkdir(exist_ok=True)
            shutil.copy2(source, dest / target)
            row[key] = target.as_posix()
            row[key + '_sha256'] = digest(dest / target)
        emitted.append(row)

    static = cfg['capture_mode'] == 'moving_camera_static_workspace' and cfg.get('scene_static_during_capture') is True
    scene_graph = {'schema': 'robot-workspace-inventory-v1', 'coordinate_units': units,
                   'instances': instances, 'inventory_complete': False,
                   'original_layout_measured': False, 'interactive_scene_validated': False,
                   'policy_equivalence_validated': False}
    dump(dest / 'scene_graph.json', scene_graph)
    capture = {'schema': cfg['schema'], 'frame_scope': 'full_workspace',
               'capture_mode': cfg['capture_mode'], 'camera_convention': cfg['camera_convention'],
               'coordinate_units': units, 'frames': emitted, 'source_manifest_sha256': digest(path),
               'scene_static_during_capture': cfg.get('scene_static_during_capture'),
               'data_origin': cfg.get('data_origin', 'USER_PROVIDED_UNVERIFIED')}
    dump(dest / 'capture.json', capture)
    if static and calibrated:
        # Deliberately no image-wide foreground mask. Unassigned and background pixels stay.
        scene = {'schema': 'calibrated-scene-v1', 'frame_scope': 'full_workspace',
                 'camera_convention': cfg['camera_convention'], 'data_origin': capture['data_origin'],
                 'frames': emitted, 'coordinate_units': units,
                 'metric_scale_source': cfg.get('metric_scale_source')}
        for key in ('bounds_m', 'depth_meters_per_unit', 'tsdf_truncation_m', 'depth_consistency_tolerance_m'):
            if key in cfg:
                scene[key] = cfg[key]
        dump(dest / 'scene.json', scene)
    issues = []
    if not calibrated:
        issues.append('Recover and validate shared-world camera intrinsics/extrinsics; no poses were invented.')
    if not static:
        issues.append('Dynamic/undeclared capture needs motion separation and tracking before static scene reconstruction.')
    if not metric:
        issues.append('Add and validate a metric scale before physics assembly.')
    if bounds is None:
        issues.append('Measure full-workspace bounds before the existing dense reconstruction backend.')
    issues.extend(['Whole-scene appearance and geometry reconstruction have NOT been run by this preparation stage.',
                   'Instance inventory, support/layout, robot alignment and occluded-surface coverage remain unverified.',
                   'Moving an object requires removing it from static appearance/collision to avoid duplicates and ghosts.',
                   'Static-scene declarations are user metadata, not independent evidence of scene rigidity.'])
    result = {'stage': 'full_workspace_input_prepared', 'frame_count': len(emitted),
              'full_frame_bytes_preserved': True, 'resized': False, 'cropped': False,
              'background_whitened': False, 'camera_poses_complete': calibrated,
              'calibrated_scene_exported': static and calibrated,
              'dense_rgb_input_ready': static and calibrated and metric and bounds is not None,
              'gaussian_training_requires_fit_derived_seed_points': True,
              'whole_scene_reconstruction_completed': False, 'photorealism_validated': False,
              'interactive_scene_validated': False, 'remaining_work': issues}
    dump(dest / 'workspace_report.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(prepare(a.config, a.out))
