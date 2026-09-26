"""Shared, fail-closed data contract. Canonical units: seconds, metres, radians, Nm."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_digest(path):
    """Hash XML plus resolved includes and local file assets; reject missing assets."""
    files={}
    def visit(p):
        p=Path(p).resolve()
        if str(p) in files:return
        files[str(p)]=digest(p)
        root=ET.parse(p).getroot();compiler=root.find('compiler')
        directories={'mesh':'','texture':''}
        if compiler is not None:
            for kind in directories:directories[kind]=compiler.get(kind+'dir',compiler.get('assetdir',''))
        for node in root.iter():
            name=node.get('file') or (node.get('filename') if node.tag=='mesh' else None)
            if not name:continue
            if name.startswith('package://'):raise ValueError('Resolve URDF package:// assets explicitly before model hashing')
            asset=Path(name)
            if not asset.is_absolute():asset=p.parent/directories.get(node.tag,'')/asset
            if node.tag=='include':visit(asset)
            else:files[str(asset.resolve())]=digest(asset)
    visit(path)
    # Include relative references through XML bytes; external content is hashed too.
    return hashlib.sha256(json.dumps(sorted(files.values())).encode()).hexdigest()


def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output must be empty; previous results are protected: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(base, value):
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


def bundle(path):
    path = Path(path)
    spec = read(path)
    if spec.get('schema') != 'adaptive-real2sim-v1':
        raise ValueError('Unsupported manifest schema')
    episodes = []
    ids, hashes, content_hashes = set(), set(), set()
    for row in spec['episodes']:
        p = resolve(path.parent, row['file'])
        h = digest(p)
        if row['id'] in ids or h in hashes:
            raise ValueError('Duplicate episode ID or duplicate episode content across splits')
        if row.get('sha256') != h:
            raise ValueError(f'Changed episode: {p}')
        if row['split'] not in ('fit', 'development', 'test'):
            raise ValueError('Split must be fit, development or test')
        with np.load(p, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        content=hashlib.sha256()
        for key,value in sorted(arrays.items()):
            content.update(key.encode());content.update(str(value.shape).encode())
            content.update(str(value.dtype).encode());content.update(value.tobytes())
        if content.hexdigest() in content_hashes:
            raise ValueError('Identical array content reused across episodes or splits')
        content_hashes.add(content.hexdigest())
        episodes.append(dict(row, arrays=arrays, path=str(p), sha256=h))
        ids.add(row['id']); hashes.add(h)
    return spec, episodes


def validate_episode(e):
    a = e['arrays']; errors = []; warnings = []
    groups = {'t': ('q', 'dq', 'tau', 'tcp'), 'action_t': ('action', 'action_valid'),
              'object_t': ('object_pose', 'object_confidence'),
              'pusher_t': ('pusher_pose',)}
    for time, signals in groups.items():
        present = [s for s in signals if s in a]
        if present and time not in a:
            errors.append(f'{present} missing {time}')
        if time not in a:
            continue
        t = a[time]
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            errors.append(f'{time}: need >=2 finite strictly increasing timestamps')
            continue
        dt = np.diff(t)
        if dt.max() > 5 * np.median(dt):
            warnings.append(f'{time}: large gap; no automatic gap filling')
        for s in present:
            if a[s].ndim == 0 or len(a[s]) != len(t) or not np.isfinite(a[s]).all():
                errors.append(f'{s}: invalid dimensions, length or nonfinite values')
    for s in ('q', 'dq', 'tau', 'tcp', 'action', 'object_pose', 'pusher_pose'):
        if s in a and a[s].ndim != 2:
            errors.append(f'{s}: expected (time, channel)')
    for s in ('object_pose', 'pusher_pose'):
        if s in a and a[s].ndim == 2 and a[s].shape[1] != 3:
            errors.append(f'{s}: planar schema requires x_m,y_m,yaw_rad')
    for s in ('dq', 'tau'):
        if s in a and 'q' in a and a[s].shape != a['q'].shape:
            errors.append(f'{s}: must follow exactly the same joint order as q')
    if 'action_valid' in a and not np.isin(a['action_valid'], [0, 1]).all():
        errors.append('action_valid must be boolean')
    if 'object_confidence' in a and ((a['object_confidence'] < 0) | (a['object_confidence'] > 1)).any():
        errors.append('object_confidence must lie in [0,1]')
    return errors, warnings


def usable(e, fields):
    errors, _ = validate_episode(e)
    if errors:
        raise ValueError(f"{e['id']}: {errors}")
    return all(k in e['arrays'] for k in fields)


def stats(error):
    x = np.asarray(error)
    return {'rmse': float(np.sqrt(np.mean(x*x))), 'p95_abs': float(np.quantile(abs(x), .95)),
            'max_abs': float(abs(x).max())}


def provenance(episodes):
    return [{'id': e['id'], 'sha256': e['sha256']} for e in episodes]


def check_independent(model, episodes):
    used = model.get('fit_episodes', [])
    ids = {e['id'] for e in used}; hashes = {e['sha256'] for e in used}
    if any(e['id'] in ids or e['sha256'] in hashes for e in episodes):
        raise ValueError('Evaluation overlaps fitting episodes')


def time_coverage(t):
    dt = np.diff(t)
    return {'samples': len(t), 'duration_s': float(t[-1]-t[0]),
            'median_dt_s': float(np.median(dt)), 'max_gap_s': float(dt.max())}
