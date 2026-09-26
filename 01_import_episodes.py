"""Explicit mappings for HDF5/Parquet/NPZ; DROID is the only built-in preset."""
import argparse
from pathlib import Path
import numpy as np
from common import read, dump, digest, fresh


def load_fields(path, mapping):
    suffix = path.suffix.lower()
    if suffix in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path) as f:
            return {dst: np.asarray(f[src]) for dst, src in mapping.items() if src in f}
    if suffix == '.parquet':
        import pyarrow.parquet as pq
        data = pq.read_table(path).to_pydict()
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            data = dict(f)
    else:
        raise ValueError('Input must be HDF5, Parquet or NPZ')
    return {dst: np.asarray(data[src]) for dst, src in mapping.items() if src in data}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); cfgpath = Path(args.config).resolve(); cfg = read(cfgpath)
    out = fresh(args.out); rows = []
    if cfg.get('preset') == 'droid':
        mapping = {'t': 'observation/timestamp/robot_state/read_start',
                   'q': 'observation/robot_state/joint_positions',
                   'dq': 'observation/robot_state/joint_velocities',
                   'tau': 'observation/robot_state/motor_torques_measured',
                   'action_t': 'observation/timestamp/control/control_start',
                   'action': 'action/joint_position',
                   'skip': 'observation/timestamp/skip_action'}
        semantics = {'state': 'measured_joint_position', 'action': 'joint_position_target',
                     'action_source': 'issued_command_log',
                     'tau_source': 'motor_torques_measured_field_not_independently_calibrated',
                     'tau_semantics_verified': False,
                     'time_basis': 'host_read_start_and_control_start_not_hardware_arrival',
                     'units': {'time': 's', 'q': 'rad', 'tau': 'Nm'},
                     'robot_model_id': 'DROID_FR3', 'video_state_sync_verified': False}
        scale = {'t': .001, 'action_t': .001}
    else:
        mapping = cfg['mapping']; semantics = cfg['semantics']; scale = cfg.get('scale', {})
    seen = set()
    for entry in cfg['episodes']:
        eid = entry['id']
        if Path(eid).name != eid or eid in seen or eid in ('', '.', '..'):
            raise ValueError('Episode IDs must be unique plain file names')
        seen.add(eid)
        source = Path(entry['source'])
        if not source.is_absolute(): source = cfgpath.parent / source
        a = load_fields(source, mapping)
        for k, factor in scale.items():
            if k in a: a[k] = a[k].astype(float) * factor
        if 'skip' in a: a['action_valid'] = ~a.pop('skip').astype(bool)
        if 'action' in a and 'action_valid' not in a:
            if not cfg.get('all_actions_issued', False):
                raise ValueError('Specify action_valid or explicitly declare all_actions_issued')
            a['action_valid'] = np.ones(len(a['action']), bool)
        # Subtract one common clock origin, never independently zero each stream.
        times = [a[k][0] for k in ('t', 'action_t', 'object_t', 'pusher_t') if k in a]
        origin = float(min(times)) if times else 0.
        for k in ('t', 'action_t', 'object_t', 'pusher_t'):
            if k in a: a[k] = a[k].astype(float) - origin
        target = out / (eid + '.npz'); np.savez_compressed(target, **a)
        rows.append({'id': eid, 'split': entry['split'], 'file': target.name,
                     'sha256': digest(target), 'source_path': str(source.resolve()),
                     'source_sha256': digest(source), 'clock_origin_s': origin,
                     'metadata': dict(semantics, **entry.get('metadata', {}))})
    result = {'schema': 'adaptive-real2sim-v1', 'data_origin': cfg.get('data_origin', 'unspecified'),
              'episodes': rows, 'source_config_sha256': digest(cfgpath),
              'missing_fields_are_not_imputed': True}
    dump(out/'manifest.json', result)
    print(f'Imported {len(rows)} episodes; missing signals were not fabricated.')


if __name__ == '__main__': main()
