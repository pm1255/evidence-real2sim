"""Route independently by signals and their semantics, not by action array shape."""
import argparse
from common import bundle, validate_episode, dump, time_coverage


def audit(manifest):
    spec, episodes = bundle(manifest); rows = []
    for e in episodes:
        a = e['arrays']; m = e.get('metadata', {})
        errors, warnings = validate_episode(e)
        controller = (not errors and all(k in a for k in ('t', 'q', 'action', 'action_t', 'action_valid'))
                      and m.get('action') == 'joint_position_target'
                      and m.get('action_source') == 'issued_command_log'
                      and a['q'].shape == (len(a['t']), a['action'].shape[-1]))
        motion = all(k in a for k in ('object_t', 'object_pose', 'pusher_t', 'pusher_pose'))
        torque = all(k in a for k in ('q', 'tau', 't'))
        rows.append({'id': e['id'], 'split': e['split'], 'errors': errors, 'warnings': warnings,
                     'present_signals': sorted(a),
                     'timing': {k: time_coverage(a[k]) for k in ('t','action_t','object_t','pusher_t')
                                if k in a and not errors},
                     'routes': {
                         'controller': 'ready_for_empirical_fit' if controller and not errors else 'missing_or_unsupported',
                         'contact_from_motion': ('ready_if_geometry_and_sync_verified' if motion else
                                                'free_slide_only_if_annotated' if 'object_pose' in a else 'missing_object_response'),
                         'torque_residual': ('ready_if_model_calibrated' if m.get('tau_semantics_verified') else
                                            'diagnostic_only_unverified_torque_semantics') if torque else 'missing_measured_torque',
                         'images': 'external_image_pipeline_required' if m.get('images') else 'not_provided',
                         'robot_geometry_calibration': 'requires_independent_TCP_observations',
                         'paired_policy_evaluation': 'requires_separate_real_sim_policy_trials'},
                     'can_certify_real_sim_equivalence': False})
    return {'data_origin': spec['data_origin'], 'episodes': rows,
            'valid_episode_count': sum(not r['errors'] for r in rows),
            'policy_equivalence_validated': False,
            'note': 'Ready means input capability, not physical validation; no reconstruction or force data is invented.'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True); a = p.parse_args()
    r = audit(a.manifest); dump(a.out, r)
    print(f"Audited {len(r['episodes'])} episodes; valid: {r['valid_episode_count']}")
    if r['valid_episode_count'] != len(r['episodes']): raise SystemExit(2)


if __name__ == '__main__': main()
