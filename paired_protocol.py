"""Freeze task-specific evaluation criteria, then audit genuinely paired trial records.

This creates a local integrity record, not an independently timestamped registry.
It cannot manufacture real trials, inspect hidden hardware resets, or establish
equivalence from synthetic records or author aggregate tables.
"""
import argparse,datetime,importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import norm
from common import read,dump,digest,fresh,resolve


def freeze(spec,out):
    p=Path(spec).resolve();cfg=read(p);out=fresh(out)
    required=('scope','policies','scoring_rule','max_success_gap_pp','max_sim_success_real_failure_rate',
              'min_real_conditions_per_task','minimum_within_task_spearman','artifacts')
    if any(k not in cfg or cfg[k] is None for k in required):raise ValueError('Define task-specific criteria before testing; no automatic default')
    if not 0<=cfg['max_success_gap_pp']<=100 or not 0<=cfg['max_sim_success_real_failure_rate']<=1:raise ValueError('Invalid tolerance')
    hashes={name:digest(resolve(p.parent,path)) for name,path in cfg['artifacts'].items()}
    record={'schema':'paired-protocol-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'spec':cfg,'artifact_sha256':hashes,'independent_preregistration_proven':False,
            'note':'Register this digest externally before viewing test outcomes for stronger preregistration evidence.'}
    dump(out/'protocol.json',record);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')


def wilson(k,n,z):
    if not n:return (0.,1.)
    p=k/n;den=1+z*z/n;centre=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,centre-half),min(1,centre+half)


def audit(protocol,trials,out):
    protocol=Path(protocol);out=fresh(out);lock=read(protocol);data=read(trials)
    if data.get('protocol_sha256')!=digest(protocol):raise ValueError('Trials not bound to this frozen protocol')
    if data.get('data_origin')!='REAL_SIM_PAIRED_CLOSED_LOOP':
        dump(out/'paired_gate.json',{'status':'not_real_paired_evidence','replacement_supported':False,
             'reason':'Synthetic fixtures, training demonstrations and author aggregates do not satisfy this gate.'});return
    cfg=lock['spec'];required=('real_episode_sha256','sim_episode_sha256','condition_measurement_id','hardware_id')
    # Measured-reset records belong to the pair. Content hashes are checked by
    # the evidence packaging step; this audit requires them, never infers them.
    for r in data['trials']:
        for k in required:
            if not r.get(k):raise ValueError(f'Missing paired provenance {k}')
        if r.get('policy_hash') not in cfg['policies'].values():raise ValueError('Unregistered policy checkpoint')
        if r.get('hardware_id') not in cfg['scope']['hardware_ids']:raise ValueError('Hardware outside registered scope')
    p=Path(__file__).with_name('09_evaluate_policies.py');s=importlib.util.spec_from_file_location('metrics',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    metrics=m.evaluate(trials);dump(out/'metrics.json',metrics)
    # Bonferroni simultaneous Wilson-based conservative difference intervals;
    # still descriptive approximation, not finite-sample exact coverage.
    groups={}
    for r in data['trials']:
        groups.setdefault((r['task_id'],r['policy_id']),{}).setdefault(r['condition_id'],{})[r['environment']]=r['success']
    comparisons=max(1,len(groups));z=norm.ppf(1-.05/(6*comparisons));rows=[]
    for (task,policy),conditions in groups.items():
        real=np.array([p['real'] for p in conditions.values()]);sim=np.array([p['sim'] for p in conditions.values()]);n=len(real)
        rl,rh=wilson(int(real.sum()),n,z);sl,sh=wilson(int(sim.sum()),n,z)
        interval=[(sl-rh)*100,(sh-rl)*100];false=int(np.sum(sim&~real));_,false_upper=wilson(false,int(sim.sum()),z)
        gap_ok=max(abs(interval[0]),abs(interval[1]))<=cfg['max_success_gap_pp']
        rows.append({'task_id':task,'policy_id':policy,'n_real':n,'conservative_gap_interval_pp':interval,
                     'false_success_rate_upper':false_upper,'success_gap_gate':gap_ok,
                     'false_success_gate':false_upper<=cfg['max_sim_success_real_failure_rate'],
                     'sample_count_gate':n>=cfg['min_real_conditions_per_task']})
    ranks=[r['within_task_spearman'] for r in metrics['tasks']]
    rank_ok=bool(ranks) and all(r is not None and r>=cfg['minimum_within_task_spearman'] for r in ranks)
    passed=bool(rows) and rank_ok and all(r['success_gap_gate'] and r['false_success_gate'] and r['sample_count_gate'] for r in rows)
    dump(out/'paired_gate.json',{'status':'criteria_met_for_supplied_scope' if passed else 'criteria_not_met',
         'configured_statistical_gates_passed':passed,'rows':rows,'ranking_gate':rank_ok,
         'replacement_supported':False,'needs_review':['Independent reset measurement and calibration evidence',
            'Session dependence and statistical assumptions','External preregistration and no policy/test leakage'],
         'universal_real_sim_equivalence_claimed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','audit']);p.add_argument('--spec');p.add_argument('--protocol');p.add_argument('--trials');p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.mode=='freeze':
        if not a.spec:p.error('--spec required')
        freeze(a.spec,a.out)
    else:
        if not a.protocol or not a.trials:p.error('--protocol and --trials required')
        audit(a.protocol,a.trials,a.out)
