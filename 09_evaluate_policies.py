"""Paired policy outcomes, not demonstration completion flags or cross-task averages.

Input JSON contains trials with task_id, condition_id, policy_id, environment
(real/sim), success (boolean), policy_hash, observation_action_contract_hash,
scoring_rule_hash and scope_hash. Bootstrap resamples paired conditions within
each task; tiny sample counts are explicitly reported.
"""
import argparse
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from common import read,dump,digest


def evaluate(path,seed=2026,repeats=2000):
    source=read(path);groups=defaultdict(dict)
    for r in source['trials']:
        if type(r['success']) is not bool:raise ValueError('Explicit boolean success required; next.done is not success')
        if r['environment'] not in ('real','sim'):raise ValueError('Environment must be real or sim')
        key=(r['task_id'],r['condition_id'],r['policy_id'])
        if r['environment'] in groups[key]:raise ValueError('Duplicate paired condition; repetitions need distinct condition IDs')
        groups[key][r['environment']]=r
    tasks=defaultdict(list)
    for key,pair in groups.items():
        if set(pair)!= {'real','sim'}:raise ValueError(f'Unpaired trial: {key}')
        for field in ('policy_hash','observation_action_contract_hash','scoring_rule_hash','scope_hash'):
            if not pair['real'].get(field) or pair['real'][field]!=pair['sim'].get(field):raise ValueError(f'Mismatch: {field}')
        tasks[key[0]].append((key[1],key[2],int(pair['real']['success']),int(pair['sim']['success']),pair['real']['policy_hash']))
    rng=np.random.default_rng(seed);rows=[]
    for task,values in tasks.items():
        policies=sorted({v[1] for v in values});conditions=sorted({v[0] for v in values})
        look={(v[0],v[1]):v[2:4] for v in values}
        if len(look)!=len(policies)*len(conditions):raise ValueError('Every policy needs the same matched condition set within each task')
        for policy in policies:
            if len({v[4] for v in values if v[1]==policy})!=1:raise ValueError('Policy checkpoint changed across conditions')
        real=np.array([[look[c,p][0] for p in policies] for c in conditions]);sim=np.array([[look[c,p][1] for p in policies] for c in conditions])
        rp=real.mean(0);sp=sim.mean(0);draw=rng.integers(0,len(conditions),size=(repeats,len(conditions)))
        gaps=(sim[draw]-real[draw]).mean(1)
        correlation=float(spearmanr(rp,sp).statistic) if len(policies)>1 and np.ptp(rp)>0 and np.ptp(sp)>0 else None
        selected=np.flatnonzero(sp==sp.max());regrets=rp.max()-rp[selected]
        rows.append({'task_id':task,'paired_conditions':len(conditions),'policies':policies,
                     'real_success_rate':rp.tolist(),'sim_success_rate':sp.tolist(),
                     'sim_minus_real_gap_pp':((sp-rp)*100).tolist(),
                     'paired_condition_bootstrap_95_gap_pp':(np.quantile(gaps,[.025,.975],axis=0)*100).T.tolist(),
                     'within_task_spearman':correlation,
                     'sim_selected_tied_policies':[policies[i] for i in selected],
                     'observed_selection_regret_range_pp':[float(regrets.min()*100),float(regrets.max()*100)],
                     'sim_success_real_failure_count':((sim==1)&(real==0)).sum(0).tolist(),
                     'small_sample_warning':len(conditions)<30,
                     'bootstrap_may_be_degenerate_for_constant_outcomes':True})
    return {'source_sha256':digest(path),'data_origin':source.get('data_origin','unspecified'),
            'tasks':rows,'thresholds':source.get('preregistered_thresholds'),
            'evaluation_equivalence_certified':False,
            'limitations':['Metadata equality does not prove physically identical resets.',
                           'Bootstrap assumes independent condition clusters; dependent sessions need session-level analysis.',
                           'No universal accuracy threshold is inserted after seeing outcomes.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();dump(a.out,evaluate(a.trials));print('Computed within-task paired metrics; no automatic equivalence certificate.')


if __name__=='__main__':main()
