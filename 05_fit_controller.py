"""Fit an empirical first-order response only for issued joint-position targets.

No torque controller is inferred. Skipped commands are ignored; target changes
are integrated at their event times, never linearly interpolated.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from common import bundle,usable,dump,fresh,provenance,stats,check_independent


def intervals(e,delay):
    a=e['arrays'];t=a['t'];valid=a['action_valid'].astype(bool)
    et=a['action_t'][valid]+delay;u=a['action'][valid]
    if not len(et):raise ValueError('No issued commands')
    segments=[]
    for left,right in zip(t[:-1],t[1:]):
        k=np.searchsorted(et,left,side='right')-1
        target=u[k] if k>=0 else a['q'][0]
        cursor=left;parts=[]
        for j in range(k+1,np.searchsorted(et,right,side='left')):
            parts.append((et[j]-cursor,target));cursor=et[j];target=u[j]
        parts.append((right-cursor,target));segments.append(parts)
    width=max(map(len,segments));dt=np.zeros((len(segments),width));targets=np.zeros((len(segments),width,a['q'].shape[1]))
    for i,parts in enumerate(segments):
        for j,(duration,target) in enumerate(parts):dt[i,j]=duration;targets[i,j]=target
    return dt,targets


def advance(q,tau,parts):
    dt,u=parts;pred=q.copy()
    for j in range(dt.shape[1]):pred+=(1-np.exp(-dt[:,j,None]/tau))*(u[:,j]-pred)
    return pred


def rollout(e,tau,delay):
    q=e['arrays']['q'];dt,u=intervals(e,delay);pred=np.zeros_like(q);pred[0]=q[0]
    for i in range(len(q)-1):pred[i+1]=advance(pred[i:i+1],tau,(dt[i:i+1],u[i:i+1]))[0]
    return pred


def fit(manifest,out):
    spec,episodes=bundle(manifest);out=fresh(out);eligible=[];skipped=[]
    for e in episodes:
        m=e.get('metadata',{})
        ok=usable(e,('t','q','action_t','action','action_valid'))
        ok=ok and m.get('action')=='joint_position_target' and m.get('action_source')=='issued_command_log'
        if ok and e['arrays']['q'].shape[1]==e['arrays']['action'].shape[1] and e['arrays']['action_valid'].any():eligible.append(e)
        else:skipped.append(e['id'])
    train=[e for e in eligible if e['split']=='fit'];tests=[e for e in eligible if e['split']=='test']
    if not train:
        result={'status':'skipped','reason':'No fit episodes with actual q and issued joint-position target semantics','skipped':skipped}
        dump(out/'controller.json',result);return result
    n=train[0]['arrays']['q'].shape[1]
    if any(e['arrays']['q'].shape[1]!=n for e in eligible):raise ValueError('Mixed joint dimensions')
    robots={e.get('metadata',{}).get('robot_model_id','unspecified') for e in eligible}
    if len(robots)>1:raise ValueError('Different robot identities require separate fits')
    choices=[]
    for delay in (0.,.02,.04,.07):
        cache=[intervals(e,delay) for e in train]
        def residual(tau):
            rows=[]
            for e,parts in zip(train,cache):
                a=e['arrays'];pred=advance(a['q'][:-1],tau,parts)
                known=a['t'][:-1]>=a['action_t'][a['action_valid'].astype(bool)][0]+delay
                if known.any():rows.append((pred[known]-a['q'][1:][known]).ravel()/np.sqrt(known.sum()))
            if not rows:raise ValueError('No observed interval after known command')
            return np.concatenate(rows)
        opt=least_squares(residual,np.full(n,.15),bounds=(.005,2.),loss='soft_l1',f_scale=.02,max_nfev=100)
        choices.append((float(np.mean(residual(opt.x)**2)),delay,opt.x))
    loss,delay,tau=min(choices,key=lambda x:x[0])
    changes=np.vstack([np.diff(e['arrays']['action'][e['arrays']['action_valid'].astype(bool)],axis=0) for e in train])
    result={'status':'empirical_estimate','data_origin':spec['data_origin'],
            'model':'uncoupled first-order target response','tau_s':tau.tolist(),'additional_delay_s':delay,
            'fit_objective':'one-step errors with measured starting states; equal episode weighting',
            'fit_episodes':provenance(train),'skipped_episodes':skipped,
            'command_change_std_rad':np.std(changes,axis=0).tolist(),
            'weakly_excited_joint_indices':np.where(np.std(changes,axis=0)<1e-4)[0].tolist(),
            'delay_search': [{'delay_s':x[1],'fit_loss':x[0]} for x in choices],
            'controller_physics_identified':False,'contact_dynamics_identified':False,
            'initial_target_before_first_log':'assumed hold at initial actual q',
            'limitations':['A zero additional delay is not proof of zero physical latency.',
                           'Response mixes control, load and unmodelled effects; no claim of torque-level or contact fidelity.']}
    check_independent(result,tests);test_rows=[]
    for e in tests:
        pred=rollout(e,tau,delay);q=e['arrays']['q']
        np.savez_compressed(out/(e['id']+'_rollout.npz'),t=e['arrays']['t'],observed=q,predicted=pred)
        test_rows.append({'id':e['id'],'whole_episode_error_rad':stats(pred-q),
                          'state_reset':'initial state only'})
    result['heldout']=test_rows;dump(out/'controller.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();r=fit(a.manifest,a.out);print(r['status'])


if __name__=='__main__':main()
