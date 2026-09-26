"""Fit motion-grounded contact parameters, or explicitly skip missing response data.

Fit and evaluation are separate commands. Evaluation checks episode overlap and
model/config hashes. No generic contact or policy equivalence certificate.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from common import bundle,read,dump,digest,model_digest,fresh,provenance,check_independent,resolve,stats
from contact import observed,free_slide,MujocoPredictor,errors,angle_error


def predictor(cfg,base):
    if cfg['backend']=='free_slide':return lambda e,x:free_slide(e,cfg,x)
    if cfg['backend']=='mujoco_prescribed_pusher':return MujocoPredictor(cfg,base)
    raise ValueError('Unknown contact backend')


def fit(manifest,config,out):
    cp=Path(config).resolve();cfg=read(cp);spec,all_e=bundle(manifest);out=fresh(out)
    train=[e for e in all_e if e['split']=='fit' and 'object_pose' in e['arrays']]
    if not train:
        r={'status':'skipped','reason':'No observed object response in fit episodes; robot q alone is insufficient'}
        dump(out/'contact_model.json',r);return r
    if cfg['position_scale_m']<=0 or cfg['yaw_scale_rad']<=0:raise ValueError('Positive observation scales required')
    predict=predictor(cfg,cp.parent)
    bounds=[p['bounds'] for p in cfg['parameters']]
    if cfg['backend']=='free_slide' and len(bounds)!=1:raise ValueError('Free slide estimates one effective sliding friction only')
    if any(lo<=0 or hi<=lo for lo,hi in bounds):raise ValueError('Positive ordered parameter bounds required')
    truths=[observed(e,cfg)[1] for e in train]
    def residual(theta):return np.concatenate([errors(predict(e,theta),truth,cfg)/np.sqrt(len(truth)) for e,truth in zip(train,truths)])
    def loss(theta):
        r=residual(theta);return float(np.mean(2*(np.sqrt(1+r*r)-1)))
    rng=np.random.default_rng(cfg.get('seed',2026));lo=np.array(bounds)[:,0];hi=np.array(bounds)[:,1]
    samples=[(lo+hi)/2]+list(rng.uniform(lo,hi,size=(cfg.get('search_samples',12),len(bounds))))
    if len(bounds)==1:
        grid=np.linspace(lo[0],hi[0],cfg.get('grid_points',41))
        samples.extend(np.array([v]) for v in grid)
    scored=[(loss(x),np.array(x)) for x in samples]
    best=min(scored,key=lambda x:x[0])
    if len(bounds)==1:
        # Contact transitions can make the 1-D objective multimodal. Refine
        # several promising brackets, not a single whole-range golden search.
        spacing=grid[1]-grid[0];refinements=[]
        seen={round(float(x[0]),12) for _,x in scored}
        for _ in range(cfg.get('grid_refinement_rounds',3)):
            proposed=[]
            for cost,x in sorted(scored,key=lambda v:v[0])[:5]:
                proposed.extend(np.linspace(max(lo[0],x[0]-spacing),min(hi[0],x[0]+spacing),9))
            for value in proposed:
                key=round(float(value),12)
                if key not in seen:
                    scored.append((loss([value]),np.array([value])));seen.add(key)
            spacing/=4
        for cost,x in sorted(scored,key=lambda v:v[0])[:5]:
            bracket=(max(lo[0],x[0]-spacing),min(hi[0],x[0]+spacing))
            candidate=minimize_scalar(lambda v:loss([v]),bounds=bracket,method='bounded',options={'xatol':1e-6})
            refinements.append(candidate);scored.append((float(candidate.fun),np.array([candidate.x])))
        opt=min(refinements,key=lambda v:v.fun);theta=np.array([opt.x])
    else:
        opt=minimize(loss,best[1],method='Powell',bounds=bounds,options={'maxiter':cfg.get('maxiter',30),'xtol':1e-4})
        theta=np.array(opt.x)
    scored.append((loss(theta),theta));best_loss,theta=min(scored,key=lambda x:x[0])
    # Local sensitivity is a diagnostic, not a proof of global identifiability.
    columns=[]
    for j in range(len(theta)):
        delta=max(1e-5,(hi[j]-lo[j])*.002);l=theta.copy();h=theta.copy()
        l[j]=max(lo[j],theta[j]-delta);h[j]=min(hi[j],theta[j]+delta)
        columns.append((residual(h)-residual(l))/(h[j]-l[j])*(hi[j]-lo[j]))
    singular=np.linalg.svd(np.array(columns).T,compute_uv=False)
    rank=int(np.sum(singular>max(1e-6,singular[0]*1e-4)))
    candidates=[]
    for cost,x in sorted(scored,key=lambda x:x[0]):
        if cost<=best_loss+cfg.get('plausible_loss_increment',.05):candidates.append({'parameters':x.tolist(),'loss':cost})
    r={'status':'estimated_not_physically_certified','data_origin':spec['data_origin'],
       'backend':cfg['backend'],'parameters':dict(zip([p['name'] for p in cfg['parameters']],theta.tolist())),
       'theta':theta.tolist(),'fit_loss':best_loss,'optimizer_success':bool(opt.success),
       'fit_episodes':provenance(train),'config_sha256':digest(cp),'config':cfg,
       'local_sensitivity_rank':rank,'local_singular_values':singular.tolist(),
       'sampled_plausible_candidates':candidates,'candidates_are_not_confidence_intervals':True,
       'parameter_near_bounds':bool(np.any((theta-lo)<.01*(hi-lo)) or np.any((hi-theta)<.01*(hi-lo))),
       'parameter_uniqueness_proven':False,'policy_equivalence_validated':False,
       'source_model_sha256':model_digest(resolve(cp.parent,cfg['model'])) if 'model' in cfg else None}
    dump(out/'contact_model.json',r);return r


def evaluate(manifest,config,model_file,out,split):
    cp=Path(config).resolve();cfg=read(cp);r=read(model_file);out=fresh(out)
    if r.get('status')=='skipped':raise ValueError('No fitted contact model')
    if r['config_sha256']!=digest(cp):raise ValueError('Configuration changed after fitting')
    if r['source_model_sha256'] and r['source_model_sha256']!=model_digest(resolve(cp.parent,cfg['model'])):raise ValueError('Physics model or asset dependency changed after fitting')
    _,episodes=bundle(manifest);selected=[e for e in episodes if e['split']==split]
    if not selected:raise ValueError('No evaluation episodes')
    check_independent(r,selected);predict=predictor(cfg,cp.parent);rows=[]
    for e in selected:
        t,truth=observed(e,cfg);pred=predict(e,r['theta'])
        position=np.linalg.norm(pred[:,:2]-truth[:,:2],axis=1);yaw=angle_error(pred[:,2]-truth[:,2])
        row={'id':e['id'],'position_m':stats(position),'yaw_rad':stats(yaw),
             'final_position_error_m':float(position[-1]),'final_yaw_error_rad':float(abs(yaw[-1]))}
        if isinstance(predict,MujocoPredictor):
            refined=predict(e,r['theta'],dt_factor=.5)
            row['half_timestep_position_change_m']=stats(np.linalg.norm(refined[:,:2]-pred[:,:2],axis=1))
        rows.append(row);np.savez_compressed(out/(e['id']+'_prediction.npz'),t=t,observed=truth,predicted=pred)
    thresholds=cfg.get('acceptance',{});needed=['max_position_rmse_m','max_yaw_rmse_rad']
    eligible=all(thresholds.get(k) is not None for k in needed)
    gate=all(row['position_m']['rmse']<=thresholds['max_position_rmse_m'] and row['yaw_rad']['rmse']<=thresholds['max_yaw_rmse_rad'] for row in rows) if eligible else None
    result={'status':'evaluated','data_origin':r['data_origin'],'split':split,'model_sha256':digest(model_file),
            'evaluation_episodes':provenance(selected),'rows':rows,'configured_numerical_gate_passed':gate,
            'gate_is_not_statistical_equivalence_proof':True,'policy_equivalence_validated':False,
            'scope':'object prediction conditional on measured pusher motion, or explicitly annotated free sliding'}
    dump(out/'contact_evaluation.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','evaluate'])
    p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model');p.add_argument('--split',choices=['development','test'],default='test');a=p.parse_args()
    if a.mode=='evaluate' and not a.model:p.error('--model is required for evaluation')
    r=fit(a.manifest,a.config,a.out) if a.mode=='fit' else evaluate(a.manifest,a.config,a.model,a.out,a.split)
    print(r['status'])


if __name__=='__main__':main()
