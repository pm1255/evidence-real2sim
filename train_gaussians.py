"""Train anisotropic 3D Gaussians from calibrated RGB, on CPU or optional gsplat/CUDA.

Reference backend: differentiable perspective covariance, depth sorted alpha
compositing, trainable centres/quaternions/scales/opacity/SH0 colour. Fixed
Gaussian budget; no claim of matching production adaptive densification or speed.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from scipy.spatial import cKDTree
from common import dump,fresh,resolve,digest
from vision_data import load_scene,scene_fingerprint


def quat_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8);w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)


class Gaussians(nn.Module):
    def __init__(self,xyz,rgb,scale):
        super().__init__();n=len(xyz)
        self.means=nn.Parameter(torch.tensor(xyz,dtype=torch.float32))
        self.log_scales=nn.Parameter(torch.tensor(np.repeat(np.asarray(scale)[:,None],3,axis=1),dtype=torch.float32).log())
        self.quats=nn.Parameter(torch.tensor(np.tile([1,0,0,0],(n,1)),dtype=torch.float32))
        self.opacity_logits=nn.Parameter(torch.full((n,),-1.))
        self.color_logits=nn.Parameter(torch.logit(torch.tensor(rgb,dtype=torch.float32).clamp(.01,.99)))
    def render(self,K,T,height,width,backend='reference',background=1.):
        if backend=='gsplat':
            from gsplat import rasterization
            image,alpha,_=rasterization(means=self.means,quats=self.quats,scales=self.log_scales.exp(),
                opacities=self.opacity_logits.sigmoid(),colors=self.color_logits.sigmoid(),
                viewmats=T[None],Ks=K[None],width=width,height=height,
                backgrounds=torch.full((1,3),background,device=K.device),packed=False)
            return image[0],alpha[0,:,:,0]
        camera=self.means@T[:3,:3].T+T[:3,3];depth=camera[:,2]
        order=torch.argsort(depth.detach());camera=camera[order];depth=depth[order]
        rotation=quat_matrix(self.quats)[order];scales=self.log_scales.exp()[order]
        cov=rotation@torch.diag_embed(scales.square())@rotation.transpose(1,2)
        cov=T[:3,:3][None]@cov@T[:3,:3].T[None]
        z=depth.clamp_min(1e-4);x,y=camera[:,0],camera[:,1]
        u=K[0,0]*x/z+K[0,1]*y/z+K[0,2];v=K[1,1]*y/z+K[1,2]
        zeros=torch.zeros_like(z)
        J=torch.stack([K[0,0]/z,K[0,1]/z,-(K[0,0]*x+K[0,1]*y)/z**2,
                       zeros,K[1,1]/z,-K[1,1]*y/z**2],-1).reshape(-1,2,3)
        screen=J@cov@J.transpose(1,2)+torch.eye(2,device=K.device)[None]*.3
        a,b,c=screen[:,0,0],screen[:,0,1],screen[:,1,1];det=(a*c-b*b).clamp_min(1e-8)
        yy,xx=torch.meshgrid(torch.arange(height,device=K.device),torch.arange(width,device=K.device),indexing='ij')
        dx=xx[None]-u[:,None,None];dy=yy[None]-v[:,None,None]
        power=(c[:,None,None]*dx*dx-2*b[:,None,None]*dx*dy+a[:,None,None]*dy*dy)/det[:,None,None]
        gaussian=torch.exp(-.5*power.clamp_min(0))
        visible=(depth>1e-3).to(gaussian.dtype)[:,None,None]
        alpha=(self.opacity_logits.sigmoid()[order,None,None]*gaussian*visible).clamp(0,.99)
        survival=torch.cumprod(1-alpha+1e-8,dim=0)
        before=torch.cat([torch.ones_like(survival[:1]),survival[:-1]],0);weights=alpha*before
        rgb=torch.einsum('nhw,nc->hwc',weights,self.color_logits.sigmoid()[order])+survival[-1,:,:,None]*background
        return rgb,1-survival[-1]


def export_ply(model,path):
    values={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    xyz=values['means'];rgb=1/(1+np.exp(-values['color_logits']));sh=(rgb-.5)/.28209479177387814
    q=values['quats'];q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
    arr=np.c_[xyz,np.zeros_like(xyz),sh,values['opacity_logits'],values['log_scales'],q]
    fields=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    with Path(path).open('w') as f:
        f.write('ply\nformat ascii 1.0\n'+f'element vertex {len(xyz)}\n')
        for key in fields:f.write(f'property float {key}\n')
        f.write('end_header\n');np.savetxt(f,arr,fmt='%.8g')


def train(scene,out,steps=160,max_side=64,max_gaussians=256,backend='reference',seed=2026):
    out=fresh(out);cfg,frames,base=load_scene(scene,max_side);torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4)
    if backend=='gsplat' and not torch.cuda.is_available():raise ValueError('gsplat backend requires CUDA; reference backend works on CPU')
    device='cuda' if backend=='gsplat' else 'cpu'
    if not cfg.get('seed_points'):raise ValueError('Need fit-only sparse/dense seed_points NPZ; never initialise from heldout scan ground truth')
    if cfg.get('seed_points_source_split')!='fit':raise ValueError('Declare and verify fit-only seed origin')
    with np.load(resolve(base,cfg['seed_points']),allow_pickle=False) as a:xyz=a['xyz'];colors=a['rgb']
    if xyz.ndim!=2 or xyz.shape[1]!=3 or colors.shape!=xyz.shape or not np.isfinite(xyz).all() or not np.isfinite(colors).all():raise ValueError('Invalid seed points')
    chosen=np.random.choice(len(xyz),min(max_gaussians,len(xyz)),replace=False);xyz=xyz[chosen];colors=colors[chosen]
    dist=cKDTree(xyz).query(xyz,k=2)[0][:,1];extent=float(np.linalg.norm(np.ptp(xyz,axis=0)))
    scales=np.clip(dist*.65,extent/1000,extent/8)
    model=Gaussians(xyz,colors,scales).to(device)
    optimizer=torch.optim.Adam([{'params':[model.means],'lr':extent*.001},
        {'params':[model.log_scales],'lr':.015},{'params':[model.quats],'lr':.005},
        {'params':[model.opacity_logits,model.color_logits],'lr':.035}])
    prepared=[]
    for f in frames:
        target=f['rgb'].copy();target[~f['mask']]=1
        prepared.append(dict(f,target=torch.tensor(target,device=device),
           Kt=torch.tensor(f['K'],dtype=torch.float32,device=device),Tt=torch.tensor(f['T'],dtype=torch.float32,device=device),
           mt=torch.tensor(f['mask'],dtype=torch.float32,device=device)))
    fit=[f for f in prepared if f['split']=='fit'];test=[f for f in prepared if f['split']=='test']
    if not test:raise ValueError('Heldout test views required to report novel-view performance')
    def evaluate(rows,prefix):
        results=[]
        with torch.no_grad():
            for i,f in enumerate(rows):
                h,w=f['mask'].shape;pred,alpha=model.render(f['Kt'],f['Tt'],h,w,backend)
                mse=float((pred-f['target']).square().mean());mask=f['mt']>0
                fmse=float((pred[mask]-f['target'][mask]).square().mean()) if mask.any() else None
                pm=alpha>.5;intersection=(pm&mask).sum();union=(pm|mask).sum()
                results.append({'view_id':f['id'],'psnr_db':-10*np.log10(max(mse,1e-12)),
                    'foreground_psnr_db':-10*np.log10(max(fmse,1e-12)) if fmse is not None else None,
                    'silhouette_iou':float(intersection/union) if union else 1.})
                if prefix:
                    composite=np.concatenate([f['target'].cpu().numpy(),pred.cpu().numpy()],axis=1)
                    cv2.imwrite(str(out/f'{prefix}_{i:02d}.png'),cv2.cvtColor(np.uint8(np.clip(composite,0,1)*255),cv2.COLOR_RGB2BGR))
        return results
    before=evaluate(test,'before');history=[];start=time.monotonic()
    for step in range(steps):
        f=fit[step%len(fit)];h,w=f['mask'].shape;optimizer.zero_grad()
        pred,alpha=model.render(f['Kt'],f['Tt'],h,w,backend)
        weights=1+2*f['mt']
        loss=((pred-f['target']).abs().mean(-1)*weights).mean()+.15*(alpha-f['mt']).abs().mean()
        loss=loss+.001*((model.means-torch.tensor(xyz,device=device))/extent).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        with torch.no_grad():model.log_scales.clamp_(np.log(extent/2000),np.log(extent/4))
        if step%20==0 or step==steps-1:
            history.append({'step':step,'fit_loss':float(loss.detach())});print(f'Gaussian step {step+1}/{steps}: fit loss {float(loss.detach()):.5f}',flush=True)
    after=evaluate(test,'heldout');fit_eval=evaluate(fit[:min(4,len(fit))],None)
    np.savez_compressed(out/'gaussians.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    export_ply(model,out/'gaussians.ply')
    result={'backend':backend,'model':'anisotropic_3D_Gaussians_SH0_fixed_budget','data_origin':cfg['data_origin'],
      'scene_sha256':scene_fingerprint(scene),'training_views':len(fit),'heldout_views':len(test),
      'gaussian_count':len(xyz),'steps':steps,'seconds':time.monotonic()-start,'image_max_side':max_side,
      'before_heldout':before,'after_heldout':after,'fit_views_diagnostic':fit_eval,'fit_history':history,
      'metric_scale_source':cfg.get('metric_scale_source'),'collision_geometry_validated':False,
      'limitations':['Fixed budget, SH0 colours, no adaptive densification; small CPU reference is not production photorealism.',
                     'Seed geometry must originate from fit images; metadata declaration alone cannot prove provenance.',
                     'Image appearance metrics do not certify contact surfaces or unobserved regions.']}
    dump(out/'training_report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--max-side',type=int,default=64)
    p.add_argument('--gaussians',type=int,default=256);p.add_argument('--backend',choices=['reference','gsplat'],default='reference')
    a=p.parse_args();train(a.scene,a.out,a.steps,a.max_side,a.gaussians,a.backend)


if __name__=='__main__':main()
