"""Joint CAD pose, common scale and scalar opening from explicit correspondences."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .common import similarity,project,validate_rig

def fit_registration(rig,landmarks,settings=None,camera=None):
    validate_rig(rig);settings=settings or {};rec=landmarks['records']
    if len(rec)<6:raise ValueError('Insufficient correspondences: need fixed body and both moving groups')
    names=[r['name'] for r in rec]
    if len(set(names))!=len(names):raise ValueError('Landmark names must be unique')
    cad=np.asarray([r['cad_mm'] for r in rec],float);world=np.asarray([r['world'] for r in rec],float)
    if cad.shape!=(len(rec),3) or world.shape!=cad.shape or not np.isfinite(cad).all() or not np.isfinite(world).all():raise ValueError('Correspondences must be finite Nx3 points')
    motion=rig['kinematics']['group_translation_per_q'];d=np.asarray([motion[r['group']] for r in rec],float)
    init_names=settings.get('initial_body_landmarks');idx=[i for i,r in enumerate(rec) if r['group']=='body' and (init_names is None or r['name'] in init_names)]
    if len(idx)<3:raise ValueError('Provide at least three non-collinear fixed-body landmarks')
    for group in set(p['group'] for p in rig['pads']):
        if sum(r['group']==group for r in rec)<3:raise ValueError('Provide at least three correspondences for moving group '+group)
    scale,R,t=similarity(cad[idx],world[idx]);lo,hi=rig['kinematics']['q_bounds_mm'];q0=np.clip(0,lo+1e-6,hi-1e-6)
    x0=np.r_[Rotation.from_matrix(R).as_rotvec(),t,q0,np.log(scale)]
    bounds=settings.get('scale_bounds_mm_per_world_unit',[scale/10,scale*10]);sigma3=float(settings.get('sigma_3d_mm',3.0))
    if sigma3<=0 or not np.isfinite(sigma3):raise ValueError('sigma_3d_mm must be positive')
    if len(bounds)!=2 or not 0<bounds[0]<scale<bounds[1]:raise ValueError('Scale bounds must be positive and contain the body initialization')
    pix_idx=[i for i,r in enumerate(rec) if 'source_pixel' in r] if camera is not None else []
    pix=np.asarray([rec[i]['source_pixel'] for i in pix_idx]);sig=np.asarray([rec[i].get('sigma_pixel',2.) for i in pix_idx],float)
    if len(sig) and (not np.isfinite(sig).all() or np.any(sig<=0)):raise ValueError('Invalid pixel weights')
    def model(x):return ((cad+d*x[6])@Rotation.from_rotvec(x[:3]).as_matrix().T+x[3:6])/np.exp(x[7])
    def residual(x):
        w=model(x);parts=[]
        if pix_idx:
            uv,z=project(w[pix_idx],camera);parts.append(((uv-pix)/sig[:,None]).ravel());parts.append(np.minimum(z,0)*1e6)
        parts.append(((w-world)*np.exp(x[7])/sigma3).ravel())
        return np.concatenate(parts)
    lower=np.r_[[-np.inf]*6,lo,np.log(bounds[0])];upper=np.r_[[np.inf]*6,hi,np.log(bounds[1])]
    opt=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',f_scale=1,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=1500)
    if not opt.success:raise ValueError('Registration optimizer did not converge: '+opt.message)
    singular=np.linalg.svd(opt.jac,compute_uv=False);rank=int(np.sum(singular>singular[0]*1e-8))
    if rank<8:raise ValueError('Registration is underconstrained (Jacobian rank '+str(rank)+'/8); add independent correspondences')
    x=opt.x;pred=model(x);err=np.linalg.norm(pred-world,axis=1)*np.exp(x[7]);per=[];pixel_errors=[]
    for i,r in enumerate(rec):
        out=dict(name=r['name'],group=r['group'],error_mm=float(err[i]),pred_world=pred[i].tolist())
        if i in pix_idx:
            uv,z=project(pred[i:i+1],camera)
            if z[0]<=0:raise ValueError('Fitted landmark behind source camera')
            e=float(np.linalg.norm(uv[0]-r['source_pixel']));out.update(error_px=e,pred_pixel=uv[0].tolist());pixel_errors.append(e)
        per.append(out)
    hit=bool(np.any(np.minimum(abs(x-lower),abs(upper-x))[6:]<1e-5))
    groups={g:dict(count=sum(r['group']==g for r in rec),rms_3d_mm=float(np.sqrt(np.mean([e*e for e,r in zip(err,rec) if r['group']==g])))) for g in set(r['group'] for r in rec)}
    return dict(scale_mm_per_world=float(np.exp(x[7])),rotation_columns=Rotation.from_rotvec(x[:3]).as_matrix().tolist(),translation_mm=x[3:6].tolist(),opening_change_mm=float(x[6]),rms_3d_mm=float(np.sqrt(np.mean(err**2))),rms_2d_px=float(np.sqrt(np.mean(np.square(pixel_errors)))) if pixel_errors else None,landmarks=per,group_residuals=groups,jacobian_rank=rank,jacobian_singular_values=singular.tolist(),parameter_bound_hit=hit,parameters=x.tolist(),initial_scale_mm_per_world=scale,scale_bounds_mm_per_world_unit=bounds,sigma_3d_mm=sigma3,object_or_contact_used_in_fit=False,interpretation='Estimated registration; residuals are not uncertainty bounds or measured tactile accuracy.')
