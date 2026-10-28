"""Discover tagged-rig correspondences from an ordered RGB cloud, without scene clicks.
Camera/grid recovery is validated on withheld points. CAD projections only guide
image-feature search: unobserved projected coordinates are never observations.
"""
from pathlib import Path
import argparse
import numpy as np,cv2,trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image
from registration.common import read,write,sha,project,validate_rig
from prepare_landmarks import prepare

def fit_projection(x,uv):
    def normalize(a):
        mu=a.mean(0);s=np.sqrt(a.shape[1])/max(np.sqrt(np.mean(np.sum((a-mu)**2,axis=1))),1e-12)
        T=np.eye(a.shape[1]+1);T[:-1,:-1]*=s;T[:-1,-1]=-mu*s
        return np.c_[a,np.ones(len(a))]@T.T,T
    X,Tx=normalize(x);U,Tu=normalize(uv);z=np.zeros_like(X)
    A=np.stack([np.c_[X,z,-U[:,0,None]*X],np.c_[z,X,-U[:,1,None]*X]],axis=1).reshape(-1,12)
    _,sv,vh=np.linalg.svd(A,full_matrices=False)
    if sv[-2]<sv[0]*1e-9:raise ValueError('Scene does not constrain a perspective camera; provide source metadata')
    return np.linalg.inv(Tu)@vh[-1].reshape(3,4)@Tx

def recover_camera(xyz):
    n=len(xyz);rng=np.random.default_rng(0);idx=rng.choice(n,min(n,1200),replace=False);a=idx[:len(idx)//2];b=idx[len(idx)//2:];candidates=[]
    for w in range(128,min(4096,n//128)+1):
        if n%w:continue
        h=n//w
        if h<128 or h>4096 or not .2<w/h<5:continue
        try:P=fit_projection(xyz[a],np.c_[a%w,a//w])
        except ValueError:continue
        q=np.c_[xyz[b],np.ones(len(b))]@P.T
        if np.any(abs(q[:,2])<1e-10):continue
        err=np.linalg.norm(q[:,:2]/q[:,2,None]-np.c_[b%w,b//w],axis=1);candidates.append((float(np.max(err)),w,h,P))
    candidates.sort(key=lambda x:x[0])
    if not candidates or candidates[0][0]>.1:raise ValueError('No verified row-major image mapping found. Supply source image/camera mapping, or let the agent identify 3D feature pairs; do not guess a grid.')
    if len(candidates)>1 and candidates[1][0]<.1:raise ValueError('Ambiguous image grid; supply source dimensions')
    error,w,h,P=candidates[0];K,R,C,*_=cv2.decomposeProjectionMatrix(P);K=K/K[2,2];center=(C[:3]/C[3]).ravel();M=np.eye(4);M[:3,:3]=R;M[:3,3]=-R@center
    if K[0,0]<0 or K[1,1]<0 or abs(K[0,1])>.01:raise ValueError('Recovered camera has unsupported sign/skew; inspect mapping')
    camera=dict(width=w,height=h,fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),world_to_camera=M.tolist())
    uv,z=project(xyz[b],camera);err=float(np.max(np.linalg.norm(uv-np.c_[b%w,b//w],axis=1)))
    if np.any(z<=0) or err>.1:raise ValueError('Decomposed camera fails withheld-point reprojection')
    return camera,dict(method='DLT over candidate row-major factor grids, disjoint fit/validation points',withheld_max_error_px=err,fit_points=len(a),withheld_points=len(b),runner_up_max_error_px=candidates[1][0] if len(candidates)>1 else None,meaning='Recovers projection encoded by this cloud, not independently calibrated physical camera or metric depth')

def tag_seed(rgb,rig,camera):
    observations=[];detected=[];K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
    for family in sorted(set(t['family'] for t in rig['tags'])):
        params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        corners,ids,_=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,family)),params).detectMarkers(rgb)
        if ids is None:continue
        for tag in rig['tags']:
            if tag['family']!=family:continue
            matches=np.flatnonzero(ids.ravel()==tag['id'])
            if len(matches)>1:raise ValueError('Duplicate required tag ID; resolve association')
            if len(matches)==1:
                detected.append(tag['id']);observations.append((tag,corners[matches[0]][0].astype(float)))
    if len(observations)<2 or len(set(t['group'] for t,uv in observations))<2:raise ValueError('Need visible known tags on both moving groups for automatic feature search; agent must use another view/3D correspondences if unavailable')
    cad=np.vstack([t['cad_corners_detector_order_mm'] for t,uv in observations]);uv=np.vstack([uv for t,uv in observations]);d=np.repeat([rig['kinematics']['group_translation_per_q'][t['group']] for t,u in observations],4,axis=0)
    lo,hi=rig['kinematics']['q_bounds_mm'];solutions=[]
    def residual(x):
        points=cad+d*x[6];p,_=cv2.projectPoints(points,x[:3],x[3:6],K,None);return (p[:,0,:]-uv).ravel()
    for q in np.linspace(lo+1,hi-1,7):
        ok,rv,tv=cv2.solvePnP(cad+d*q,uv,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tv[2,0]<=0:continue
        opt=least_squares(residual,np.r_[rv.ravel(),tv.ravel(),q],bounds=(np.r_[[-np.inf]*5,1,lo],np.r_[[np.inf]*6,hi]),loss='soft_l1',xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=1000)
        R=Rotation.from_rotvec(opt.x[:3]).as_matrix();depth=((cad+d*opt.x[6])@R.T+opt.x[3:6])[:,2]
        if opt.success and np.all(depth>0):solutions.append((np.mean(residual(opt.x)**2),opt.x))
    if not solutions:raise ValueError('No valid articulated tag pose hypothesis')
    solutions.sort(key=lambda x:x[0]);cost,x=solutions[0]
    if np.sqrt(cost)>3:raise ValueError('Tags do not agree with configured rig/print geometry (RMS >3 px)')
    return x,K,detected,dict(tag_coordinate_rms_px=float(np.sqrt(cost)),initial_opening_change_mm=float(x[6]),method='Known tag CAD corners, joint perspective body pose and opening; no object/contact input')

def circular_candidates(gray,predicted,radius_px,search_px=14):
    u,v=predicted;h,w=gray.shape;u0=max(0,int(u-search_px-12));v0=max(0,int(v-search_px-12));u1=min(w,int(u+search_px+13));v1=min(h,int(v+search_px+13))
    if u1-u0<12 or v1-v0<12:return []
    crop=gray[v0:v1,u0:u1];blur=cv2.GaussianBlur(crop,(3,3),.7);edges=cv2.Canny(blur,40,100)
    candidates=[]
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour)<10:continue
        area=cv2.contourArea(contour);length=cv2.arcLength(contour,True)
        circularity=4*np.pi*area/max(length*length,1)
        if area<12 or circularity<.70:continue
        (x,y),(major,minor),angle=cv2.fitEllipse(contour)
        if min(major,minor)<3 or max(major,minor)>radius_px*3.2 or min(major,minor)/max(major,minor)<.4:continue
        r=(major+minor)/4;pt=np.array([x+u0,y+v0]);dist=np.linalg.norm(pt-predicted)
        if dist>search_px or not .45<r/max(radius_px,1)<1.6:continue
        score=dist/max(search_px,1)+.25*abs(np.log(r/max(radius_px,1)))+.5*(1-circularity)
        candidates.append(dict(pixel=pt.tolist(),radius_px=float(r),closed_contour_circularity=float(circularity),distance_from_tag_prediction_px=float(dist),score=float(score),detector='Canny closed contour + ellipse fit'))
    return sorted(candidates,key=lambda x:x['score'])

def discover(scene,rig_path,output,camera_json=None):
    out=Path(output)
    if out.exists():raise FileExistsError('Choose a fresh discovery directory')
    rig=read(rig_path);validate_rig(rig);cloud=trimesh.load(scene,process=False);xyz=np.asarray(cloud.vertices)
    if not np.isfinite(xyz).all():raise ValueError('Automatic mapping currently needs a finite ordered cloud; use explicit metadata/correspondences for clouds with missing vertices')
    if camera_json:
        info=read(camera_json);camera=info.get('camera',info);audit={'method':'Supplied mapping; verified against point ordering'}
    else:camera,audit=recover_camera(xyz)
    w,h=camera['width'],camera['height']
    if len(xyz)!=w*h:raise ValueError('RGB cloud does not match source grid')
    index=np.arange(0,len(xyz),max(1,len(xyz)//10000));check,z=project(xyz[index],camera)
    if np.any(z<=0) or np.max(np.linalg.norm(check-np.c_[index%w,index//w],axis=1))>.1:raise ValueError('Source mapping fails cloud reprojection')
    rgb=np.asarray(cloud.colors)[:,:3].reshape(h,w,3)
    out.mkdir(parents=True);Image.fromarray(rgb).save(out/'source_rgb.png');write(out/'source_camera.json',camera)
    try:seed,K,ids,tag_audit=tag_seed(rgb,rig,camera)
    except ValueError as error:
        write(out/'discovery.json',dict(camera_recovery=audit,status='tag_initialization_failed',reason=str(error),source_cloud_sha256=sha(scene),rig_sha256=sha(rig_path),object_used=False))
        raise
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);matches=[];reviews=[]
    R=Rotation.from_rotvec(seed[:3]).as_matrix()
    for feature in rig['features']:
        if feature.get('automatic_detector')!='circular_screw':continue
        p=np.array(feature['cad_mm'])+np.asarray(rig['kinematics']['group_translation_per_q'][feature['group']])*seed[6]
        pix,_=cv2.projectPoints(p.reshape(1,3),seed[:3],seed[3:6],K,None);pix=pix[0,0];depth=(R@p+seed[3:6])[2]
        radius=feature['head_radius_mm']*np.sqrt(camera['fx']*camera['fy'])/depth
        candidates=circular_candidates(gray,pix,radius,search_px=feature.get('search_radius_px',14))
        status='no_image_feature'
        if candidates:
            best=candidates[0]
            # Distinct equally ranked image candidates remain unresolved; preserve them for agent review.
            ambiguous=any(c['score']<best['score']+.12 and np.linalg.norm(np.array(c['pixel'])-best['pixel'])>3 for c in candidates[1:])
            status='ambiguous' if ambiguous else 'matched'
            if not ambiguous:matches.append(dict(name=feature['name'],source_pixel=best['pixel'],sigma_pixel=2.,discovery='Closed image-edge contour and ellipse fit; CAD projection only supplies search region'))
        reviews.append(dict(name=feature['name'],group=feature['group'],predicted_pixel=pix.tolist(),expected_radius_px=float(radius),status=status,candidates=candidates))
    ann=dict(organized_cloud={'width':w,'height':h},camera=camera,detect_tag_ids=ids,features=matches)
    write(out/'annotations.json',ann);write(out/'discovery.json',dict(camera_recovery=audit,tag_seed=tag_audit,features=reviews,requires_agent_visual_review=True,source_cloud_sha256=sha(scene),rig_sha256=sha(rig_path),object_used=False))
    draw_discovery(out,rgb,reviews)
    body_names={f['name'] for f in rig['features'] if f['group']=='body'}
    if sum(m['name'] in body_names for m in matches)<3:raise ValueError('Fewer than three unambiguous body features. Review correspondence_candidates.png and discovery.json; agent should refine real observed features, never accept projected positions as measurements.')
    prepare(scene,rig_path,out/'annotations.json',out/'prepared')
    print(out/'annotations.json');return ann

def draw_discovery(out,rgb,reviews):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,11));ax.imshow(rgb)
    for r in reviews:
        p=r['predicted_pixel'];ax.plot(*p,'+',color='yellow',ms=10)
        if r['candidates']:
            c=r['candidates'][0]['pixel'];color='lime' if r['status']=='matched' else 'orange';ax.plot(*c,'o',mfc='none',mec=color,ms=9);ax.text(c[0]+5,c[1],r['name'],color=color,fontsize=8,bbox=dict(facecolor='black',alpha=.5,pad=1))
    ax.set_title('Image feature candidates: green = matched; orange = ambiguous\nYellow crosses are CAD predictions, not measured correspondences');ax.axis('off');fig.tight_layout();fig.savefig(out/'correspondence_candidates.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots((len(reviews)+2)//3,3,figsize=(12,4*((len(reviews)+2)//3)),squeeze=False)
    for ax,r in zip(axes.ravel(),reviews):
        u,v=r['predicted_pixel'];ax.imshow(rgb);ax.set_xlim(u-22,u+22);ax.set_ylim(v+22,v-22);ax.plot(u,v,'+',color='yellow')
        for i,c in enumerate(r['candidates']):ax.plot(*c['pixel'],'o',mfc='none',mec='lime');ax.text(*c['pixel'],str(i),color='red')
        ax.set_title(r['name']+' | '+r['status']);ax.set_xlabel('Original image column');ax.set_ylabel('Original image row')
    for ax in axes.ravel()[len(reviews):]:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'feature_crops.png',dpi=160);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--rig',required=True);p.add_argument('--output',required=True);p.add_argument('--camera-json');a=p.parse_args();discover(a.scene,a.rig,a.output,a.camera_json)
