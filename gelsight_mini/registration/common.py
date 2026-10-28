"""Portable registration data conventions; no machine-specific paths."""
from pathlib import Path
import json,hashlib
import numpy as np

def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolve(base,value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()
def camera_matrix(camera):
    m=np.asarray(camera['world_to_camera'],float)
    if m.shape!=(4,4) or not np.isfinite(m).all():raise ValueError('world_to_camera must be a finite 4x4 rigid matrix')
    np.testing.assert_allclose(m[3],[0,0,0,1],atol=1e-8)
    np.testing.assert_allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-8)
    if not np.isclose(np.linalg.det(m[:3,:3]),1):raise ValueError('Camera rotation must be proper')
    for k in ['fx','fy']:
        if not np.isfinite(camera[k]) or camera[k]<=0:raise ValueError('Invalid camera focal length')
    return m

def project(world,camera):
    """OpenCV camera: +x right, +y down, +z forward; translation in scene units."""
    m=camera_matrix(camera);cam=np.asarray(world)@m[:3,:3].T+m[:3,3]
    z=np.maximum(cam[:,2],1e-12)
    return np.c_[camera['fx']*cam[:,0]/z+camera['cx'],camera['fy']*cam[:,1]/z+camera['cy']],cam[:,2]

def rays_world(pixels,camera):
    m=camera_matrix(camera);uv=np.asarray(pixels)
    rays=np.c_[(uv[:,0]-camera['cx'])/camera['fx'],(uv[:,1]-camera['cy'])/camera['fy'],np.ones(len(uv))]@m[:3,:3]
    origin=-m[:3,:3].T@m[:3,3]
    return origin,rays

def similarity(cad,world):
    x=np.asarray(cad)-np.mean(cad,axis=0);y=np.asarray(world)-np.mean(world,axis=0)
    if np.linalg.matrix_rank(x)<2 or np.linalg.matrix_rank(y)<2:raise ValueError('Need at least three non-collinear fixed-body correspondences to initialize scale/pose')
    u,sv,vt=np.linalg.svd(x.T@y);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
    row=u@d@vt;k=np.sum(sv*np.diag(d))/np.sum(x*x)
    if k<=0:raise ValueError('Invalid initial scale')
    t=np.mean(world,axis=0)-np.mean(cad,axis=0)@row*k
    return 1/k,row.T,t/k

def validate_rig(rig):
    if rig['kinematics']['type']!='scalar_prismatic':raise ValueError('Only explicitly defined scalar-prismatic articulation is supported; do not use for rotary/linkage mechanisms')
    g=rig['kinematics']['group_translation_per_q'];a=np.asarray(list(g.values()),float)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():raise ValueError('Invalid motion vectors')
    if not np.allclose(g['body'],0):raise ValueError('Body must be stationary relative to CAD frame')
    if np.allclose(a,0):raise ValueError('Opening is unobservable with zero group motion')
    lo,hi=rig['kinematics']['q_bounds_mm']
    if not np.isfinite([lo,hi]).all() or lo>=hi:raise ValueError('Invalid opening limits')
    used=[]
    for group,ids in rig.get('part_groups',{}).items():
        if group not in g:raise ValueError('Unknown part group')
        used.extend(ids)
    if len(set(used))!=len(used):raise ValueError('CAD part assigned to multiple groups')
    for pad in rig['pads']:
        if 'apex_from_extremum' in pad:
            a=pad['apex_from_extremum']
            if a['axis'] not in [0,1,2] or a['which'] not in ['min','max']:raise ValueError('Invalid apex extremum selector')
        b=np.asarray(pad['basis_columns_cad'],float);n=np.asarray(pad['inward_cad'],float)
        if b.shape!=(3,3) or not np.isfinite(b).all():raise ValueError('Invalid Mini basis')
        np.testing.assert_allclose(b.T@b,np.eye(3),atol=1e-8)
        if np.linalg.det(b)<.999999:raise ValueError('Mini basis must be right handed')
        if n.shape!=(3,) or np.linalg.norm(n)==0 or np.dot(n/np.linalg.norm(n),b[:,2])<.999999:raise ValueError('Mini inward axis disagrees with housing direction')
        if pad['group'] not in g:raise ValueError('Unknown pad group')
