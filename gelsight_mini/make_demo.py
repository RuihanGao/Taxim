"""Generate a self-contained synthetic CAD/scene case; no private example data."""
from pathlib import Path
import argparse,sys
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'mini_depth'))
from profile import load_profile
from registration.common import write,sha

def make_demo(output):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Choose a fresh demo directory')
    out.mkdir(parents=True);(out/'cad').mkdir();_,gv,gf,_=load_profile(ROOT)
    R=Rotation.from_rotvec([.1,-.2,.15]).as_matrix();t=np.array([15.,-35.,80.]);scale=1000.;q=-2.
    groups={'body':[0,0,0],'left':[.5,0,0],'right':[-.5,0,0]};pads=[];parts=[]
    for idx,(name,sign) in enumerate([('left',1),('right',-1)]):
        apex=np.array([15.*sign,0,0]);b=np.array([[0,0,sign],[0,-sign,0],[1,0,0.]])
        v=apex+gv@b.T;path=out/f'cad/{name}.npz';np.savez_compressed(path,vertices=v,faces=gf)
        parts.append(dict(index=idx,name=name,mesh_file=path.name,mesh_sha256=sha(path)))
        pads.append(dict(name=name,group=name,gel_part_index=idx,rest_apex_cad_mm=apex.tolist(),basis_columns_cad=b.tolist(),inward_cad=[sign,0,0],orientation_source='Synthetic canonical fixture',mount_check_tolerance_mm=.05))
    manifest=dict(source_sha256='synthetic-CAD-generated-from-pinned-Mini-surface',units='mm',assembly_transforms_applied=True,parts=parts);write(out/'cad/assembly.json',manifest)
    rig=dict(name='Synthetic symmetric translating gripper',kinematics=dict(type='scalar_prismatic',group_translation_per_q=groups,q_bounds_mm=[-10,10]),part_groups={'left':[0],'right':[1]},pads=pads)
    write(out/'rig.json',rig)
    body=[[-20,-35,-15],[20,-35,-15],[0,-15,-10],[12,-25,8]];left=[[17,-10,-5],[17,10,-5],[20,0,8]];right=[[-17,-10,-5],[-17,10,-5],[-20,0,8]];records=[]
    for group,points in [('body',body),('left',left),('right',right)]:
        for i,p in enumerate(points):
            w=(R@(np.array(p)+np.array(groups[group])*q)+t)/scale
            records.append(dict(name=f'{group}_{i}',group=group,cad_mm=p,world=w.tolist(),method='Synthetic exact CAD/scene pair'))
    obj=trimesh.creation.icosphere(subdivisions=3,radius=14.4);obj.vertices=(obj.vertices@R.T+t)/scale;obj.export(out/'object.obj')
    xyz=np.vstack([obj.vertices,np.array([r['world'] for r in records])]);trimesh.points.PointCloud(xyz,colors=np.tile([60,140,80,255],(len(xyz),1))).export(out/'scene.ply')
    write(out/'landmarks.json',dict(source_cloud_sha256=sha(out/'scene.ply'),records=records))
    cfg=dict(case_name='synthetic_sphere',object_mesh='object.obj',scene_cloud='scene.ply',cad_manifest='cad/assembly.json',rig_profile='rig.json',landmarks='landmarks.json',alignment_verified=True,alignment_evidence='Synthetic object, CAD and cloud transformed with the same known rigid transform; numerical test, not physical data.',registration={'sigma_3d_mm':3.},assumptions=['Synthetic fixture; both pads should have approximately 0.4 mm maximum normal overlap.'])
    write(out/'case.json',cfg);write(out/'truth.json',dict(scale_mm_per_world=scale,rotation_columns=R.tolist(),translation_mm=t.tolist(),opening_change_mm=q,expected_peak_mm=.4));print(out/'case.json');return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);make_demo(p.parse_args().output)
