"""Portable CAD registration -> frozen rest poses -> full-mesh depth -> Mini RGB."""
from pathlib import Path
import argparse,sys,subprocess
import numpy as np,trimesh
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'mini_pose'))
from run_case import run,load_mesh,load_profile,validate_config
from registration.common import read,write,resolve,sha,project,validate_rig
from registration.fit import fit_registration

def run_pipeline(config_path,output,cad_python=None):
    config_path=Path(config_path).resolve();cfg=read(config_path);base=config_path.parent;out=Path(output).resolve()
    if out.exists():raise FileExistsError('Choose a new pipeline output directory; existing work is preserved')
    mesh_path=resolve(base,cfg['object_mesh']);cloud_path=resolve(base,cfg['scene_cloud']);rig_path=resolve(base,cfg['rig_profile'])
    rig=read(rig_path);validate_rig(rig)
    inputs={str(p):sha(p) for p in [config_path,mesh_path,cloud_path,rig_path]}
    if cfg.get('alignment_verified') is not True or not cfg.get('alignment_evidence'):raise ValueError('Verify the existing object/cloud alignment and record evidence; pipeline never realigns the object')
    cloud=trimesh.load(cloud_path,process=False)
    if not isinstance(cloud,trimesh.points.PointCloud):raise ValueError('scene_cloud must be a point cloud')
    out.mkdir(parents=True)
    if cfg.get('landmarks'):
        landmark_path=resolve(base,cfg['landmarks']);landmarks=read(landmark_path)
    else:
        from prepare_landmarks import prepare
        annotation_path=resolve(base,cfg['annotations']);inputs[str(annotation_path)]=sha(annotation_path)
        landmarks=prepare(cloud_path,rig_path,annotation_path,out/'landmarks');landmark_path=out/'landmarks/landmarks.json'
        if not cfg.get('source_image') and (out/'landmarks/source_rgb.png').exists():cfg['source_image']=str(out/'landmarks/source_rgb.png')
    inputs[str(landmark_path)]=sha(landmark_path)
    if landmarks.get('source_cloud_sha256') not in [None,sha(cloud_path)]:raise ValueError('Landmarks belong to a different scene cloud')
    if landmarks.get('rig_sha256') not in [None,sha(rig_path)]:raise ValueError('Landmarks belong to a different rig definition')
    if cfg.get('source_image'):
        image_path=resolve(base,cfg['source_image']);inputs[str(image_path)]=sha(image_path)
    if cfg.get('cad_manifest'):
        manifest_path=resolve(base,cfg['cad_manifest'])
    else:
        step=resolve(base,cfg['cad_step']);inputs[str(step)]=sha(step)
        subprocess.run([cad_python or sys.executable,str(ROOT/'export_step.py'),'--input',str(step),'--output',str(out/'cad')],check=True)
        manifest_path=out/'cad/assembly.json'
    manifest=read(manifest_path);inputs[str(manifest_path)]=sha(manifest_path)
    if cfg.get('cad_step'):
        step=resolve(base,cfg['cad_step']);inputs[str(step)]=sha(step)
        if manifest['source_sha256']!=sha(step):raise ValueError('CAD export does not match the supplied STEP')
    if rig.get('cad_sha256') and manifest['source_sha256']!=rig['cad_sha256']:raise ValueError('Rig definition belongs to a different CAD revision; identify actual mounts/features before reuse')
    if not manifest.get('units','').startswith('mm'):raise ValueError('CAD export must use millimetres')
    parts={p['index']:p for p in manifest['parts']};cache={}
    def part_mesh(index):
        if index not in cache:
            p=parts[index];path=resolve(manifest_path.parent,p['mesh_file']);digest=sha(path)
            if p.get('mesh_sha256') and digest!=p['mesh_sha256']:raise ValueError('CAD mesh hash mismatch')
            inputs[str(path)]=digest
            with np.load(path,allow_pickle=False) as a:cache[index]=trimesh.Trimesh(a['vertices'],a['faces'],process=False)
        return cache[index]
    camera=cfg.get('camera',landmarks.get('camera'))
    reg=fit_registration(rig,landmarks,cfg.get('registration',{}),camera);write(out/'registration.json',reg)
    scale=reg['scale_mm_per_world'];R=np.array(reg['rotation_columns']);t=np.array(reg['translation_mm']);q=reg['opening_change_mm'];motions=rig['kinematics']['group_translation_per_q']
    profile,gv,gf,profile_audit=load_profile(ROOT);pads=[];mounting=[]
    for pad in rig['pads']:
        cad=part_mesh(pad['gel_part_index']);b=np.asarray(pad['basis_columns_cad']);v=np.asarray(cad.vertices)
        if 'rest_apex_cad_mm' in pad:apex=np.asarray(pad['rest_apex_cad_mm'],float)
        else:
            spec=pad['apex_from_extremum'];axis=spec['axis'];idx=np.argmin(v[:,axis]) if spec['which']=='min' else np.argmax(v[:,axis]);apex=v[idx]
        local=(v-apex)@b;target=trimesh.Trimesh(local,cad.faces,process=False);samples=gv[::20]
        distances=np.concatenate([trimesh.proximity.closest_point_naive(target,p)[1] for p in np.array_split(samples,max(1,len(samples)//8))])
        tolerance=float(pad.get('mount_check_tolerance_mm',.05))
        if distances.max()>tolerance:raise ValueError(f'{pad["name"]}: pinned surface disagrees with CAD mount ({distances.max():.4f} mm > {tolerance}); review apex/axes/CAD rather than resizing gel')
        shifted=apex+np.asarray(motions[pad['group']])*q;o=(R@shifted+t)/scale;B=R@b
        pads.append(dict(name=pad['name'],origin_world=o.tolist(),basis_columns_world=B.tolist(),normal_toward_sensor_world=(R@np.asarray(pad['inward_cad'])).tolist(),origin_definition='undeformed_gel_rest_apex',pose_source='Joint body/moving-group landmark registration and actual CAD mount; see registration.json and mounting_audit.json',orientation_source=pad['orientation_source'],CAD_rest_apex_before_opening_mm=apex.tolist(),basis_columns_CAD=b.tolist()))
        mounting.append(dict(name=pad['name'],part_index=pad['gel_part_index'],apex_cad_mm=apex.tolist(),profile_to_CAD_sample_count=len(samples),distance_quantiles_mm=np.quantile(distances,[.5,.95,1]).tolist(),tolerance_mm=tolerance))
    frozen=dict(case_name=cfg['case_name'],object_mesh=str(mesh_path),scene_cloud=str(cloud_path),mm_per_world_unit=scale,scale_source='One global articulated CAD/scene landmark fit; scale not inferred from object contact',alignment_verified=True,alignment_evidence=cfg['alignment_evidence'],pads=pads,mode='fixed_rest_poses_from_articulated_CAD_registration',assumptions=cfg.get('assumptions',[])+['Registration is estimated; fit residuals are not physical accuracy.','No object/contact fitting, first-touch reset, pressing offset or rim normalization.'])
    if validate_config(frozen):raise ValueError('Invalid derived rest-pose configuration')
    write(out/'mounting_audit.json',mounting);write(out/'rest_pose_config.json',frozen)
    report=run(out/'rest_pose_config.json',out/'prediction')
    scene=trimesh.Scene();obj=load_mesh(mesh_path);scene.add_geometry(obj.copy(),node_name='unchanged_object');transforms=[];draw=[]
    for group,indices in rig.get('part_groups',{}).items():
        delta=np.asarray(motions[group])*q
        for idx in indices:
            m=part_mesh(idx).copy();m.vertices=(m.vertices+delta)@R.T/scale+t/scale;m.visual.face_colors=[30,175,210,190] if group=='body' else [240,174,35,190]
            scene.add_geometry(m,node_name=f'{group}_{idx}');draw.append(m)
            matrix=np.eye(4);matrix[:3,:3]=R/scale;matrix[:3,3]=(R@delta+t)/scale
            transforms.append(dict(index=idx,group=group,cad_mm_to_world=matrix.tolist()))
    gels=[]
    for pad in pads:
        w=np.array(pad['origin_world'])+gv@np.array(pad['basis_columns_world']).T/scale;m=trimesh.Trimesh(w,gf,process=False);m.visual.face_colors=[240,40,145,255];scene.add_geometry(m,node_name=pad['name']+'_rest_gel');gels.append(m)
    bounds=np.vstack([obj.bounds]+[m.bounds for m in draw+gels]);low=bounds.min(0)-20/scale;high=bounds.max(0)+20/scale;points=np.asarray(cloud.vertices);idx=np.flatnonzero(np.all((points>=low)&(points<=high),axis=1));idx=idx[::max(1,len(idx)//30000)]
    if len(idx):scene.add_geometry(trimesh.points.PointCloud(points[idx],colors=cloud.colors[idx]),node_name='scene_sample_for_review')
    scene.export(out/'registered_scene.glb');write(out/'part_transforms.json',transforms)
    plot_overlay(out,obj,draw,gels,cloud,idx,reg,landmarks,camera,cfg,base)
    warnings=[]
    limit=cfg.get('registration',{}).get('diagnostic_rms_mm',2.0)
    if reg['rms_3d_mm']>limit:warnings.append(f'Registration RMS {reg["rms_3d_mm"]:.3f} mm exceeds chosen review threshold {limit} mm; this threshold is not a tactile-accuracy guarantee.')
    if reg['parameter_bound_hit']:warnings.append('Registration reached an opening/scale bound; inspect constraints.')
    if any(p['large_overlap_warning'] for p in report['pads']):warnings.append('Object overlap exceeds the rest surface z extent; geometry requires review.')
    if any(p['no_contact'] for p in report['pads']):warnings.append('At least one pad has no contact; retained without forcing an indentation.')
    if not all(sha(Path(path))==digest for path,digest in inputs.items()):raise AssertionError('An input changed during processing')
    status=dict(status='geometry_diagnostic' if warnings else 'conditional_simulation',warnings=warnings,physical_accuracy_validated=False,object_unchanged=True,poses_frozen_before_depth=True,registration_rms_mm=reg['rms_3d_mm'],registration_rms_px=reg['rms_2d_px'],scale_mm_per_world=scale,opening_change_mm=q,input_sha256=inputs,pads=[{k:p[k] for k in ['name','normal_peak_indentation_mm','camera_peak_depth_delta_mm','normal_contact_area_mm2','no_contact']} for p in report['pads']])
    write(out/'report.json',status)
    (out/'README.md').write_text('# CAD registration and Mini simulation\n\nStatus: '+status['status']+'. All poses are estimates; physical accuracy is unvalidated.\n\n'+ '\n'.join('- '+w for w in warnings)+'\n\nSee registration.json for per-landmark/group residuals and fitted opening; mounting_audit.json for CAD/rest-surface checks; rest_pose_config.json for the frozen frames; registered_scene.glb and scene_overlay.png for placement. Prediction maps, masks, native RGB, contacts, calibration/profile metadata and depth comparison are in prediction/.\n\nRerun the original portable case JSON with run_pipeline.py into a new directory. Paths in that input are relative to the case JSON. The runtime frozen configuration records absolute paths on this machine; regenerate it after relocating the case. No object geometry or poses were adjusted to manufacture contact.\n')
    print(out/'report.json');return status

def plot_overlay(out,obj,draw,gels,cloud,idx,reg,landmarks,camera,cfg,base):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import LineCollection,PolyCollection
    fig=plt.figure(figsize=(13,7));ax=fig.add_subplot(121,projection='3d');origin=obj.bounds.mean(0);scale=reg['scale_mm_per_world'];points=np.asarray(cloud.vertices)
    if len(idx):ax.scatter(*((points[idx]-origin)*scale).T,c=cloud.colors[idx,:3]/255,s=.3,alpha=.3)
    for m,color,alpha in [(obj,'gray',.25)]+[(m,'#f0ad24',.12) for m in draw]+[(m,'#ed339b',.8) for m in gels]:
        v=(m.vertices-origin)*scale;ax.add_collection3d(Poly3DCollection(v[m.faces],facecolor=color,edgecolor='none',alpha=alpha))
    bb=np.vstack([obj.bounds]+[m.bounds for m in draw+gels]);bb=(bb-origin)*scale
    for k,fn in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):fn(bb[:,k].min(),bb[:,k].max())
    ax.set_box_aspect(np.maximum(np.ptp(bb,axis=0),1));ax.set(xlabel='Scene X (mm)',ylabel='Scene Y (mm)',zlabel='Scene Z (mm)',title='CAD + unchanged object + rest gels');ax.view_init(elev=20,azim=-60)
    b=fig.add_subplot(122)
    if camera:
        if cfg.get('source_image'):
            from PIL import Image
            b.imshow(Image.open(resolve(base,cfg['source_image'])))
        for m in draw:
            merged=m.copy();merged.merge_vertices();uv,z=project(merged.vertices,camera);e=merged.face_adjacency_edges[merged.face_adjacency_angles>np.deg2rad(25)];e=e[np.all(z[e]>0,axis=1)];b.add_collection(LineCollection(uv[e],colors='#16b8db',linewidths=.35,alpha=.55))
        for m in gels:
            uv,z=project(m.vertices,camera);f=m.faces[np.all(z[m.faces]>0,axis=1)];b.add_collection(PolyCollection(uv[f],facecolors='#ec3399',edgecolors='none',alpha=.3))
        for a,r in zip(landmarks['records'],reg['landmarks']):
            if 'source_pixel' in a and 'pred_pixel' in r:
                u,v=a['source_pixel'],r['pred_pixel'];b.plot([u[0],v[0]],[u[1],v[1]],color='red');b.plot(*u,'o',mfc='none',mec='red',ms=4)
        b.set_xlim(0,camera['width']);b.set_ylim(camera['height'],0);b.set_title('Source camera: CAD and observed features');b.set_aspect('equal')
    else:
        b.bar([r['name'] for r in reg['landmarks']],[r['error_mm'] for r in reg['landmarks']]);b.tick_params(axis='x',rotation=90);b.set_ylabel('Landmark residual mm');b.set_title('3D-only registration residuals')
    fig.suptitle(f'Estimated registration: {reg["rms_3d_mm"]:.3f} mm 3D RMS; no contact-driven adjustment');fig.tight_layout();fig.savefig(out/'scene_overlay.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--cad-python',help='Optional separate interpreter with requirements-cad.txt installed');a=p.parse_args();run_pipeline(a.config,a.output,a.cad_python)
