"""Read pinned Mini assets, without importing the optical simulator."""
from pathlib import Path
import ast,hashlib,json,subprocess,xml.etree.ElementTree as ET
import numpy as np
import trimesh,yaml

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_profile(root,mode='runtime_horizontal_fov'):
    root=Path(root);repo=root/'vendor/hydroshear';base=repo/'rl/tacsl_sensors'
    paths={'camera_yaml':base/'configs/gs_mini.yaml','renderer_config':base/'tactile_utils/gelsight_render.py','polycalib':base/'tactile_utils/gs_mini_data/polycalib.npz','surface':repo/'assets/tacsl/mesh/gs_mini_meshes/gsmini_elastomer_transformed.obj','urdf':base/'examples/assets/urdf/gelsight_description/robots/gsmini_finger.urdf'}
    cfg=yaml.safe_load(paths['camera_yaml'].read_text());opt={}
    for node in ast.walk(ast.parse(paths['renderer_config'].read_text())):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='conf_gs_mini' for t in node.targets):
            opt={k.value:ast.literal_eval(v) for k,v in zip(node.value.keys,node.value.values) if isinstance(v,ast.Constant)}
    tree=ET.parse(paths['urdf']);tip=None
    for joint in tree.findall('joint'):
        if joint.find('child').attrib.get('link')=='elastomer_tip':tip=np.fromstring(joint.find('origin').attrib['xyz'],sep=' ')
    if tip is None:raise ValueError('Missing tip frame')
    gel=trimesh.load(paths['surface'],force='mesh',process=False);v=(np.asarray(gel.vertices)-tip)*1000
    p={'width':cfg['width'],'height':cfg['height'],'distance_mm':cfg['camera_dist']*1000,'near_mm':cfg['near_plane']*1000,'mode':mode}
    p['fx']=p['width']*cfg['camera_dist']/(2*cfg['cx'])
    p['fy']=p['fx'] if mode=='runtime_horizontal_fov' else p['height']*cfg['camera_dist']/(2*cfg['cy'])
    if mode not in ['runtime_horizontal_fov','declared_xy_extents']:raise ValueError(mode)
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    expected='f815b82fdf3451852acd918933020a82cede1f3b'
    if revision!=expected:raise ValueError('Hydroshear revision changed; review profile before proceeding')
    table=np.load(paths['polycalib']);bins=int(table['bins'])
    audit={'hydroshear_revision':revision,'camera':p,'camera_yaml':cfg,'optical_parameters_not_used_for_depth':opt,'polycalib_bins':bins,'polycalib_shapes':{k:list(table[k].shape) for k in table.files},'optical_bin_mismatch':bins!=opt['numBins'],'gel_extent_xyz_mm':np.ptp(v,axis=0).tolist(),'tip_origin_parent_m':tip.tolist(),'tip_plane_fov_mm':[p['width']*p['distance_mm']/p['fx'],p['height']*p['distance_mm']/p['fy']],'sources':{k:{'path':str(f),'sha256':sha(f)} for k,f in paths.items()},'notes':['Default reproduces horizontal-FOV camera parameterization with square pixels; source computes vertical FOV but does not assign it. Isaac Gym has not been run for pixel parity.','Canonical frame: x image right, y image up, z into sensor; camera at z=26.7 mm looks along -z. Actual scene camera roll is inferred.','Use physical primary contact surface, not 0.1 mm offset duplicate visual surface or enclosed backing cap.','0.065 mm/pixel is an optical slope conversion, not the perspective depth-map calibration.','No optical rendering, marker motion, pressure prediction or elastic mechanics is performed.']}
    return p,v,np.asarray(gel.faces),audit
