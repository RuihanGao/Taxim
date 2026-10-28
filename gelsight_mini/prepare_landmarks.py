"""Create CAD/scene correspondences from tags and annotated rigid feature points."""
from pathlib import Path
import argparse
import numpy as np,trimesh,cv2
from scipy.ndimage import binary_erosion
from matplotlib.path import Path as Polygon
from PIL import Image
from registration.common import read,write,sha,resolve,project,rays_world,validate_rig

def prepare(scene_path,rig_path,annotations_path,output):
    scene_path=Path(scene_path).resolve();annotations_path=Path(annotations_path).resolve();out=Path(output)
    if out.exists():raise FileExistsError('Use a fresh landmark output directory')
    rig=read(rig_path);validate_rig(rig);ann=read(annotations_path);cloud=trimesh.load(scene_path,process=False)
    if not isinstance(cloud,trimesh.points.PointCloud):raise ValueError('Scene must be a point cloud')
    xyz=np.asarray(cloud.vertices);camera=ann.get('camera');grid=None;rgb=None;records=[];audit=[]
    if ann.get('organized_cloud'):
        h,w=ann['organized_cloud']['height'],ann['organized_cloud']['width']
        if len(xyz)!=h*w:raise ValueError('Cloud vertex order/size does not match declared organized grid')
        if camera is None:raise ValueError('Declare source camera when using an organized RGB cloud')
        if (camera['width'],camera['height'])!=(w,h):raise ValueError('Camera and grid dimensions differ')
        grid=xyz.reshape(h,w,3);rgb=np.asarray(cloud.colors)[:,:3].reshape(h,w,3)
        idx=np.arange(0,len(xyz),max(1,len(xyz)//10000));uv,z=project(xyz[idx],camera);expected=np.c_[idx%w,idx//w]
        if not np.isfinite(uv).all() or np.any(z<=0) or np.max(np.linalg.norm(uv-expected,axis=1))>ann.get('max_cloud_reprojection_px',.25):raise ValueError('Declared camera/order does not reproduce source cloud pixels; provide correct mapping or explicit 3D landmarks')
    if ann.get('source_image'):
        rgb=np.asarray(Image.open(resolve(annotations_path.parent,ann['source_image'])).convert('RGB'))
        if camera and rgb.shape[:2]!=(camera['height'],camera['width']):raise ValueError('Source image size disagrees with camera')
    tag_ids=ann.get('detect_tag_ids',[])
    if tag_ids:
        if grid is None or rgb is None:raise ValueError('Automatic tag depth requires a verified organized cloud; use explicit 3D pairs otherwise')
        defs={int(t['id']):t for t in rig.get('tags',[])};families=set(defs[i]['family'] for i in tag_ids);detections={}
        for family in families:
            if not family.startswith('DICT_') or not hasattr(cv2.aruco,family):raise ValueError('Unknown OpenCV tag dictionary: '+family)
            params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
            corners,ids,_=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,family)),params).detectMarkers(rgb)
            if ids is not None:
                for tag,poly in zip(ids.ravel(),corners):
                    key=(family,int(tag))
                    if key in detections:raise ValueError('Duplicate tag ID in source image; resolve association explicitly')
                    detections[key]=poly[0]
        yy,xx=np.indices(grid.shape[:2]);uvall=np.c_[xx.ravel(),yy.ravel()]
        for tag in tag_ids:
            definition=defs[tag];key=(definition['family'],tag)
            if key not in detections:raise ValueError('Required tag not detected: '+str(key))
            poly=detections[key];mask=binary_erosion(Polygon(poly).contains_points(uvall).reshape(grid.shape[:2]),iterations=int(ann.get('tag_erode_pixels',3)))
            points=grid[mask];points=points[np.isfinite(points).all(axis=1)]
            if len(points)<12:raise ValueError('Insufficient valid rigid tag-interior points')
            weights=np.ones(len(points))
            for _ in range(10):
                center=np.average(points,axis=0,weights=weights);_,sv,vh=np.linalg.svd((points-center)*np.sqrt(weights[:,None]),full_matrices=False);normal=vh[-1];res=(points-center)@normal;sigma=max(np.median(abs(res-np.median(res)))*1.4826,1e-8);weights=np.minimum(1,2*sigma/np.maximum(abs(res),1e-12))
            if sv[1]<1e-8:raise ValueError('Degenerate tag plane')
            origin,rays=rays_world(poly,camera);den=rays@normal
            if np.any(abs(den)<1e-8):raise ValueError('Tag plane is edge-on to source rays')
            depth=((center-origin)@normal)/den
            if np.any(depth<=0):raise ValueError('Tag plane lies behind source camera')
            world=origin+rays*depth[:,None];cad=np.asarray(definition['cad_corners_detector_order_mm'])
            if cad.shape!=(4,3):raise ValueError('Each tag needs four CAD corners in decoder-returned order, not image top-left order')
            for j in range(4):records.append(dict(name=f'tag_{tag}_decoder_corner_{j}',group=definition['group'],cad_mm=cad[j].tolist(),world=world[j].tolist(),source_pixel=poly[j].tolist(),sigma_pixel=1.,method='Decoded tag corner ray intersecting robust rigid-tag interior plane'))
            audit.append(dict(tag_id=tag,family=definition['family'],interior_points=len(points),plane_rms_world=float(np.sqrt(np.mean(res**2))),source_pixels=poly.tolist(),corner_order='OpenCV decoded marker order, as explicitly mapped in rig definition'))
    features={f['name']:f for f in rig.get('features',[])}
    for item in ann.get('features',[]):
        definition=features[item['name']];row=dict(name=item['name'],group=definition['group'],cad_mm=definition['cad_mm'],sigma_pixel=item.get('sigma_pixel',2.))
        if 'world' in item:
            row.update(world=item['world'],method='Explicit scene-coordinate 3D correspondence supplied by user')
        elif 'scene_point_index' in item:
            i=int(item['scene_point_index'])
            if not 0<=i<len(xyz):raise ValueError('Scene point index outside cloud')
            row.update(world=xyz[i].tolist(),method='Explicit point index in original scene cloud')
        elif grid is not None and 'source_pixel' in item:
            u,v=np.rint(item['source_pixel']).astype(int)
            if not (0<=u<grid.shape[1] and 0<=v<grid.shape[0]):raise ValueError('Feature pixel outside cloud grid')
            pts=grid[max(0,v-2):v+3,max(0,u-2):u+3].reshape(-1,3);pts=pts[np.isfinite(pts).all(axis=1)]
            if len(pts)<3:raise ValueError('Insufficient valid depth near rigid feature')
            row.update(world=np.median(pts,axis=0).tolist(),method='Annotated rigid feature; median 5x5 original cloud neighborhood')
        else:raise ValueError('Feature needs explicit world coordinate, point index, or pixel in verified organized cloud')
        if 'source_pixel' in item:row['source_pixel']=item['source_pixel']
        records.append(row)
    result=dict(source_cloud_sha256=sha(scene_path),rig_sha256=sha(rig_path),camera=camera,records=records,tag_plane_audit=audit,object_or_contact_used=False)
    out.mkdir(parents=True);write(out/'landmarks.json',result)
    if rgb is not None:Image.fromarray(rgb).save(out/'source_rgb.png')
    print(out/'landmarks.json');return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--rig',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.scene,a.rig,a.annotations,a.output)
