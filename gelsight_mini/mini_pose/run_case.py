"""Fixed-pose Mini prediction. No pose fitting, contact reset, or approach offset."""
from pathlib import Path
import argparse, copy, json, sys
import numpy as np
import trimesh, torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mini_depth'))
sys.path.insert(0, str(ROOT / 'mini_optical'))
from geometry import normal_grid, camera, project, unproject
from profile import load_profile, sha
from adapter import MiniOptical


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_mesh(path):
    asset = trimesh.load(path, process=False)
    if isinstance(asset, trimesh.Scene):
        # dump applies each scene-node world transform, including repeated instances.
        meshes = [g for g in asset.dump() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError('No triangle meshes in scene')
        asset = trimesh.util.concatenate(meshes)
    if not isinstance(asset, trimesh.Trimesh) or not len(asset.faces):
        raise ValueError('Input must contain a triangle mesh')
    if not np.isfinite(asset.vertices).all():
        raise ValueError('Nonfinite mesh vertices')
    return asset


def validate_config(c):
    missing = []
    scale = c.get('mm_per_world_unit')
    if scale is None or not np.isfinite(scale) or scale <= 0:
        missing.append('positive independently supported mm_per_world_unit')
    if not c.get('scale_source'):
        missing.append('scale_source')
    if c.get('alignment_verified') is not True or not c.get('alignment_evidence'):
        missing.append('mesh/cloud alignment evidence')
    names = [p.get('name') for p in c.get('pads', [])]
    if not names or len(set(names)) != len(names):
        missing.append('unique pad names')
    for pad in c.get('pads', []):
        name = pad['name']
        if not isinstance(name, str) or not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Unsafe pad name')
        for key in ('origin_world', 'basis_columns_world', 'pose_source', 'orientation_source', 'normal_toward_sensor_world'):
            if pad.get(key) is None or pad.get(key) == '':
                missing.append(name + '.' + key)
        if pad.get('origin_world') is None or pad.get('basis_columns_world') is None:
            continue
        o = np.asarray(pad['origin_world'], float)
        b = np.asarray(pad['basis_columns_world'], float)
        if o.shape != (3,) or b.shape != (3, 3) or not np.isfinite(o).all() or not np.isfinite(b).all():
            raise ValueError(name + ': invalid apex/basis')
        np.testing.assert_allclose(b.T @ b, np.eye(3), atol=1e-8, rtol=0)
        if not np.isclose(np.linalg.det(b), 1, atol=1e-8, rtol=0):
            raise ValueError(name + ': basis must be right handed')
        if pad.get('normal_toward_sensor_world') is not None:
            n = np.asarray(pad['normal_toward_sensor_world'], float)
            if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                raise ValueError(name + ': invalid independent normal reference')
            if np.dot(n / np.linalg.norm(n), b[:, 2]) < .999999:
                raise ValueError(name + ': z disagrees with supplied inward normal')
        if pad.get('origin_definition') != 'undeformed_gel_rest_apex':
            missing.append(name + ': resolve mounting frame to undeformed gel rest apex')
    return missing


def run(config_path, output=None):
    c = json.loads(Path(config_path).read_text())
    missing = validate_config(c)
    if missing:
        raise ValueError('Prediction blocked; unresolved: ' + '; '.join(missing))
    out = Path(output) if output else ROOT / 'outputs/mini_pose' / c['case_name']
    if out.resolve() == (ROOT / 'outputs/mini_pose/cube').resolve():
        raise ValueError('Approved cube reference directory is protected')
    if (out / 'pose_report.json').exists() or any((out / p['name']).exists() for p in c['pads']):
        raise FileExistsError('Use a new output directory; existing predictions are preserved')
    meshfile, cloudfile = ROOT / c['object_mesh'], ROOT / c['scene_cloud']
    input_hashes = {'mesh': sha(meshfile), 'cloud': sha(cloudfile), 'config': sha(config_path)}
    mesh = load_mesh(meshfile)
    original_v = mesh.vertices.copy()
    p, gv, gf, profile = load_profile(ROOT)
    assert (p['height'], p['width'], p['mode']) == (320, 240, 'runtime_horizontal_fov')
    np.testing.assert_allclose([p['fx'], p['fy'], p['distance_mm']], [308.8192771084337, 308.8192771084337, 26.7], atol=1e-10, rtol=0)
    rest, rid = normal_grid(gv, gf)
    d0, di = camera(gv, gf, p)
    valid, nvalid = di >= 0, rid >= 0
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    opt = MiniOptical(ROOT, device='cpu')
    assert opt.metadata['effective_bins'] == 125
    scale = c['mm_per_world_unit']
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'config.json', c)
    report = {'mode': c.get('mode', 'existing_absolute_pose'), 'first_contact_reset_applied': False, 'added_approach_mm': 0,
              'input_sha256': input_hashes, 'mm_per_world_unit': scale, 'scale_source': c['scale_source'],
              'alignment_evidence': c['alignment_evidence'], 'profile': profile, 'optical': opt.metadata,
              'physical_accuracy_validated': False, 'pads': [], 'assumptions': c.get('assumptions', []),
              'interpretation': 'Simulated geometric intersections conditional on reconstruction, scale and fixed rest poses; no measured tactile validation, elastic mechanics or force solution.',
              'source_sha256': {str(q.relative_to(ROOT)): sha(q) for q in [Path(__file__), ROOT/'mini_depth/geometry.py', ROOT/'mini_depth/profile.py', ROOT/'mini_optical/adapter.py']}}
    display = []
    scene = trimesh.Scene()
    scene.add_geometry(mesh.copy(), node_name='unchanged_object')
    try:
        if cloudfile.suffix == '.npz':
            with np.load(cloudfile) as data:
                cloud = trimesh.points.PointCloud(data['xyz'])
        else:
            cloud = trimesh.load(cloudfile, process=False)
        if isinstance(cloud, trimesh.points.PointCloud):
            take = np.arange(0, len(cloud.vertices), max(1, len(cloud.vertices)//50000))
            scene.add_geometry(trimesh.points.PointCloud(cloud.vertices[take], colors=cloud.colors[take] if len(cloud.colors)==len(cloud.vertices) else None), node_name='scene_rgb_sample')
    except (ValueError, TypeError):
        pass  # Cube reference cloud is NPZ; its scene is still represented by object/pads.
    for pad in c['pads']:
        saved_pad = copy.deepcopy(pad)
        o, b = np.asarray(pad['origin_world']), np.asarray(pad['basis_columns_world'])
        v, f = (original_v - o) @ b * scale, np.asarray(mesh.faces)
        obj, oid = normal_grid(v, f)
        hit = nvalid & (oid >= 0)
        h = np.zeros_like(rest)
        h[hit] = np.maximum(obj[hit] - rest[hit], 0)
        objd, ids = camera(v, f, p)  # Fails explicitly if any geometry crosses near plane.
        dh = np.zeros_like(d0)
        support = valid & (ids >= 0)
        dh[support] = np.maximum(d0[support] - objd[support], 0)
        d1 = d0 - dh
        a = (dh / 1000).astype(np.float32)
        assert np.isfinite(a).all() and (a >= 0).all() and (a[~valid] == 0).all()
        np.testing.assert_allclose(a.astype(np.float64) * 1000, dh, atol=1e-9, rtol=np.finfo(np.float32).eps)
        rgb = opt.render(a)[0]
        contact = valid & (dh > 1e-5)
        ncontact = nvalid & (h > 1e-5)
        folder = out / pad['name']
        folder.mkdir()
        arrays = {'normal_indentation_mm': np.where(nvalid, h, np.nan), 'nominal_camera_depth_mm': np.where(valid, d0, np.nan),
                  'deformed_camera_depth_mm': np.where(valid, d1, np.nan), 'camera_depth_delta_mm': dh,
                  'hydroshear_depth_delta_m': a, 'valid_gel_mask': valid, 'normal_valid_gel_mask': nvalid,
                  'contact_mask': contact, 'normal_contact_mask': ncontact}
        for key, arr in arrays.items():
            np.save(folder / (key + '.npy'), arr)
        Image.fromarray(rgb).save(folder / 'tactile_rgb.png')
        with Image.open(folder / 'tactile_rgb.png') as im:
            im.load()
            assert im.mode == 'RGB' and im.size == (240, 320)
            np.testing.assert_array_equal(np.asarray(im), opt.render(a)[0])
        # Unproject only valid contact pixels, avoiding inf*zero outside gel support.
        safe = np.where(valid, d1, 0)
        world = o + unproject(safe, p)[contact] @ b.T / scale
        np.save(folder / 'contact_points_world.npy', world)
        if len(world):
            # Preserve original-world coordinates in the saved PLY as float64.
            header='ply\nformat ascii 1.0\nelement vertex '+str(len(world))+'\nproperty double x\nproperty double y\nproperty double z\nend_header'
            np.savetxt(folder/'contact_in_original_scene.ply',world,fmt='%.17g',header=header,comments='')
            reloaded=np.asarray(trimesh.load(folder/'contact_in_original_scene.ply',process=False).vertices)
            np.testing.assert_array_equal(reloaded,world)
            # Work in mm: trimesh's absolute degeneracy tolerance can misclassify
            # tiny triangles in arbitrary world units. Verify finite triangles, not just planes.
            triangles = v[f[ids[contact]]]
            local_points = (world - o) @ b * scale
            nearest = trimesh.triangles.closest_point(triangles, local_points)
            err = np.linalg.norm(nearest - local_points, axis=1)
            maxerr = float(err.max())
            assert maxerr < 1e-7
        else:
            (folder / 'contact_in_original_scene.ply').write_text('ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
            maxerr = 0.
        uv = project(v, p)
        inview = (uv[:,0]>=0)&(uv[:,0]<p['width'])&(uv[:,1]>=0)&(uv[:,1]<p['height'])
        meta = {'name': pad['name'], 'pose': pad, 'basis_determinant': float(np.linalg.det(b)),
                'normal_peak_indentation_mm': float(h.max()), 'camera_peak_depth_delta_mm': float(dh.max()),
                'float32_metres_roundtrip_max_error_mm': float(np.max(np.abs(a.astype(np.float64)*1000-dh))),
                'rest_surface_z_extent_mm': float(np.ptp(gv[:,2])),
                'normal_samples_penetrating_beyond_full_rest_surface_z_extent': int(np.sum(h>np.ptp(gv[:,2]))),
                'large_overlap_warning': bool(np.any(h>np.ptp(gv[:,2]))),
                'normal_contact_area_mm2': float(ncontact.sum() * .05**2), 'area_definition': 'projected sensor-normal sample footprint at 0.05 mm spacing; not curved surface area',
                'camera_contact_pixels': int(contact.sum()), 'normal_contact_samples': int(ncontact.sum()),
                'normal_contact_samples_outside_camera_fov': None,
                'surface_point_triangle_distance_max_mm': maxerr, 'saved_ply_points_roundtrip_exact': True, 'pose_unchanged': pad == saved_pad,
                'mesh_vertices_outside_camera_fov': int((~inview).sum()), 'gel_pixels_without_object_support': int((valid & ~support).sum()),
                'normal_gel_samples_without_object_support': int((nvalid & ~hit).sum()),
                'no_contact': not bool(ncontact.any() or contact.any()), 'rgb_sha256': sha(folder/'tactile_rgb.png'),
                'renderer_input_finite_nonnegative': True, 'optical_replay_exact': True,
                'invalid_depth_convention': 'NaN for unsupported absolute depth/normal height; zero for optical delta',
                **opt.gradient_diagnostics(a)}
        yy, xx = np.indices(rest.shape)
        normal_points = np.c_[((xx-(rest.shape[1]-1)/2)*.05)[ncontact], (-(yy-(rest.shape[0]-1)/2)*.05)[ncontact], obj[ncontact]]
        if len(normal_points):
            nq = project(normal_points,p)
            meta['normal_contact_samples_outside_camera_fov'] = int(((nq[:,0]<0)|(nq[:,0]>=p['width'])|(nq[:,1]<0)|(nq[:,1]>=p['height'])).sum())
        else:
            meta['normal_contact_samples_outside_camera_fov'] = 0
        write(folder / 'metadata.json', meta)
        report['pads'].append(meta)
        display.append((pad['name'], np.where(nvalid,h,np.nan), dh, rgb))
        gelworld = o + gv @ b.T / scale
        gelmesh = trimesh.Trimesh(gelworld, gf, process=False)
        gelmesh.visual.face_colors = [40, 160, 230, 150]
        scene.add_geometry(gelmesh, node_name=pad['name']+'_fixed_rest_gel')
    np.testing.assert_array_equal(mesh.vertices, original_v)
    assert input_hashes == {'mesh': sha(meshfile), 'cloud': sha(cloudfile), 'config': sha(config_path)}
    scene.export(out / 'scene_overlay.glb')
    vmax = max(max(float(np.nanmax(row[1])), float(row[2].max())) for row in display) or 1e-6
    fig, axes = plt.subplots(len(display), 3, figsize=(12, 4.5*len(display)), squeeze=False)
    for row, (name, h, dh, rgb) in enumerate(display):
        for col, (a0, title) in enumerate(zip([h, dh, rgb], ['Normal indentation (mm)', 'Camera depth difference (mm)', 'Taxim RGB, 240 x 320'])):
            ax=axes[row,col]
            if col<2:
                kw={'extent':[-10.375,10.375,-12.625,12.625]} if col==0 else {}
                im=ax.imshow(a0, cmap='magma', vmin=0, vmax=vmax, **kw)
                fig.colorbar(im,ax=ax,label='mm',shrink=.8)
            else:
                ax.imshow(a0)
            ax.set_title(name+' | '+title)
            ax.set_xlabel('Sensor x (mm)' if col==0 else 'Image column (px)')
            ax.set_ylabel('Sensor y (mm)' if col==0 else 'Image row (px)')
    fig.suptitle(c['case_name']+': fixed poses, no first-contact reset or added approach')
    fig.tight_layout()
    fig.savefig(out/'fixed_pose_estimate.png',dpi=180)
    plt.close(fig)
    write(out/'pose_report.json',report)
    write(out/'validation.json',{'mesh_unchanged':True,'input_hashes_unchanged':True,'physical_accuracy_validated':False,
                                'checks':[{k:v for k,v in x.items() if k in ['name','pose_unchanged','renderer_input_finite_nonnegative','optical_replay_exact','surface_point_triangle_distance_max_mm']} for x in report['pads']]})
    print(json.dumps({'output':str(out),'pads':[{k:x[k] for k in ['name','normal_peak_indentation_mm','rgb_sha256']} for x in report['pads']]},indent=2))
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    ap.add_argument('--output',help='New destination; existing predictions are never overwritten')
    ap.add_argument('--check-config',action='store_true')
    args=ap.parse_args()
    if args.check_config:
        missing=validate_config(json.loads(Path(args.config).read_text()))
        print(json.dumps({'ready':not missing,'unresolved':missing},indent=2))
        sys.exit(2 if missing else 0)
    run(args.config,args.output)
