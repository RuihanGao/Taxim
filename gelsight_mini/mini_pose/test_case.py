"""Regression coverage for scene transforms, unsafe poses, and no contact."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from run_case import ROOT,load_mesh,validate_config,run,write,MiniOptical

class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/"outputs/mini_pose").mkdir(parents=True,exist_ok=True)
    def config(self):
        return {'case_name':'synthetic_no_contact_test','object_mesh':'unused.obj','scene_cloud':'unused.ply',
                'mm_per_world_unit':1.,'scale_source':'Synthetic numerical fixture in millimetres',
                'alignment_verified':True,'alignment_evidence':'Synthetic identity-frame fixture',
                'pads':[{'name':'test','origin_world':[0,0,0],'basis_columns_world':np.eye(3).tolist(),
                         'normal_toward_sensor_world':[0,0,1],'origin_definition':'undeformed_gel_rest_apex',
                         'pose_source':'Synthetic identity','orientation_source':'Synthetic identity axes'}]}
    def test_missing_pose_is_blocked(self):
        c=self.config();c['pads'][0]['origin_world']=None
        self.assertTrue(validate_config(c))
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/mini_pose',prefix='_test_missing_') as temp:
            path=Path(temp)/'config.json';write(path,c)
            with self.assertRaisesRegex(ValueError,'Prediction blocked'):
                run(path,Path(temp)/'result')
            self.assertFalse((Path(temp)/'result').exists())
    def test_left_handed_basis_rejected(self):
        c=self.config();c['pads'][0]['basis_columns_world'][0][0]=-1
        with self.assertRaises(ValueError):validate_config(c)
    def test_reversed_normal_rejected(self):
        c=self.config();c['pads'][0]['normal_toward_sensor_world']=[0,0,-1]
        with self.assertRaises(ValueError):validate_config(c)
    def test_glb_node_transform_is_applied(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/mini_pose',prefix='_test_nodes_') as temp:
            scene=trimesh.Scene();mesh=trimesh.creation.box(extents=[2,4,6])
            transform=np.eye(4);transform[:3,3]=[10,20,30]
            scene.add_geometry(mesh,transform=transform);path=Path(temp)/'transformed.glb';scene.export(path)
            np.testing.assert_allclose(load_mesh(path).bounds,mesh.bounds+[10,20,30],atol=1e-6)
    def test_no_contact_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/mini_pose',prefix='_test_empty_') as temp:
            base=Path(temp);mesh=trimesh.creation.icosphere(subdivisions=1,radius=1)
            mesh.apply_translation([0,0,-8]);mesh.export(base/'object.obj')
            trimesh.points.PointCloud(mesh.vertices).export(base/'cloud.ply')
            c=self.config();c['object_mesh']=str(base/'object.obj');c['scene_cloud']=str(base/'cloud.ply')
            write(base/'config.json',c);report=run(base/'config.json',base/'result')
            pad=report['pads'][0];self.assertTrue(pad['no_contact']);self.assertEqual(pad['camera_contact_pixels'],0)
            folder=base/'result/test';a=np.load(folder/'hydroshear_depth_delta_m.npy')
            self.assertEqual(np.count_nonzero(a),0)
            np.testing.assert_array_equal(np.asarray(Image.open(folder/'tactile_rgb.png')),MiniOptical(ROOT).render(np.zeros((320,240),np.float32))[0])
            self.assertEqual(np.load(folder/'contact_points_world.npy').shape,(0,3))
            self.assertTrue((folder/'contact_in_original_scene.ply').exists())
            with self.assertRaises(FileExistsError):run(base/'config.json',base/'result')
if __name__=='__main__':unittest.main(verbosity=2)
