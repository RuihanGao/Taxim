from pathlib import Path
import sys,tempfile,unittest,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from registration.fit import fit_registration
from registration.common import read,write,validate_rig
from make_demo import make_demo
from run_pipeline import run_pipeline
from prepare_landmarks import prepare

class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.case=make_demo(self.base/'case');self.rig=read(self.case/'rig.json');self.landmarks=read(self.case/'landmarks.json');self.truth=read(self.case/'truth.json')
    def tearDown(self):self.temp.cleanup()
    def test_recovers_scale_body_and_opening(self):
        fit=fit_registration(self.rig,self.landmarks)
        self.assertAlmostEqual(fit['scale_mm_per_world'],1000,places=6);self.assertAlmostEqual(fit['opening_change_mm'],-2,places=7)
        np.testing.assert_allclose(fit['rotation_columns'],self.truth['rotation_columns'],atol=1e-9)
        np.testing.assert_allclose(fit['translation_mm'],self.truth['translation_mm'],atol=1e-7)
    def test_missing_fixed_body_is_rejected(self):
        self.landmarks['records']=[r for r in self.landmarks['records'] if r['group']!='body']
        with self.assertRaisesRegex(ValueError,'fixed-body'):fit_registration(self.rig,self.landmarks)
    def test_degenerate_body_is_rejected(self):
        for r in self.landmarks['records']:
            if r['group']=='body':r['cad_mm']=[0,0,0]
        with self.assertRaisesRegex(ValueError,'non-collinear'):fit_registration(self.rig,self.landmarks)
    def test_unknown_articulation_is_rejected(self):
        self.rig['kinematics']['type']='rotary'
        with self.assertRaisesRegex(ValueError,'scalar-prismatic'):validate_rig(self.rig)
    def test_left_handed_mount_is_rejected(self):
        self.rig['pads'][0]['basis_columns_cad'][2][0]=-1
        with self.assertRaisesRegex(ValueError,'right handed'):validate_rig(self.rig)
    def test_unorganized_cloud_point_indices(self):
        import trimesh
        cloud=trimesh.load(self.case/'scene.ply',process=False);n=len(self.landmarks['records']);self.rig['features']=[];annotations={'features':[]}
        for i,r in enumerate(self.landmarks['records']):
            self.rig['features'].append({k:r[k] for k in ['name','group','cad_mm']});annotations['features'].append({'name':r['name'],'scene_point_index':len(cloud.vertices)-n+i})
        write(self.case/'rig.json',self.rig);write(self.case/'annotations.json',annotations)
        result=prepare(self.case/'scene.ply',self.case/'rig.json',self.case/'annotations.json',self.base/'prepared')
        fit=fit_registration(self.rig,result);self.assertAlmostEqual(fit['opening_change_mm'],-2,places=4)
    def test_wrong_landmark_rig_hash_rejected(self):
        self.landmarks['rig_sha256']='wrong-rig';write(self.case/'landmarks.json',self.landmarks)
        with self.assertRaisesRegex(ValueError,'different rig'):run_pipeline(self.case/'case.json',self.base/'wrong_rig_result')
    def test_rotated_tag_retains_decoded_corner_order(self):
        import cv2,trimesh
        h=w=200;image=np.full((h,w),255,np.uint8)
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),7,100)
        image[50:150,50:150]=np.rot90(marker,2);yy,xx=np.mgrid[:h,:w]
        xyz=np.c_[(xx.ravel()-100)/100,(yy.ravel()-100)/100,np.ones(h*w)]
        trimesh.points.PointCloud(xyz,colors=np.repeat(image.reshape(-1,1),3,axis=1)).export(self.case/'tag_scene.ply')
        cad=[[0,0,0],[10,0,0],[10,10,0],[0,10,0]]
        self.rig['tags']=[dict(id=7,family='DICT_APRILTAG_36h11',group='left',cad_corners_detector_order_mm=cad)]
        write(self.case/'rig.json',self.rig)
        ann=dict(organized_cloud={'width':w,'height':h},camera=dict(width=w,height=h,fx=100,fy=100,cx=100,cy=100,world_to_camera=np.eye(4).tolist()),detect_tag_ids=[7])
        write(self.case/'tag_annotations.json',ann)
        result=prepare(self.case/'tag_scene.ply',self.case/'rig.json',self.case/'tag_annotations.json',self.base/'tag_prepared')
        first=result['records'][0];self.assertGreater(first['source_pixel'][0],140);self.assertGreater(first['source_pixel'][1],140);self.assertEqual(first['cad_mm'],cad[0])
        self.assertAlmostEqual(first['world'][2],1,places=8)
    def test_portable_end_to_end_and_existing_output_protection(self):
        result=run_pipeline(self.case/'case.json',self.base/'result')
        self.assertFalse(result['physical_accuracy_validated']);self.assertTrue(result['object_unchanged'])
        for pad in result['pads']:
            self.assertGreater(pad['normal_peak_indentation_mm'],.38);self.assertLess(pad['normal_peak_indentation_mm'],.41);self.assertFalse(pad['no_contact'])
            a=np.load(self.base/'result/prediction'/pad['name']/'hydroshear_depth_delta_m.npy');self.assertEqual(a.shape,(320,240));self.assertTrue(np.isfinite(a).all() and np.all(a>=0))
        with self.assertRaises(FileExistsError):run_pipeline(self.case/'case.json',self.base/'result')
if __name__=='__main__':unittest.main(verbosity=2)
