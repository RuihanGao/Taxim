import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from discover_correspondences import recover_camera,circular_candidates
from registration.common import project

class DiscoveryTests(unittest.TestCase):
    def scene(self):
        u,v=np.meshgrid(np.arange(192),np.arange(256));z=1+.2*np.sin(u/13)*np.cos(v/17)
        xyz=np.stack([(u-95.5)*z/220,(v-127.5)*z/220,z],axis=-1).reshape(-1,3)
        R=Rotation.from_rotvec([.1,-.2,.3]).as_matrix();return xyz@R.T+np.array([.2,.3,-.1]),np.c_[u.ravel(),v.ravel()]
    def test_camera_and_grid_recovered_from_withheld_rays(self):
        xyz,uv=self.scene();camera,audit=recover_camera(xyz);pred,z=project(xyz,camera)
        self.assertEqual((camera['width'],camera['height']),(192,256));self.assertLess(np.max(np.abs(pred-uv)),1e-6);self.assertTrue(np.all(z>0));self.assertGreater(audit['withheld_points'],0)
    def test_unordered_cloud_rejected(self):
        xyz,_=self.scene();xyz=np.random.default_rng(9).permutation(xyz)
        with self.assertRaisesRegex(ValueError,'No verified'):recover_camera(xyz)
    def test_predicted_position_is_not_an_observation(self):
        self.assertEqual(circular_candidates(np.zeros((100,100),np.uint8),[50,50],6),[])
    def test_observed_circle_displaced_from_prediction(self):
        gray=np.zeros((100,100),np.uint8);cv2.circle(gray,(47,54),6,220,-1)
        candidates=circular_candidates(gray,[50,50],6);self.assertTrue(candidates)
        self.assertLess(np.linalg.norm(np.array(candidates[0]['pixel'])-[47,54]),1)
        self.assertGreater(np.linalg.norm(np.array(candidates[0]['pixel'])-[50,50]),3)
if __name__=='__main__':unittest.main()
