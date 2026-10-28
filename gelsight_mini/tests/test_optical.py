from pathlib import Path
import sys,json,unittest,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mini_optical'))
from adapter import MiniOptical,calibration_bins,sha,SOURCE_SHA

class CalibrationTests(unittest.TestCase):
    def data(self,n=5):return {'bins':np.array(n),**{k:np.zeros((n,n,6)) for k in ['grad_r','grad_g','grad_b']}}
    def test_authoritative_bins(self):self.assertEqual(calibration_bins(self.data()),5)
    def test_inconsistent_bins_fail(self):
        a=self.data();a['bins']=4
        with self.assertRaises(ValueError):calibration_bins(a)
    def test_inconsistent_channels_fail(self):
        a=self.data();a['grad_g']=np.zeros((5,4,6))
        with self.assertRaises(ValueError):calibration_bins(a)
    def test_nonfinite_coefficients_fail(self):
        a=self.data();a['grad_r'][0,0,0]=np.nan
        with self.assertRaises(ValueError):calibration_bins(a)

class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        cls.r=MiniOptical(ROOT);cls.data=np.load(ROOT/'tests/synthetic_depth_m.npz',allow_pickle=False);cls.golden=json.loads((ROOT/'tests/golden.json').read_text());cls.zero=cls.data['zero']
    def test_bins_and_source_preserved(self):
        self.assertEqual(self.r.metadata['effective_bins'],125);self.assertEqual(self.r.metadata['original_config_bins'],120)
        self.assertAlmostEqual(self.r.renderer.x_binr,np.pi/2/124);self.assertAlmostEqual(self.r.renderer.y_binr,2*np.pi/124)
        self.assertEqual(sha(ROOT/'vendor/hydroshear/rl/tacsl_sensors/tactile_utils/gelsight_render.py'),SOURCE_SHA)
    def test_reject_invalid_inputs(self):
        for a in [np.zeros((240,320)),np.full((320,240),np.nan),np.full((320,240),-1.)]:
            with self.assertRaises(ValueError):self.r.render(a)
    def test_zero_and_subthreshold_baseline(self):
        np.testing.assert_array_equal(self.r.render(np.full_like(self.zero,5e-7)),self.r.render(self.zero))
    def test_approved_golden_and_input_unchanged(self):
        for name,item in self.golden['cases'].items():
            a=self.data[name].copy();original=a.copy();rgb=self.r.render(a)[0]
            self.assertEqual(hashlib.sha256(a.tobytes()).hexdigest(),item['depth_float32_bytes_sha256'])
            self.assertEqual(hashlib.sha256(rgb.tobytes()).hexdigest(),item['rgb_uint8_bytes_sha256']);np.testing.assert_array_equal(a,original)
    def test_single_batch_parity(self):
        a=self.data['asymmetric'];b=self.r.render(np.stack([self.zero,a]));np.testing.assert_array_equal(b[0],self.r.render(self.zero)[0]);np.testing.assert_array_equal(b[1],self.r.render(a)[0])
    def test_numpy_reference_interior(self):
        for name in ['asymmetric','boundary']:
            a=self.data[name];t=self.r.render(a)[0];n=np.clip(self.r.renderer.render(a),0,255).astype(np.uint8)
            self.assertLess(np.abs(t[3:-3,3:-3].astype(float)-n[3:-3,3:-3].astype(float)).mean(),.02)
    def test_lookup_ranges_and_dimensions(self):
        for name in self.data.files:
            a=self.data[name];d=self.r.gradient_diagnostics(a)
            for k in ['magnitude_index_range','direction_index_range']:self.assertGreaterEqual(d[k][0],0);self.assertLess(d[k][1],125)
            im=self.r.render(a)[0];self.assertEqual(im.shape,(320,240,3));self.assertEqual(im.dtype,np.uint8)
    def test_bin_correction_and_response(self):
        a=self.data['asymmetric'];rgb=self.r.render(a)[0];legacy=MiniOptical(ROOT,legacy=True).render(a)[0]
        self.assertTrue(np.any(rgb!=legacy));self.assertTrue(np.any(rgb!=self.r.render(self.zero)[0]))
if __name__=='__main__':unittest.main(verbosity=2)
