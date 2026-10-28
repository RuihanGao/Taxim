"""Invoke pinned upstream Mini optical renderer with data-authoritative bin count.

No upstream implementation is vendored or rewritten here. Input is positive
nominal-minus-current camera depth in metres; source performs sign conversion.
"""
from pathlib import Path
import importlib.util,hashlib,subprocess
import numpy as np
import torch

REV='f815b82fdf3451852acd918933020a82cede1f3b'
SOURCE_SHA='ea7db8c81791f5915b7b92fe20b3eff2ec36617f0979d763ae8dabeb7272969c'
CALIB_SHA='34937ac09579b259fd344f99fbc71ebb41a6ad8bbe5f75118d016ada8fbb044a'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def calibration_bins(data):
    value=np.asarray(data['bins'])
    if value.ndim!=0 or not np.isfinite(value) or float(value)!=int(value) or int(value)<2:raise ValueError('Calibration bins must be an integer scalar >=2')
    n=int(value)
    for c in ('grad_r','grad_g','grad_b'):
        a=np.asarray(data[c])
        if a.shape!=(n,n,6) or not np.isfinite(a).all():raise ValueError(f'Invalid {c} coefficient array: expected {(n,n,6)} finite values')
    return n

class MiniOptical:
    def __init__(self,root,legacy=False,device='cpu'):
        root=Path(root);repo=root/'vendor/hydroshear'
        if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=REV:raise ValueError('Unexpected Hydroshear revision')
        source=repo/'rl/tacsl_sensors/tactile_utils/gelsight_render.py'
        if sha(source)!=SOURCE_SHA:raise ValueError('Upstream renderer differs from the audited source')
        spec=importlib.util.spec_from_file_location('hydroshear_mini_optical',source);self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        config=dict(self.module.conf_options['gs_mini'])
        if sha(config['calib_path'])!=CALIB_SHA:raise ValueError('Mini calibration differs from the audited source')
        with np.load(config['calib_path'],allow_pickle=False) as data:n=calibration_bins(data)
        original=int(config['numBins']);config['numBins']=original if legacy else n
        self.module.conf_options['gs_mini']=config
        self.renderer=self.module.gelsightRender('gs_mini',device=device)
        self.config=config;self.device=device;self.bin_count=n
        self.metadata={'hydroshear_revision':REV,'upstream_source_sha256':sha(source),'calibration_sha256':sha(config['calib_path']),'background_path':str(config['background_path']),'background_sha256':sha(config['background_path']),'original_config_bins':original,'calibration_bins':n,'effective_bins':int(config['numBins']),'coefficient_shape':[n,n,6],'legacy_diagnostic_only':legacy,'height':config['h'],'width':config['w'],'optical_mm_per_pixel':config['pixmm'],'magnitude_bin_step_rad':float(self.renderer.x_binr),'direction_bin_step_rad':float(self.renderer.y_binr),'renderer_method':'gelsightRender.render_tensorized','device':device,'channel_order':'RGB','input_units':'metres; positive nominal depth minus current depth','output':'upstream-clipped uint8 RGB; no display contrast normalization'}
        if self.renderer.background.shape!=(config['h'],config['w'],3):raise ValueError('Background shape is inconsistent')
    def render(self,depth_delta_m):
        a=np.asarray(depth_delta_m,dtype=np.float32)
        if a.ndim==2:a=a[None]
        if a.ndim!=3 or a.shape[1:]!=(self.config['h'],self.config['w']):raise ValueError('Expected (N,320,240) or (320,240) depth change in metres')
        if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Depth change must be finite and nonnegative; mask invalid pixels to zero upstream')
        with torch.inference_mode():result=self.renderer.render_tensorized(torch.from_numpy(a.copy()).to(self.device))
        return result.cpu().numpy()
    def gradient_diagnostics(self,depth_delta_m):
        # Observe the upstream pre-lookup stages without changing its renderer.
        h=torch.as_tensor(np.asarray(depth_delta_m,dtype=np.float32),device=self.device).clone()
        if h.ndim==2:h=h[None]
        h[h.abs()<1e-6]=0;h=h*(-1000/self.config['pixmm'])
        filtered=self.module.gaussian_filtering(h.unsqueeze(-1),self.renderer.kernel).squeeze(-1)
        mag,direction,_=self.module.generate_normals_tensor(filtered)
        ix=torch.floor(mag/self.renderer.x_binr).long();iy=torch.floor((direction+np.pi)/self.renderer.y_binr).long()
        if ix.min()<0 or iy.min()<0 or ix.max()>=self.bin_count or iy.max()>=self.bin_count:raise ValueError('Angular lookup outside calibration table')
        return {'magnitude_index_range':[int(ix.min()),int(ix.max())],'direction_index_range':[int(iy.min()),int(iy.max())],'max_slope_angle_degrees':float(torch.rad2deg(mag).max())}
