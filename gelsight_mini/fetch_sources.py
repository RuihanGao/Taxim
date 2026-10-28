"""Fetch immutable external Mini assets. Never reset an existing checkout."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
REV='f815b82fdf3451852acd918933020a82cede1f3b'
REPO=ROOT/'vendor/hydroshear'
def git(*args):return subprocess.check_output(['git',*map(str,args)],text=True).strip()
def main():
    if not REPO.exists():
        REPO.parent.mkdir(parents=True,exist_ok=True)
        git('clone','--filter=blob:none','--no-checkout','https://github.com/MMintLab/hydroshear.git',REPO)
        git('-C',REPO,'sparse-checkout','init','--cone')
        git('-C',REPO,'sparse-checkout','set','rl/tacsl_sensors','assets/tacsl/mesh/gs_mini_meshes')
        git('-C',REPO,'fetch','--depth=1','origin',REV)
        git('-C',REPO,'checkout','--detach',REV)
    if git('-C',REPO,'rev-parse','HEAD')!=REV:
        raise RuntimeError('Existing dependency differs from the pin. Preserve it and use a fresh checkout.')
    required=['rl/tacsl_sensors/tactile_utils/gelsight_render.py','rl/tacsl_sensors/tactile_utils/gs_mini_data/polycalib.npz','rl/tacsl_sensors/tactile_utils/gs_mini_data/mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png','rl/tacsl_sensors/configs/gs_mini.yaml','assets/tacsl/mesh/gs_mini_meshes/gsmini_elastomer_transformed.obj','rl/tacsl_sensors/examples/assets/urdf/gelsight_description/robots/gsmini_finger.urdf']
    missing=[p for p in required if not (REPO/p).is_file()]
    if missing:raise RuntimeError('Existing sparse checkout is missing required assets: '+repr(missing))
    print('Pinned external Mini renderer, calibration and geometry available.')
if __name__=='__main__':main()
