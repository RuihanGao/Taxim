"""Render a native Mini camera-depth difference, in metres, as RGB."""
from pathlib import Path
import argparse,json,sys
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'mini_optical'))
from adapter import MiniOptical,sha
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError('Choose a new output PNG; previous results are preserved.')
    delta=np.load(a.input,allow_pickle=False)
    if delta.shape!=(320,240):raise ValueError('One native (320,240) camera-depth-difference map in metres is required.')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    optical=MiniOptical(ROOT,device='cpu');rgb=optical.render(delta)[0]
    a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(a.output)
    with Image.open(a.output) as image:np.testing.assert_array_equal(np.array(image),rgb)
    report={'input_sha256':sha(a.input),'output_sha256':sha(a.output),'input_units':'metres, nominal minus deformed camera depth','input_shape':list(delta.shape),'max_depth_delta_m':float(np.max(delta)),'deterministic_cpu_threads':4,'optical':optical.metadata,'diagnostics':optical.gradient_diagnostics(delta),'physical_accuracy_validated':False}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
