# GelSight Mini: fixed geometry and tensorized Taxim optics

This optional module reproduces the approved GelSight Mini path from the VTCotraining experiments. It adds a checked adapter around the **pinned Hydroshear tensorized Taxim implementation**. It does not replace Taxim's original optical renderer or modify its other sensor modes.

- [Complete standalone CAD registration workflow](docs/STANDALONE.md)
- [Deployment instructions and full change list](docs/DEPLOYMENT.md)
- [Copyable prompt for a new object/scene](docs/NEW_CHAT_PROMPT.md)
- [Frozen-pose configuration template](case.template.json)
- [Recorded numerical validation](VALIDATION.json)

The workflow is: register the full articulated gripper CAD to scene features → derive both undeformed gel rest frames from their CAD mounts → freeze poses and a shared scale → rasterize the full object and curved gel → pass positive camera-depth difference in metres to Taxim → save native RGB and diagnostics. Registration remains an estimate; numerical replay does not prove physical tactile accuracy.

## Quick start on another Linux server

Use Python 3.11 and a separate environment. These commands do not require CUDA, Isaac Gym or a STEP importer for optical/depth replay.

```bash
git clone --branch gelsight-mini https://github.com/RuihanGao/Taxim.git
cd Taxim
python3.11 -m venv .mini-env
source .mini-env/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r gelsight_mini/requirements.txt
python gelsight_mini/fetch_sources.py
python -m unittest discover -s gelsight_mini/tests -p 'test_*.py' -v
python -m unittest discover -s gelsight_mini/mini_pose -p test_case.py -v
```

`requirements.txt` here is for this module, not Taxim's legacy root requirements. The tested Baymax environment is recorded in `environment.baymax.json`. A fresh external dependency checkout and all tests were verified on Baymax's existing Python environment; installation into a newly created environment on another OS/hardware has not been tested. Run the golden tests there before claiming byte-identical replay.

## Render a depth map already computed

```bash
python gelsight_mini/render_depth.py \
  --input /absolute/path/camera_depth_delta_m.npy \
  --output /absolute/path/new_result/tactile_rgb.png
```

Input: one `(320,240)` finite, nonnegative map in **metres**, representing nominal minus deformed camera depth. Invalid gel/object support must be zero. Output: native 240x320 **RGB** PNG and JSON provenance. Do not use absolute depth, millimetres, normal-grid penetration, or a pre-negated map. The wrapper preserves existing outputs.

## Register CAD and compute depth end to end

The standalone registration stage is now included. Start with a synthetic case requiring no private inputs:

```bash
python gelsight_mini/make_demo.py --output demo_case
python gelsight_mini/run_pipeline.py --config demo_case/case.json --output demo_result
```

For real inputs, use `pipeline.template.json`, the matching rig definition and scene annotations or paired CAD/scene landmarks. Run:

```bash
python gelsight_mini/run_pipeline.py --config /path/case.json --output /path/new_result
```

See [STANDALONE.md](docs/STANDALONE.md) for organized-image and unorganized-cloud workflows, STEP import, input schemas and limits. Registration estimates common scale/body pose/opening, derives both actual CAD rest-gel frames and then calls the unchanged numerical depth/optical runner. No private paths or prior fitted scale/pose are required.

## Compute depth from known fixed rest poses

Copy `case.template.json` and fill in the real case paths, independently supported common scale, both rest-apex origins, right-handed sensor basis columns, mounting/orientation sources and alignment evidence. Use absolute data paths to avoid ambiguity. Empty/null fields deliberately fail validation.

```bash
python gelsight_mini/mini_pose/run_case.py --config /path/case.json --check-config
python gelsight_mini/mini_pose/run_case.py \
  --config /path/case.json --output /path/new_case_result
```

`run_case.py` consumes already frozen poses. `run_pipeline.py` adds supervised registration from declared CAD/scene correspondences. It is not an arbitrary-CAD recognition system: use the matching rig geometry and annotate identifiable features in each new scene. Supported articulation is a scalar prismatic opening; do not reuse avocado pixels/camera parameters for another capture.

Each pad result includes normal indentation in mm; nominal/deformed camera depths in mm; camera depth difference in mm; float32 renderer input in metres; gel/contact masks; `tactile_rgb.png`; contact points on the full object transformed back to the original world; and pose/scale/numerical diagnostics. The runner also creates a common-scale comparison figure and a scene GLB. The normal-grid footprint is a sampled projected area, not measured pressure or curved contact area.

Native frame: x=image right, y=image up, z=into sensor. Rest apex `[0,0,0]`; camera `[0,0,26.7]` mm, looking along -z. With basis columns B, `local_mm = (world-origin) @ B * mm_per_world_unit`. Native rendering remains 240x320; physical orientation/display rotations belong in explicit case metadata.

## Source and data scope

Hydroshear is fetched at `f815b82fdf3451852acd918933020a82cede1f3b`; renderer/calibration hashes are checked. External source, calibration, backgrounds, and meshes are not copied into this branch. Their original notices and terms remain with the fetched dependency. The original Taxim files and licensing remain unchanged. This module includes only our integration/helpers, synthetic test inputs and documentation; no user photographs, object scenes, gripper STEP files or per-scene results are published.
