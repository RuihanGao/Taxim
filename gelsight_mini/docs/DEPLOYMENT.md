# Deployment handoff: what changed for GelSight Mini

This branch preserves the approved backend: Hydroshear's `gelsightRender.render_tensorized`, using its Taxim example-based optical model. It is an additive integration in `gelsight_mini/`, not a rewrite of the legacy Taxim renderer. Baymax's `mini_optical/adapter.py`, `mini_depth/profile.py`, `mini_depth/geometry.py` and `mini_pose/run_case.py` are copied without numerical changes.

## Changes and preserved behavior

| Area | Integration behavior |
|---|---|
| Backend selection | Explicitly load `gs_mini` from pinned Hydroshear revision `f815b82fdf3451852acd918933020a82cede1f3b`. Invoke `render_tensorized`; CUDA/Isaac Gym are not required. |
| Calibration-bin correction | Upstream Mini config declares 120 bins. Read `polycalib.npz`, validate scalar `bins=125` and finite `grad_r/grad_g/grad_b` arrays each `(125,125,6)`, and set `numBins=125` **before** renderer construction. Angular spacing uses denominators 124, not 119. No coefficients are modified. |
| Source checks | Require the pinned Git revision, renderer SHA-256 `ea7db8c81791f5915b7b92fe20b3eff2ec36617f0979d763ae8dabeb7272969c`, and calibration SHA-256 `34937ac09579b259fd344f99fbc71ebb41a6ad8bbe5f75118d016ada8fbb044a`. Record background and geometry hashes. |
| Mini background | Preserve `mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png`. The renderer does not use the config's `real_bg.npy` path; no replacement is invented. Both pads share the supplied surrogate calibration unless separately calibrated profiles are explicitly introduced. |
| Input contract | Accept finite nonnegative `(320,240)` or `(N,320,240)` camera-depth differences in metres. Convert to float32, copy inputs so upstream mutation cannot change the caller's array, and reject bad shape/NaN/negative values. The CLI accepts one map. |
| Sign and units | Input is `(nominal_camera_depth_mm - deformed_camera_depth_mm)/1000`. Do not pre-negate it. Upstream applies its own negative sign and metre-to-mm conversion once. Normal indentation and absolute depth are not optical inputs. |
| Reproducibility | CPU, four Torch threads, deterministic algorithms, inference mode; unchanged upstream blur, derivative, angular lookup, threshold, clipping and uint8 casting. Save RGB, not OpenCV BGR. |
| Physical surface | Load the actual primary curved Mini surface through `profile.py`, with the URDF elastomer-tip transform and metre-to-mm conversion. Use 20.75 x 25.25 mm physical extent, not the larger body flange, a flat patch or duplicated offset surface. |
| Perspective camera | Preserve runtime_horizontal_fov, width 240, height 320, fx=fy=308.8192771084337 px, principal point (120,160), distance 26.7 mm. Optical `pixmm=0.065` is a slope-conversion parameter, not a replacement camera model. |
| Geometry adapter | Apply all GLB scene-node transforms; transform the entire object to each frozen rest frame using one common scale. Rasterize normal and camera views independently. Never crop the object to contact clouds. |
| Depth/contact policy | Normal map is `max(z_object-z_rest,0)` on valid intersecting support. Camera deformed depth is the nearer rest/object depth where both exist; elsewhere retain rest. Optical delta is zero outside valid support. Preserve genuine no contact. No first-contact reset, approach sweep or ROI-rim reference. |
| Output validation | Check right-handed orthonormal bases and inward direction, fixed poses/mesh, finite nonnegative float32 metres, RGB replay, contact points on finite original mesh triangles, input hashes and saved image decoding. Report out-of-view and excessive overlap rather than modifying geometry to hide them. |
| Portable packaging | Add external-source fetcher including optical **and geometry** asset paths, single-map CLI, explicit unresolved pose template, pinned dependencies, synthetic golden fixtures, and tests that do not require private cube/avocado data. |

## Unchanged upstream behavior

The tensorized renderer suppresses depth changes below 1 micrometre, converts the positive input to its internal signed slope units, applies its upstream 5x5 filtering, computes normal magnitude/direction, evaluates the angular/spatial polynomial calibration and clips/casts RGB. The adapter does not add material texture, marker motion, force prediction or elastic deformation. The NumPy reference and tensorized paths have different boundary handling; tests compare their interior rather than silently switching renderers.

`legacy=True` exists solely for the 120-bin comparison test. Production entry points always use the corrected 125-bin mode. Switching calibration requires an explicit profile/source-check update and fresh validation, not disabling checks.

## CAD-registration stage versus optical deployment

The optical server can start with a valid camera-depth-difference map; it does not need CAD software. Full depth estimation additionally needs object geometry and reliable rest poses/scale. Our selected UMI workflow registers the **full assembly**, one common body transform/scale and synchronized CAD-X finger translation. Markers and Mini mounts move with each finger. Rest apices are obtained from both actual Mini instances after assembly transforms; generic SHELL/COMPOUND names must not be filtered out.

This stage requires scene-specific correspondences. It is not a general automatic STEP-to-point-cloud solver. Tag family/IDs, print dimensions, camera mapping, gripper articulation, metric scale and sensor image orientation need checking for each case. The new-chat prompt points to the case preparation implementation on Baymax and requires those values to be generalized, not copied from avocado.

The CAD defines the undeformed shape and mount offsets, but registration makes its world pose **estimated**, not exact measured ground truth. The avocado fit still produced incompatible object intersections despite a close image overlay. That case demonstrates the need to report 3D residuals and relative mesh/gripper uncertainty; it is not a tactile accuracy benchmark.

## New-server commands

```bash
git clone --branch gelsight-mini https://github.com/RuihanGao/Taxim.git
cd Taxim
python3.11 -m venv .mini-env
source .mini-env/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r gelsight_mini/requirements.txt
python gelsight_mini/fetch_sources.py
python -m unittest discover -s gelsight_mini/tests -p test_optical.py -v
python -m unittest discover -s gelsight_mini/mini_pose -p test_case.py -v
python gelsight_mini/render_depth.py \
  --input /path/camera_depth_delta_m.npy --output /path/new_output/tactile_rgb.png
```

The fetcher refuses an existing checkout with a different revision; it does not reset local work. For a missing asset in an existing sparse checkout, inspect that checkout before expanding it. A fresh fetch includes `rl/tacsl_sensors` and `assets/tacsl/mesh/gs_mini_meshes`.

Baymax uses Python 3.11 with Torch 2.5.1+cu121 but renders on CPU. The installation example uses the CPU wheel. `environment.baymax.json` records all tested versions; CPU wheel/new-server portability must be checked with the included tests. No fresh pip environment was installed on Baymax, and its working `.env`, `.cad-env` and `.taxim-env` were not modified.

## Tests and transfer validation

The portable tests use synthetic depth inputs generated independently of any user scene. `tests/golden.json` stores SHA-256 of the exact uint8 RGB array bytes from the approved Baymax adapter. It does not redistribute the background or generated RGB images. Tests cover calibration consistency, malformed inputs, zero/subthreshold depth, input preservation, single/batch parity, angular bounds, bin-correction response and NumPy interior agreement. Geometry tests cover GLB transforms, missing/reversed/left-handed frames, true no contact, and existing-output protection.

The port is also replayed against the approved fixed-pose cube on Baymax, using its original private inputs without publishing them. Both native PNG hashes and all five depth arrays must match. See `VALIDATION.json` for the actual validation performed. Passing these tests establishes numerical consistency; physical validation still requires real tactile measurements and reliable metric geometry.
