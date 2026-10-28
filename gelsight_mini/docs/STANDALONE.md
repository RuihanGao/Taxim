# Standalone CAD-to-depth-to-Taxim pipeline

Everything required by the code is in this repository or fetched from the pinned public Hydroshear dependency. No access to the original experiment server, laptop, historical fit, previous scale estimate or previous contact outputs is required. Input paths are resolved relative to your case JSON. A private shareable avocado case bundle additionally includes that case's original mesh, cloud, full STEP, fresh CAD export and case configuration; those data are not published in this repository.

## What must be provided

1. An aligned object mesh and scene point cloud, with documented common coordinates.
2. The full gripper assembly STEP, **or** an exported CAD manifest plus the mounting/articulation definition. A standalone Mini gel CAD alone defines its undeformed shape but cannot determine the finger mounting offsets or its pose in a scene.
3. A rig definition: moving groups, opening axis/model, CAD landmark coordinates, both actual gel part indices/apex definitions and canonical sensor bases. `rigs/umi_gs_mini.json` supplies this for the exact UMI STEP SHA recorded in that file. It is reusable for other objects with the same assembly. A different CAD revision, tag print layout or mechanism must be checked and adapted.
4. Observable matching scene features. Start with automatic tag/screw discovery for the supported UMI rig and an ordered RGB cloud; the agent reviews detections. Reviewed image annotations or explicit CAD-feature/scene-point pairs are alternatives. See [CORRESPONDENCES.md](CORRESPONDENCES.md) for discovery, camera recovery, and fallbacks. No prior scene pose/scale or manual clicks are required by the supported automatic path.

The current articulation implementation supports a single scalar prismatic opening with explicitly specified group translations. It rejects rotary/linkage mechanisms. The UMI profile translates left/right fingers by +/-q/2 along CAD X; markers, shells and gels follow their own finger. Hidden gear/linkage angles are not solved or drawn as if fixed.

## Install and check a clean server

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

This module's requirements are separate from the legacy Taxim root requirements. CPU is sufficient. The fetcher gets renderer, calibration, background and physical geometry at the pinned revision. Internet access is needed for installation/fetching; this is not an offline dependency bundle. Original dependency notices/terms remain applicable.

CAD import is optional when a CAD export is supplied. For a new STEP, use a separate CAD environment so OCP's native libraries do not affect optical rendering:

```bash
python3.11 -m venv .cad-env
.cad-env/bin/python -m pip install -r gelsight_mini/requirements-cad.txt
.cad-env/bin/python gelsight_mini/export_step.py \
  --input /path/assembled_umi.step --output /path/case/cad
```

The exporter traverses **all** leaf instances, including SHELL/COMPOUND, applying assembly transforms and storing mm geometry. It records source/mesh hashes, primitive face information and bounds in `cad/assembly.json`. In a case, set `cad_manifest` to that file. Alternatively omit `cad_manifest`, specify `cad_step`, and pass `--cad-python /path/.cad-env/bin/python` to the pipeline for automatic isolated import.

## Start with the public synthetic example

```bash
python gelsight_mini/make_demo.py --output demo_case
python gelsight_mini/run_pipeline.py --config demo_case/case.json --output demo_result
```

This constructs a fully specified synthetic moving-gripper CAD, cloud, object and 3D feature pairs. The true scale is 1000 mm/world unit, opening change -2 mm and nominal sphere overlap 0.4 mm per pad. Rasterized faceted geometry produces about 0.397 mm. This is a numerical example, not a measured tactile sample. All filenames in the generated case are relative; move the directory elsewhere and run it again to check portability.

## Prepare a real case

For automatic discovery, copy `pipeline.auto.template.json` to your case directory, set its input paths and matching rig, and record alignment evidence. Run the same `run_pipeline.py` command below. It discovers the source camera/grid when supported, detects tags and visible screw contours, prepares landmarks, and runs registration/depth/optics. Review `correspondences/feature_crops.png`, `correspondence_candidates.png` and `discovery.json` before accepting associations. If discovery stops, refine actual observations from the saved evidence or use the alternatives below. It never substitutes CAD predictions for observed pixels.

For a separate discovery-and-review pass:

```bash
python gelsight_mini/discover_correspondences.py --scene my_case/inputs/scene.ply --rig my_case/rig.json --output discovery
```

Then set `annotations` to the reviewed `discovery/annotations.json` (relative to the case JSON) instead of enabling automatic discovery. See [NEW_CHAT_PROMPT.md](NEW_CHAT_PROMPT.md) for an agent-driven workflow covering all five stages.

### Explicit annotations alternative

Copy `pipeline.template.json` to `my_case/case.json` and `annotations.template.json` to `my_case/annotations.json`. Put your data under `my_case/inputs/`, and copy the matching rig definition to `my_case/rig.json`. Verify the object/cloud alignment and set `alignment_verified=true` with the evidence. The pipeline never changes object alignment to create contact.

### Organized RGB cloud plus source camera

In annotations, provide `organized_cloud: {"width": W, "height": H}` only if the PLY vertex order is actually row-major image order. Provide the camera:

```json
{
  "width": 756, "height": 1008,
  "fx": 757.1657678474378, "fy": 757.1657678474378,
  "cx": 377.5, "cy": 503.5,
  "world_to_camera": [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
}
```

These numbers demonstrate the avocado source convention; **replace them for another capture**. Pixel coordinates use zero-based integer pixel centres. Camera axes follow OpenCV: x right, y down, z forward; matrix translation is in the same units as the cloud. This source camera is separate from the fixed internal Mini camera. The preparation code checks cloud-to-pixel reprojection before reshaping/using the mapping; it rejects mismatched dimensions or conventions. Optional `source_image` is relative to annotations.json; otherwise cloud RGB reconstructs it.

Specify `detect_tag_ids` from the matching rig (7 and 4 only for our unchanged UMI print layout). Detection uses the declared OpenCV tag dictionary, retains **decoded marker corner order**, and uses the matching CAD corners in that order. A rotated image does not justify reordering by image top-left. Corner 3D points are ray intersections with robust planes fitted to eroded rigid printed-tag interiors. No gray gel/contact fitting is used.

Annotate visible screws/features, for example:

```json
{"name": "cover_left", "source_pixel": [172, 504], "sigma_pixel": 2.0}
```

Feature names refer to CAD coordinates and rigid groups in rig.json. The pixel above is only an example; annotate your new capture. Three non-collinear fixed-body features initialize the common similarity. At least three correspondences per moving pad group constrain opening and pose; use more well-spread features where possible.

### Unorganized cloud or missing source image

Leave camera/grid unset and use explicit scene coordinates or point indices:

```json
{"name": "cover_left", "scene_point_index": 12345}
```

or

```json
{"name": "cover_left", "world": [-0.1, 0.2, -1.3]}
```

These must identify the corresponding physical CAD feature in your actual cloud. The numbers here are illustrative, not usable annotations. This path requires no image, known camera or ordered vertices. You can alternatively supply a ready `landmarks.json` containing records with `name`, `group`, `cad_mm`, and `world`, plus optional `source_pixel`/`sigma_pixel`; set `landmarks` instead of `annotations` in case.json. A standalone surface with no observable corresponding features cannot establish a unique world rest pose.

## Run the whole case

```bash
python gelsight_mini/run_pipeline.py \
  --config my_case/case.json --output my_case_result
```

If STEP extraction is needed:

```bash
python gelsight_mini/run_pipeline.py \
  --config my_case/case.json --output my_case_result \
  --cad-python /path/to/.cad-env/bin/python
```

The pipeline discovers correspondences when `auto_correspondences` is enabled or prepares supplied `annotations`; fits a common body rotation/translation, shared metric scale and scalar opening; checks both actual CAD gel mounts against the pinned Mini surface; freezes the rest frames; and calls the unchanged full-mesh depth/optical runner. There is no historical initialization file. Body features provide the initialization; optional scale bounds must contain it. `initial_body_landmarks` may select a stable subset of fixed-body landmarks, never a prior fitted pose.

Registration uses robust soft-L1 residuals. Defaults are 3 mm per 3D coordinate, 1 px for detected corners and 2 px for manual features. These are fit weights, not claimed accuracy. Opening bounds belong to the rig. The 2 mm default diagnostic RMS threshold is a review threshold, not proof that smaller residuals guarantee tactile accuracy. The optimizer checks Jacobian rank and flags bound hits.

## Outputs and review

- `registration.json`: scale/opening/body transform, landmark and group residuals, image RMS where available, rank/bound diagnostics.
- `mounting_audit.json`: both apex definitions and pinned-surface-to-CAD discrepancies.
- `rest_pose_config.json`: frozen rest origins and right-handed basis columns. Runtime absolute paths describe the current machine; after moving inputs, rerun the portable case JSON to regenerate them.
- `cad_alignment_and_rest_gels.png`: two source-camera panels showing articulated CAD and actual rest gels; explicitly labelled 3D overview if a source image/mapping is unavailable.
- `part_transforms.json`, `registered_scene.glb`, `scene_overlay.png`: registered rigid groups, actual rest surfaces, unchanged object and a cloud sample.
- `prediction/<pad>/`: normal indentation mm; nominal/deformed camera depths mm; camera delta mm; float32 renderer delta metres; support/contact masks; native RGB; contacts back in original scene; per-pad diagnostics.
- `prediction/fixed_pose_estimate.png`: both depth maps and Taxim outputs on comparable colour scales.
- `report.json`: input hashes, warning/status summary and numerical peaks/footprints. `README.md` explains the result.

The original object and cloud are never resized or moved. No first-contact reset, approach offset, rim subtraction or contact-driven pose adjustment is applied. Native Mini image shape remains (320,240), and camera deltas are divided by 1000 exactly once before rendering. The 125-bin fix and pinned source checks remain unchanged.

Registration, reconstruction and mounting uncertainty can exceed the predicted indentation. Large overlap, missing contact or a good image overlay with poor 3D residuals must remain visible. Our avocado replay is intentionally retained as a geometry-inconsistency case, not a validated tactile prediction. Marker motion, elasticity, pressure and force are outside this optical pipeline.
