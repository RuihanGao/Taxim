Reproduce our CAD-registration → undeformed GelSight Mini rest poses → mesh depth → Taxim optical pipeline for a new object and scene.

INPUTS (replace placeholders)
- Case name: <CASE_NAME>
- Aligned full object mesh: <OBJECT_MESH_PATH>
- Scene RGB point cloud: <SCENE_POINT_CLOUD_PATH>
- Full gripper assembly CAD: <ASSEMBLY_STEP_PATH>; for the same UMI gripper, use assembled_umi.step, not the single Schunk adapter.
- Original source image and image-to-cloud correspondence/camera metadata: <PATHS_OR_UNKNOWN>
- Optional previous estimated contact points/annotations: <PATHS_OR_NONE>
- Known tag family/IDs/printed dimensions, metric reference, jaw opening, pose metadata: <VALUES_OR_UNKNOWN>
- Sensor orientation reference: <PATH_OR_SAME_UMI_SETUP>

Execute experiments on baymax.ri.cmu.edu. Develop code under /ssd/ruihan/code/VTCotraining and use /ssd/ruihan/code/VTCotraining/.env/bin/python. Preserve existing code, inputs, outputs and environments. Do not reinstall or upgrade dependencies unnecessarily. The approved tensorized optical path does not use .taxim-env. The portable implementation is on https://github.com/RuihanGao/Taxim/tree/gelsight-mini/gelsight_mini; consult its deployment documentation if working on another server.

1. RECOVER THE APPROVED IMPLEMENTATION
Read these paths relative to /ssd/ruihan/code/VTCotraining:
- outputs/active_prediction.json
- mini_pose/README.md and mini_pose/CASE_RUNNER_README.md
- mini_pose/run_case.py and mini_pose/verify_cube_case.py
- mini_depth/geometry.py and mini_depth/profile.py
- mini_optical/adapter.py, mini_optical/environment.json and mini_optical/test_optical.py
- mini_pose/register_umi_articulated.py, mini_pose/inspect_umi_full.py, mini_pose/freeze_umi_registration.py
- outputs/mini_pose/avocado/articulated_registration/README.md, landmarks.json, rest_pose_config.json, mounting_audit.json and validation.json

The selected policy is one prediction at frozen inferred poses, without first-contact reset or added approach. The original mini_pose/run.py is cube-specific. The CAD preparation scripts contain avocado-specific paths, landmarks, part indices and image dimensions. Generalize those for this case; never run them unchanged on new inputs. The old mini_optical/README.md documents an earlier pressing-depth sweep: its adapter details remain useful, but the sweep is not the approved pose policy. Do not use the contact_roi rim-reference heuristic or the rejected pad_refit workflow.

Run the optical tests and verify the approved cube reference without changing its inputs or poses. Reproduce into a separate directory if needed and compare with outputs/active_prediction.json. Investigate mismatches before altering dependencies or calibration.

2. INSPECT THE INPUT COORDINATES
Hash the inputs. Verify the supplied object mesh and scene share the intended frame, using metadata, source-view projection and 3D consistency. Respect all GLB scene-node transforms. Preserve the reconstructed object; do not run SAM3D, TripoSR or a shape prior, and do not move or reshape it to obtain contact. Sharing coordinates does not prove metric accuracy.

Inspect the full CAD assembly hierarchy and instance transforms, including generically named SHELL and COMPOUND leaves. Our UMI assembly contains both Mini instances; an earlier name-only extraction missed them. Check the actual structure rather than assuming a sensor is absent. CAD import must preserve units and assembly transforms. On Baymax, existing OCP/CadQuery libraries can be reused in an isolated CAD-import process; do not change the optical environment.

3. REGISTER THE ARTICULATED GRIPPER
Infer one common scale and body pose from rigid CAD/scene correspondences, using both detected tag corners and visible feature points such as cover, rail and finger screws. Use the source photograph and its mapping to the cloud where available. Identify the actual tag family/IDs and corner order; do not assume all tags are ArUco or blindly reuse the avocado IDs. Confirm printed marker size versus CAD recess size and record any approximation.

Treat opening as an articulated parameter. Every tag, holder, housing and gel moves with its own finger. For our UMI rack-and-guide mechanism, the validated model was symmetric CAD-X translation of the two finger groups, with the body fixed. Confirm that model for the supplied CAD; do not substitute unrelated linkage motion or independent unconstrained pad fits. Keep CAD dimensions rigid and use one scene-to-mm scale; never resize pads independently.

Fit the common body transform, opening and scale jointly. Document landmark selection, image/3D residual weights and the source of each correspondence. Estimate depth from rigid printed-tag interiors if corner depth is unstable; never equate observed gray gel/contact pixels with the undeformed gel. Save both reprojection residuals in pixels and 3D residuals in mm, per landmark and per rigid group. Inspect a source-view overlay and a 3D scene overlay; a small image residual alone is insufficient. Use held-out rigid features and registration sensitivity checks when informative.

Object surfaces and previous contact patches are held-out consistency evidence, not targets for moving the pads into contact. Do not reuse avocado's scale, opening, coordinates, tag corners, ROI depths, or cube's scale and 90-degree roll. If registration is underdetermined, show available evidence and ask for the smallest missing correspondence or measurement while continuing independent preparation. Do not call the inferred poses exact measurements.

4. DERIVE AND FREEZE THE REST-GEL POSES
Recover each undeformed rest apex and sensor axes from the actual CAD Mini instance and its mounting transform. Keep the physical curved Mini surface at 20.75 x 25.25 mm. The larger CAD body/flange is not the active gel footprint. Compare the canonical pinned surface to both CAD instances; do not mirror the gel or use a backing cap/duplicate offset surface.

Canonical axes are x=image-right, y=image-up, z=inward from object toward sensor; rest apex [0,0,0]; camera [0,0,26.7] mm looking along -z. For world-axis basis columns B:
local_mm = (world_xyz - origin_world) @ B * mm_per_world_unit.
Validate finite orthonormal right-handed bases, normals pointing into the housing, mounting offsets and image orientation. For the same photographed UMI setup, the red marked long edge is the 320-pixel edge. Preserve native 240x320 RGB; a separate landscape preview may rotate that edge to the top. Record the convention and distinguish it from measured hardware pixel parity.

Freeze the inferred poses and scale in a per-case configuration before computing object penetration. No first-touch reset, approach offset, pressing sweep, rim normalization or adjustment based on the desired tactile pattern is allowed.

5. COMPUTE DEPTH AND OPTICS
Use vendor/hydroshear at f815b82fdf3451852acd918933020a82cede1f3b and the existing mini_depth/profile.py loader. Preserve runtime_horizontal_fov: width=240, height=320, fx=fy=308.8192771084337 px, principal point=(120,160), camera distance=26.7 mm.

Transform the full mesh into each frozen sensor frame. Rasterize the actual undeformed gel and full object with existing geometry helpers. On valid normal-grid support compute max(z_object-z_rest,0) in mm. Separately rasterize nominal gel and object camera depths. Where both are valid, deformed depth is the nearer of the two; otherwise retain nominal gel depth. Set camera_depth_delta_mm = max(nominal_depth_mm-deformed_depth_mm,0), with zero outside valid gel/object support.

Pass camera_depth_delta_mm/1000 as finite nonnegative float32 metres to the existing MiniOptical adapter and upstream gelsightRender.render_tensorized. Do not pass absolute depth, normal indentation, millimetres or pre-negated values. The renderer performs its internal sign/unit conversion once. Preserve its no-marker background, RGB order and deterministic four-thread CPU execution.

Keep the calibration-bin correction: configuration says 120, but calibration has 125 bins and RGB arrays shaped (125,125,6). Validate and set 125 BEFORE constructing the renderer. Never truncate, rebin, refit or modify coefficients. The optical 0.065 mm/pixel parameter is only for slope conversion; it does not replace perspective geometry. Preserve true no-contact results.

6. DELIVER AND VALIDATE
Save a new case under /ssd/ruihan/code/VTCotraining/outputs/mini_pose/<CASE_NAME>/ without overwriting prior results. Include:
- Registered CAD/scene overlay, per-part transforms, landmark annotations, fitted opening, scale provenance and residuals.
- Frozen rest-apex poses, sensor bases and CAD-to-Mini mounting checks.
- Per-pad normal indentation mm, nominal/deformed camera depths mm, camera depth difference mm, renderer input metres, valid-gel/contact masks, native tactile_rgb.png, and estimated contact points transformed into the original scene.
- A figure comparing both pads with common depth colour scales, plus a readable scene-placement figure.
- Report with input/code hashes, revision/configuration, bin correction, peak depths, contact footprint, unsupported/out-of-view geometry, checks and limitations.
- Reusable configuration, exact rerun command and README.

Verify poses and original object remain unchanged after freezing, optical inputs are finite/nonnegative with correct units, contact points lie on the original finite mesh triangles, images decode and deterministic replay agrees. Separate numerical consistency from physical accuracy. If the fitted arrangement produces implausible overlap or misses expected contact, retain and label the result as a geometry diagnostic, explain the conflicting evidence and do not tune the poses to manufacture contact. These are simulations conditional on reconstructed geometry and registration, not measured indentation, force, pressure or validated real tactile images.
