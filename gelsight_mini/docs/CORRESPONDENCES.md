# How to obtain identifiable CAD-to-scene pairs

A pair must refer to the **same physical feature** in CAD and scene. Arbitrary nearest points do not establish that identity. This implementation combines decoded tags with observed screw contours, then verifies the pixels' connection to the cloud.

## Reusable hardware knowledge
`rigs/umi_gs_mini.json` contains CAD-space screw centres, tag corners in decoder order, group membership, prismatic jaw kinematics and gel mounts for the supplied full UMI assembly. It contains no avocado pixels, camera, scale, opening or fitted body pose. Tag IDs and physical print orientation are installation properties that CAD geometry alone cannot identify. Check them when changing hardware.

## Implemented automatic path
1. For a complete finite row-major RGB cloud, try candidate image dimensions and normalized perspective DLT. Fit up to 600 points and test 600 disjoint points; reject ambiguity and require <0.1 px maximum reprojection error. Check a larger distributed subset afterward. Grid dimensions are bounded to 128–4096 and aspect ratios to 0.2–5. This recovers the projection encoded in the reconstruction, not physically calibrated depth.
2. Reconstruct RGB only after that check. Detect declared AprilTag/ArUco IDs, retain decoder corner order, and jointly estimate a tag-based body pose/opening to guide searches.
3. Project known CAD screw centres to define search regions. Detect actual closed image-edge contours, fit ellipses and filter size/shape/location. Competing near-equal matches remain unresolved. An unobserved CAD prediction is never an observation. This detector targets visible round screw heads; it is not general CAD recognition.
4. Save predictions, candidates and crops for the agent to inspect. A green detector match is not independent proof of identity.
5. Lift tag corners through source-ray intersections with robust planes fitted to tag interiors. Lift screw pixels using local cloud neighborhoods. Save CAD and scene coordinates with provenance. Fit the final common similarity/opening from body features and moving tags; do not accept the tag-only initializer as the result.

```bash
python gelsight_mini/discover_correspondences.py \
  --scene case/inputs/scene.ply --rig case/rig.json --output discovery
```

Optional `--camera-json` supplies an already verified source camera. Outputs: `annotations.json`, `prepared/landmarks.json`, `source_rgb.png`, `discovery.json`, `correspondence_candidates.png`, `feature_crops.png`. If insufficient body features are found, candidates remain for review but prepared landmarks are not produced. The pipeline also accepts `auto_correspondences: true`.

## Fallbacks the agent should attempt
- **Known image/mapping:** verify metadata by reprojection. For partial/reordered clouds, preserve a pixel-index map and implement its lifting explicitly or use 3D feature pairs. The automatic script requires a complete finite grid.
- **Unorganized cloud:** render colored cloud views with point IDs, identify the same physical holes, screws, tag corners or distinctive geometry, then write actual point indices/world coordinates. `prepare_landmarks.py` supports these without a camera. Generic 3D descriptor matching is not implemented here; proposals from such methods still need symmetry/occlusion checks.
- **Ambiguous screws:** inspect crops, refine real observed centres or use other visible CAD features. Never write predicted positions as observed data.
- **Different CAD:** export all parts, derive groups/kinematics, identify features and verify housing-to-gel transforms and canonical axes. The fitter supports scalar prismatic opening; other mechanisms require explicit implementation and validation. Different tag layouts require their actual IDs, dimensions and corner order.
- **Unobservable geometry:** request the smallest missing source image/mapping, extra view, or named 3D feature. No prompt can recover absent information.

At least three non-collinear body pairs initialize the common similarity, and at least three observations on each moving group are required. More widely spread pairs and independent checks are preferable; three body points provide little outlier redundancy.

## Validation scope
On the avocado cloud, discovery used no scene clicks, source dimensions/intrinsics, previous scale or opening. It recovered the 756×1008 grid, detected both tags and three cover screws; two rail screw features were rejected. Agent review confirmed the selected locations lie on visible screws. Held-out mapping error was about 0.00003 px. The scene/object still have substantial 3D inconsistency, so this demonstrates data preparation and numerical execution, not accurate physical indentation or reliability on arbitrary scenes.
