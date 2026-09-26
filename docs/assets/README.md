# Gallery provenance / 图集来源

These are plots of executed artifacts, not generated concept art. The procedural
images, reconstructed meshes and robot motion in this gallery are synthetic.
The YCB panel contains numeric experiment results only. No third-party source
photographs or robot meshes are redistributed.

| File | Actual source and scope |
|---|---|
| input_views.png | Six fit frames from make_vision_example.py: IDs 001, 005, 009, 013, 017, 021. |
| gaussian_views.png | All four heldout views, reference / initialization / optimized output. Images enlarged with nearest-neighbour sampling; no enhancement. |
| appearance_metrics.png | Per-view PSNR in reports/synthetic_gaussians.json and reports/ycb_gaussians.json; includes background. |
| dense_surfaces.png | Actual RGB and RGB-D exported triangle meshes, common metric axes, two viewpoints, flat shading. No hole filling. |
| controller_response.png | Case A, controller_3 heldout rollout and valid issued commands. Synthetic first-order system, same model family as fit. |
| motion_contact.png | Case C slide_3 and case D push_2 heldout x trajectories. Both synthetic; D conditions on prescribed pusher motion. |
| torque_diagnostic.png | Case E, joint 1, 7- and 11-sample derivative smoothing windows. Synthetic inverse dynamics diagnostic. |
| contact_feedback.gif | Actual stored MuJoCo body poses for the stable three-DOF fixture and no-contact replay; every 3 control frames, 60 ms per GIF frame. Top-down schematic, not native renderer or camera footage. |
| feedback_traces.png | Same stable robot episodes: q[0], object_xyz[0], generalized_contact_force[0]. The legacy generalized_contact_force field actually records qfrc_constraint, including joint limits; it is not isolated contact force. First joint is prismatic: position in metres, force in newtons. |

[provenance.json](provenance.json) records SHA-256 hashes of the source artifacts.
Paths are experiment-relative names. Hashes support traceability; they are not
evidence of physical validity.

## Executed display configurations

- Synthetic Gaussian experiment: 24 cameras (20 fit / 4 test), 256 fixed Gaussians,
  160 steps, 64 px maximum side. Initialization uses fit-view synthetic depth.
- YCB Gaussian experiment: 80 fit / 16 test cameras, 320 fixed Gaussians, 320 steps,
  64 px maximum side. Seeds carved from fit silhouettes.
- Synthetic RGB surface: 16 fit views; exported 6,594 vertices / 12,342 triangles.
- Synthetic RGB-D surface: 20 fit views; exported 9,333 vertices / 18,183 triangles.
- A–E: outputs of run_examples.py. The data generator supplies known dynamics;
  very small prediction errors are software checks, not real accuracy.
- F: stable fixture using implicitfast integration, 1 ms physics step,
  20 ms control period, 5 s duration, 18 mm task success tolerance.

The quick A–I runner uses smaller vision settings than the display experiments.
RGB and RGB-D mesh runs also differ in camera count and resolution; their metrics
must not be interpreted as a controlled depth-only ablation. Directed distances
from mesh vertices to analytic synthetic surfaces do not certify completeness,
worst-case surface error or smartphone sensing accuracy.

All gallery content remains subject to the validation limits in
[VALIDATION.md](../../reports/VALIDATION.md). In particular, there is no paired
real-robot equivalence claim and no connected, real-tested visual-policy loop.
