# Validation record — 2026-09-26

**35 software checks passed (27 core + 8 extended). Nine portable input cases executed.**
**Real-robot evaluation equivalence is NOT established.**

| Experiment | Evidence | Executed result | Scope |
|---|---|---|---|
| Gaussian training, YCB object | Real RGB + published camera calibration; 80 fit, 16 heldout; fit-derived visual-hull seeds | Mean heldout PSNR 15.58 → 23.09 dB; silhouette IoU 0.924 | 320 fixed Gaussians, 64px max side, 320 steps; appearance baseline, not production photorealism |
| Gaussian training, two-object fixture | Synthetic RGB, known cameras and fit-depth seeds | PSNR 17.18 → 21.57 dB | Second scene and input contract check, not real data |
| RGB dense reconstruction, YCB | Depth inferred from 16 fit RGB views | 25,864 cross-view retained points; 5,373 vertices / 9,145 faces | Partial non-watertight surface; metric accuracy not independently certified |
| RGB dense reconstruction, fixture | Synthetic images; analytic geometry used only for scoring | Mean / p95 sampled surface distance 0.734 / 2.105 mm | Directed sampled geometry error; no full coverage guarantee |
| RGB-D fusion, fixture | Synthetic registered depths | Mean / p95 sampled surface distance 0.433 / 1.016 mm | Checks TSDF fusion independently of learned imagery |
| Three-DOF actuator feedback | Simulated robot + object + closed-loop state policy | 1,395 contact samples; task passed in simulation; identical target replay without collision failed | Disabling contact changes actual robot x by up to 10.01 mm; no mocap or state overwrites |
| Seven-joint Panda adapter | Existing local Panda/YCB simulation assets | 3,775 contact samples, nonzero joint reaction; target error 63.80 mm; task failed | Demonstrates integrated feedback execution, NOT task success or real dynamics alignment |
| Unknown-pose SfM | Synthetic object views, poses withheld | Only 2 of 20 fit images registered, 30 sparse points | **Insufficient coverage; failed as a complete reconstruction path** |
| Paired real policy evaluation | No matching independent hardware trials connected | Evidence audit and rejection of synthetic substitution implemented | **Not executed on real paired trials** |

The test suite detects skipped-command handling, duplicate fit/test data, parameter
gauge degeneracy, false object replay, sparse input downgrade, camera corner shape
compatibility, differentiable Gaussian gradients, large OpenCV sampling batches,
analytical surface error and two-way contact feedback.

## Known failures and limits

- Initial dense fusion hit OpenCV's signed-short remap dimension limit; batch
  sampling fixes it and a 40,000-point regression test passes.
- Explicit integration destabilized the small high-gain yaw servo in the fixture;
  the fixture now uses MuJoCo implicitfast. This does not calibrate real stiffness.
- Native video rendering is unavailable in this headless macOS environment;
  numeric physics rollouts and traces still execute. Optional video errors are recorded.
- Sparse SfM did not cover the unknown-pose fixture. No successful generic
  unposed-phone pipeline is claimed.
- CUDA/gsplat is implemented as an optional backend but has not run on this host.
- The complete loop does not yet connect reconstructed Gaussian images to an
  independently tested real visual policy. Gaussian training and feedback
  experiments remain separate evidence branches.
- No real robot is available, and no matched per-condition policy test set for
  this new simulator has been supplied. Public author tables are not our validation.

See the JSON reports for actual metrics, provenance hashes and limitations.
Third-party photographs and meshes are not included; only numeric summaries are
published. Synthetic fixtures can be generated from source.
