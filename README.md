# Evidence Real2Sim

Data-aware robot simulation calibration, Gaussian appearance training, dense
reconstruction, actuator-driven contact feedback, and scoped evaluation audits.

**Research prototype. Real-to-sim policy evaluation equivalence has not been
validated.** This project separates executed software, synthetic checks, real
image/log experiments and missing real-robot evidence.

[中文说明](docs/README_zh.md) · [Input examples](docs/INPUT_CASES.md) ·
[Validation record](reports/VALIDATION.md) · [Third-party notices](THIRD_PARTY.md)

## What runs

- Explicit HDF5 / Parquet / NPZ episode mapping; no guessing what `action` means.
- Optional camera calibration, metric marker tracking and sparse SfM.
- Independent TCP alignment, selected joint-zero fitting and rank checks.
- Empirical command-response identification and motion-grounded contact fitting.
- Measured-torque inventory and conditional inverse-dynamics diagnostics.
- **Trainable anisotropic 3D Gaussians** with a CPU reference rasterizer, or an
  optional gsplat/CUDA backend. Trainable centres, rotations, scales, opacity and
  SH0 colour; fixed Gaussian budget, no adaptive densification.
- **RGB plane-sweep stereo + TSDF**, and registered RGB-D TSDF fusion. Unobserved
  regions remain unfilled. Dense meshes are not automatically collision-certified.
- **Actuator-driven robot and object simulation**, with policy feedback or logged
  target commands. States are only initialized at reset; robot states are not
  forced to follow recorded motion during this stage.
- Paired policy metrics and an explicit protocol/evidence audit. Synthetic
  examples and author aggregate results cannot pass as new real trials.

## Quick start

Python 3.12 is the tested local version. Work in a clean virtual environment.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-vision.txt
python test_checks.py
python tests/test_extended.py
python run_examples.py --out outputs/examples --vision
```

Without PyTorch / vision dependencies:

```sh
pip install -r requirements.txt
python run_examples.py --out outputs/signals
```

Every output directory must be new or empty. Examples write only local files and
do not command physical robots. Sample-count and accuracy gates are task-specific;
the software does not insert a universal threshold after observing results.

## Train Gaussian appearance

```sh
python make_vision_example.py --out outputs/vision_input
python train_gaussians.py --scene outputs/vision_input/scene.json \
  --out outputs/gaussians --steps 160 --max-side 64 --gaussians 256
```

The scene manifest uses OpenCV world-to-camera matrices, per-view intrinsics,
fit/test splits, optional masks and **fit-derived seed points**. Camera poses must
already be available. Unknown-pose SfM is a separate stage and may fail on weak or
repeated texture; there is no automatic claim of universal phone-photo recovery.

The CPU implementation is intentionally small and slow. The optional
`--backend gsplat` uses the public gsplat rasterization API and requires a working
CUDA installation. That GPU path has not been executed on the local Apple Silicon
host. A successful small CPU run is not production photorealism.

## Reconstruct observed surfaces

```sh
python dense_reconstruct.py --scene outputs/vision_input/scene.json \
  --mode rgb --out outputs/rgb_surface
python dense_reconstruct.py --scene outputs/vision_input/scene.json \
  --mode sensor-depth --out outputs/rgbd_surface
```

RGB mode estimates depth from fit images. Sensor mode consumes registered metric
depth. Both reject inconsistent depth, save observation coverage, and export
partial surfaces. They never fill unseen regions and mark them measured.

## Run two-way robot contact feedback

```sh
python make_feedback_example.py --out outputs/feedback_input
python robot_feedback.py --config outputs/feedback_input/config.json \
  --out outputs/feedback
python robot_feedback.py --config outputs/feedback_input/no_contact.json \
  --commands outputs/feedback/episode.npz --out outputs/no_contact
```

The second run applies the recorded targets with robot-object collision disabled.
The difference in actual robot motion tests whether reaction forces affect the
robot. This is a software invariant, not a real-robot dynamics measurement.
`examples/panda_push_policy.py` also supports a matching seven-joint Panda model;
robot assets are intentionally not bundled.

## What is not complete

The current evidence does **not** establish that our simulator can replace real
policy evaluation. No matching hardware platform or heldout per-condition,
per-policy real trial set has been connected. The repository contains the audit
and data contract, not invented real trial outcomes.

Visual fidelity, measured contact geometry, controller response, contact
prediction and policy-ranking agreement are separate validation stages. See
`reports/VALIDATION.md` for executed results and failures. Hardware-specific
controller interfaces, sensor uncertainty and physical reset procedures still
require an experimental deployment.

## License

MIT for this repository's code. Third-party software and data retain their own
terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
