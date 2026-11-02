# SPR-SAC: setup and six Control Suite experiments

This guide describes the current `spr_sac` implementation and the experiment
presets in [`src/configs`](src/configs). Training writes scalar metrics to the
console and local JSON logs.

## 1. Create a conda environment

The commands below use Bash on Linux. Install conda first if necessary, following
the [conda installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
Run all subsequent commands from the repository root, where `requirements.txt`
and `src/` are located.

```bash
conda create -n dmcontrol-spr python=3.12 pip -y
conda activate dmcontrol-spr
python -m pip install --upgrade pip
```

Install **one** PyTorch build. For a Linux machine with an NVIDIA GPU:

```bash
python -m pip install torch
```

For a specific CUDA build, use the command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/) that matches
your system. This project only needs the `torch` package from that command.

For a CPU-only installation, use this command instead:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the project dependencies into the active conda environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

`dm-control` installs its MuJoCo Python dependency through pip. Use this setup
for the current codebase. Environment creation and activation follow the
[conda environment workflow](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## 2. Configure rendering and verify installation

SPR learns from rendered pixel observations. `dm_control` chooses an OpenGL
backend in this order: GLFW, then EGL, then OSMesa. The
[Control Suite rendering instructions](https://github.com/google-deepmind/dm_control#rendering)
list the system libraries for each backend.

`libglfw3` and `libglew2.2` are the windowed GLFW backend. GLFW needs a
display. On a headless machine `DISPLAY` is unset, and the Python `glfw`
package only warns about that: `glfw.init()` returns failure instead of
raising. `dm_control` still selects GLFW, and MuJoCo then fails with
`an OpenGL platform library has not been loaded into this process`.

Headless GPU rendering uses EGL. `src/env.py` sets `MUJOCO_GL=egl` when that
variable is unset and neither `DISPLAY` nor `WAYLAND_DISPLAY` is set. Set
`MUJOCO_GL` yourself to choose another backend. `MUJOCO_EGL_DEVICE_ID` selects
the EGL GPU.

CPU-only software rendering needs OSMesa, which is a separate library from
GLFW and GLEW:

```bash
sudo apt-get install libosmesa6
export MUJOCO_GL=osmesa
```

Check the installed versions and CUDA availability:

```bash
python - <<'PY'
import sys
import torch
import numpy as np
import kornia

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Kornia:", kornia.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

python -m unittest discover -s tests -v
```

The compatibility checks have passed with Python 3.12, PyTorch 2.15.0 (CPU),
NumPy 2.5.4, Kornia 0.9.0, Gymnasium 1.4.0, and dm-control 1.0.50.
The requirements use version ranges; record the actual installed versions when
comparing experiments.

Run this small training smoke test to check pixel rendering, evaluation, and
optimization together:

```bash
SMOKE_ROOT="$PWD/spr_runs/smoke-$(date +%Y%m%d-%H%M%S)"
OMP_NUM_THREADS=1 python src/train.py \
  --config-path configs --config-name cartpole_swingup \
  seed=1 gpuid=0 work_dir=. hydra.run.dir="$SMOKE_ROOT" \
  num_env_steps=48 init_steps=4 action_repeat=4 \
  eval_freq=32 num_eval_episodes=1 \
  replay_buffer_capacity=32 batch_size=2 auxiliary_task_batch_size=2 \
  jumps=1 hidden_dim=16 encoder_feature_dim=8 latent_dim=16 \
  transition_model_layer_width=16 num_layers=2 num_filters=4 log_interval=1
```

This smoke test uses a small network and buffer. Use the presets below for the
full experiments. Training automatically uses CUDA when available and CPU
otherwise.

## 3. Experiment presets

| Config name | Control Suite environment | Action repeat |
| --- | --- | ---: |
| `ball_in_cup_catch` | `ball_in_cup/catch` | 4 |
| `cartpole_swingup` | `cartpole/swingup` | 8 |
| `cheetah_run` | `cheetah/run` | 4 |
| `finger_spin` | `finger/spin` | 2 |
| `reacher_easy` | `reacher/easy` | 4 |
| `walker_walk` | `walker/walk` | 2 |

Each preset already selects `agent: spr_sac`. All six use a training horizon of
`105000`, a warmup of `1000` agent decisions, and evaluation every `5000`
training steps for `10` episodes. Cheetah uses `2e-4` for the actor, critic,
encoder, and auxiliary learning rates; the other presets use `1e-3`.

The step counter advances by `action_repeat` after each agent action.
`num_env_steps` and `eval_freq` therefore count underlying Control Suite steps;
evaluation interaction is additional. The code converts `init_steps`,
`log_interval`, `actor_update_freq`, and `critic_target_update_freq` from agent
decisions to this step scale internally.

The training loop excludes its upper bound. These horizons include evaluation
at the desired budget:

| Desired evaluation step | `num_env_steps` |
| --- | ---: |
| 100,000 steps | `105000` (preset default) |
| 500,000 steps | `505000` |

The default replay capacity of `100000` reserves approximately **18 GB
(16.8 GiB)** for the two pixel-observation arrays alone. Allow additional host
memory for training. On smaller machines, overrides such as
`replay_buffer_capacity=20000 batch_size=128 auxiliary_task_batch_size=32`
reduce memory use, but change the experimental settings.

## 4. Run one environment

Every task preset has `work_dir: ???`, which must be supplied. Hydra changes
into its run directory, so `work_dir=.` keeps logs beneath the
chosen `hydra.run.dir`. The `configs` path is resolved relative to
`src/train.py`, as described in the
[Hydra command-line documentation](https://hydra.cc/docs/advanced/hydra-command-line-flags/).

For cartpole with seed 1 and CUDA device 0:

```bash
RUN_ROOT="$PWD/spr_runs/single-$(date +%Y%m%d-%H%M%S)"
python src/train.py --config-path configs --config-name cartpole_swingup \
  seed=1 gpuid=0 work_dir=. \
  hydra.run.dir="$RUN_ROOT/cartpole_swingup/seed1"
```

Use any config name from the table with the same command structure. Supply
`seed` explicitly for reproducible runs; the preset fallback chooses a random
seed. `gpuid` is the CUDA index visible to PyTorch. EGL rendering-device
selection can be controlled separately with `MUJOCO_EGL_DEVICE_ID` when needed.

To inspect a configuration without starting an experiment:

```bash
python src/train.py --config-path configs --config-name cartpole_swingup \
  seed=1 gpuid=0 work_dir=. --cfg job --resolve
```

## 5. Run all six environments with multiple seeds

Paste this Bash block from the repository root after activating the conda
environment. Headless runs use EGL unless `MUJOCO_GL` is already set. It runs
**18 experiments sequentially**:
six environments, each with seeds 1, 2, and 3. Edit the seed list and horizon
as needed. Each invocation gets a separate output directory.

```bash
(
  set -euo pipefail
  RUN_ROOT="$PWD/spr_runs/six-envs-$(date +%Y%m%d-%H%M%S)"
  train_steps=105000  # Set to 505000 to include evaluation at 500k steps.
  gpu_id=0
  mkdir -p "$RUN_ROOT"
  python -m pip freeze > "$RUN_ROOT/packages.txt"

  for task_config in \
    ball_in_cup_catch cartpole_swingup cheetah_run \
    finger_spin reacher_easy walker_walk
  do
    for seed in 1 2 3
    do
      python src/train.py --config-path configs --config-name "$task_config" \
        seed="$seed" gpuid="$gpu_id" num_env_steps="$train_steps" \
        work_dir=. hydra.run.dir="$RUN_ROOT/$task_config/seed$seed"
    done
  done
)
```

For a single-seed sweep, change `for seed in 1 2 3` to `for seed in 1`.
For ten seeds, use `for seed in {1..10}`.

## 6. Find logs

With the commands above, a task/seed run has this layout:

```text
<RUN_ROOT>/<task_config>/seed<seed>/
├── .hydra/
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── train.log
└── spr_sac_dmc/<generated-experiment-name>/
    ├── train_info.log
    └── eval_info.log
```

`train_info.log` and `eval_info.log` contain one JSON object per line. Evaluation
records include `step`, `episode_reward`, `mean_episode_reward`, and
`best_episode_reward`. Use `mean_episode_reward` at the same `step` to compare
seeds and environments. Training writes these scalar logs only. It does not
save model checkpoints or replay-buffer data.
