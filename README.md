# AGS-DD: Adaptive Guidance Allocation for Training-Free Diffusion Dataset Distillation

Local anonymous **preview for author review**, not an approved public release or a verified reproduction of the paper. Publication scope, licenses, anonymity, and experiment instructions still require review. See `PREVIEW_STATUS.md` and `THIRD_PARTY_NOTICES.md` before sharing externally.

## Contents

This is a selected source-code package. It includes the AGS implementation, generation and evaluation entry points, class lists, configurations, analysis helpers, and the custom `diffusers` runtime. It excludes the manuscript, figures, raw experiment logs, generated images, result snapshots, checkpoints, datasets, caches, internal notes, and Git history.

| Entry | Purpose |
| --- | --- |
| `ags/` | Class complexity, adaptive stopping, timestep schedules, sampler |
| `gen_single_config.py` | Explicit single-configuration DiT generation |
| `duration_sweep.py` | Helpers used by generation; also contains a separate legacy sweep |
| `quick_eval_v3.py` | Classifier evaluation with best-over-training accuracy tracking |
| `sample_ags.py` | Earlier general DiT entry; defaults are not a paper reproduction recipe |
| `sample_ags_sd.py` | Stable Diffusion extension; requires separate model/data assets |
| `gen_d4m.py`, `gen_cags_d4m_hybrid.py` | Baseline and hybrid generation paths |
| `configs/`, `misc/` | Configurations, class lists and utility dependencies |
| `train.py`, `train_models/`, `data.py`, `tsne_plots.py` | Shared loader, feature and evaluator code |
| `diffusers/` | Bundled custom implementation; preserve its local editable installation |

The training helpers are included because feature-loader and evaluation imports depend on them. "Training-free" concerns the diffusion generator, not the downstream classifier evaluation.

## Reference installation (not executed in a fresh environment)

Reference target: Python 3.10, PyTorch 2.7.1, torchvision 0.22.1, NVIDIA CUDA 12.8 wheels. Dependency lower bounds outside PyTorch are not a tested lockfile. The commands below are a proposed installation recipe, not evidence that a clean installation has passed.

Run in PowerShell **from this extracted package root**, in a new directory without an existing `.venv`:

```powershell
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index https://download.pytorch.org/whl/cu128
uv pip check --python .venv\Scripts\python.exe
& .\.venv\Scripts\python.exe -c "import diffusers; print(diffusers.__version__); print(diffusers.__file__)"
```

The last path must end in this package's `diffusers/src/diffusers/__init__.py`; it must not point to a different checkout or an ordinary site-packages copy. Keep `-e ./diffusers` in the requirements. Stop and diagnose a resolver conflict instead of silently changing the Python/PyTorch/CUDA target. A CUDA Toolkit installation is not part of this recipe. On Linux the interpreter path is `.venv/bin/python`; Linux execution has not been tested for this preview.

Installation syntax references: [uv package management](https://docs.astral.sh/uv/pip/packages/) and [PyTorch previous-version wheels](https://pytorch.org/get-started/previous-versions/).

## Inspect before running experiments

These commands display options only:

```powershell
& .\.venv\Scripts\python.exe -B gen_single_config.py --help
& .\.venv\Scripts\python.exe -B quick_eval_v3.py --help
& .\.venv\Scripts\python.exe -B sample_ags.py --help
& .\.venv\Scripts\python.exe -B sample_ags_sd.py --help
```

Equivalent `--help` checks passed using an existing environment with this preview's source paths forced first and networking blocked. No fresh environment was installed for those checks. Imports and help output do not establish that model loading, sampling, or classifier training works.

## Assets and paths

Always run from the package root: several helpers read `./misc/...` and other relative paths. Supply dataset paths explicitly, and use a new output directory for each run to avoid overwriting previous outputs or mixing cluster caches.

The DiT entry expects a dataset with `train/<class-id>/...` and `val/<class-id>/...` directories, class lists under `misc/`, the `DiT-XL-2-256x256.pt` checkpoint under `pretrained_models/`, and access to the `stabilityai/sd-vae-ft-mse` VAE. Missing model assets may trigger network downloads during generation. No model or dataset is bundled or downloaded by preparation of this preview. The Stable Diffusion extension requires its own SD 1.5 model and dataset. Obtain assets under their applicable terms separately.

Some utility scripts execute data preparation at module scope: do not import or run all scripts indiscriminately. Do not reuse untrusted pickle caches or checkpoint files.

## Configuration and metric cautions

- For the two-factor CAGS configuration, specify `--alpha 0 --beta 0.5 --gamma 0.5 --delta 0` explicitly in `gen_single_config.py`. Its defaults are the four-factor configuration.
- Select the intended `--window` and `--guidance-steps` explicitly. `gen_single_config.py` uses a constant schedule and enables IAST only with `--use-iast`. The earlier `sample_ags.py` and SD entry have different defaults; do not assume they enable the same ablation.
- CAGS is the primary method in the current manuscript; retaining IAST/TAGS code does not assert that enabling them improves results. Match each table's configuration separately.
- `--no-cags --fixed-scale 0` disables this additional mode-guidance term; it does not by itself disable the base classifier-free guidance in the diffusion model. Cluster work may still occur in the generation path.
- `gen_single_config.py` creates one generated dataset under `<save-base>/<tag>/dataset_0/`. Evaluator `--seeds` are classifier-training seeds, not additional independently generated datasets.
- `quick_eval_v3.py` reports the best recorded validation Top-1 during training, then mean and population standard deviation across evaluator seeds. It does not report final-epoch Top-1. Specify `--class-file`, `--nclass`, `--epochs`, architecture, depth, and seeds explicitly; its defaults are not the complete main-table protocol.
- Do not treat the legacy 20-epoch sweep evaluator in `duration_sweep.py` as equivalent to the longer `quick_eval_v3.py` protocol.

Exact table-to-command mappings, asset versions, generation/evaluation seeds, and full experiment reproducibility remain pending author review. No paper results were regenerated while making this package.

## Known portability issue

`tsne_plots.get_loader` still contains a local `lambda` transform with `num_workers=4`. This pattern can fail to pickle under Windows multiprocessing spawn. The preview preserves the selected source version and does not incorporate machine-specific loader fixes. Successful imports and `--help` are not a successful Windows data-loading test. Resolve and validate this separately before attempting a Windows end-to-end run.
