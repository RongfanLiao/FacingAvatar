# train_motion_diffusion.py Args

This document records the command-line arguments for `train_motion_diffusion.py` and explains how they affect training.

Current diffusion training in this repository uses the aligned input contract:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- FLAME targets from `flame_param.npz`

The training entrypoint is:

```bash
python train_motion_diffusion.py [args...]
```

## Typical command

```bash
python train_motion_diffusion.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_diffusion_lookingface \
  --epochs 10 \
  --batch_size 2 \
  --val_interval 2 \
  --video_canvas_size 400
```

## Data and split args

### `--predefined_splits_dir`

- Type: `str`
- Default: `None`
- Meaning: directory containing `train.json`, `valid.json`, and optionally `test.json`
- Current use: recommended for LookingFace training

If provided, the script reads predefined splits from the given directory. If `test.json` exists, it is used as the evaluation split; otherwise `valid.json` is used.

### `--split_path`

- Type: `str`
- Default: `data/benchmark_split.json`
- Meaning: output or input path for the fallback random train/validation split

This is used only when `--predefined_splits_dir` is not supplied.

### `--max_sequences`

- Type: `int`
- Default: `0`
- Meaning: limit the number of train and validation sequences after split selection

Useful for smoke tests or quick debugging runs.

### `--max_eval_samples`

- Type: `int`
- Default: `None`
- Meaning: limit the number of validation sequences used for evaluation and metric generation

Useful when you want a faster validation pass or a small evaluation smoke test without reducing the training set.

### `--train_val_same`

- Flag
- Default: `False`
- Meaning: force validation to use the same sequence IDs as training

Useful only for debugging or convergence checks.

### `--eval_only`

- Flag
- Default: `False`
- Meaning: skip training and only run evaluation / metric generation

Useful when a checkpoint already exists in `--checkpoint_dir`.

### `--fast_eval`

- Flag
- Default: `False`
- Meaning: use a reduced reverse diffusion step count for a faster evaluation run

This is a convenience switch for evaluation-heavy workflows. It leaves the model weights unchanged and simply reduces the number of sampling steps used at inference time.

### `--fast_eval_timesteps`

- Type: `int`
- Default: `10`
- Meaning: reverse diffusion steps to use when `--fast_eval` is enabled

When `--fast_eval` is active, the effective `--inference_timesteps` becomes:

```text
min(inference_timesteps, fast_eval_timesteps)
```

### `--video_canvas_size`

- Type: `int`
- Default: `512`
- Meaning: spatial size used when fitting raw left video frames into a square canvas

This affects memory use directly. Smaller values reduce GPU memory and speed up training, but may reduce visual fidelity.

Recommended starting values:

- `224` for smoke tests and limited-memory runs
- `400` or `512` for more faithful training

## Optimization args

### `--epochs`

- Type: `int`
- Default: `100`
- Meaning: number of training epochs

### `--batch_size`

- Type: `int`
- Default: `4`
- Meaning: number of sequences per batch

Because raw video is loaded and encoded online, this usually needs to be much smaller than embedding-based training.

### `--lr`

- Type: `float`
- Default: `1e-4`
- Meaning: AdamW learning rate

### `--weight_decay`

- Type: `float`
- Default: `1e-5`
- Meaning: AdamW weight decay

### `--grad_clip`

- Type: `float`
- Default: `1.0`
- Meaning: gradient clipping threshold applied after backpropagation

Useful for stabilizing diffusion training.

### `--num_workers`

- Type: `int`
- Default: value from `config.NUM_WORKERS`
- Meaning: dataloader worker count

When debugging, use `0`. For training, increase this only if raw-video decoding remains stable.

### `--log_interval`

- Type: `int`
- Default: `1`
- Meaning: print per-iteration training and validation progress every N batches

This uses the same clearer logging style as the TransVAE training path. The logs include:

- timestamp
- train or validation label
- epoch and iteration progress
- elapsed time and ETA
- detailed loss breakdown

Use larger values such as `10` or `20` when you want less frequent console output.

## Model architecture args

### `--feature_dim`

- Type: `int`
- Default: `256`
- Meaning: internal latent channel size used throughout the denoiser

Smaller values reduce memory and compute. Larger values increase model capacity.

### `--n_heads`

- Type: `int`
- Default: `8`
- Meaning: number of transformer attention heads

### `--num_layers`

- Type: `int`
- Default: `4`
- Meaning: number of transformer layers in the denoiser blocks

### `--dropout`

- Type: `float`
- Default: `0.1`
- Meaning: dropout used in transformer layers

The current diffusion benchmark path predicts the repository's 112-d FLAME content target.

## Diffusion process args

### `--train_timesteps`

- Type: `int`
- Default: `1000`
- Meaning: number of forward diffusion steps used during training

Larger values more closely match standard DDPM-style setups, but increase diffusion schedule resolution.

### `--inference_timesteps`

- Type: `int`
- Default: `50`
- Meaning: number of reverse sampling steps used during evaluation / generation

Lower values make evaluation faster.
If `--fast_eval` is enabled, this value is capped by `--fast_eval_timesteps`.

### `--beta_start`

- Type: `float`
- Default: `1e-4`
- Meaning: starting beta in the linear noise schedule

### `--beta_end`

- Type: `float`
- Default: `2e-2`
- Meaning: ending beta in the linear noise schedule

### `--clip_sample`

- Type: `float`
- Default: `5.0`
- Meaning: clamp range applied to predicted samples during diffusion

### `--timestep_spacing`

- Choices: `leading`, `linspace`, `trailing`, `full`
- Default: `leading`
- Meaning: strategy for selecting reverse-process timesteps during inference

Use the default unless comparing scheduler variants.

### `--ddim_eta`

- Type: `float`
- Default: `0.0`
- Meaning: DDIM stochasticity parameter during sampling

`0.0` gives deterministic DDIM-style sampling.

## Conditioning dropout and guidance args

These control classifier-free style conditioning dropout in the denoiser.

### `--guidance_scale`

- Type: `float`
- Default: `1.5`
- Meaning: conditioning strength used during sampling

Higher values push predictions to depend more strongly on conditioning signals.

### `--audio_drop_prob`

- Type: `float`
- Default: `0.2`
- Meaning: training-time dropout probability for audio conditioning

### `--video_drop_prob`

- Type: `float`
- Default: `0.2`
- Meaning: training-time dropout probability for video conditioning

### `--latent_drop_prob`

- Type: `float`
- Default: `0.2`
- Meaning: training-time dropout probability for the pooled latent conditioning token

## Checkpoint and output args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/motion_diffusion_port`
- Meaning: directory where checkpoints and `metrics.json` are written

Files typically written there:

- `best.pt`
- `last.pt`
- `metrics.json`

### `--resume_checkpoint`

- Type: `str`
- Default: `None`
- Meaning: checkpoint path used to resume training state or to provide weights for `--eval_only`

If supplied during training, the script restores the model and optimizer state and continues from the saved epoch. If supplied with `--eval_only`, the script evaluates that checkpoint directly.

## Validation cadence

### `--val_interval`

- Type: `int`
- Default: `5`
- Meaning: run validation every N epochs

If `--epochs 1`, set `--val_interval 1` so validation and checkpoint writing happen.

## Logging behavior

Training and validation now print TransVAE-style progress lines during the epoch.

Typical examples:

```text
[2026-04-05 07:29:36] [train] epoch 1/1 iter 1/2 (50.0%) elapsed=00:03 eta=00:03 | loss_total_with_clip=0.97863 | loss_total=0.97863 | loss_rec=0.96014 | loss_vel=0.03698 | loss_exp=0.31652 | loss_jaw=0.03422 | loss_neck=0.09320 | loss_eyes=0.03612 | mean_timestep=11.00000
```

```text
[2026-04-05 07:29:44] [test] epoch 1/1 iter 1/2 (50.0%) elapsed=00:03 eta=00:03 | loss_total=3.46999 | loss_rec=3.45856 | loss_vel=0.02285 | loss_exp=1.49357 | loss_jaw=0.05759 | loss_neck=0.13142 | loss_eyes=0.04670 | mean_timestep=1.00000
```

This is controlled by `--log_interval`.

## Recommended presets

## Smoke test

```bash
python train_motion_diffusion.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_diffusion_smoke \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --log_interval 1 \
  --val_interval 1 \
  --video_canvas_size 224 \
  --feature_dim 64 \
  --n_heads 4 \
  --num_layers 1 \
  --train_timesteps 50 \
  --inference_timesteps 10 \
  --max_sequences 8
```

## Small real run

```bash
python train_motion_diffusion.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_diffusion_small \
  --epochs 5 \
  --batch_size 1 \
  --num_workers 0 \
  --log_interval 1 \
  --val_interval 1 \
  --video_canvas_size 224 \
  --feature_dim 128 \
  --n_heads 4 \
  --num_layers 2 \
  --train_timesteps 200 \
  --inference_timesteps 20 \
  --max_sequences 32
```

## Fast eval-only checkpoint test

```bash
python train_motion_diffusion.py \
  --eval_only \
  --resume_checkpoint checkpoints/motion_diffusion_port/best.pt \
  --checkpoint_dir checkpoints/motion_diffusion_port_fast_eval \
  --batch_size 1 \
  --num_workers 0 \
  --fast_eval \
  --fast_eval_timesteps 10
```

## Full training starting point

```bash
python train_motion_diffusion.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_diffusion_lookingface \
  --epochs 100 \
  --batch_size 2 \
  --log_interval 10 \
  --val_interval 5 \
  --video_canvas_size 224 \
  --feature_dim 256 \
  --n_heads 8 \
  --num_layers 4 \
  --train_timesteps 1000 \
  --inference_timesteps 50
```

## Notes

1. This diffusion path no longer depends on Qwen video embeddings.
2. It expects wav2vec features to exist in `data/wav2vec_embeddings`.
3. It loads raw left video frames directly from the LookingFace videos through the manifest.
4. If a run says no training sequences are available, check:
   - `data/manifest.json`
   - `data/wav2vec_embeddings`
   - `data/LookingFace/dataset_splits`
5. For quick validation, always start with `--batch_size 1`, `--num_workers 0`, and `--video_canvas_size 224`.