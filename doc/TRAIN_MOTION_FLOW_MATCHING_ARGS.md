# train_motion_flow_matching.py Args

This document records the command-line arguments for `train_motion_flow_matching.py` and explains the first LookingFace flow-matching benchmark implementation.

Current flow-matching training in this repository uses the aligned LookingFace input contract:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- full 118-d FLAME targets from `flame_param.npz`

The training entrypoint is:

```bash
python train_motion_flow_matching.py [args...]
```

## Typical command

```bash
python train_motion_flow_matching.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_flow_matching_lookingface \
  --epochs 10 \
  --batch_size 1 \
  --val_interval 2 \
  --video_canvas_size 224 \
  --video_chunk_size 16 \
  --amp \
  --solver heun \
  --solver_steps 40
```

## Low-memory command

```bash
python train_motion_flow_matching.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_flow_matching_lowmem \
  --batch_size 1 \
  --video_canvas_size 128 \
  --video_chunk_size 8 \
  --amp
```

This is the first configuration to try when training fails in the raw-video Conv3D path with a CUDA OOM.

## What this model is

This benchmark reuses the conditioning stack from the current motion diffusion port, but changes the generative core:

- diffusion predicts a denoising target over discrete timesteps
- flow matching predicts a continuous-time velocity field over interpolated states

At sampling time, the model integrates an ODE from source noise to motion using Euler or Heun updates.

## Data and split args

### `--predefined_splits_dir`

- Type: `str`
- Default: `None`
- Meaning: directory containing `train.json`, `valid.json`, and optionally `test.json`

Recommended for LookingFace benchmarking.

### `--split_path`

- Type: `str`
- Default: `data/benchmark_split.json`
- Meaning: output or input path for the fallback random train/validation split

### `--max_sequences`

- Type: `int`
- Default: `0`
- Meaning: limit the number of train and validation sequences after split selection

### `--max_eval_samples`

- Type: `int`
- Default: `None`
- Meaning: limit the number of validation sequences used for final metric evaluation

### `--train_val_same`

- Flag
- Default: `False`
- Meaning: force validation to use the same sequence IDs as training

### `--eval_only`

- Flag
- Default: `False`
- Meaning: skip training and only run final metric evaluation

### `--video_canvas_size`

- Type: `int`
- Default: repository default from `config.py`
- Meaning: spatial size used when fitting raw left video frames into a square canvas

Recommended starting values:

- `128` or `224` for smoke tests
- `224` or `400` for normal experimentation

If you hit OOM in the video encoder, lowering `--video_canvas_size` is usually the fastest fix.

## Optimization args

### `--epochs`

- Type: `int`
- Default: `100`

### `--val_interval`

- Type: `int`
- Default: `5`
- Meaning: run validation every N epochs

### `--batch_size`

- Type: `int`
- Default: `1`

### `--lr`

- Type: `float`
- Default: `1e-4`

### `--weight_decay`

- Type: `float`
- Default: `1e-5`

### `--grad_clip`

- Type: `float`
- Default: `1.0`

### `--num_workers`

- Type: `int`
- Default: value from `config.NUM_WORKERS`

### `--log_interval`

- Type: `int`
- Default: `1`
- Meaning: print per-iteration progress every N batches

## Model args

### `--feature_dim`

- Type: `int`
- Default: `256`
- Meaning: shared internal latent channel size in the conditional velocity field model

### `--n_heads`

- Type: `int`
- Default: `8`

### `--num_layers`

- Type: `int`
- Default: `4`
- Meaning: number of transformer layers in the conditioning and decoding stack

### `--dropout`

- Type: `float`
- Default: `0.1`

### `--video_chunk_size`

- Type: `int`
- Default: `32`
- Meaning: maximum number of frames processed at once by the raw-video Conv3D encoder

Smaller values reduce peak GPU memory and are the most direct fix for OOM in the video path.

### `--amp` / `--no_amp`

- Default: `--amp`
- Meaning: enable or disable CUDA automatic mixed precision for training, validation, and final metric evaluation

Keep `--amp` enabled unless you are debugging numerical instability.

### `--clip_sample`

- Type: `float`
- Default: `5.0`
- Meaning: clamp range applied during ODE integration and target reconstruction diagnostics

## Flow matching args

### `--solver`

- Choices: `euler`, `heun`
- Default: `heun`
- Meaning: ODE solver used at generation time

`euler` is simpler and faster. `heun` is usually a better default for evaluation quality.

### `--solver_steps`

- Type: `int`
- Default: `40`
- Meaning: number of ODE integration steps during sampling

Lower values run faster. Higher values may improve generation quality.

### `--time_embed_scale`

- Type: `float`
- Default: `1000.0`
- Meaning: scale applied to continuous times before sinusoidal embedding

This keeps the continuous-time embedding numerically similar to the existing diffusion-style timestep embedding.

### `--time_sampling`

- Choices: `uniform`, `beta`
- Default: `uniform`
- Meaning: how training times $t \in [0, 1]$ are sampled

### `--time_beta_alpha`

- Type: `float`
- Default: `1.5`
- Meaning: alpha parameter for Beta time sampling when `--time_sampling beta`

### `--time_beta_beta`

- Type: `float`
- Default: `1.5`
- Meaning: beta parameter for Beta time sampling when `--time_sampling beta`

## Classifier-free guidance args

### `--guidance_scale`

- Type: `float`
- Default: `1.5`
- Meaning: conditional guidance scale used during ODE sampling

### `--audio_drop_prob`

- Type: `float`
- Default: `0.2`

### `--video_drop_prob`

- Type: `float`
- Default: `0.2`

### `--latent_drop_prob`

- Type: `float`
- Default: `0.2`

These probabilities control condition dropout during training for classifier-free guidance.

## Loss args

### `--flow_weight`

- Type: `float`
- Default: `1.0`
- Meaning: weight for the primary flow-matching supervision term

### `--reconstruction_weight`

- Type: `float`
- Default: `0.0`
- Meaning: optional weight for reconstructed target diagnostics used as an auxiliary training regularizer

### `--velocity_weight`

- Type: `float`
- Default: `0.0`
- Meaning: optional weight for frame-to-frame motion consistency in reconstructed targets

The current recommended starting point is to keep training dominated by the flow objective and only add reconstruction/velocity regularizers after basic stability is confirmed.

## Checkpoint args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/motion_flow_matching_port`

### `--resume_checkpoint`

- Type: `str`
- Default: `None`

## Smoke test recommendation

Use this command before longer runs:

```bash
python train_motion_flow_matching.py \
  --epochs 1 \
  --val_interval 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_sequences 1 \
  --train_val_same \
  --video_canvas_size 128 \
  --feature_dim 128 \
  --num_layers 2 \
  --solver_steps 8 \
  --checkpoint_dir output/motion_flow_matching_smoke
```

## Notes

- This first implementation predicts the full `118`-d FLAME target.
- Metric evaluation uses the same shared `evaluate_motion_metrics()` path as the other benchmark models.
- Validation loss is deterministic and uses a midpoint interpolation path for more stable reporting.
- Final metrics are based on sampled sequences from the configured ODE solver, not on the midpoint training objective alone.