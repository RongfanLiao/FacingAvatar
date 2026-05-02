# train_motion_transvae.py Args

This document records the command-line arguments for `train_motion_transvae.py` and explains the current training and evaluation behavior of the TransVAE baseline.

Current TransVAE training in this repository uses the shared LookingFace benchmark contract:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- FLAME targets from `flame_param.npz`

The training entrypoint is:

```bash
python train_motion_transvae.py [args...]
```

## Typical commands

### Train on predefined splits

```bash
python train_motion_transvae.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_transvae_lookingface \
  --epochs 20 \
  --batch_size 2 \
  --val_interval 2 \
  --video_canvas_size 224
```

### Evaluate only from a checkpoint

```bash
python train_motion_transvae.py \
  --eval_only \
  --resume_checkpoint checkpoints/motion_transvae_lookingface/best.pt \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --batch_size 1
```

### Minimal smoke test

```bash
python train_motion_transvae.py \
  --epochs 1 \
  --val_interval 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_eval_samples 1 \
  --train_val_same \
  --video_canvas_size 128 \
  --feature_dim 128 \
  --checkpoint_dir output/motion_transvae_smoke
```

## Data and split args

### `--predefined_splits_dir`

- Type: `str`
- Default: `None`
- Meaning: directory containing `train.json`, `valid.json`, and optionally `test.json`

Recommended for LookingFace benchmarking.

If `test.json` exists, the script uses it as the evaluation split and labels the final metrics as `test_*`. Otherwise it uses `valid.json` and labels metrics as `valid_*`.

### `--split_path`

- Type: `str`
- Default: `data/benchmark_split.json`
- Meaning: output path for the fallback random train/validation split

This is used only when `--predefined_splits_dir` is not supplied.

### `--train_val_same`

- Flag
- Default: `False`
- Meaning: force validation to use the same sequence IDs as training

Useful only for debugging or convergence checks.

### `--eval_only`

- Flag
- Default: `False`
- Meaning: skip training and only run evaluation / metric generation

Behavior:

- if `--resume_checkpoint` is provided, that checkpoint is loaded for evaluation
- otherwise, if `best.pt` exists under `--checkpoint_dir`, that checkpoint is used

The current TransVAE benchmark path predicts the repository's 112-d FLAME content target.

### `--video_canvas_size`

- Type: `int`
- Default: `512`
- Meaning: square canvas size used for raw left video frame fitting

This affects memory use directly. Smaller values reduce GPU memory and data-loading cost.

Recommended starting values:

- `128` for smoke tests
- `224` for practical debugging runs
- `400` or `512` for more faithful training if memory allows

### `--documentary`

- Flag
- Default: `False`
- Meaning: use the documentary manifest instead of the standard LookingFace manifest

Only use this when you intentionally want the documentary dataset path.

### `--max_eval_samples`

- Type: `int`
- Default: `None`
- Meaning: truncate the evaluation split to the first N sequences

Useful for quick validation checks and smoke tests.

## Optimization args

### `--epochs`

- Type: `int`
- Default: `100`

### `--batch_size`

- Type: `int`
- Default: `2`

Because raw video is decoded and processed online, this usually needs to stay small.

### `--lr`

- Type: `float`
- Default: `1e-5`
- Meaning: AdamW learning rate

This is intentionally conservative for the current raw-video TransVAE port.

### `--div_p`

- Type: `float`
- Default: `10.0`
- Meaning: weight applied to the diversity regularization term

This term encourages different latent samples to produce meaningfully different motion predictions for the same conditioning input.

### `--num_workers`

- Type: `int`
- Default: value from `config.NUM_WORKERS`
- Meaning: dataloader worker count

When debugging, use `0`. For training, increase cautiously because raw video decoding can become the limiting factor.

### `--val_interval`

- Type: `int`
- Default: `5`
- Meaning: run validation every N epochs

Important behavior:

- `best.pt` and `last.pt` are only written on epochs where validation runs
- if `--val_interval` is greater than `--epochs`, training can finish without writing those validation checkpoints

### `--log_interval`

- Type: `int`
- Default: `1`
- Meaning: print per-iteration progress every N batches

The logs include:

- timestamp
- train or evaluation label
- epoch and iteration progress
- elapsed time and ETA
- loss breakdown

Use larger values such as `10` or `20` when you want less console output.

## Model args

### `--feature_dim`

- Type: `int`
- Default: `128`
- Meaning: shared hidden size used by the speaker encoder, latent VAE, and decoder

Smaller values reduce memory and compute. Larger values increase capacity.

### `--n_heads`

- Type: `int`
- Default: `4`
- Meaning: number of attention heads in the TransVAE transformer blocks

### `--max_seq_len`

- Type: `int`
- Default: `1024`
- Meaning: initial maximum sequence length used to build the decoder attention mask

If a longer sequence appears at runtime, the decoder expands the mask automatically.

## Checkpoint args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/motion_transvae_documentary`
- Meaning: output directory for checkpoints and final metrics

The script may write:

- `best.pt`: best validation checkpoint by `loss_total`
- `last.pt`: most recent validation checkpoint
- `final.pt`: final training-state checkpoint at the end of the run
- `metrics.json`: final evaluation metrics for the chosen evaluation split

### `--resume_checkpoint`

- Type: `str`
- Default: `None`
- Meaning: resume training from a saved checkpoint or load a checkpoint for eval-only mode

Training behavior:

- restores model weights
- restores optimizer state when available
- restores AMP scaler state when available
- resumes from the stored epoch number

## Training behavior summary

The current training flow is:

1. Build train and evaluation splits.
2. Create a `LookingFaceBenchmarkDataset` that loads wav2vec plus raw left video frames.
3. Train `MotionOnlyTransformerVAE` with masked reconstruction loss, KL loss, and diversity loss.
4. Run validation every `--val_interval` epochs.
5. Save checkpoints.
6. Run final paired motion metrics and write `metrics.json`.

The final metric report comes from `evaluate_motion_metrics(...)` and includes:

- MAE / RMSE
- feature-wise MAE / RMSE by target group
- type-conditioned concordance correlation (`frcorr_type`)
- type-conditioned motion distance (`frdist_type`)
- temporal-delta metrics
- Fréchet-style distribution metrics where applicable

## Practical starting points

### Safe debugging run

```bash
python train_motion_transvae.py \
  --epochs 1 \
  --val_interval 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_eval_samples 1 \
  --train_val_same \
  --video_canvas_size 128
```

### More realistic baseline run

```bash
python train_motion_transvae.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/motion_transvae_lookingface \
  --epochs 50 \
  --batch_size 2 \
  --val_interval 5 \
  --feature_dim 128 \
  --n_heads 4 \
  --video_canvas_size 224
```

## Notes

- The current script always uses wav2vec audio, not Whisper.
- The current script always uses raw left video frames, not precomputed video embeddings.
- The optimizer is AdamW with the default PyTorch `weight_decay`, because this script does not expose a separate weight-decay argument.