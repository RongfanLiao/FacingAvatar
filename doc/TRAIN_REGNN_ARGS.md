# train_regnn.py Args

This document records the command-line arguments for `train_regnn.py` and explains how they affect training, evaluation, and generation behavior.

Current REGNN training in this repository uses the same aligned input contract as the current `motion_diffusion` and `motion_transvae` ports:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- FLAME targets from `flame_param.npz`

The training entrypoint is:

```bash
python train_regnn.py [args...]
```

## Typical commands

### Train on predefined splits

```bash
python train_regnn.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/regnn_lookingface \
  --epochs 20 \
  --batch_size 1 \
  --val_period 2 \
  --video_canvas_size 224
```

### Evaluate only from a checkpoint

```bash
python train_regnn.py \
  --eval_only \
  --resume_checkpoint checkpoints/regnn_lookingface/best.pt \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --batch_size 1
```

### Minimal smoke test

```bash
python train_regnn.py \
  --epochs 1 \
  --val_period 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_sequences 1 \
  --train_val_same \
  --checkpoint_dir output/regnn_smoke
```

## Data and split args

### `--predefined_splits_dir`

- Type: `str`
- Default: `None`
- Meaning: directory containing `train.json`, `valid.json`, and optionally `test.json`
- Current use: recommended for LookingFace benchmarking

If provided, the script reads predefined splits from that directory. If `test.json` exists, it becomes the evaluation split; otherwise `valid.json` is used.

### `--split_path`

- Type: `str`
- Default: `data/benchmark_split.json`
- Meaning: output path for the fallback random train/validation split

This is used only when `--predefined_splits_dir` is not supplied.

### `--max_sequences`

- Type: `int`
- Default: `0`
- Meaning: limit the number of train and validation sequences after split selection

Useful for smoke tests, debugging, and memory checks.

### `--train_val_same`

- Flag
- Default: `False`
- Meaning: force validation to use the same sequence IDs as training

Useful for convergence checks or tiny smoke runs. It is not appropriate for real benchmark reporting.

### `--eval_only`

- Flag
- Default: `False`
- Meaning: skip gradient-based training and only run checkpoint loading plus final metric evaluation

Use this when you already have a saved checkpoint and want to produce metric outputs without another training pass.

### `--video_canvas_size`

- Type: `int`
- Default: `512`
- Meaning: spatial size used when fitting raw left video frames into a square canvas

This directly affects memory use because REGNN now loads and encodes raw video online.

Recommended starting values:

- `224` for smoke tests and constrained GPUs
- `400` for a middle ground
- `512` for the default full setting

## Optimization args

### `--epochs`

- Type: `int`
- Default: `100`
- Meaning: number of training epochs

### `--val_period`

- Type: `int`
- Default: `5`
- Meaning: run validation every N epochs

Lower values give faster feedback but increase runtime.

### `--batch_size`

- Type: `int`
- Default: `4`
- Meaning: number of sequences per batch

Because raw video is decoded and encoded online, practical batch sizes are usually much smaller than embedding-only training.

### `--lr`

- Type: `float`
- Default: `1e-4`
- Meaning: Adam learning rate

### `--weight_decay`

- Type: `float`
- Default: `5e-4`
- Meaning: Adam weight decay

### `--grad_clip`

- Type: `float`
- Default: `1.0`
- Meaning: gradient clipping threshold applied after backpropagation

Useful when the invertible graph or raw-video encoder makes training unstable.

### `--num_workers`

- Type: `int`
- Default: value from `config.NUM_WORKERS`
- Meaning: dataloader worker count

Use `0` for debugging and smoke tests. Increase this only after confirming raw-video decoding is stable.

### `--log_interval`

- Type: `int`
- Default: `1`
- Meaning: print per-iteration training and validation progress every N batches

The current REGNN training path now uses the same timestamped progress style as the diffusion and TransVAE training loops. Each emitted line includes:

- timestamp
- train or validation label
- epoch and batch progress
- elapsed time and ETA
- detailed loss breakdown

Use larger values such as `10` or `20` to reduce console verbosity during longer runs.

## Model architecture args

### `--fused_dim`

- Type: `int`
- Default: `64`
- Meaning: hidden feature size produced by the perceptual fusion stage before the graph modules

Smaller values reduce memory and compute. Larger values increase model capacity.

### `--num_frames`

- Type: `int`
- Default: `50`
- Meaning: fixed clip length used by REGNN during training and chunked inference

This matches the baseline REGNN setup. During training, the loader builds a fixed window of this size from each padded full sequence. During evaluation and prediction, long sequences are processed in chunks of this length.

### `--edge_dim`

- Type: `int`
- Default: `8`
- Meaning: number of edge channels in the cognitive graph builder

### `--neighbors`

- Type: `int`
- Default: `6`
- Meaning: top-k neighborhood size used when sparsifying graph edges

### `--layers`

- Type: `int`
- Default: `2`
- Meaning: number of graph layers in the invertible motor processor

### `--act_type`

- Choices: `ELU`, `ReLU`, `GeLU`, `None`
- Default: `ELU`
- Meaning: activation used inside graph attention blocks

`ELU` is the current default and closest to the current local REGNN port behavior.

### `--dropout`

- Type: `float`
- Default: `0.1`
- Meaning: retained for API consistency when constructing the REGNN model

In the current raw-video REGNN path, the perceptual stage uses the baseline-style speaker encoder and this argument does not materially change behavior.

### `--target_variant`

- Choices: `content`, `motion58`
- Default: `content`
- Meaning: which FLAME target representation REGNN predicts

`content`:

- predicts the reduced FLAME content target used by this repo
- shape is currently 112 per frame

`motion58`:

- predicts the reduced 58-d motion target

Use `content` unless you are explicitly benchmarking the 58-d representation.

## REGNN objective args

### `--latent_weight`

- Type: `float`
- Default: `1.0`
- Meaning: weight for the main latent matching objective between speaker and listener graph features

This is the primary training loss in the current REGNN port.

### `--neighbor_pattern`

- Choices: `all`, `nearest`
- Default: `all`
- Meaning: reduction strategy used when matching speaker features to listener features in the latent loss

`all` is the current default and matches the existing local training setup.

### `--no_mid_loss`

- Flag
- Default: `False`
- Meaning: disable the auxiliary mid-loss term

If set, REGNN trains without the extra feature-centering style loss.

### `--mid_weight`

- Type: `float`
- Default: `1.0`
- Meaning: multiplier for the auxiliary mid-loss term

### `--logdet_weight`

- Type: `float`
- Default: `0.0`
- Meaning: weight for the motor processor log-determinant term

The default keeps this disabled.

### `--reconstruction_weight`

- Type: `float`
- Default: `0.0`
- Meaning: weight for masked FLAME reconstruction loss on predicted target sequences

The default local REGNN training is latent-match focused rather than reconstruction focused.

### `--vel_weight`

- Type: `float`
- Default: `0.0`
- Meaning: weight for velocity consistency loss on FLAME target trajectories

Enable this only if you want extra temporal smoothing pressure.

## Checkpoint and resume args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/regnn_port`
- Meaning: directory used for `best.pt`, `last.pt`, and `metrics.json`

### `--resume_checkpoint`

- Type: `str`
- Default: `None`
- Meaning: load a previous checkpoint or raw model state before training or evaluation

If used with training enabled, the script resumes from the stored epoch and tries to restore the optimizer. If used with `--eval_only`, the checkpoint is loaded and then metric evaluation runs.

## Inference and evaluation behavior

REGNN now supports the shared `inference.py` entrypoint for single-sequence prediction and FLAME export. The current repository has two relevant entrypoints:

- `train_regnn.py` for training and benchmark evaluation
- `inference.py` for single-sequence FLAME prediction export

### Training mode

- default behavior when `--eval_only` is not set
- trains for `--epochs`
- validates every `--val_period`
- saves `last.pt` and `best.pt`
- runs final metric evaluation after training

### Evaluation-only mode

- enabled with `--eval_only`
- skips gradient updates
- loads a checkpoint from `--resume_checkpoint` or, if absent, tries `best.pt` in `--checkpoint_dir`
- runs final metric evaluation and writes `metrics.json`

### Single-sequence inference mode

Use `inference.py` with a REGNN checkpoint:

```bash
python inference.py \
  --seq_id 0012 \
  --checkpoint checkpoints/regnn_lookingface/best.pt \
  --output_dir output/regnn_pred \
  --gen_vis
```

This writes the same FLAME `.npz` contract used by the motion TransVAE inference path, so downstream video generation and visualization code can reuse it directly.

### Sequence prediction behavior

Inside the REGNN model, prediction uses chunked sequence inference:

- input sequences are split into windows of `--num_frames`
- short tail clips are zero-padded
- the model predicts FLAME sequences in the selected `--target_variant`

### `--noise_threshold`

- Type: `float`
- Default: `0.0`
- Meaning: optional cap used when sampling noise during the model's internal generation path

In the current implementation:

- values `<= 0` are treated as disabled
- positive values enable capped-noise sampling in the REGNN speaker feature space

This is mainly an inference-generation knob rather than a core optimizer setting.

When using `inference.py`, the corresponding flag is `--regnn_noise_threshold`.

## Practical recommendations

For smoke tests:

- `--batch_size 1`
- `--num_workers 0`
- `--max_sequences 1`
- `--train_val_same`
- `--video_canvas_size 224`

For a more realistic initial run:

- `--predefined_splits_dir data/LookingFace/dataset_splits`
- `--batch_size 1`
- `--video_canvas_size 224` or `400`
- `--val_period 1` or `2`

For evaluation only:

- `--eval_only`
- `--resume_checkpoint path/to/best.pt`
- keep `--target_variant` the same as training
