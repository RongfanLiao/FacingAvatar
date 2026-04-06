# train_dyadic_dim.py Args

This document records the command-line arguments for `train_dyadic_dim.py` and explains the current training and evaluation behavior of the Dyadic ContinuousTransformer port.

Current Dyadic training in this repository uses the same shared benchmark contract as `motion_transvae`:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- FLAME targets from `flame_param.npz`

The training entrypoint is:

```bash
python train_dyadic_dim.py [args...]
```

## Typical commands

### Train on predefined splits

```bash
python train_dyadic_dim.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/dyadic_dim \
  --epochs 20 \
  --batch_size 1 \
  --val_interval 2 \
  --video_canvas_size 224
```

### Evaluate only from a checkpoint

```bash
python train_dyadic_dim.py \
  --eval_only \
  --resume_checkpoint checkpoints/dyadic_dim/best.pt \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --batch_size 1
```

### Minimal smoke test

```bash
python train_dyadic_dim.py \
  --epochs 1 \
  --val_interval 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_sequences 1 \
  --train_val_same \
  --video_canvas_size 224 \
  --checkpoint_dir output/dyadic_dim_smoke
```

## Data and split args

### `--predefined_splits_dir`

- Type: `str`
- Default: `None`
- Meaning: directory containing `train.json`, `valid.json`, and optionally `test.json`

Recommended for LookingFace benchmarking.

### `--split_path`

- Type: `str`
- Default: `data/benchmark_split.json`
- Meaning: output path for the fallback random train/validation split

### `--max_sequences`

- Type: `int`
- Default: `0`
- Meaning: limit the number of train and validation sequences after split selection

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
- Default: `512`
- Meaning: square canvas size used for raw video frame fitting

Recommended starting values:

- `224` for smoke tests
- `400` or `512` for larger runs

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
- Default: `2`
- Meaning: number of sequences per batch

### `--lr`

- Type: `float`
- Default: `1e-4`

### `--weight_decay`

- Type: `float`
- Default: `1e-5`

### `--num_workers`

- Type: `int`
- Default: value from `config.NUM_WORKERS`

### `--log_interval`

- Type: `int`
- Default: `1`
- Meaning: print per-iteration progress every N batches

## Model args

### `--target_variant`

- Choices: `content`, `motion58`
- Default: `content`
- Meaning: which FLAME target representation the Dyadic port predicts

### `--feature_dim`

- Type: `int`
- Default: `256`

### `--n_heads`

- Type: `int`
- Default: `8`

### `--num_encoder_layers`

- Type: `int`
- Default: `6`

### `--num_decoder_layers`

- Type: `int`
- Default: `6`

### `--dropout`

- Type: `float`
- Default: `0.1`

### `--video_chunk_size`

- Type: `int`
- Default: `8`
- Meaning: temporal chunk size used by the raw-video encoder to control memory use

## Loss args

### `--vel_weight`

- Type: `float`
- Default: `0.5`
- Meaning: weight applied to the velocity consistency term

## Checkpoint args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/dyadic_dim`

### `--resume_checkpoint`

- Type: `str`
- Default: `None`

## Inference support

Single-sequence FLAME export is supported through the shared `inference.py` entrypoint.

Example:

```bash
python inference.py \
  --seq_id 0012 \
  --checkpoint checkpoints/dyadic_dim/best.pt \
  --output_dir output/dyadic_pred \
  --gen_vis
```

This writes the same FLAME `.npz` contract used by the motion TransVAE, REGNN, and ListenFormer inference paths.