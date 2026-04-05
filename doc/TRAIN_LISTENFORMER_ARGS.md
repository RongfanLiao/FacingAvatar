# train_listenformer.py Args

This document records the command-line arguments for `train_listenformer.py` and explains the current training and evaluation behavior of the first ListenFormer-style port.

Current ListenFormer training in this repository uses the shared LookingFace benchmark contract:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- FLAME content targets from `flame_param.npz`

The current port is intentionally scoped to the `content` target only.

The training entrypoint is:

```bash
python train_listenformer.py [args...]
```

## Typical commands

### Train on predefined splits

```bash
python train_listenformer.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/listenformer_content \
  --epochs 20 \
  --batch_size 1 \
  --val_interval 2 \
  --video_canvas_size 224
```

### Evaluate only from a checkpoint

```bash
python train_listenformer.py \
  --eval_only \
  --resume_checkpoint checkpoints/listenformer_content/best.pt \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --batch_size 1
```

### Minimal smoke test

```bash
python train_listenformer.py \
  --epochs 1 \
  --val_interval 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_sequences 1 \
  --train_val_same \
  --video_canvas_size 224 \
  --checkpoint_dir output/listenformer_smoke
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

Used only when `--predefined_splits_dir` is not supplied.

### `--max_sequences`

- Type: `int`
- Default: `0`
- Meaning: limit the number of train and validation sequences after split selection

Useful for smoke tests and memory checks.

### `--train_val_same`

- Flag
- Default: `False`
- Meaning: force validation to use the same sequence IDs as training

Useful for debugging only.

### `--eval_only`

- Flag
- Default: `False`
- Meaning: skip training and only run final metric evaluation

### `--video_canvas_size`

- Type: `int`
- Default: `512`
- Meaning: square canvas size used for raw video frame fitting

This directly affects memory use.

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

Because the model uses raw video frames, practical batch sizes are usually small.

### `--lr`

- Type: `float`
- Default: `1e-4`

### `--weight_decay`

- Type: `float`
- Default: `1e-5`

### `--num_workers`

- Type: `int`
- Default: value from `config.NUM_WORKERS`

Use `0` for debugging or smoke tests.

### `--log_interval`

- Type: `int`
- Default: `1`
- Meaning: print per-iteration progress every N batches

Logs include timestamp, epoch progress, elapsed time, ETA, and loss breakdown.

## Model args

### `--feature_dim`

- Type: `int`
- Default: `256`
- Meaning: hidden size used throughout the adapted ListenFormer model

### `--n_heads`

- Type: `int`
- Default: `8`
- Meaning: attention heads used in the audio encoder, cross-attention, and decoder

### `--num_encoder_layers`

- Type: `int`
- Default: `3`
- Meaning: number of transformer encoder layers for the audio branch

### `--num_decoder_layers`

- Type: `int`
- Default: `3`
- Meaning: number of transformer decoder layers for the autoregressive target branch

### `--dropout`

- Type: `float`
- Default: `0.1`

### `--video_chunk_size`

- Type: `int`
- Default: `8`
- Meaning: temporal chunk size used by the raw-video encoder to control peak memory use

Lower values reduce memory at the cost of more encoder passes.

## Loss args

### `--vel_weight`

- Type: `float`
- Default: `0.5`
- Meaning: weight applied to the velocity consistency term

The main reconstruction loss is always the masked FLAME content loss.

## Checkpoint args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/listenformer_content`
- Meaning: directory used for `best.pt`, `last.pt`, and `metrics.json`

### `--resume_checkpoint`

- Type: `str`
- Default: `None`
- Meaning: load a saved checkpoint or raw state dict before training or evaluation

## Inference support

Single-sequence FLAME export is supported through the shared `inference.py` entrypoint.

Example:

```bash
python inference.py \
  --seq_id 0012 \
  --checkpoint checkpoints/listenformer_content/best.pt \
  --output_dir output/listenformer_pred \
  --gen_vis
```

This writes the same FLAME `.npz` contract used by the motion TransVAE and REGNN inference paths, so downstream video generation code can reuse it directly.
