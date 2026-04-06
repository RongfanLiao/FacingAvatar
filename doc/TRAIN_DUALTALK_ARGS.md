# train_dualtalk.py Args

This document records the command-line arguments for `train_dualtalk.py` and explains the current training and evaluation behavior of the first DualTalk-style port.

Current DualTalk training in this repository uses the shared LookingFace benchmark contract:

- wav2vec audio features from `data/wav2vec_embeddings`
- raw left video frames loaded from the LookingFace videos
- FLAME content targets from `flame_param.npz`

The current port is intentionally scoped to the `content` target only.

The training entrypoint is:

```bash
python train_dualtalk.py [args...]
```

## Typical commands

### Train on predefined splits

```bash
python train_dualtalk.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --checkpoint_dir checkpoints/dualtalk_content \
  --epochs 20 \
  --batch_size 2 \
  --val_interval 2 \
  --video_canvas_size 224
```

### Evaluate only from a checkpoint

```bash
python train_dualtalk.py \
  --eval_only \
  --resume_checkpoint checkpoints/dualtalk_content/best.pt \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --batch_size 1
```

### Minimal smoke test

```bash
python train_dualtalk.py \
  --epochs 1 \
  --val_interval 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_sequences 1 \
  --train_val_same \
  --video_canvas_size 128 \
  --feature_dim 128 \
  --interaction_layers 2 \
  --checkpoint_dir output/dualtalk_smoke
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
- Default: `224`
- Meaning: square canvas size used for raw video frame fitting

Recommended starting values:

- `128` or `224` for smoke tests
- `224` as the current default and recommended starting point for routine training
- `400` or `512` only after confirming memory headroom for your chosen batch size

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

### `--feature_dim`

- Type: `int`
- Default: `256`
- Meaning: hidden size of the shared temporal feature stream

### `--n_heads`

- Type: `int`
- Default: `8`

### `--interaction_layers`

- Type: `int`
- Default: `3`
- Meaning: number of transformer encoder layers in the interaction module

### `--decoder_layers`

- Type: `int`
- Default: `1`
- Meaning: number of transformer decoder layers in the synthesis module

### `--dropout`

- Type: `float`
- Default: `0.1`

### `--video_chunk_size`

- Type: `int`
- Default: `8`
- Meaning: temporal chunk size used by the raw-video encoder to control memory use

### `--modulation_factor`

- Type: `float`
- Default: `0.1`
- Meaning: strength of the adaptive modulation term in the synthesis head

## Loss args

### `--vel_weight`

- Type: `float`
- Default: `0.5`
- Meaning: weight applied to the velocity consistency term

## Checkpoint args

### `--checkpoint_dir`

- Type: `str`
- Default: `checkpoints/dualtalk_content`

### `--resume_checkpoint`

- Type: `str`
- Default: `None`

## Inference support

Single-sequence FLAME export is supported through the shared `inference.py` entrypoint.

Example:

```bash
python inference.py \
  --seq_id 0012 \
  --checkpoint checkpoints/dualtalk_content/best.pt \
  --output_dir output/dualtalk_pred \
  --gen_vis
```

This writes the same FLAME `.npz` contract used by the motion TransVAE, REGNN, ListenFormer, and Dyadic DIM inference paths.