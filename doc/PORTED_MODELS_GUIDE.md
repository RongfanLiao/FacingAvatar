# Ported Model Benchmark Guide

This document describes the imported benchmark models currently ported into this repository, how they consume LookingFace data, how training and evaluation flow through the codebase, and how to run each training script.

## Scope

The following baseline models have been ported for benchmarking on the LookingFace dataset in this repository:

1. `motion_transvae`
2. `motion_diffusion`
3. `REGNN`

These ports are not byte-for-byte copies of `../baseline_react2025`. They preserve the main modeling ideas where practical, but they are adapted to the data contract available in this repository:

1. left-side audio embedding
2. left-side video embedding
3. right-side FLAME targets

The shared benchmark code lives under `benchmark/`.

## Files

Core shared files:

1. `benchmark/lookingface.py`: shared dataset adapter and split builder
2. `benchmark/targets.py`: FLAME target conversion helpers
3. `manifest.py`: paired sample discovery and manifest loading

Ported model modules:

1. `benchmark/motion_transvae.py`
2. `benchmark/motion_diffusion.py`
3. `benchmark/regnn.py`

Training entrypoints:

1. `train_motion_transvae.py`
2. `train_motion_diffusion.py`
3. `train_regnn.py`

Utility script:

1. `scripts/compare_benchmarks.py`

## LookingFace Data Input Format

The benchmark ports assume a paired sample with the following conceptual structure:

1. `video_left`: source or speaker-side video
2. `video_right`: target or listener-side video
3. `video_right_dir`: directory containing right-side FLAME parameter files

In this repository, the shared benchmark adapter resolves each usable sequence into:

1. `left_mp4`
2. `right_mp4`
3. `flame_npz`
4. `left_audio_emb`
5. `left_video_emb`
6. optional `right_video_emb`

The manifest entry is normalized by `manifest.py`, and the benchmark dataset then expects the following files to exist:

1. `data/audio_embeddings/<seq_id>_left_whisper.npy`
2. `data/video_embeddings/<seq_id>_left.npy`
3. `LookingFace/.../<seq_id>_left.mp4`
4. `LookingFace/.../<seq_id>_right.mp4`
5. `LookingFace/.../<seq_id>_right/flame_param.npz`

### Loaded Tensors

`benchmark/lookingface.py` exposes batches with these main keys:

1. `left_audio_feat`: padded temporal tensor shaped roughly `[B, T, 1280]`
2. `left_video_feat`: fixed tensor shaped `[B, 3584]`
3. `flame_target_118`: padded target tensor `[B, T, 118]`
4. `flame_target_58`: padded target tensor `[B, T, 58]` when requested
5. `flame_target_content`: padded target tensor `[B, T, 112]` when requested
6. `lengths`: valid frame counts per sample
7. `padding_mask`: padded-frame mask shaped `[B, T]`

### Temporal Alignment

Left audio embeddings are aligned to FLAME target length in `benchmark/lookingface.py`:

1. the audio duration is read from `data/audio/<seq_id>_left.wav`
2. valid Whisper frames are estimated from chunk duration and model frame count
3. the left audio embedding is truncated to valid length
4. the embedding is linearly interpolated to match the number of FLAME frames

This gives each benchmark model a time-aligned left-audio sequence and right-target sequence.

## Target Variants

Target conversion is implemented in `benchmark/targets.py`.

### Native FLAME Target

`flame_target_118` concatenates the repository-native FLAME keys in this order:

1. `expr` (100)
2. `jaw_pose` (3)
3. `rotation` (3)
4. `neck_pose` (3)
5. `eyes_pose` (6)
6. `translation` (3)

Total dimension: `118`

### REACT-style Reduced Motion Target

`flame_target_58` contains:

1. `expr[:52]`
2. `rotation`
3. `translation`

Total dimension: `58`

This variant exists mainly for compatibility with legacy reduced-motion assumptions from the baseline code.

### Content Target

`flame_target_content` is the default benchmark target for new runs. It contains:

1. `expr` (100)
2. `jaw_pose` (3)
3. `neck_pose` (3)
4. `eyes_pose` (6)

Total dimension: `112`

This target is preferred because it emphasizes the face-content components that matter most for the current benchmark.

## Shared Training and Evaluation Data Flow

All three benchmark training scripts follow the same high-level flow:

1. build or load a reproducible train/validation split with `build_benchmark_split`
2. create `LookingFaceBenchmarkDataset` instances
3. collate batches with `collate_benchmark_batch`
4. train the model for some number of epochs
5. save `last.pt` and `best.pt` checkpoints
6. load the best checkpoint if it exists
7. run validation metrics
8. save metrics to `metrics.json`

### Split Construction

Split generation happens through `benchmark/lookingface.py`:

1. usable sequences are discovered from the manifest
2. sequences missing required audio, video, or FLAME files are filtered out
3. the remaining IDs are split according to `TRAIN_RATIO` and `SPLIT_SEED`
4. the split can be written to `data/benchmark_split.json`

### Checkpoint Convention

Each training script writes into its own checkpoint directory, typically:

1. `best.pt`
2. `last.pt`
3. `metrics.json`

## Model-specific Training Data Flow

### 1. motion_transvae

Implementation:

1. `benchmark/motion_transvae.py`
2. `train_motion_transvae.py`

Data flow:

1. `left_audio_feat` and `left_video_feat` are loaded from the shared benchmark adapter
2. the speaker context encoder fuses temporal audio tokens with a static left-video token
3. the fused temporal features are passed into a latent VAE
4. a transformer decoder predicts the target motion sequence
5. the loss combines:
   1. masked reconstruction loss
   2. KL divergence loss
   3. diversity penalty during training

Training target:

1. default: `flame_target_content`
2. optional: `flame_target_58`

### 2. motion_diffusion

Implementation:

1. `benchmark/motion_diffusion.py`
2. `train_motion_diffusion.py`

Data flow:

1. `left_audio_feat` and `left_video_feat` are loaded from the shared benchmark adapter
2. the target motion sequence is corrupted with diffusion noise during training
3. a transformer decoder denoiser receives:
   1. noisy target queries
   2. timestep embedding
   3. left-audio memory
   4. left-video memory
   5. pooled latent memory token
4. the model predicts the clean target sequence
5. sampling uses DDIM-style reverse diffusion with configurable timestep spacing

Training target:

1. default: `flame_target_content`
2. optional: `flame_target_58`

Training loss:

1. masked reconstruction loss
2. optional velocity loss

Inference details:

1. classifier-free guidance is supported through conditional dropout and `guidance_scale`
2. DDIM sampling parameters are exposed in the CLI

### 3. REGNN

Implementation:

1. `benchmark/regnn.py`
2. `train_regnn.py`

Data flow:

1. a fixed-length clip is sampled from each padded full sequence for training
2. `LookingFacePercepProcessor` fuses left-audio sequence and left-video embedding into a temporal feature stream
3. `REGNNCognitiveProcessor` converts fused features into node-wise speaker graph features and learned graph edges
4. `LipschitzGraph` acts as the motor processor
5. training primarily aligns speaker graph features with listener graph features produced from the target clip
6. decoded sequence reconstruction is available as an auxiliary term
7. full-sequence evaluation is performed by sliding over the sequence in `num_frames` chunks and stitching predictions back together

Training target:

1. default: `flame_target_content`
2. optional: `flame_target_58`

Training objective:

The more faithful default objective in the current port is:

1. latent graph matching loss as the primary term
2. optional mid-loss regularizer
3. optional logdet term
4. optional decoded reconstruction auxiliary
5. optional velocity auxiliary

Note that the baseline REACT training uses multi-candidate targets more heavily than the current LookingFace pairing setup. This port keeps the graph structure and the latent objective shape, but it is still adapted to a single paired target per sample in this repository.

## Metric Computation

Shared metric computation is driven by `evaluate_motion_metrics` in `benchmark/motion_transvae.py`.

`motion_transvae` uses it directly.

`motion_diffusion` and `REGNN` wrap their sampling or prediction functions so they can reuse the same metric stack.

### Reported Metrics

The current benchmark evaluation computes:

1. `mae`
2. `rmse`
3. `frcorr`
4. `frdist`
5. `delta_mae`
6. `delta_rmse`
7. `fid_delta_fm`
8. `snd`

It also reports per-group metrics depending on the target variant.

For `content`:

1. `mae_expr`, `rmse_expr`
2. `mae_jaw`, `rmse_jaw`
3. `mae_neck`, `rmse_neck`
4. `mae_eyes`, `rmse_eyes`

For `motion58`:

1. `mae_expr`, `rmse_expr`
2. `mae_rot`, `rmse_rot`
3. `mae_tran`, `rmse_tran`

### Metric Inputs

Metric evaluation always compares predicted temporal target sequences against ground-truth temporal target sequences after trimming each sample to its valid frame length.

### Metric Output File

Each training script saves the final validation metrics to:

1. `<checkpoint_dir>/metrics.json`

The JSON also stores:

1. `target_variant`

## Training Script Usage

The examples below assume the configured Python environment:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python
```

### motion_transvae

Script:

1. `train_motion_transvae.py`

Example training command:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python train_motion_transvae.py \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-5 \
  --target_variant content \
  --checkpoint_dir checkpoints/motion_transvae_content
```

Example eval-only command:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python train_motion_transvae.py \
  --eval_only \
  --batch_size 2 \
  --num_workers 0 \
  --target_variant content \
  --checkpoint_dir checkpoints/motion_transvae_content
```

Important arguments:

1. `--target_variant {content,motion58}`
2. `--feature_dim`
3. `--n_heads`
4. `--num_layers`
5. `--dropout`
6. `--div_p`
7. `--train_val_same`
8. `--eval_only`

### motion_diffusion

Script:

1. `train_motion_diffusion.py`

Example training command:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python train_motion_diffusion.py \
  --epochs 100 \
  --batch_size 4 \
  --target_variant content \
  --checkpoint_dir checkpoints/motion_diffusion_content
```

Example smoke-test command:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python train_motion_diffusion.py \
  --epochs 1 \
  --val_period 1 \
  --batch_size 2 \
  --num_workers 0 \
  --target_variant content \
  --feature_dim 64 \
  --n_heads 4 \
  --num_layers 2 \
  --train_timesteps 16 \
  --inference_timesteps 4 \
  --guidance_scale 1.5 \
  --max_sequences 2 \
  --train_val_same \
  --checkpoint_dir checkpoints/motion_diffusion_smoke_refined
```

Important arguments:

1. `--target_variant {content,motion58}`
2. `--train_timesteps`
3. `--inference_timesteps`
4. `--guidance_scale`
5. `--audio_drop_prob`
6. `--video_drop_prob`
7. `--latent_drop_prob`
8. `--timestep_spacing`
9. `--ddim_eta`
10. `--max_sequences`
11. `--eval_only`

### REGNN

Script:

1. `train_regnn.py`

Example training command:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python train_regnn.py \
  --epochs 100 \
  --batch_size 4 \
  --target_variant content \
  --checkpoint_dir checkpoints/regnn_content
```

Example faithful-objective smoke-test command:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python train_regnn.py \
  --epochs 1 \
  --val_period 1 \
  --batch_size 2 \
  --num_workers 0 \
  --target_variant content \
  --fused_dim 32 \
  --num_frames 16 \
  --edge_dim 4 \
  --neighbors 4 \
  --layers 1 \
  --max_sequences 2 \
  --train_val_same \
  --checkpoint_dir checkpoints/regnn_smoke_faithful
```

Important arguments:

1. `--target_variant {content,motion58}`
2. `--num_frames`
3. `--edge_dim`
4. `--neighbors`
5. `--layers`
6. `--neighbor_pattern {all,nearest}`
7. `--no_mid_loss`
8. `--mid_weight`
9. `--logdet_weight`
10. `--reconstruction_weight`
11. `--vel_weight`
12. `--max_sequences`
13. `--eval_only`

## Recommended Defaults

For new LookingFace benchmark runs, use:

1. `--target_variant content`
2. a dedicated checkpoint directory per model and run
3. `--num_workers 0` for first-pass debugging if anything fails in the dataloader

Suggested pattern:

1. run a small smoke test first with `--max_sequences 2` or a small batch size where available
2. confirm `metrics.json` is written
3. scale up to the real run

## Output Artifacts

After a successful run, the main artifacts are:

1. `best.pt`
2. `last.pt`
3. `metrics.json`

These are written inside the script-specific checkpoint directory.

## Benchmark Comparison Script

You can compare multiple benchmark runs side by side with:

1. `scripts/compare_benchmarks.py`

The script accepts checkpoint directories or direct `metrics.json` paths.

Example:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python scripts/compare_benchmarks.py \
  checkpoints/motion_diffusion_smoke_refined \
  checkpoints/regnn_smoke_faithful
```

Example with explicit labels and markdown output:

```bash
/home/r/rongfan/micromamba/envs/qwen_vl/bin/python scripts/compare_benchmarks.py \
  checkpoints/motion_transvae_content \
  checkpoints/motion_diffusion_smoke_refined \
  checkpoints/regnn_smoke_faithful \
  --labels transvae diffusion regnn \
  --format markdown
```

Useful options:

1. `--labels`
2. `--metrics`
3. `--show_all`
4. `--format {plain,markdown,csv}`
5. `--output`
6. `--precision`

## Limitations and Adaptation Notes

1. The imported models are adapted to the data layout available in this repository, not the exact REACT dataset contract.
2. `content` is the preferred benchmark target for new runs.
3. `motion58` exists for compatibility and comparison, but it is not the preferred target for current LookingFace evaluation.
4. `REGNN` remains the most structurally adapted port because the original baseline depends more heavily on multi-candidate listener targets and a different feature extraction setup.

## Summary

If you only need the shortest operational guidance:

1. use `content` target by default
2. start with the shared split and `LookingFaceBenchmarkDataset`
3. train with one of:
   1. `train_motion_transvae.py`
   2. `train_motion_diffusion.py`
   3. `train_regnn.py`
4. read `<checkpoint_dir>/metrics.json` after training or eval-only runs
