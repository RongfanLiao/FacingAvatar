# head_avatar

Audio-visual FLAME parameter prediction from pre-extracted audio and video features.

This repository predicts per-frame FLAME face parameters from:

- Whisper-Large-v3 audio encoder features
- Qwen2.5-VL-7B video features
- A transformer-based audio-video fusion model

The model predicts these dynamic FLAME parameter groups:

- expr: 100
- jaw_pose: 3
- rotation: 3
- neck_pose: 3
- eyes_pose: 6
- translation: 3

Total predicted FLAME dimension: 118

The shape vector and canonical FLAME parameters are not predicted. During inference, they are copied from the label file so the result can be reused by downstream visualization code.


## Quick start

### Train the current motion TransVAE on LookingFace

The current recommended training entrypoint for LookingFace is `train_motion_transvae.py`.

This workflow:

- uses raw left video frames plus wav2vec audio features
- trains on `data/LookingFace/dataset_splits/train.json`
- evaluates on `data/LookingFace/dataset_splits/test.json`
- writes checkpoints and final metrics to a dedicated checkpoint directory

Recommended environment:

```bash
conda activate avatar
```

If your network cannot reach Hugging Face directly, set a mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Rebuild the manifest after any dataset changes:

```bash
python manifest.py --rebuild
```

Precompute wav2vec features for LookingFace if `data/wav2vec_embeddings` is missing or incomplete:

```bash
python scripts/preprocess_lookingface.py --skip_existing --device cuda:0
```

Start training on the predefined train split and evaluate on the predefined test split:

```bash
python train_motion_transvae.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --epochs 100 \
  --lr 1e-5 \
  --checkpoint_dir checkpoints/motion_transvae_lookingface_predefined
```

Resume the same run from the last saved checkpoint:

```bash
python train_motion_transvae.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --epochs 150 \
  --lr 1e-5 \
  --checkpoint_dir checkpoints/motion_transvae_lookingface_predefined \
  --resume_checkpoint checkpoints/motion_transvae_lookingface_predefined/last.pt
```

Useful notes:

- current defaults are `--epochs 100`, `--lr 1e-5`, and `--batch_size 2`
- when resuming, `--epochs` is the total target epoch count for the run, not the number of extra epochs
- `train_motion_transvae.py` labels predefined evaluation as `test` in logs and saved metrics when `test.json` is present
- the script only uses samples that have a valid manifest entry, right-side FLAME target, right MP4, and wav2vec feature file
- final metrics are written to `checkpoints/motion_transvae_lookingface_predefined/metrics.json`

### Run draft reaction annotation on paired LookingFace clips

Use `tools/annotate_reactions_vlm.py` to generate draft HRNC reaction-type annotations for the `right` videos in `data/LookingFace`.

This workflow:

- reads paired `left_mp4` and `right_mp4` entries from `data/manifest.json`
- derives `content_type` from the LookingFace top-level folder (`documentary`, `game`, `movie`, `music`, `sports`, `talk show`)
- runs a local VLM with the prompt in `HRNC_reaction_type_vlm_prompt_template.md`
- writes one JSON result per clip to `data/reaction_annotations/`

Recommended environment:

```bash
conda activate avatar
```

Run the default 24-clip pilot:

```bash
python tools/annotate_reactions_vlm.py --pilot --device cuda:0
```

Inspect the selected pilot clips without running inference:

```bash
python tools/annotate_reactions_vlm.py --pilot --dry_run
```

Restrict the pilot to specific categories:

```bash
python tools/annotate_reactions_vlm.py --pilot --categories documentary movie sports
```

Annotate all eligible clips instead of the pilot subset:

```bash
python tools/annotate_reactions_vlm.py --all --device cuda:0
```

Useful notes:

- the current implementation supports the local `qwen2_5_vl` backend only, but keeps `--model_id` configurable
- existing annotation JSON files are skipped unless `--overwrite` is set
- malformed model output is saved as an error record for the clip instead of aborting the whole batch
- the saved JSON includes the parsed annotation, the raw model response, and provenance fields for review
- these labels are draft annotations and should be validated by a human before downstream use

### Use prepared embeddings and labels

If `data/audio_embeddings`, `data/video_embeddings`, and `data/video_labels` are already populated:

```bash
micromamba activate qwen_vl
python train.py
python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt --smooth 5 --gen_vis
```

### Run the current 5-video convergence test

The current `config.py` is set up to train and validate on the same 5 videos:

- `2920`
- `2921`
- `2922`
- `2923`
- `2924`

Start training:

```bash
micromamba activate qwen_vl
python -u train.py | tee output/train_5vid_convergence.log
```

Run inference on the same 5 videos:

```bash
for sid in 2920 2921 2922 2923 2924; do
  python inference.py \
    --seq_id "$sid" \
    --checkpoint checkpoints/best_model.pt \
    --output_dir output/convergence_5vid \
    --smooth 5 \
    --gen_vis
done
```

### Run the full preprocessing pipeline

If you only have raw MP4 files under `data/`:

```bash
micromamba activate qwen_vl
python split_audio_video.py
python encode_audio_whisper.py
python encode_video_qwen.py
python train.py
python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt --smooth 5 --gen_vis
```


## Repository layout

```text
head_avatar/
├── config.py                  # Paths, hyperparameters, split settings
├── dataset.py                 # Sequence discovery, alignment, dataloaders
├── model.py                   # Audio-video fusion model and FLAME heads
├── train.py                   # Training loop and checkpoint saving
├── train_motion_transvae.py   # Motion-only TransVAE training on LookingFace
├── inference.py               # Single-sequence prediction entry point
├── split_audio_video.py       # Raw MP4 -> WAV + video-only MP4
├── encode_audio_whisper.py    # WAV -> Whisper features
├── encode_video_qwen.py       # MP4 -> Qwen video embedding
├── benchmark/                 # LookingFace benchmark datasets and model ports
├── checkpoints/               # Saved model checkpoints
├── output/                    # Logs and inference outputs
└── data/
    ├── audio/                 # Extracted WAV files
    ├── video/                 # Extracted video-only MP4 files
    ├── audio_embeddings/      # Whisper features and projector init weights
    ├── video_embeddings/      # Qwen video embeddings
    └── video_labels/          # FLAME labels and visualization assets
```

### Core runtime files

- `config.py`: central place for paths, device selection, loss weights, and split mode
- `dataset.py`: discovers valid sequences and aligns audio features to FLAME frame counts
- `model.py`: transformer-based fusion network with separate prediction heads per FLAME group
- `train.py`: trains the model and writes checkpoints to `checkpoints/`
- `train_motion_transvae.py`: trains the motion-only TransVAE using raw left video frames and wav2vec audio
- `inference.py`: loads a checkpoint and writes predicted FLAME parameters for one sequence

### Preprocessing files

- `split_audio_video.py`: extracts 16 kHz WAV audio and video-only MP4 files from raw MP4 inputs
- `encode_audio_whisper.py`: converts WAV audio to Whisper encoder features
- `encode_video_qwen.py`: converts video to a single Qwen embedding per sequence
- `scripts/preprocess_lookingface.py`: extracts WAV audio and wav2vec features for LookingFace training

### Data and generated artifacts

- `data/audio`: extracted WAV files used for Whisper encoding
- `data/video`: extracted MP4 files used for Qwen video encoding
- `data/audio_embeddings`: `*_whisper.npy`, `*_proj.npy`, and `audio_projector_init.pt`
- `data/wav2vec_embeddings`: wav2vec features used by `train_motion_transvae.py`
- `data/video_embeddings`: one `*_left.npy` file per video sequence
- `data/video_labels`: FLAME labels expected at `<seq_id>_right/flame_param.npz`
- `checkpoints`: `best_model.pt`, `final_model.pt`, and periodic epoch checkpoints
- `output`: predicted FLAME files, visualization-ready outputs, and training logs


## Expected data layout

The code assumes the following naming convention:

- raw MP4 files live under `data/` and contain `left` in the filename
- extracted audio is saved as `data/audio/<seq_id>_left.wav`
- extracted video is saved as `data/video/<seq_id>_left.mp4`
- audio embeddings are saved as `data/audio_embeddings/<seq_id>_left_whisper.npy`
- video embeddings are saved as `data/video_embeddings/<seq_id>_left.npy`
- FLAME labels are loaded from `data/video_labels/<seq_id>_right/flame_param.npz`

Each valid training sample must have all of the following:

- audio embedding
- video embedding
- FLAME label file

If you already have embeddings and labels, you can skip the preprocessing steps and go straight to training or inference.


## Environment requirements

The scripts are written as standalone Python files and are expected to be run from the repository root.

Recommended environment:

- Linux
- CUDA GPU
- Python 3.10
- ffmpeg available on PATH

Python packages used by the repository include:

- torch
- numpy
- scipy
- librosa
- transformers
- qwen-vl-utils

If you use micromamba, a typical workflow is:

```bash
micromamba activate qwen_vl
```


## End-to-end workflow

### LookingFace motion TransVAE workflow

Use this workflow when training `train_motion_transvae.py` on the official LookingFace splits.

### 1. Rebuild the manifest after dataset edits

```bash
python manifest.py --rebuild
```

This rescans `data/LookingFace` and regenerates `data/manifest.json`.

### 2. Precompute wav2vec features

```bash
python scripts/preprocess_lookingface.py --skip_existing --device cuda:0
```

Outputs:

- `data/audio/<seq_id>_left.wav`
- `data/wav2vec_embeddings/<seq_id>_left.npy`

If direct Hugging Face access is blocked, run with:

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/preprocess_lookingface.py --skip_existing --device cuda:0
```

### 3. Train on predefined train split and evaluate on predefined test split

```bash
python train_motion_transvae.py \
  --predefined_splits_dir data/LookingFace/dataset_splits \
  --epochs 100 \
  --lr 1e-5 \
  --checkpoint_dir checkpoints/motion_transvae_lookingface_predefined
```

Behavior:

- train split comes from `data/LookingFace/dataset_splits/train.json`
- evaluation split comes from `data/LookingFace/dataset_splits/test.json`
- samples missing FLAME targets, right MP4 files, or wav2vec features are filtered out automatically
- checkpoints are written under the provided checkpoint directory
- final metrics are saved as `metrics.json` in that checkpoint directory

Common training knobs:

- `--epochs`: total number of epochs, default `100`
- `--lr`: AdamW learning rate, default `1e-5`
- `--batch_size`: batch size, default `2`
- `--val_interval`: run evaluation every N epochs, default `5`

### 1. Split raw MP4 files into audio and video

This looks for `*left*.mp4` files under `data/`.

```bash
python split_audio_video.py
```

What it does:

- extracts 16 kHz WAV audio into `data/audio`
- extracts video-only MP4 files into `data/video`


### 2. Encode audio with Whisper

```bash
python encode_audio_whisper.py
```

Outputs:

- `*_whisper.npy`: raw Whisper encoder features, shape `(T, 1280)`
- `*_proj.npy`: projected features, shape `(T, 3584)`
- `audio_projector_init.pt`: initial projector weights

Important note: current training uses `*_whisper.npy`, not `*_proj.npy`.


### 3. Encode video with Qwen

```bash
python encode_video_qwen.py
```

Output:

- `data/video_embeddings/<seq_id>_left.npy`, shape `(3584,)`


### 4. Train the model

```bash
python train.py
```

Training behavior:

- dataloaders are built from discovered valid sequences
- audio features are cropped to valid duration and interpolated to the FLAME frame count
- checkpoints are written to `checkpoints/`
- best checkpoint is saved as `checkpoints/best_model.pt`
- periodic checkpoints are saved every 100 epochs
- final checkpoint is saved as `checkpoints/final_model.pt`


### 5. Run inference

Single sequence:

```bash
python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt
```

Useful options:

- `--smooth N`: apply temporal moving-average smoothing to predicted parameters
- `--gen_vis`: save output in a per-sequence directory and copy `foreground_image.png` and `transforms.json`
- `--output_dir DIR`: choose output directory
- `--n_frames N`: manually override output frame count

Example:

```bash
python inference.py \
  --seq_id 2920 \
  --checkpoint checkpoints/best_model.pt \
  --output_dir output/predicted \
  --smooth 5 \
  --gen_vis
```

With `--gen_vis`, the output is:

- `output/<seq_id>_right/flame_param.npz`
- copied `foreground_image.png`
- copied `transforms.json`


## Current convergence-test setup

The current `config.py` is set up for a same-video convergence test.

Relevant settings:

- `TRAIN_VAL_SAME_SEQS = True`
- `CONVERGENCE_SEQ_IDS = ["2920", "2921", "2922", "2923", "2924"]`

This means:

- train and validation use the same sequences
- only those 5 sequence IDs are used

This is useful for checking whether the model can overfit a small set and converge.

To run training in that mode:

```bash
python -u train.py | tee output/train_5vid_convergence.log
```

To monitor progress:

```bash
tail -f output/train_5vid_convergence.log
```

To run inference on the same 5 videos:

```bash
for sid in 2920 2921 2922 2923 2924; do
  python inference.py \
    --seq_id "$sid" \
    --checkpoint checkpoints/best_model.pt \
    --output_dir output/convergence_5vid \
    --smooth 5 \
    --gen_vis
done
```


## Returning to normal hold-out validation

To switch back to a standard train/validation split:

1. Set `TRAIN_VAL_SAME_SEQS = False` in `config.py`
2. Set `CONVERGENCE_SEQ_IDS = []` if you want to use all discovered sequences
3. Adjust `VAL_SEQS` in `config.py` to choose the hold-out sequences


## Important implementation notes

### Translation prediction

Translation is now predicted by the model instead of being copied directly from labels.

The current behavior is:

- translation head is initialized with small weights and zero bias for stable startup
- translation loss weight is lower than expression loss weight
- expression loss weight is increased to improve expression sensitivity


### Inference with older checkpoints

Older checkpoints created before the translation head was added can still be loaded by `inference.py`.

In that case, inference prints a warning and uses the initialized translation head. That is acceptable for compatibility, but for meaningful translation prediction you should retrain and use a new checkpoint.


### Frame count alignment

Inference uses the label file's actual expression frame count by default when labels are available. This avoids frame-count mismatch from estimating frames with a fixed fps.


### Smoothing

Use `--smooth N` to reduce frame-to-frame jitter in predicted parameters. A value of `5` is a reasonable starting point.


## Output summary

Training outputs:

- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`
- `checkpoints/epoch_100.pt`, `epoch_200.pt`, and so on

Inference outputs:

- by default: `output/predicted/<seq_id>_flame_pred.npz`
- with `--gen_vis`: `output/<seq_id>_right/flame_param.npz` plus copied visualization assets


## Common pitfalls

- `config.py` currently points to `cuda:0`; change it if needed
- `split_audio_video.py` only processes MP4 filenames containing `left`
- training only uses sequences that have all required audio embeddings, video embeddings, and FLAME labels
- `encode_audio_whisper.py` generates projected audio embeddings, but `train.py` currently uses the raw Whisper features
- if `ffmpeg` is missing, preprocessing will fail

