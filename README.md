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
├── inference.py               # Single-sequence prediction entry point
├── split_audio_video.py       # Raw MP4 -> WAV + video-only MP4
├── encode_audio_whisper.py    # WAV -> Whisper features
├── encode_video_qwen.py       # MP4 -> Qwen video embedding
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
- `inference.py`: loads a checkpoint and writes predicted FLAME parameters for one sequence

### Preprocessing files

- `split_audio_video.py`: extracts 16 kHz WAV audio and video-only MP4 files from raw MP4 inputs
- `encode_audio_whisper.py`: converts WAV audio to Whisper encoder features
- `encode_video_qwen.py`: converts video to a single Qwen embedding per sequence

### Data and generated artifacts

- `data/audio`: extracted WAV files used for Whisper encoding
- `data/video`: extracted MP4 files used for Qwen video encoding
- `data/audio_embeddings`: `*_whisper.npy`, `*_proj.npy`, and `audio_projector_init.pt`
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

