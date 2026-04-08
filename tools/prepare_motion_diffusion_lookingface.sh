#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/avatar/bin/python}"
STAGE="${1:-all}"

cd "$ROOT_DIR"

case "$STAGE" in
  manifest)
    exec "$PYTHON_BIN" manifest.py --rebuild
    ;;
  wav2vec)
    exec "$PYTHON_BIN" scripts/preprocess_lookingface.py --skip_existing
    ;;
  check)
    exec "$PYTHON_BIN" -c "from manifest import load_manifest; from benchmark.lookingface import load_predefined_splits, discover_benchmark_sequences; m=load_manifest(); s=load_predefined_splits('data/LookingFace/dataset_splits', manifest=m, require_wav2vec_audio=True); print({k: len(v) for k,v in s.items()}); print('train_usable', len(discover_benchmark_sequences(seq_ids=s.get('train', []), require_left_audio=False, require_left_video_embedding=False, require_wav2vec_audio=True, require_right_mp4=True, manifest=m)))"
    ;;
  all)
    "$PYTHON_BIN" manifest.py --rebuild
    "$PYTHON_BIN" scripts/preprocess_lookingface.py --skip_existing
    exec "$PYTHON_BIN" -c "from manifest import load_manifest; from benchmark.lookingface import load_predefined_splits, discover_benchmark_sequences; m=load_manifest(); s=load_predefined_splits('data/LookingFace/dataset_splits', manifest=m, require_wav2vec_audio=True); print({k: len(v) for k,v in s.items()}); print('train_usable', len(discover_benchmark_sequences(seq_ids=s.get('train', []), require_left_audio=False, require_left_video_embedding=False, require_wav2vec_audio=True, require_right_mp4=True, manifest=m)))"
    ;;
  *)
    echo "Usage: $0 [manifest|wav2vec|check|all]" >&2
    exit 1
    ;;
esac
