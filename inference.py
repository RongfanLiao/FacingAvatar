"""Inference entrypoint for legacy, motion TransVAE, REGNN, ListenFormer, Dyadic DIM, and DualTalk checkpoints.

Usage:
    python inference.py --seq_id 2920
    python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/motion_transvae_lookingface/best.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/regnn_lookingface/best.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/listenformer_content/best.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/dyadic_dim/best.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/dualtalk_content/best.pt
"""

import argparse
import os
import shutil

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch.nn.utils.rnn import pad_sequence

from benchmark.lookingface import _fit_frame_to_canvas
from benchmark.dualtalk import LookingFaceDualTalk
from benchmark.dyadic_dim import DyadicContinuousTransformer
from benchmark.motion_transvae import MotionTransformerVAE
from benchmark.listenformer import LookingFaceListenFormer
from benchmark.regnn import LookingFaceREGNN
from benchmark.targets import FLAME_CONTENT_DIM, FLAME_58_DIM
from config import (
    AUDIO_EMB_DIR, VIDEO_EMB_DIR,
    WAV2VEC_EMB_DIR, WHISPER_MAX_FRAMES, WHISPER_CHUNK_SEC, CKPT_DIR, DEVICE,
)
from dataset import _get_audio_duration, _interpolate_features, FLAME_KEYS
from manifest import load_manifest
from decord import VideoReader, cpu as decord_cpu


def _load_checkpoint(checkpoint: str, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def _detect_checkpoint_family(checkpoint_state: dict[str, torch.Tensor]) -> str:
    if "decoder.output_head.weight" in checkpoint_state:
        return "motion_transvae"
    if "context_encoder.audio_feature_map.weight" in checkpoint_state and "encoder.layers.0.self_attn.in_proj_weight" in checkpoint_state:
        return "dyadic_dim"
    if "joint_encoder.speaker_encoder.audio_feature_map.weight" in checkpoint_state and "synthesis_module.modulation_layer.weight" in checkpoint_state:
        return "dualtalk"
    if "start_token" in checkpoint_state and "condition_proj.weight" in checkpoint_state:
        return "listenformer"
    if "cognitive_processor.convert_layer.weight1" in checkpoint_state:
        return "regnn"
    if "flame_head.expr_head.weight" in checkpoint_state:
        return "legacy_av"
    raise RuntimeError("Unsupported checkpoint format")


def _infer_listenformer_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int]:
    feature_dim = int(checkpoint_state["audio_proj.weight"].shape[0])
    audio_dim = int(checkpoint_state["audio_proj.weight"].shape[1])
    output_dim = int(checkpoint_state["output_head.3.weight"].shape[0])
    encoder_layer_indices = {
        int(parts[2])
        for key in checkpoint_state
        if key.startswith("audio_encoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 3 and parts[2].isdigit()
    }
    decoder_layer_indices = {
        int(parts[2])
        for key in checkpoint_state
        if key.startswith("decoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 3 and parts[2].isdigit()
    }
    num_encoder_layers = max(encoder_layer_indices) + 1 if encoder_layer_indices else 3
    num_decoder_layers = max(decoder_layer_indices) + 1 if decoder_layer_indices else 3
    return {
        "feature_dim": feature_dim,
        "audio_dim": audio_dim,
        "output_dim": output_dim,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
    }


def _infer_dyadic_dim_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int | str]:
    feature_dim = int(checkpoint_state["context_encoder.audio_feature_map.weight"].shape[0])
    audio_dim = int(checkpoint_state["context_encoder.audio_feature_map.weight"].shape[1])
    output_dim = int(checkpoint_state["output_head.3.weight"].shape[0])
    encoder_layer_indices = {
        int(parts[2])
        for key in checkpoint_state
        if key.startswith("encoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 3 and parts[2].isdigit()
    }
    decoder_layer_indices = {
        int(parts[2])
        for key in checkpoint_state
        if key.startswith("decoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 3 and parts[2].isdigit()
    }
    num_encoder_layers = max(encoder_layer_indices) + 1 if encoder_layer_indices else 6
    num_decoder_layers = max(decoder_layer_indices) + 1 if decoder_layer_indices else 6
    if output_dim == FLAME_CONTENT_DIM:
        target_variant = "content"
    elif output_dim == FLAME_58_DIM:
        target_variant = "motion58"
    else:
        raise RuntimeError(f"Unsupported Dyadic DIM output dim: {output_dim}")
    return {
        "feature_dim": feature_dim,
        "audio_dim": audio_dim,
        "output_dim": output_dim,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "target_variant": target_variant,
    }


def _infer_dualtalk_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int]:
    feature_dim = int(checkpoint_state["joint_encoder.speaker_encoder.audio_feature_map.weight"].shape[0])
    audio_dim = int(checkpoint_state["joint_encoder.speaker_encoder.audio_feature_map.weight"].shape[1])
    output_dim = int(checkpoint_state["synthesis_module.output_head.3.weight"].shape[0])
    interaction_layer_indices = {
        int(parts[3])
        for key in checkpoint_state
        if key.startswith("interaction_module.interaction_encoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 4 and parts[3].isdigit()
    }
    decoder_layer_indices = {
        int(parts[3])
        for key in checkpoint_state
        if key.startswith("synthesis_module.synthesis_decoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 4 and parts[3].isdigit()
    }
    interaction_layers = max(interaction_layer_indices) + 1 if interaction_layer_indices else 3
    decoder_layers = max(decoder_layer_indices) + 1 if decoder_layer_indices else 1
    return {
        "feature_dim": feature_dim,
        "audio_dim": audio_dim,
        "output_dim": output_dim,
        "interaction_layers": interaction_layers,
        "decoder_layers": decoder_layers,
    }


def _resolve_dualtalk_config(ckpt: dict[str, object], state_dict: dict[str, torch.Tensor]) -> dict[str, int | float]:
    saved_config = ckpt.get("model_config")
    if isinstance(saved_config, dict) and saved_config.get("family") == "dualtalk":
        required_keys = {
            "audio_dim",
            "output_dim",
            "feature_dim",
            "n_heads",
            "interaction_layers",
            "decoder_layers",
            "dropout",
            "video_chunk_size",
            "modulation_factor",
        }
        if required_keys.issubset(saved_config.keys()):
            return {
                "audio_dim": int(saved_config["audio_dim"]),
                "output_dim": int(saved_config["output_dim"]),
                "feature_dim": int(saved_config["feature_dim"]),
                "n_heads": int(saved_config["n_heads"]),
                "interaction_layers": int(saved_config["interaction_layers"]),
                "decoder_layers": int(saved_config["decoder_layers"]),
                "dropout": float(saved_config["dropout"]),
                "video_chunk_size": int(saved_config["video_chunk_size"]),
                "modulation_factor": float(saved_config["modulation_factor"]),
            }

    inferred = _infer_dualtalk_config(state_dict)
    return {
        "audio_dim": int(inferred["audio_dim"]),
        "output_dim": int(inferred["output_dim"]),
        "feature_dim": int(inferred["feature_dim"]),
        "n_heads": 8,
        "interaction_layers": int(inferred["interaction_layers"]),
        "decoder_layers": int(inferred["decoder_layers"]),
        "dropout": 0.1,
        "video_chunk_size": 8,
        "modulation_factor": 0.1,
    }


def _infer_regnn_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int | str]:
    fused_dim = int(checkpoint_state["perceptual_processor.encoder.audio_feature_map.weight"].shape[0])
    audio_dim = int(checkpoint_state["perceptual_processor.encoder.audio_feature_map.weight"].shape[1])
    num_frames = int(checkpoint_state["cognitive_processor.convert_layer.weight1"].shape[0])
    target_dim = int(checkpoint_state["cognitive_processor.convert_layer.weight2"].shape[-1])
    edge_qk_shape = checkpoint_state["cognitive_processor.edge_layer.qk.weight"].shape
    edge_dim = int(edge_qk_shape[0] // max(2 * num_frames, 1))
    layer_indices = {
        int(parts[2])
        for key in checkpoint_state
        if key.startswith("motor_processor.layers.")
        for parts in [key.split(".")]
        if len(parts) > 3 and parts[2].isdigit()
    }
    layers = max(layer_indices) + 1 if layer_indices else 2
    if target_dim == FLAME_CONTENT_DIM:
        target_variant = "content"
    elif target_dim == FLAME_58_DIM:
        target_variant = "motion58"
    else:
        raise RuntimeError(f"Unsupported REGNN target dim: {target_dim}")

    return {
        "audio_dim": audio_dim,
        "fused_dim": fused_dim,
        "num_frames": num_frames,
        "edge_dim": edge_dim,
        "layers": layers,
        "target_variant": target_variant,
    }


def _build_motion_transvae_inputs(seq_id: str, n_frames: int, video_canvas_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = load_manifest()
    entry = manifest.get(seq_id)
    if entry is None:
        raise KeyError(f"Sequence {seq_id} not found in manifest")

    wav2vec_path = os.path.join(WAV2VEC_EMB_DIR, f"{seq_id}_left.npy")
    if not os.path.exists(wav2vec_path):
        raise FileNotFoundError(f"Missing wav2vec feature file: {wav2vec_path}")

    wav2vec_feat = np.load(wav2vec_path)
    wav2vec_feat = _interpolate_features(wav2vec_feat, n_frames)
    audio_tensor = torch.from_numpy(np.asarray(wav2vec_feat, dtype=np.float32)).float()

    vr = VideoReader(entry["left_mp4"], ctx=decord_cpu(0))
    total_video_frames = len(vr)
    if n_frames > 0 and total_video_frames != n_frames:
        indices = np.linspace(0, total_video_frames - 1, n_frames).astype(int).tolist()
    else:
        indices = list(range(total_video_frames))

    frames = []
    for frame_index in indices:
        frame_np = vr[frame_index].asnumpy()
        frames.append(_fit_frame_to_canvas(frame_np, video_canvas_size))
    video_tensor = torch.stack(frames)

    lengths = torch.tensor([n_frames], dtype=torch.long)
    padding_mask = torch.zeros((1, n_frames), dtype=torch.bool)
    return audio_tensor.unsqueeze(0), video_tensor.unsqueeze(0), lengths, padding_mask


def _motion_prediction_to_flame(prediction: np.ndarray, label_npz: np.lib.npyio.NpzFile, output_dim: int) -> dict[str, np.ndarray]:
    results = {key: np.asarray(label_npz[key]) for key in label_npz.files}

    if output_dim == FLAME_CONTENT_DIM:
        results["expr"] = prediction[:, :100]
        results["jaw_pose"] = prediction[:, 100:103]
        results["neck_pose"] = prediction[:, 103:106]
        results["eyes_pose"] = prediction[:, 106:112]
        return results

    if output_dim == FLAME_58_DIM:
        expr = np.asarray(label_npz["expr"]).copy()
        expr[:, :52] = prediction[:, :52]
        results["expr"] = expr
        results["rotation"] = prediction[:, 52:55]
        results["translation"] = prediction[:, 55:58]
        return results

    raise RuntimeError(f"Unsupported motion TransVAE output dim: {output_dim}")


def predict_motion_transvae(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    output_dim = int(ckpt["model_state_dict"]["decoder.output_head.weight"].shape[0])
    model = MotionTransformerVAE(output_dim=output_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    audio_t, video_t, lengths, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    lengths = lengths.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction, _ = model(audio_t, video_t, lengths=lengths, padding_mask=padding_mask)

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, output_dim)
    finally:
        label_npz.close()


def predict_regnn(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
    neighbors: int,
    act_type: str,
    noise_threshold: float | None,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = ckpt["model_state_dict"]
    regnn_config = _infer_regnn_config(state_dict)
    target_variant = str(regnn_config["target_variant"])
    target_dim = FLAME_CONTENT_DIM if target_variant == "content" else FLAME_58_DIM

    model = LookingFaceREGNN(
        audio_dim=int(regnn_config["audio_dim"]),
        fused_dim=int(regnn_config["fused_dim"]),
        target_variant=target_variant,
        num_frames=int(regnn_config["num_frames"]),
        edge_dim=int(regnn_config["edge_dim"]),
        neighbors=neighbors,
        layers=int(regnn_config["layers"]),
        act_type=act_type,
        noise_threshold=noise_threshold,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio_t, video_t, lengths, _ = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction = model.predict_sequence(
                left_audio_feat=audio_t,
                left_video=video_t,
                lengths=lengths,
                target_variant=target_variant,
            )

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, target_dim)
    finally:
        label_npz.close()


def predict_listenformer(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
    n_heads: int,
    dropout: float,
    video_chunk_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = ckpt["model_state_dict"]
    listenformer_config = _infer_listenformer_config(state_dict)

    output_dim = int(listenformer_config["output_dim"])
    if output_dim != FLAME_CONTENT_DIM:
        raise RuntimeError(f"Unsupported ListenFormer output dim: {output_dim}")

    model = LookingFaceListenFormer(
        audio_dim=int(listenformer_config["audio_dim"]),
        output_dim=output_dim,
        feature_dim=int(listenformer_config["feature_dim"]),
        n_heads=n_heads,
        num_encoder_layers=int(listenformer_config["num_encoder_layers"]),
        num_decoder_layers=int(listenformer_config["num_decoder_layers"]),
        dropout=dropout,
        video_chunk_size=video_chunk_size,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio_t, video_t, lengths, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    lengths = lengths.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction, _ = model(
                left_audio_feat=audio_t,
                left_video_frames=video_t,
                lengths=lengths,
                padding_mask=padding_mask,
                target=None,
            )

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, output_dim)
    finally:
        label_npz.close()


def predict_dyadic_dim(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
    n_heads: int,
    dropout: float,
    video_chunk_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = ckpt["model_state_dict"]
    dyadic_config = _infer_dyadic_dim_config(state_dict)
    target_variant = str(dyadic_config["target_variant"])
    output_dim = int(dyadic_config["output_dim"])

    model = DyadicContinuousTransformer(
        audio_dim=int(dyadic_config["audio_dim"]),
        target_variant=target_variant,
        feature_dim=int(dyadic_config["feature_dim"]),
        n_heads=n_heads,
        num_encoder_layers=int(dyadic_config["num_encoder_layers"]),
        num_decoder_layers=int(dyadic_config["num_decoder_layers"]),
        dropout=dropout,
        video_chunk_size=video_chunk_size,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio_t, video_t, lengths, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    lengths = lengths.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction, _ = model(
                left_audio_feat=audio_t,
                left_video_frames=video_t,
                lengths=lengths,
                padding_mask=padding_mask,
                target=None,
            )

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, output_dim)
    finally:
        label_npz.close()


def predict_dualtalk(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
    n_heads: int,
    dropout: float,
    video_chunk_size: int,
    modulation_factor: float,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = ckpt["model_state_dict"]
    dualtalk_config = _resolve_dualtalk_config(ckpt, state_dict)
    output_dim = int(dualtalk_config["output_dim"])

    if output_dim != FLAME_CONTENT_DIM:
        raise RuntimeError(f"Unsupported DualTalk output dim: {output_dim}")

    model = LookingFaceDualTalk(
        audio_dim=int(dualtalk_config["audio_dim"]),
        output_dim=output_dim,
        feature_dim=int(dualtalk_config["feature_dim"]),
        n_heads=int(dualtalk_config["n_heads"]),
        interaction_layers=int(dualtalk_config["interaction_layers"]),
        decoder_layers=int(dualtalk_config["decoder_layers"]),
        dropout=float(dualtalk_config["dropout"]),
        video_chunk_size=int(dualtalk_config["video_chunk_size"]),
        modulation_factor=float(dualtalk_config["modulation_factor"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio_t, video_t, lengths, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    lengths = lengths.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction, _ = model(
                left_audio_feat=audio_t,
                left_video_frames=video_t,
                lengths=lengths,
                padding_mask=padding_mask,
            )

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, output_dim)
    finally:
        label_npz.close()


def main():
    parser = argparse.ArgumentParser(description="Predict FLAME params from audio+video")
    parser.add_argument("--seq_id", type=str, required=True, help="Sequence ID (e.g. 2920)")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CKPT_DIR, "best_model.pt"))
    parser.add_argument("--n_frames", type=int, default=None, help="Number of output frames (default: auto from duration)")
    parser.add_argument("--output_dir", type=str, default="work_dir/predicted", help="Output directory")
    parser.add_argument("--gen_vis", action="store_true", help="Pack output into a per-sequence dir with foreground_image.png and transforms.json for visualization")
    parser.add_argument("--smooth", type=int, default=5, help="Temporal smoothing window size (0=disabled)")
    parser.add_argument("--video_canvas_size", type=int, default=400, help="Canvas size for motion TransVAE raw-frame inference")
    parser.add_argument("--dyadic_n_heads", type=int, default=8, help="Dyadic ContinuousTransformer attention head count used when reconstructing the model for inference")
    parser.add_argument("--dyadic_dropout", type=float, default=0.1, help="Dyadic ContinuousTransformer dropout used when reconstructing the model for inference")
    parser.add_argument("--dyadic_video_chunk_size", type=int, default=8, help="Dyadic ContinuousTransformer video encoder chunk size for inference")
    parser.add_argument("--listenformer_n_heads", type=int, default=8, help="ListenFormer attention head count used when reconstructing the model for inference")
    parser.add_argument("--listenformer_dropout", type=float, default=0.1, help="ListenFormer dropout used when reconstructing the model for inference")
    parser.add_argument("--listenformer_video_chunk_size", type=int, default=8, help="ListenFormer video encoder chunk size for inference")
    parser.add_argument("--dualtalk_n_heads", type=int, default=8, help="DualTalk attention head count used when reconstructing the model for inference")
    parser.add_argument("--dualtalk_dropout", type=float, default=0.1, help="DualTalk dropout used when reconstructing the model for inference")
    parser.add_argument("--dualtalk_video_chunk_size", type=int, default=8, help="DualTalk video encoder chunk size for inference")
    parser.add_argument("--dualtalk_modulation_factor", type=float, default=0.1, help="DualTalk modulation factor used when reconstructing the model for inference")
    parser.add_argument("--regnn_neighbors", type=int, default=6, help="Neighbor count used by REGNN edge sparsification at inference time")
    parser.add_argument("--regnn_act_type", choices=["ELU", "ReLU", "GeLU", "None"], default="ELU", help="REGNN graph activation used when reconstructing the model for inference")
    parser.add_argument("--regnn_noise_threshold", type=float, default=0.0, help="Optional REGNN noise threshold for internal sampling; <=0 disables it")
    args = parser.parse_args()

    # Load label npz for static keys and frame count
    manifest = load_manifest()
    entry = manifest[args.seq_id]
    label_npz = np.load(entry["flame_npz"])
    label_dir = os.path.dirname(entry["flame_npz"])
    checkpoint = _load_checkpoint(args.checkpoint, DEVICE)
    checkpoint_family = _detect_checkpoint_family(checkpoint["model_state_dict"])

    # Use label frame count if not overridden, so all params stay aligned
    n_frames = args.n_frames or label_npz["expr"].shape[0]

    print(f"Predicting FLAME params for sequence {args.seq_id} ({n_frames} frames)...")
    print(f"Checkpoint family: {checkpoint_family}")
    if checkpoint_family == "motion_transvae":
        results = predict_motion_transvae(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
        )
    elif checkpoint_family == "dyadic_dim":
        results = predict_dyadic_dim(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
            n_heads=args.dyadic_n_heads,
            dropout=args.dyadic_dropout,
            video_chunk_size=args.dyadic_video_chunk_size,
        )
    elif checkpoint_family == "listenformer":
        results = predict_listenformer(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
            n_heads=args.listenformer_n_heads,
            dropout=args.listenformer_dropout,
            video_chunk_size=args.listenformer_video_chunk_size,
        )
    elif checkpoint_family == "dualtalk":
        results = predict_dualtalk(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
            n_heads=args.dualtalk_n_heads,
            dropout=args.dualtalk_dropout,
            video_chunk_size=args.dualtalk_video_chunk_size,
            modulation_factor=args.dualtalk_modulation_factor,
        )
    elif checkpoint_family == "regnn":
        results = predict_regnn(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
            neighbors=args.regnn_neighbors,
            act_type=args.regnn_act_type,
            noise_threshold=(args.regnn_noise_threshold if args.regnn_noise_threshold > 0 else None),
        )
    else:
        results = predict(args.seq_id, args.checkpoint, n_frames)

    # Optional temporal smoothing on predicted params
    if args.smooth > 0:
        for key in FLAME_KEYS:
            results[key] = uniform_filter1d(results[key], size=args.smooth, axis=0)

    # Copy non-predicted keys (shape, canonical_*) from labels
    for k in label_npz.keys():
        if k not in results:
            results[k] = label_npz[k]

    if args.gen_vis:
        # Pack into a per-sequence directory matching label layout
        seq_out_dir = os.path.join(args.output_dir, f"{args.seq_id}_right")
        os.makedirs(seq_out_dir, exist_ok=True)

        out_path = os.path.join(seq_out_dir, "flame_param.npz")
        np.savez(out_path, **results)

        # Copy visualization assets from label dir
        for fname in ("foreground_image.png", "transforms.json"):
            src = os.path.join(label_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, seq_out_dir)
                print(f"  Copied {fname}")

    else:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{args.seq_id}_flame_pred.npz")
        np.savez(out_path, **results)

    print(f"Saved to {out_path}")
    for key, arr in results.items():
        print(f"  {key}: shape={arr.shape}")


if __name__ == "__main__":
    main()
