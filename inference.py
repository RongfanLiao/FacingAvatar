"""Inference entrypoint for legacy, motion diffusion, motion flow matching, motion TransVAE, REGNN, ListenFormer, Dyadic DIM, and DualTalk checkpoints.

Usage:
    python inference.py --seq_id 2920
    python inference.py --seq_id 2920 --checkpoint checkpoints/best_model.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/motion_diffusion_lookingface/best.pt
    python inference.py --seq_id 0012 --checkpoint checkpoints/motion_flow_matching_lookingface/best.pt
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
from scipy.ndimage import median_filter, uniform_filter1d
from torch.nn.utils.rnn import pad_sequence

from benchmark.lookingface import _fit_frame_to_canvas
from benchmark.dualtalk import LookingFaceDualTalk
from benchmark.dyadic_dim import DyadicContinuousTransformer
from benchmark.motion_diffusion import MotionDiffusionModel
from benchmark.motion_flow_matching import MotionFlowMatchingModel
from benchmark.motion_transvae import MotionTransformerVAE, checkpoint_state_dict
from benchmark.listenformer import LookingFaceListenFormer
from benchmark.regnn import LookingFaceREGNN
from benchmark.targets import FLAME_118_DIM, FLAME_CONTENT_DIM
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
    if (
        "velocity_field.audio_proj.weight" in checkpoint_state
        and "velocity_field.target_proj.weight" in checkpoint_state
    ):
        return "motion_flow_matching"
    if (
        "denoiser.audio_proj.weight" in checkpoint_state
        and "denoiser.target_proj.weight" in checkpoint_state
        and "alpha_bars" in checkpoint_state
    ):
        return "motion_diffusion"
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


def _infer_dyadic_dim_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int]:
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
    if output_dim != FLAME_CONTENT_DIM:
        raise RuntimeError(f"Unsupported Dyadic DIM output dim: {output_dim}")
    return {
        "feature_dim": feature_dim,
        "audio_dim": audio_dim,
        "output_dim": output_dim,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
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


def _infer_regnn_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int]:
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
    if target_dim not in {FLAME_CONTENT_DIM, FLAME_118_DIM}:
        raise RuntimeError(f"Unsupported REGNN target dim: {target_dim}")

    return {
        "audio_dim": audio_dim,
        "fused_dim": fused_dim,
        "target_dim": target_dim,
        "num_frames": num_frames,
        "edge_dim": edge_dim,
        "layers": layers,
    }


def _infer_motion_diffusion_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int]:
    feature_dim = int(checkpoint_state["denoiser.audio_proj.weight"].shape[0])
    audio_dim = int(checkpoint_state["denoiser.audio_proj.weight"].shape[1])
    output_dim = int(checkpoint_state["denoiser.output_head.3.weight"].shape[0])

    encoder_layer_indices = {
        int(parts[3])
        for key in checkpoint_state
        if key.startswith("denoiser.audio_encoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 4 and parts[3].isdigit()
    }
    decoder_layer_indices = {
        int(parts[3])
        for key in checkpoint_state
        if key.startswith("denoiser.decoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 4 and parts[3].isdigit()
    }
    num_layers = max(encoder_layer_indices | decoder_layer_indices) + 1 if (encoder_layer_indices or decoder_layer_indices) else 4

    if output_dim not in {FLAME_CONTENT_DIM, FLAME_118_DIM}:
        raise RuntimeError(f"Unsupported motion diffusion output dim: {output_dim}")

    return {
        "audio_dim": audio_dim,
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
    }


def _infer_motion_flow_matching_config(checkpoint_state: dict[str, torch.Tensor]) -> dict[str, int]:
    feature_dim = int(checkpoint_state["velocity_field.audio_proj.weight"].shape[0])
    audio_dim = int(checkpoint_state["velocity_field.audio_proj.weight"].shape[1])
    output_dim = int(checkpoint_state["velocity_field.output_head.3.weight"].shape[0])

    encoder_layer_indices = {
        int(parts[3])
        for key in checkpoint_state
        if key.startswith("velocity_field.audio_encoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 4 and parts[3].isdigit()
    }
    decoder_layer_indices = {
        int(parts[3])
        for key in checkpoint_state
        if key.startswith("velocity_field.decoder.layers.")
        for parts in [key.split(".")]
        if len(parts) > 4 and parts[3].isdigit()
    }
    num_layers = max(encoder_layer_indices | decoder_layer_indices) + 1 if (encoder_layer_indices or decoder_layer_indices) else 4

    return {
        "audio_dim": audio_dim,
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
    }


def _resolve_motion_flow_matching_config(ckpt: dict[str, object], state_dict: dict[str, torch.Tensor]) -> dict[str, int | float | str]:
    saved_config = ckpt.get("model_config")
    if isinstance(saved_config, dict) and saved_config.get("family") == "motion_flow_matching":
        required_keys = {
            "audio_dim",
            "target_dim",
            "feature_dim",
            "n_heads",
            "num_layers",
            "dropout",
            "solver",
            "solver_steps",
            "clip_sample",
            "guidance_scale",
            "time_embed_scale",
        }
        if required_keys.issubset(saved_config.keys()):
            return {
                "audio_dim": int(saved_config["audio_dim"]),
                "output_dim": int(saved_config["target_dim"]),
                "feature_dim": int(saved_config["feature_dim"]),
                "n_heads": int(saved_config["n_heads"]),
                "num_layers": int(saved_config["num_layers"]),
                "dropout": float(saved_config["dropout"]),
                "solver": str(saved_config["solver"]),
                "solver_steps": int(saved_config["solver_steps"]),
                "clip_sample": float(saved_config["clip_sample"]),
                "guidance_scale": float(saved_config["guidance_scale"]),
                "time_embed_scale": float(saved_config["time_embed_scale"]),
            }

    inferred = _infer_motion_flow_matching_config(state_dict)
    return {
        "audio_dim": int(inferred["audio_dim"]),
        "output_dim": int(inferred["output_dim"]),
        "feature_dim": int(inferred["feature_dim"]),
        "n_heads": 8,
        "num_layers": int(inferred["num_layers"]),
        "dropout": 0.1,
        "solver": "heun",
        "solver_steps": 40,
        "clip_sample": 5.0,
        "guidance_scale": 1.5,
        "time_embed_scale": 1000.0,
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

    if output_dim == FLAME_118_DIM:
        results["expr"] = prediction[:, :100]
        results["jaw_pose"] = prediction[:, 100:103]
        results["rotation"] = prediction[:, 103:106]
        results["neck_pose"] = prediction[:, 106:109]
        results["eyes_pose"] = prediction[:, 109:115]
        results["translation"] = prediction[:, 115:118]
        return results

    if output_dim == FLAME_CONTENT_DIM:
        results["expr"] = prediction[:, :100]
        results["jaw_pose"] = prediction[:, 100:103]
        results["neck_pose"] = prediction[:, 103:106]
        results["eyes_pose"] = prediction[:, 106:112]
        return results

    raise RuntimeError(f"Unsupported motion prediction output dim: {output_dim}")


def _stabilize_pose_tracks(
    results: dict[str, np.ndarray],
    noise_window: int,
    smooth_window: int,
    keys: tuple[str, ...] = ("rotation", "translation"),
) -> None:
    if noise_window <= 1 and smooth_window <= 1:
        return

    for key in keys:
        values = results.get(key)
        if values is None or values.ndim == 0 or values.shape[0] <= 1:
            continue

        stabilized = values.astype(np.float32, copy=False)
        if noise_window > 1:
            stabilized = median_filter(stabilized, size=(noise_window, 1), mode="nearest")
        if smooth_window > 1:
            stabilized = uniform_filter1d(stabilized, size=smooth_window, axis=0, mode="nearest")
        results[key] = stabilized.astype(values.dtype, copy=False)


def predict_motion_flow_matching(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
    n_heads: int,
    dropout: float,
    solver: str,
    solver_steps: int,
    clip_sample: float,
    guidance_scale: float,
    time_embed_scale: float,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = checkpoint_state_dict(ckpt)
    flow_config = _resolve_motion_flow_matching_config(ckpt, state_dict)
    output_dim = int(flow_config["output_dim"])

    model = MotionFlowMatchingModel(
        audio_dim=int(flow_config["audio_dim"]),
        target_dim=output_dim,
        feature_dim=int(flow_config["feature_dim"]),
        n_heads=int(flow_config.get("n_heads", n_heads)),
        num_layers=int(flow_config["num_layers"]),
        dropout=float(flow_config.get("dropout", dropout)),
        solver=str(flow_config.get("solver", solver)),
        solver_steps=int(flow_config.get("solver_steps", solver_steps)),
        clip_sample=float(flow_config.get("clip_sample", clip_sample)),
        guidance_scale=float(flow_config.get("guidance_scale", guidance_scale)),
        audio_drop_prob=0.0,
        video_drop_prob=0.0,
        latent_drop_prob=0.0,
        time_embed_scale=float(flow_config.get("time_embed_scale", time_embed_scale)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio_t, video_t, _, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction = model.sample(
                left_audio_feat=audio_t,
                left_video_frames=video_t,
                padding_mask=padding_mask,
            )

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, output_dim)
    finally:
        label_npz.close()


def predict_motion_transvae(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = checkpoint_state_dict(ckpt)
    output_dim = int(state_dict["decoder.output_head.weight"].shape[0])
    model = MotionTransformerVAE(output_dim=output_dim).to(device)
    model.load_state_dict(state_dict)
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
    state_dict = checkpoint_state_dict(ckpt)
    regnn_config = _infer_regnn_config(state_dict)
    target_dim = int(regnn_config["target_dim"])

    model = LookingFaceREGNN(
        audio_dim=int(regnn_config["audio_dim"]),
        target_dim=target_dim,
        fused_dim=int(regnn_config["fused_dim"]),
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
    state_dict = checkpoint_state_dict(ckpt)
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
    state_dict = checkpoint_state_dict(ckpt)
    dyadic_config = _infer_dyadic_dim_config(state_dict)
    output_dim = int(dyadic_config["output_dim"])

    model = DyadicContinuousTransformer(
        audio_dim=int(dyadic_config["audio_dim"]),
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
    state_dict = checkpoint_state_dict(ckpt)
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


def predict_motion_diffusion(
    seq_id: str,
    checkpoint: str,
    n_frames: int,
    video_canvas_size: int,
    n_heads: int,
    dropout: float,
    train_timesteps: int,
    inference_timesteps: int,
    beta_start: float,
    beta_end: float,
    clip_sample: float,
    guidance_scale: float,
    timestep_spacing: str,
    ddim_eta: float,
) -> dict[str, np.ndarray]:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint, device)
    state_dict = checkpoint_state_dict(ckpt)
    diffusion_config = _infer_motion_diffusion_config(state_dict)

    model = MotionDiffusionModel(
        audio_dim=int(diffusion_config["audio_dim"]),
        target_dim=int(diffusion_config["output_dim"]),
        feature_dim=int(diffusion_config["feature_dim"]),
        n_heads=n_heads,
        num_layers=int(diffusion_config["num_layers"]),
        dropout=dropout,
        train_timesteps=train_timesteps,
        inference_timesteps=inference_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=clip_sample,
        guidance_scale=guidance_scale,
        audio_drop_prob=0.0,
        video_drop_prob=0.0,
        latent_drop_prob=0.0,
        timestep_spacing=timestep_spacing,
        ddim_eta=ddim_eta,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio_t, video_t, _, padding_mask = _build_motion_transvae_inputs(seq_id, n_frames, video_canvas_size)
    audio_t = audio_t.to(device)
    video_t = video_t.to(device)
    padding_mask = padding_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            prediction = model.sample(
                left_audio_feat=audio_t,
                left_video_frames=video_t,
                padding_mask=padding_mask,
            )

    pred_np = prediction[0, :n_frames].detach().cpu().numpy().astype(np.float32)
    manifest = load_manifest()
    label_npz = np.load(manifest[seq_id]["flame_npz"])
    try:
        return _motion_prediction_to_flame(pred_np, label_npz, int(diffusion_config["output_dim"]))
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
    parser.add_argument("--pose_noise_filter", type=int, default=5, help="Median-filter window used to remove rotation/translation spikes before export (<=1 disables it)")
    parser.add_argument("--pose_smooth", type=int, default=9, help="Additional smoothing window applied to rotation/translation before export (<=1 disables it)")
    parser.add_argument(
        "--zero_rotation_translation",
        action="store_true",
        help="Set FLAME rotation and translation to zeros in the final exported output",
    )
    parser.add_argument("--video_canvas_size", type=int, default=300, help="Canvas size for motion TransVAE raw-frame inference")
    parser.add_argument("--motion_diffusion_n_heads", type=int, default=8, help="Motion diffusion attention head count used when reconstructing the model for inference")
    parser.add_argument("--motion_diffusion_dropout", type=float, default=0.1, help="Motion diffusion dropout used when reconstructing the model for inference")
    parser.add_argument("--motion_diffusion_train_timesteps", type=int, default=1000, help="Motion diffusion training schedule length used when reconstructing the sampler")
    parser.add_argument("--motion_diffusion_inference_timesteps", type=int, default=50, help="Motion diffusion reverse sampling steps used during inference")
    parser.add_argument("--motion_diffusion_beta_start", type=float, default=1e-4, help="Motion diffusion beta schedule start used for inference")
    parser.add_argument("--motion_diffusion_beta_end", type=float, default=2e-2, help="Motion diffusion beta schedule end used for inference")
    parser.add_argument("--motion_diffusion_clip_sample", type=float, default=5.0, help="Motion diffusion sample clamp range used during inference")
    parser.add_argument("--motion_diffusion_guidance_scale", type=float, default=1.5, help="Motion diffusion classifier-free guidance scale used during inference")
    parser.add_argument("--motion_diffusion_timestep_spacing", choices=["leading", "linspace", "trailing", "full"], default="leading", help="Motion diffusion timestep spacing strategy used during inference")
    parser.add_argument("--motion_diffusion_ddim_eta", type=float, default=0.0, help="Motion diffusion DDIM eta used during inference")
    parser.add_argument("--motion_flow_matching_n_heads", type=int, default=8, help="Motion flow matching attention head count used when reconstructing the model for inference")
    parser.add_argument("--motion_flow_matching_dropout", type=float, default=0.1, help="Motion flow matching dropout used when reconstructing the model for inference")
    parser.add_argument("--motion_flow_matching_solver", choices=["euler", "heun"], default="heun", help="Motion flow matching ODE solver used during inference")
    parser.add_argument("--motion_flow_matching_solver_steps", type=int, default=40, help="Motion flow matching ODE integration steps used during inference")
    parser.add_argument("--motion_flow_matching_clip_sample", type=float, default=5.0, help="Motion flow matching sample clamp range used during inference")
    parser.add_argument("--motion_flow_matching_guidance_scale", type=float, default=1.5, help="Motion flow matching classifier-free guidance scale used during inference")
    parser.add_argument("--motion_flow_matching_time_embed_scale", type=float, default=1000.0, help="Motion flow matching continuous-time embedding scale used during inference")
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
    state_dict = checkpoint_state_dict(checkpoint)
    checkpoint_family = _detect_checkpoint_family(state_dict)

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
    elif checkpoint_family == "motion_diffusion":
        results = predict_motion_diffusion(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
            n_heads=args.motion_diffusion_n_heads,
            dropout=args.motion_diffusion_dropout,
            train_timesteps=args.motion_diffusion_train_timesteps,
            inference_timesteps=args.motion_diffusion_inference_timesteps,
            beta_start=args.motion_diffusion_beta_start,
            beta_end=args.motion_diffusion_beta_end,
            clip_sample=args.motion_diffusion_clip_sample,
            guidance_scale=args.motion_diffusion_guidance_scale,
            timestep_spacing=args.motion_diffusion_timestep_spacing,
            ddim_eta=args.motion_diffusion_ddim_eta,
        )
    elif checkpoint_family == "motion_flow_matching":
        results = predict_motion_flow_matching(
            args.seq_id,
            args.checkpoint,
            n_frames,
            video_canvas_size=args.video_canvas_size,
            n_heads=args.motion_flow_matching_n_heads,
            dropout=args.motion_flow_matching_dropout,
            solver=args.motion_flow_matching_solver,
            solver_steps=args.motion_flow_matching_solver_steps,
            clip_sample=args.motion_flow_matching_clip_sample,
            guidance_scale=args.motion_flow_matching_guidance_scale,
            time_embed_scale=args.motion_flow_matching_time_embed_scale,
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
        raise RuntimeError(
            f"Checkpoint family {checkpoint_family!r} is not supported by the current inference entrypoint"
        )

    # Optional temporal smoothing on predicted params
    if args.smooth > 0:
        for key in FLAME_KEYS:
            results[key] = uniform_filter1d(results[key], size=args.smooth, axis=0, mode="nearest")

    _stabilize_pose_tracks(
        results,
        noise_window=args.pose_noise_filter,
        smooth_window=args.pose_smooth,
    )

    # Copy non-predicted keys (shape, canonical_*) from labels
    for k in label_npz.keys():
        if k not in results:
            results[k] = label_npz[k]

    if args.zero_rotation_translation:
        for key in ("rotation", "translation"):
            if key in results:
                results[key] = np.zeros_like(results[key])
            elif key in label_npz:
                results[key] = np.zeros_like(label_npz[key])

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
