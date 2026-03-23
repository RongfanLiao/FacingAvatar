# Metrics for Reaction Head Avatars

## 1. Scope and Goal

This repository predicts per-frame FLAME parameters from audio and video features, with paired ground-truth FLAME targets available during training and validation. Because the model output is motion in FLAME space rather than RGB video, the evaluation should treat motion quality and temporal behavior as the primary evidence. Rendered-video metrics are still useful, but they should be reported as secondary evidence of final visual quality rather than the main proof of appropriate reaction behavior.

## 2. Document Summary

This document covers 31 metric entries across motion, rendering, lip-sync, identity, synchrony, semantic, and human-evaluation categories.

## 3. Recommended Evaluation Framework

The evaluation should follow a layered structure.

| Layer | Goal | Role in the Paper |
|---|---|---|
| Motion accuracy | Check whether predicted FLAME parameters match the paired target | Primary evidence |
| Temporal realism | Check whether the predicted reaction evolves naturally over time | Primary evidence |
| Visual quality | Check whether rendered outputs look realistic | Secondary evidence |
| Semantic appropriateness | Check whether the reaction suits the source content | Claim-level evidence |

## 4. Metric Glossary

### 4.1 Motion Reconstruction Metrics

| Metric | Full Name | What It Measures |
|---|---|---|
| FD | Feature Distance or Fréchet Distance, depending on the paper | A distance between generated motion features and reference motion features. In listener-generation papers, it often means the distance between predicted and ground-truth motion representations. |
| MAE | Mean Absolute Error | The average absolute difference between prediction and ground truth. It is a direct paired reconstruction error. |
| MSE | Mean Squared Error | The average squared difference between prediction and ground truth. Larger errors are penalized more heavily than with MAE. |
| RMSE | Root Mean Squared Error | The square root of MSE. It stays in the original unit scale while still emphasizing larger mistakes. |
| P-FD | Paired Fréchet Distance | A paired version of feature-distance evaluation that compares generated and real samples with explicit pairing. |

### 4.2 Motion Dynamics and Diversity Metrics

| Metric | Full Name | What It Measures |
|---|---|---|
| FID△fm | Fréchet Inception Distance on frame-to-frame motion differences | A distributional metric computed on motion deltas between adjacent frames. It measures how realistic the temporal dynamics are. |
| SND | Sequence Naturalness Distance | A sequence-level metric that measures how close generated motion sequences are to real motion sequences in distribution and dynamics. |
| Var | Variance | Measures how much motion varies over time or across generated samples. It is often used as a simple diversity indicator. |
| SID | SI for Diversity | A diversity metric used in talking-head and listener-generation work to assess whether generated motions span a reasonable range instead of collapsing to average behavior. |
| FDD | Face Dynamic Deviation or upper-face dynamic deviation, depending on the paper | A mesh-dynamics metric that measures whether upper-face movements follow the expected dynamic pattern. |

### 4.3 Rendered Video and Frame Metrics

| Metric | Full Name | What It Measures |
|---|---|---|
| FVD | Fréchet Video Distance | A video-level realism metric that compares distributions of generated and real videos while accounting for temporal information. |
| LPIPS | Learned Perceptual Image Patch Similarity | A perceptual similarity metric based on deep features, intended to align better with human visual judgment than pixel-wise metrics. |
| FID | Fréchet Inception Distance | A distributional realism metric on image features. It measures whether generated frames resemble real frames at the dataset level. |
| SSIM | Structural Similarity Index Measure | A frame-level similarity metric that compares local luminance, contrast, and structure between generated and real frames. |
| PSNR | Peak Signal-to-Noise Ratio | A pixel-level reconstruction metric derived from MSE. Higher values mean the generated frame is numerically closer to the reference frame. |

### 4.4 Lip-Sync and Speech-Articulation Metrics

| Metric | Full Name | What It Measures |
|---|---|---|
| SyncScore | SyncNet confidence score | A lip-sync metric that measures whether mouth motion matches the audio track. |
| Sync-C | Synchronization Confidence | A lip-sync metric where higher values indicate stronger audio-mouth alignment. |
| Sync-D | Synchronization Distance | A lip-sync metric where lower values indicate better audio-mouth alignment. |
| LSE-C | Lip Sync Error Confidence | Another confidence-style lip-sync metric derived from SyncNet-like models. |
| LSE-D | Lip Sync Error Distance | Another distance-style lip-sync metric for audio-lip alignment. |
| LVE | Lip Vertex Error | A mesh-based metric that measures the error of lip vertex positions against ground truth. |
| MOD | Mouth Opening Distance | A metric that measures how similar the mouth opening pattern is to ground truth, often used for speech articulation style. |

### 4.5 Identity and Interaction Synchrony Metrics

| Metric | Full Name | What It Measures |
|---|---|---|
| CSIM | Cosine Similarity for identity features | A feature similarity score used to measure identity preservation between a reference image and generated frames. |
| RPCC or rPCC | Residual Pearson Correlation Coefficient | Measures correlation between two motion signals after removing mean trends, often for interaction synchrony. |
| RTLCC | Residual Time Lagged Cross Correlation | Measures whether generated motion follows an expected lagged relationship with a driving signal over time. |
| RWTLCC | Residual Windowed Time Lagged Cross Correlation | A windowed version of RTLCC that measures local lagged response behavior over short temporal windows. |

### 4.6 Semantic and Human Evaluation Metrics

| Metric | Full Name | What It Measures |
|---|---|---|
| Ground-truth reaction embedding similarity | Semantic similarity in a pretrained embedding space | Measures whether the generated reaction is semantically close to the paired real reaction using a pretrained facial-expression or affect encoder. |
| Reaction emotion agreement | Emotion-label or emotion-trajectory agreement | Measures whether the affect expressed by the generated reaction matches the expected affective response. |
| Cross-modal appropriateness retrieval | Content-reaction matching accuracy | Measures whether the reaction generated for a source clip matches that source better than mismatched clips in a shared embedding space. |
| Human appropriateness study | Human judgment of appropriateness | Measures perceived suitability of the generated reaction for the source content using raters. |
| Emotion-feature Fréchet metric | Fréchet distance in affect-feature space | Measures whether the distribution of generated reactions matches the real reaction distribution in an emotion-sensitive feature space. |

## 5. Recommended Metrics

### 5.1 Primary Metrics

| Metric | Representation | Why It Is Recommended | Reporting Advice |
|---|---|---|---|
| FD / MAE | FLAME parameters | Directly measures paired motion error in the space the model predicts. | Report both overall and per-group results for `expr`, `jaw_pose`, `rotation`, `neck_pose`, `eyes_pose`, and `translation`. |
| MSE / RMSE | FLAME parameters | Complements MAE by penalizing larger mistakes more strongly. | Use together with MAE rather than as a replacement. |
| FID△fm | FLAME motion deltas | Evaluates frame-to-frame motion realism and temporal dynamics. | Important for detecting stiff, jittery, or over-smoothed motion. |
| SND | FLAME sequences | Measures sequence-level naturalness and distributional similarity of motion. | Strong fit for long-form listener or reaction behavior. |

### 5.2 Secondary Metrics

| Metric | Representation | Why It Is Recommended | Reporting Advice |
|---|---|---|---|
| FVD | Rendered videos | Best video-level realism metric once FLAME outputs are rendered. | Prefer this as the main rendered-video metric. |
| LPIPS | Rendered frames | Measures perceptual frame similarity more meaningfully than pixel-only metrics. | Strong complement to FVD. |
| FID | Rendered frames | Standard realism metric for generated frames. | Use for compatibility with prior work, but keep secondary. |
| SSIM | Rendered frames | Standard frame similarity metric. | Supplementary only. |
| PSNR | Rendered frames | Standard reconstruction metric for pixel similarity. | Supplementary only. |

### 5.3 Conditional or Optional Metrics

| Metric | Representation | When to Use It | Caution |
|---|---|---|---|
| P-FD | FLAME motion features | When you want an extra paired realism metric in motion space. | Useful, but not required for the first benchmark version. |
| Var | FLAME sequences | When you want to check for motion collapse or over-smoothing. | Not strong enough as primary evidence of appropriateness. |
| SID | FLAME sequences | When comparing multiple generators or one-to-many generation settings. | Weak as standalone evidence in paired settings. |
| Ground-truth reaction embedding similarity | Rendered videos or expression embeddings | When you need an automatic semantic metric for a paired setup. | Best semantic choice for the current setup. |
| Reaction emotion agreement | Rendered videos | When content differences are strongly affective and expected reactions are emotion-driven. | Requires a defensible definition of expected reaction emotion. |
| Cross-modal appropriateness retrieval | Source content plus generated reactions | When the paper strongly emphasizes content-conditioned appropriateness. | Strong idea, but harder to implement and justify. |
| Human appropriateness study | Human judgments | When the core claim is subjective reaction appropriateness. | Best used together with automatic metrics. |
| Emotion-feature Fréchet metric | Affect-model feature space | When you want a distribution-level semantic metric. | Does not prove per-sample appropriateness. |

## 6. Not Recommended as Main Metrics

| Metric | Representation | Why It Is Not Recommended As Main Evidence | When It Could Still Be Used |
|---|---|---|---|
| SyncScore / Sync-C / Sync-D | Audio-video lip sync | These target speaking-head lip synchronization, not reaction appropriateness. | Only use if the generated head is also speaking and lip alignment is part of the claim. |
| LSE-C / LSE-D | Audio-video lip sync | Same issue as SyncNet-style metrics: they measure articulation alignment, not reaction suitability. | Use only in speech-generation settings. |
| LVE | Lip vertices | Too specific to speaking articulation and mouth shape accuracy. | Relevant only if lip motion fidelity is a central contribution. |
| MOD | Mouth opening motion | Focuses on mouth opening style, which is not the main target for listener or reaction generation. | Possible auxiliary metric for speech-driven talking heads, not for content-appropriate reactions. |
| FDD | Upper-face vertex dynamics | Too narrow for the main claim and tied to specific mesh-region protocols. | Could be used for direct comparison with FLAME mesh-based talking-head methods. |
| CSIM | Identity features | Identity preservation is not the main research question here. | Use only if identity consistency is a stated contribution. |
| RPCC / rPCC / RTLCC / RWTLCC by themselves | Driver-response alignment | Correlation alone does not prove semantic appropriateness, and the repo does not currently use explicit framewise driver motion as input. | Use only if you can define a defensible driver-response signal from the source content. |
| FID alone | Rendered frames | Frame realism alone does not show the reaction is appropriate to the content. | Still useful as a secondary rendering metric. |
| SSIM alone or PSNR alone | Rendered frames | Pixel similarity can reward blur or average-looking results and says little about semantics. | Keep only as supplementary comparisons. |

## 7. Suggested Default Bundle

If you want one compact and defensible evaluation suite for a paper, use the following bundle.

| Role | Metrics |
|---|---|
| Primary motion quality | FD / MAE, RMSE |
| Primary temporal quality | FID△fm, SND |
| Secondary visual quality | FVD, LPIPS, FID, SSIM, PSNR |
| Semantic appropriateness | Ground-truth reaction embedding similarity |
| Subjective validation | Human appropriateness study |

## 8. Practical Recommendation

### 8.1 Recommended Order of Adoption

Use the metrics in the following order.

| Stage | Metrics | Why This Stage Comes Here |
|---|---|---|
| Stage 1 | FD / MAE, RMSE | Establishes whether the model predicts the correct FLAME motion. |
| Stage 2 | FID△fm, SND | Establishes whether the reaction dynamics are natural over time. |
| Stage 3 | FVD, LPIPS, optionally FID or SSIM or PSNR | Establishes whether rendered outputs are visually plausible. |
| Stage 4 | Ground-truth reaction embedding similarity, human study | Establishes whether the reaction is actually appropriate to the source content. |

### 8.2 What to Report at Each Stage

#### Stage 1: Motion Accuracy

Report:

1. One overall FD or MAE score.
2. One overall RMSE score.
3. One per-parameter-group table for `expr`, `jaw_pose`, `rotation`, `neck_pose`, `eyes_pose`, and `translation`.
4. Optionally, mean and standard deviation across validation sequences.

This stage proves that the generated reactions are numerically close to the paired targets in the model's native prediction space.

#### Stage 2: Temporal Realism

Report:

1. FID△fm on the same validation split.
2. SND on the same validation split.

This stage proves that the generated reactions evolve naturally rather than only matching framewise averages.

#### Stage 3: Visual Quality After Rendering

Report:

1. FVD as the main rendered-video metric.
2. LPIPS as the main rendered-frame perceptual metric.
3. FID, SSIM, and PSNR only as supplementary metrics for comparison with prior work.

This stage proves that the predicted FLAME motion still looks visually plausible when turned into videos.

#### Stage 4: Semantic Appropriateness

Report:

1. Ground-truth reaction embedding similarity as the main automatic semantic metric.
2. A small human appropriateness study if appropriateness is the central paper claim.

This stage proves that the reaction is not only realistic but also suitable for the source content or consistent with the target reaction behavior.

### 8.3 Minimal Strong Benchmark

If you need the smallest strong set for a first submission or internal benchmark, use:

1. FD / MAE
2. RMSE
3. FID△fm
4. SND
5. FVD
6. LPIPS
7. Ground-truth reaction embedding similarity

If you have time for one additional component, add a small human study.

### 8.4 What to Avoid in the Main Table

Avoid putting the following into the main headline table unless they are directly tied to your stated contribution:

1. SyncScore, Sync-C, Sync-D, LSE-C, LSE-D
2. LVE, MOD, FDD
3. CSIM
4. RPCC, rPCC, RTLCC, and RWTLCC by themselves

These metrics answer different questions such as lip articulation, identity preservation, or generic synchrony. They can distract from the core claim of content-appropriate facial reaction.

## 9. Recommended Paper Framing

When writing the evaluation section, the cleanest framing is:

1. Primary metrics evaluate paired motion correctness in FLAME space.
2. Temporal metrics evaluate whether the reaction evolves naturally.
3. Video metrics evaluate the quality of rendered outputs.
4. Semantic or human metrics evaluate whether the reaction is appropriate for the source content.

This layered structure gives you evidence for motion accuracy, temporal realism, final visual quality, and semantic appropriateness without over-relying on metrics that are better suited to speech articulation or identity preservation.