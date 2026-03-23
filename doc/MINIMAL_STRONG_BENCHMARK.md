# Minimal Strong Benchmark for Reaction Head Avatars

## 1. Purpose

This document defines a minimal but strong benchmark for evaluating reaction head avatars in this repository. The goal is to keep the benchmark small enough to be practical, while still covering the four questions that matter most:

1. Is the predicted reaction motion correct?
2. Does the reaction evolve naturally over time?
3. Does the rendered output look visually plausible?
4. Is the reaction semantically appropriate to the source content?

The recommended minimal benchmark contains the following seven metrics:

1. FD or MAE
2. RMSE
3. FID△fm
4. SND
5. FVD
6. LPIPS
7. Ground-truth reaction embedding similarity

## 2. Why These Metrics Form a Strong Minimal Set

This benchmark is intentionally small, but each metric covers a different failure mode.

| Metric | Main Question It Answers | Role |
|---|---|---|
| FD or MAE | Is the predicted FLAME motion close to the paired target? | Motion accuracy |
| RMSE | Are there large motion errors or unstable failures? | Motion robustness |
| FID△fm | Are frame-to-frame motion changes realistic? | Temporal realism |
| SND | Does the overall motion sequence look natural? | Sequence realism |
| FVD | Does the rendered video behave like a real video? | Video realism |
| LPIPS | Do rendered frames look perceptually similar to the reference? | Frame perceptual quality |
| Ground-truth reaction embedding similarity | Is the generated reaction semantically close to the paired real reaction? | Semantic appropriateness |

Together, these metrics cover reconstruction quality, temporal behavior, rendered quality, and semantic appropriateness. That is enough to support a strong evaluation section without overloading the paper with metrics that answer unrelated questions such as lip-sync or identity preservation.

## 3. Metric Details

### 3.1 FD or MAE

#### What it is

FD in listener-generation papers is sometimes used loosely for a motion-space feature distance. In this repository, the most stable and interpretable version for a minimal benchmark is MAE, the mean absolute error between predicted and ground-truth FLAME parameters.

#### How it is computed

For each validation sequence:

1. Predict the FLAME parameters for every frame.
2. Concatenate the target parameter groups in a consistent order:
   1. `expr`
   2. `jaw_pose`
   3. `rotation`
   4. `neck_pose`
   5. `eyes_pose`
   6. `translation`
3. Compute the element-wise absolute difference between prediction and ground truth.
4. Average over valid frames and dimensions.

In formula form:

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|
$$

where $\hat{y}_i$ is the predicted FLAME value and $y_i$ is the ground-truth FLAME value over all valid frame-dimension entries.

#### Why it is suitable

1. It directly matches the model output space.
2. It uses the paired supervision already available in the repo.
3. It is easy to interpret and easy for reviewers to trust.
4. It should be reported both overall and per parameter group, because different groups capture different aspects of behavior.

#### What it captures well

1. Framewise reconstruction quality
2. Parameter-level accuracy
3. Per-component errors such as expression versus head motion

#### What it does not capture well

1. Temporal smoothness
2. Distributional realism
3. Semantic appropriateness

### 3.2 RMSE

#### What it is

RMSE is the root mean squared error between predicted and ground-truth FLAME parameters.

#### How it is computed

For all valid frame-dimension entries:

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}
$$

Like MAE, it can be reported overall and per parameter group.

#### Why it is suitable

1. It complements MAE by penalizing large mistakes more strongly.
2. It is useful for detecting occasional unstable frames that MAE may understate.
3. It remains in the original unit scale after the square root, so it is interpretable.

#### What it captures well

1. Sensitivity to large prediction errors
2. Unstable or spiky motion prediction failures

#### What it does not capture well

1. Temporal naturalness by itself
2. Whether the reaction is semantically appropriate

### 3.3 FID△fm

#### What it is

FID△fm is a Fréchet-style distance computed on frame-to-frame motion differences rather than raw frames or raw parameters. Instead of asking whether single frames are close, it asks whether the dynamics of motion look realistic.

#### How it is computed

For each sequence:

1. Compute the motion difference between adjacent frames for both prediction and ground truth:

$$
\Delta \hat{y}_t = \hat{y}_t - \hat{y}_{t-1}, \qquad \Delta y_t = y_t - y_{t-1}
$$

2. Collect these motion-delta vectors over the evaluation set.
3. Estimate the mean and covariance of predicted motion deltas and ground-truth motion deltas.
4. Compute a Fréchet distance between the two Gaussian statistics.

Conceptually:

$$
\text{FID}\Delta fm = ||\mu_p - \mu_g||_2^2 + \operatorname{Tr}(\Sigma_p + \Sigma_g - 2(\Sigma_p \Sigma_g)^{1/2})
$$

where $(\mu_p, \Sigma_p)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of predicted and ground-truth motion-delta features.

#### Why it is suitable

1. Reaction quality depends on how the face changes over time, not only on framewise closeness.
2. It detects stiff, jittery, or over-smoothed motion that MAE and RMSE may miss.
3. It is directly aligned with the motion-generation literature you summarized.

#### What it captures well

1. Local temporal realism
2. Naturalness of motion transitions
3. Motion statistics over time

#### What it does not capture well

1. Exact framewise pairing
2. Semantic appropriateness by itself

### 3.4 SND

#### What it is

SND, or Sequence Naturalness Distance, is a sequence-level metric that evaluates whether generated motion sequences behave like real motion sequences in a more holistic way than framewise error.

#### How it is computed

Exact implementations differ by paper, but the usual structure is:

1. Represent each generated and real motion sequence with a sequence-level motion descriptor or embedding.
2. Compare the distribution of generated sequences and real sequences.
3. Summarize the discrepancy into one score, often using statistics related to both raw motion and motion differences.

In practice for this repo, the important point is to compute SND over full FLAME motion sequences on the same validation split used for the reconstruction metrics.

#### Why it is suitable

1. It checks whether complete reaction sequences look natural rather than merely close frame by frame.
2. It is useful for long-form reactions where timing and continuity matter.
3. It complements FID△fm by operating at the sequence level rather than only at adjacent-frame differences.

#### What it captures well

1. Whole-sequence naturalness
2. Distributional similarity of generated and real reactions
3. Long-range behavioral plausibility

#### What it does not capture well

1. Fine-grained semantic fit to the content
2. Visual rendering quality

### 3.5 FVD

#### What it is

FVD, or Fréchet Video Distance, is a realism metric computed on video features rather than static images. It compares distributions of generated videos and real videos, including temporal information.

#### How it is computed

1. Render the predicted FLAME outputs into videos.
2. Collect the corresponding ground-truth rendered videos.
3. Pass both sets of videos through a pretrained video feature extractor.
4. Estimate the feature mean and covariance for generated and real videos.
5. Compute the Fréchet distance between the two distributions.

The form is analogous to FID, but the features come from videos rather than images.

#### Why it is suitable

1. It is the strongest single video-space realism metric in this minimal set.
2. It reflects temporal behavior in the rendered domain.
3. It is more suitable than frame-only metrics when the final artifact is a video.

#### What it captures well

1. Overall realism of rendered videos
2. Temporal consistency in the visual domain
3. Dataset-level video quality

#### What it does not capture well

1. Paired motion accuracy in FLAME space
2. Whether a specific reaction is semantically appropriate to the source clip

### 3.6 LPIPS

#### What it is

LPIPS is a perceptual image similarity metric based on deep visual features. It is designed to align better with human perception than simple pixel-wise scores such as PSNR.

#### How it is computed

1. Render predicted frames.
2. Render or collect the paired ground-truth frames.
3. Pass both through a pretrained perceptual network.
4. Compute the distance between deep features at one or more layers.
5. Average the distances across frames.

#### Why it is suitable

1. It is a stronger frame-level metric than SSIM or PSNR for perceptual quality.
2. It helps show that the final visual output is not only numerically similar but visually plausible.
3. It is a common complement to FVD in video-generation evaluation.

#### What it captures well

1. Perceptual frame similarity
2. Appearance-level visual closeness

#### What it does not capture well

1. Sequence-level temporal realism
2. Semantic appropriateness

### 3.7 Ground-truth Reaction Embedding Similarity

#### What it is

This metric compares the generated reaction and the paired real reaction in a pretrained semantic embedding space, such as a facial-expression encoder, affect encoder, or video-level reaction encoder.

#### How it is computed

1. Choose a pretrained encoder that captures expression or affective behavior.
2. Encode each generated reaction video or expression sequence into an embedding vector.
3. Encode the paired ground-truth reaction in the same embedding space.
4. Compute similarity, usually cosine similarity:

$$
\operatorname{sim}(z_g, z_r) = \frac{z_g^\top z_r}{||z_g||\,||z_r||}
$$

5. Average the similarity over the evaluation set, or report retrieval-style ranking if you prefer a matching-based formulation.

#### Why it is suitable

1. It is the best automatic semantic metric for the current paired setup.
2. It goes beyond low-level motion or image similarity and asks whether the generated reaction carries the same reaction meaning as the paired real one.
3. It is more directly tied to the claim of appropriate response than MAE, RMSE, or FVD.

#### What it captures well

1. Semantic closeness of reactions
2. Expression-level or affect-level similarity
3. Appropriateness relative to the paired target reaction

#### What it does not capture well

1. Low-level pixel accuracy
2. Fine-grained physical correctness in FLAME space
3. True human preference if the encoder is biased or limited

## 4. Why These Metrics Are Suitable for This Repository

This repository predicts FLAME parameters directly and only later converts them into visualization-ready outputs. That makes the following evaluation logic the most defensible:

1. First evaluate in FLAME space, because that is the model's native output space.
2. Then evaluate temporal realism in FLAME space, because reaction quality depends strongly on dynamics.
3. Then evaluate rendered videos, because the final artifact may still be shown visually.
4. Finally, evaluate semantic appropriateness, because the core research claim is not only realism but reaction suitability to different content.

The minimal benchmark follows exactly this logic:

| Evaluation Need | Metric That Covers It |
|---|---|
| Paired motion correctness | FD or MAE, RMSE |
| Temporal naturalness | FID△fm, SND |
| Rendered realism | FVD, LPIPS |
| Semantic appropriateness | Ground-truth reaction embedding similarity |

This is why the benchmark is both minimal and strong. It does not waste space on metrics that are mainly about lip articulation, identity preservation, or unrelated synchrony definitions.

## 5. Reporting Recommendations

For a clean paper presentation, report the benchmark in four blocks.

### 5.1 Motion Accuracy Block

Report:

1. FD or MAE
2. RMSE
3. Per-group scores for `expr`, `jaw_pose`, `rotation`, `neck_pose`, `eyes_pose`, and `translation`

### 5.2 Temporal Realism Block

Report:

1. FID△fm
2. SND

### 5.3 Visual Quality Block

Report:

1. FVD
2. LPIPS

### 5.4 Semantic Appropriateness Block

Report:

1. Ground-truth reaction embedding similarity

If possible, add a small human study to support this block.

## 6. Final Recommendation

If you can only afford one compact benchmark that still looks technically serious, use exactly this set:

1. FD or MAE
2. RMSE
3. FID△fm
4. SND
5. FVD
6. LPIPS
7. Ground-truth reaction embedding similarity

This set is strong because:

1. it matches the model's actual output space
2. it evaluates both framewise accuracy and temporal realism
3. it includes rendered-video quality without letting rendering dominate the evaluation
4. it adds one semantic metric so the benchmark speaks to reaction appropriateness rather than only reconstruction