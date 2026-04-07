# Benchmark Metrics for Reaction Head Avatars

## 1. Purpose

This document defines a minimal but strong benchmark for evaluating reaction head avatars in this repository. The goal is to keep the benchmark small enough to be practical, while still covering the four questions that matter most:

1. Is the predicted reaction motion correct?
2. Does the reaction evolve naturally over time?
3. Does the reaction preserve the right temporal behavior beyond framewise error?
4. Does the whole motion trajectory stay behaviorally close to the paired target?

The recommended minimal benchmark contains the following six metrics:

1. MAE
2. RMSE
3. FID△fm
4. SND
5. frcorr
6. frdist

## 2. Why These Metrics Form a Strong Minimal Set

This benchmark is intentionally small, but each metric covers a different failure mode.

| Metric | Main Question It Answers | Role |
|---|---|---|
| MAE | Is the predicted FLAME motion close to the paired target? | Motion accuracy |
| RMSE | Are there large motion errors or unstable failures? | Motion robustness |
| FID△fm | Are frame-to-frame motion changes realistic? | Temporal realism |
| SND | Does the overall motion sequence look natural? | Sequence realism |
| frcorr | Does the predicted motion trajectory covary with the paired target? | Temporal agreement |
| frdist | Is the predicted motion path close to the paired target even with timing offsets? | Trajectory alignment |

Together, these metrics cover reconstruction quality, temporal behavior, and paired sequence agreement. That is enough to support a strong evaluation section without overloading the paper with metrics that answer unrelated questions such as lip-sync or identity preservation.

## 3. Metric Details

### 3.1 MAE

#### What it is

MAE, the mean absolute error between predicted and ground-truth FLAME parameters.

#### How it is computed

For each validation sequence:

1. Predict the FLAME parameters for every frame.
2. Concatenate the target parameter groups in a consistent order:
   1. `expr`
   2. `jaw_pose`
   3. `rotation`
   4. `neck_pose`
   5. `eyes_pose`
   6. `translation` (not used in the training)
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

### 3.5 frcorr

#### What it is

`frcorr` is a frame-sequence correlation score derived from the concordance correlation coefficient (CCC) between predicted and ground-truth motion trajectories.

#### How it is computed

For each validation sequence:

1. Take the predicted FLAME sequence and the paired ground-truth FLAME sequence.
2. For each output dimension, compute the concordance correlation coefficient:

$$
\operatorname{CCC}(x, y) = \frac{2\rho\sigma_x\sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}
$$

3. Average the per-dimension CCC values.
4. Average the per-sequence results over the evaluation split.

#### Why it is suitable

1. It checks whether the predicted motion follows the same temporal trend as the paired target.
2. It is stricter than simple correlation because it also penalizes mean and scale mismatch.
3. It stays in FLAME space, which matches the model output space directly.

#### What it captures well

1. Temporal agreement with the paired target
2. Whether rises and falls in motion happen in the right pattern
3. Similarity of trajectory shape, mean, and variance

#### What it does not capture well

1. Distribution-level realism across the whole dataset
2. Large local misalignments that could still preserve broad trend agreement

### 3.6 frdist

#### What it is

`frdist` is a trajectory-distance metric built from grouped Dynamic Time Warping (DTW) comparisons between predicted and ground-truth FLAME sequences.

#### How it is computed

For each validation sequence:

1. Split the FLAME target into meaningful groups.
2. For each group, compute a DTW distance between the predicted sequence and the paired ground-truth sequence.
3. Weight the group distances and sum them.
4. Average over the evaluation set.

Conceptually, this measures the minimum alignment cost between two motion trajectories while allowing small timing shifts.

#### Why it is suitable

1. It captures path-level similarity rather than only framewise error.
2. It is more tolerant to slight timing offsets than strict per-frame metrics.
3. It is useful for motion sequences where two reactions may be similar but slightly phase-shifted in time.

#### What it captures well

1. Whole-trajectory alignment
2. Similarity of motion paths even with mild timing mismatch
3. Grouped motion closeness in expression and pose space

#### What it does not capture well

1. Whether the generated motion distribution looks realistic at the dataset level
2. Visual realism of any rendered output

## 4. Why These Metrics Are Suitable for This Repository

This repository predicts FLAME parameters directly and only later converts them into visualization-ready outputs. That makes the following evaluation logic the most defensible:

1. First evaluate in FLAME space, because that is the model's native output space.
2. Then evaluate temporal realism in FLAME space, because reaction quality depends strongly on dynamics.
3. Then evaluate paired trajectory agreement in FLAME space, because the task has paired supervision.

The minimal benchmark follows exactly this logic:

| Evaluation Need | Metric That Covers It |
|---|---|
| Paired motion correctness | MAE, RMSE |
| Temporal naturalness | FID△fm, SND |
| Paired trajectory agreement | frcorr, frdist |

This is why the benchmark is both minimal and strong. It does not waste space on metrics that are mainly about lip articulation, identity preservation, unrelated synchrony definitions, or rendering-heavy side tasks.

