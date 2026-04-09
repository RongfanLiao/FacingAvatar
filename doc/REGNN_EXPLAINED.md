# REGNN Explained For A Beginner

This note has two goals:

1. explain what a graph neural network (GNN) is in plain language
2. explain what the REGNN model in this repository is doing, step by step

The explanation below is tied to the actual implementation in this repo, mainly:

- `benchmark/regnn.py`
- `train_regnn.py`
- `benchmark/lookingface.py`
- `benchmark/motion_transvae.py`

## 1. What Is A GNN?

A graph neural network is a neural network that works on **graphs** instead of only grids or sequences.

A graph has:

- **nodes**: the entities you care about
- **edges**: the relationships between those entities
- **node features**: numbers stored on each node
- sometimes **edge features**: numbers stored on each connection

Examples:

- social network: people are nodes, friendships are edges
- molecule: atoms are nodes, bonds are edges
- human skeleton: joints are nodes, bones are edges
- face motion model: facial components or output channels can be treated as nodes

The basic GNN idea is:

1. each node starts with a feature vector
2. nodes send information to their neighbors
3. each node updates its own state using messages from connected nodes
4. after several rounds, each node contains information about both itself and its neighborhood

This is often called **message passing**.

In a compact form, one update looks like:

$$
h_i^{(l+1)} = \text{Update}\left(h_i^{(l)}, \sum_{j \in \mathcal{N}(i)} \text{Message}(h_i^{(l)}, h_j^{(l)}, e_{ij})\right)
$$

Where:

- $h_i^{(l)}$ is node $i$'s feature at layer $l$
- $\mathcal{N}(i)$ is the neighbor set of node $i$
- $e_{ij}$ is the edge between node $i$ and node $j$

## 2. Why Use A GNN?

GNNs are useful when the problem is not just “what happened at this position?” but also “how do parts influence each other?”

That matters when:

- outputs are strongly correlated
- structure matters more than plain order
- some dimensions should communicate directly

For face motion, this is natural because different motion channels are related:

- jaw movement affects mouth-related expression
- neck and head motion interact with facial dynamics
- some output dimensions co-vary much more than others

So instead of predicting every output dimension independently, a graph-based model tries to learn the dependency structure between them.

## 3. How Is A GNN Different From A CNN Or Transformer?

Very roughly:

- **CNN**: best when data lives on a regular grid, like images
- **Transformer**: best when everything can attend to everything, usually in a sequence or token set
- **GNN**: best when you want to model explicit relational structure between entities

A GNN does not replace time modeling by itself. It gives you a way to model relationships among nodes. In this repo, REGNN combines temporal clips with graph reasoning.

## 4. What Problem Does REGNN Solve In This Repo?

The REGNN port here predicts the listener-side FLAME face motion target from speaker-side inputs.

The inputs come from the left side of a LookingFace pair:

- wav2vec audio features
- raw left video frames

The target comes from the right side:

- FLAME motion parameters loaded from `flame_param.npz`

At training time, the dataset is built in `train_regnn.py` and `benchmark/lookingface.py`.

The data path is:

1. load sequence IDs
2. load left audio features
3. load raw left video frames
4. load right-side FLAME target
5. pad variable-length sequences into a batch
6. cut each sequence into a fixed clip for training

## 5. High-Level Picture Of REGNN In This Codebase

The current REGNN implementation has three main stages:

1. **Perceptual processor**
2. **Cognitive processor**
3. **Motor processor**

The pipeline is:

$$
\text{audio+video clip} \rightarrow \text{fused per-frame features} \rightarrow \text{graph nodes + learned edges} \rightarrow \text{invertible graph mapping} \rightarrow \text{FLAME prediction}
$$

### REGNN Pipeline Diagram

The SVG figure below should render in standard Markdown preview.

If the SVG does not render in your viewer, the plain-text fallback is below.

```text
LookingFace batch
  |- left_audio_feat
  |- left_video_frames
  `- flame_target_118
          |
          v
build_regnn_clips
  |- left_audio_clip: B x T x audio_dim
  |- left_video_clip: B x T x 3 x H x W
  `- target_clip:     B x T x 118

Speaker-driven path:

left_audio_clip + left_video_clip
          |
          v
LookingFacePercepProcessor
          |
          v
BaselineSpeakerEncoder
  |- Conv3D video encoder
  |- audio linear map
  `- fusion layer
          |
          v
fused features: B x T x fused_dim
          |
          v
REGNNCognitiveProcessor
  |
  +--> MultiNodeMlp
  |       |
  |       `-> node features: B x target_dim x T
  |
  `--> EdgeLayer
          |
          `-> learned top-k edge graph

node features + edge graph
          |
          v
LipschitzGraph.inverse
          |
          v
prediction: B x T x target_dim

Training-only target path:

target_clip
   |
   v
transpose to target nodes: B x target_dim x T
   |
   v
LipschitzGraph.forward
   |
   v
listener_feature

Losses:

speaker_feature + listener_feature --> latent matching loss
prediction + target_clip          --> reconstruction loss
prediction + target_clip          --> velocity loss

all enabled loss terms ----------> total REGNN loss
```

How to read this diagram:

- the top path is the speaker-driven prediction path
- the lower target path exists during training so the model can align speaker and listener latent features
- the graph is built over target channels, not over time steps

In code, the main model is `LookingFaceREGNN` in `benchmark/regnn.py`.

## 6. Input And Tensor Shapes

With the default setup in `train_regnn.py`:

- `num_frames = 50`
- `target_dim = 118`
- `fused_dim = 64`

The training clip tensors are approximately:

- audio clip: `(B, 50, audio_dim)`
- video clip: `(B, 50, 3, H, W)`
- target clip: `(B, 50, 118)`

Where:

- `B` is batch size
- `50` is the fixed clip length
- `118` is the full FLAME target dimension in this training entrypoint

Important detail:

- the current `train_regnn.py` creates the model with `target_dim=118`
- the normal training path uses `flame_target_118`

## 7. Stage 1: Perceptual Processor

This stage is `LookingFacePercepProcessor` in `benchmark/regnn.py`.

It delegates to `BaselineSpeakerEncoder` from `benchmark/motion_transvae.py`.

That encoder does three things:

1. encode raw video frames with a Conv3D video encoder
2. project wav2vec audio features with a linear layer
3. concatenate video and audio features, then fuse them with another linear layer

So after this stage, each frame has a fused feature vector.

If the clip length is `T=50` and fused feature size is `F=64`, the output shape is:

$$
(B, T, F) = (B, 50, 64)
$$

This stage is not graph reasoning yet. It is multimodal feature extraction.

## 8. Stage 2: Cognitive Processor

This is `REGNNCognitiveProcessor` in `benchmark/regnn.py`.

It has two subparts:

- `MultiNodeMlp`
- `EdgeLayer`

### 8.1 What `MultiNodeMlp` is doing

This is the first place where the representation becomes graph-like.

Input to `MultiNodeMlp`:

- fused features of shape `(B, num_frames, fused_dim)`

The module is parameterized with:

- `n_nodes = num_frames`
- `out_dim = target_dim`

It transforms each frame position into a vector over target dimensions, then the result is transposed:

```python
converted = self.convert_layer(inputs)
node_features = converted.transpose(1, 2)
```

After the transpose, the tensor shape becomes:

$$
(B, target\_dim, num\_frames)
$$

With defaults, that is:

$$
(B, 118, 50)
$$

That means this port treats:

- each FLAME output dimension as a **graph node**
- each node's feature vector as its values across the clip window

This is a key design idea in this implementation.

### 8.2 What `EdgeLayer` is doing

`EdgeLayer` learns the relationships between those nodes.

Given node features, it:

1. computes query/key projections
2. forms attention scores between nodes
3. converts them into soft edge strengths
4. keeps only the top-$k$ neighbors per node, where $k = \text{neighbors}$
5. normalizes the resulting edge matrix

So the graph is **learned from the input**, not fixed in advance.

This is important: the model is not told that “jaw connects to mouth” manually. It learns a data-driven graph over FLAME channels.

The output of this stage is:

- `speaker_feature`: node features of shape `(B, target_dim, num_frames)`
- `edge`: learned graph connectivity

## 9. Stage 3: Motor Processor

This is `LipschitzGraph` in `benchmark/regnn.py`.

It is made of repeated `GraphLayer` blocks, and each `GraphLayer` uses `GraphAttention`.

The idea here is:

- use graph attention to propagate information between FLAME nodes
- keep the mapping stable by constraining it in a Lipschitz-friendly way
- use a reversible or approximately invertible structure

### 9.1 Graph attention here

`GraphAttention` takes:

- node features `x`
- learned edges `edge`

It then:

1. applies a nonlinearity
2. builds attention weights between nodes
3. combines these attention weights with the learned edge tensor
4. aggregates neighbor information
5. scales the result using a norm estimate for stability

In simpler terms: each FLAME channel updates itself by looking at related FLAME channels.

### 9.2 Residual graph block

Each `GraphLayer` does:

$$
y = x + f(x, edge)
$$

That is a residual update. Residual structures are easier to optimize and easier to invert approximately.

### 9.3 Why there is an `inverse()`

One unusual part of REGNN is that the motor processor is used in two directions:

- `forward(...)`: map target nodes into listener feature space
- `inverse(...)`: decode speaker features back into target motion

In the model's `forward()` method:

- `speaker_feature, edge = self.forward_features(...)`
- `decoded = self.motor_processor.inverse(speaker_feature, edge=edge, cal_norm=True)`

So prediction is obtained by running the inverse graph mapping.

If the target is available during training, the model also computes:

- `target_nodes = target_clip.transpose(1, 2)`
- `listener_feature, logdets = self.motor_processor(target_nodes, edge=edge)`

This gives a target-side latent representation that can be matched against the speaker-side latent representation.

That is the core REGNN idea in this port:

- build a speaker-driven graph latent
- build a target-driven listener latent
- train them to align
- use the inverse graph mapping to decode motion

## 10. What Happens In One Forward Pass?

For one training clip:

1. take left audio clip and left video clip
2. fuse them into per-frame speaker features
3. convert those features into graph node features over FLAME dimensions
4. learn an edge graph among FLAME nodes
5. invert the motor graph to produce predicted target motion
6. if ground-truth target is present, pass target nodes through the forward graph to obtain listener features
7. compute losses between prediction and target, and between speaker and listener latent features

## 11. How Clips Are Built

The helper `build_regnn_clips()` in `benchmark/regnn.py` creates fixed-size windows.

Training and validation differ slightly:

- training: uses a random start offset per sequence
- validation: always uses the first window

If a sequence is shorter than `num_frames`, the clip is padded with zeros.

The function returns:

- `left_audio_clip`
- `left_video_clip`
- `target_clip`
- `clip_lengths`
- `padding_mask`

This is how the model can train on variable-length source sequences while keeping a fixed graph size.

## 12. How The Loss Works

The loss class is `REGNNLoss` in `benchmark/regnn.py`.

The total loss is:

$$
L = \lambda_{latent} L_{latent} + \lambda_{mid} L_{mid} + \lambda_{logdet} L_{logdet} + \lambda_{rec} L_{rec} + \lambda_{vel} L_{vel}
$$

In practice, the important parts are:

### 12.1 Latent matching loss

This is the main term.

It compares:

- `speaker_feature`
- `listener_feature`

The idea is that the latent representation inferred from the speaker should match the latent representation extracted from the true listener motion.

This is why the logged key is `loss_match`.

### 12.2 Mid loss

This is an auxiliary regularization term.

It computes a per-sequence mean latent feature and encourages features within the sequence to stay close to that mean.

Intuition:

- reduce noisy or unstable latent variation
- make the listener latent space more coherent

### 12.3 Reconstruction loss

This compares predicted motion to target motion using a masked MSE over valid frames.

In the current code, the reconstruction breakdown is written for the content-related slices:

- expression
- jaw
- neck
- eyes

This term is present in the code, but in `train_regnn.py` the default weight is:

- `reconstruction_weight = 0.0`

So by default it is **off**.

### 12.4 Velocity loss

This penalizes differences in frame-to-frame motion velocity:

$$
(\hat{y}_t - \hat{y}_{t-1}) - (y_t - y_{t-1})
$$

It encourages temporal smoothness and correct motion dynamics.

In `train_regnn.py`, its default weight is also `0.0`, so it is off unless enabled.

### 12.5 Log-determinant loss

The code includes a `logdet` term for the invertible-style architecture.

In the current local implementation, `GraphLayer.forward()` returns `0.0` for the log-determinant contribution, so this term is effectively a placeholder unless the graph block is extended later.

## 13. What The Training Loop Actually Optimizes

The training loop in `train_regnn.py` calls `train_regnn()`.

For each batch:

1. build random fixed-size clips
2. move clips to device
3. run the model
4. compute the REGNN loss
5. backpropagate
6. apply gradient clipping
7. update the optimizer

Validation uses `validate_regnn()` with deterministic first-window clips.

One implementation detail worth noticing:

- `candidate_lengths` is set to all ones in both training and validation

That means the current latent loss is operating on one clip candidate per sample, not on a larger candidate pool.

## 14. How Inference Works

At inference time, the model uses `predict_sequence()`.

Instead of only using one clip, it:

1. splits the full sequence into chunks of length `num_frames`
2. pads the last chunk if needed
3. predicts each chunk independently
4. removes padding from the last chunk
5. concatenates the chunk predictions back into one full sequence

So training is clip-based, but evaluation can cover full sequences by chunking.

## 15. What Is The “Graph” In This REGNN, Exactly?

This is the most important conceptual point for a beginner.

In this implementation, the graph is **not** over frames.

It is over **target channels**.

More specifically:

- nodes correspond to FLAME output dimensions
- node features summarize the clip over time
- edges describe which output dimensions should exchange information

That means REGNN is modeling dependency structure among facial motion channels, not just temporal order.

Time is still present, but mostly as the per-node feature axis after the transpose.

## 16. Why This Design Can Make Sense

Suppose the model wants to predict one output dimension like jaw opening.

Jaw opening is not independent of:

- mouth expression coefficients
- some eye and neck motion patterns
- speaking rhythm in audio

REGNN tries to capture this by:

1. extracting multimodal speaker evidence from audio and video
2. turning target dimensions into interacting nodes
3. learning which nodes should communicate
4. decoding the final motion from that structured latent space

That is the “relational” part of the model.

## 17. Important Repo-Specific Notes

These are worth knowing because they affect how you should read the code.

### 17.1 The current training script uses 118-d targets

`train_regnn.py` instantiates:

```python
model = LookingFaceREGNN(target_dim=118, ...)
```

So the main training entrypoint is configured for full FLAME targets.

### 17.2 The reconstruction breakdown currently emphasizes content slices

Inside `REGNNLoss._masked_reconstruction_loss()`, the named reconstruction pieces explicitly cover:

- expression
- jaw
- neck
- eyes

Rotation and translation are not broken out there as separate named components.

That does not stop the model from predicting 118 dimensions, but it is a sign that this port still carries some assumptions from a content-focused setup.

### 17.3 The default training objective is mostly latent alignment

Because the default weights are:

- `latent_weight = 1.0`
- `mid_weight = 1.0`
- `reconstruction_weight = 0.0`
- `vel_weight = 0.0`
- `logdet_weight = 0.0`

the default objective is dominated by latent matching plus mid-loss.

That is not a bug, but it is important for understanding what REGNN is being asked to learn.

## 18. Mental Model To Keep In Your Head

If you want one short mental model, use this:

> REGNN in this repo first turns speaker audio-video clips into a set of interacting FLAME-channel nodes, learns a graph over those nodes, aligns that speaker graph with a target-side listener graph during training, and uses the inverse graph mapping to decode facial motion.

## 19. When REGNN Might Help

A graph-based model like this can be attractive when:

- output channels are highly correlated
- you want a learned dependency structure among output dimensions
- you want something more structured than framewise MLP prediction

## 20. When REGNN Might Be Hard To Train

Compared with simpler baselines, this design is harder because it combines:

- raw video encoding
- audio-video fusion
- learned graph construction
- invertible-style graph blocks
- multiple optional loss terms

If training is unstable, typical levers are:

- reduce `video_canvas_size`
- reduce `batch_size`
- use `num_workers=0` while debugging
- keep `grad_clip` enabled
- start with short smoke tests

## 21. Practical Reading Guide For The Code

If you want to read the code in a sensible order, use this sequence:

1. `train_regnn.py`: see how data loaders, model, loss, and optimizer are created
2. `benchmark/lookingface.py`: see what the batch actually contains
3. `benchmark/regnn.py`:
   - `LookingFaceREGNN`
   - `REGNNCognitiveProcessor`
   - `EdgeLayer`
   - `GraphAttention`
   - `REGNNLoss`
   - `train_regnn()` and `validate_regnn()`
4. `benchmark/motion_transvae.py`: see `BaselineSpeakerEncoder` and `VideoEncoder`

## 22. Short Summary

In one paragraph:

REGNN is a talking-face model that uses graph reasoning over FLAME output dimensions. It first encodes speaker audio and video into fused frame features, converts those features into graph nodes where each node corresponds to one facial-motion channel, learns a graph of dependencies among those channels, and then uses a Lipschitz-styled invertible graph module to both align speaker and listener latent spaces during training and decode final FLAME predictions. In this repository, the default training setup uses full 118-d FLAME targets, fixed 50-frame clips, learned top-$k$ graph edges, and a loss dominated by latent matching rather than direct reconstruction.