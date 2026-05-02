# HRNC Reaction Type VLM Prompt Template

Draft prompt template for automatic reaction type annotation in the current HRNC paper.
This template replaces the older "functional label" wording with the current
"reaction type" framing.

---

## Purpose

Use a vision-language model (VLM) to generate draft reaction type annotations for
HRNC clips. The output is intended as a preliminary annotation track and should be
validated or corrected by human annotators before being treated as ground truth.

The VLM should label the observable reaction pattern expressed by the reactor in
relation to the stimulus content. The task is not to infer the reactor's private
emotional state.

---

## Required Inputs

- `clip_id`: unique identifier of the NCFR clip.
- `stimulus_clip`: the stimulus video crop.
- `reaction_clip`: the facial-reaction video crop, temporally synchronised with
  the stimulus clip.
- `content_type`: one of documentary, game, movie, music, sports, talk show.
- `optional_context`: short human- or model-generated description of the stimulus
  content, if available.

The stimulus and reaction clips should cover the same time interval from the
original NCFR video.

---

## Reaction Type Inventory

Use only the following main reaction type families. Multiple families may be
selected when clearly supported, but one family must be marked as dominant.

| Code | Reaction type | Short description |
|---|---|---|
| A | Endorsement | agreement, affirmation, appreciation, approval, or personal resonance |
| B | Disendorsement | disagreement, rejection, skepticism, doubt, or confusion |
| C | Affective resonance | empathy, sympathy, or being moved by depicted people/events |
| D | Enjoyment | amusement, laughter, or sustained positive enjoyment |
| E | Surprise | unexpectedness, shock, disbelief, or pleasant surprise |
| F | Negative reaction | discomfort, disgust, aversion, tension, or anxiety |
| G | Cognitive engagement | attentive viewing, effortful processing, or absorption |
| H | Low engagement | minimal reaction, weak response, distraction, or disengagement |
| I | Other / unclear | compound, transitional, ambiguous, or insufficiently visible reaction |

---

## Decision Rules

1. Base the annotation on observable facial and head behaviour: gaze, brow, eyes,
   mouth, jaw, head pose, head movement and visible reaction timing.
2. Use the stimulus clip only to judge what the reaction is responding to. Do not
   label the stimulus content itself.
3. Do not infer private emotion, intention, personality or preference unless it is
   directly supported by visible behaviour.
4. Select the main family that best describes the dominant visible reaction
   pattern over the clip.
5. Use H when the reactor remains near baseline, shows only weak response, or
   appears disengaged from the stimulus.
6. Use I when the clip contains a clear transition between multiple families, the
   face is too occluded, the behaviour is too subtle, or no family can be assigned
   reliably.

---

## Copy-Ready VLM Prompt

```text
You are annotating a short natural-content facial reaction clip from the HRNC
dataset. You will be given two temporally synchronised videos:

1. stimulus_clip: the natural audio-visual content being watched.
2. reaction_clip: the reactor's facial reaction to the stimulus.

Your task is to assign reaction type labels to the reactor's visible facial and
head behaviour. Label the observable reaction pattern in relation to the stimulus,
not the reactor's private emotional state.

Use only visible evidence from the reaction_clip, with the stimulus_clip used as
context for what the reactor is responding to. Consider gaze, brow movement, eye
opening, mouth shape, jaw movement, head pose, head motion, stillness, timing and
changes across the clip.

Reaction type labels:
- A Endorsement: agreement, affirmation, appreciation, approval, or personal resonance
- B Disendorsement: disagreement, rejection, skepticism, doubt, or confusion
- C Affective resonance: empathy, sympathy, or being moved by depicted people/events
- D Enjoyment: amusement, laughter, or sustained positive enjoyment
- E Surprise: unexpectedness, shock, disbelief, or pleasant surprise
- F Negative reaction: discomfort, disgust, aversion, tension, or anxiety
- G Cognitive engagement: attentive viewing, effortful processing, or absorption
- H Low engagement: minimal reaction, weak response, distraction, or disengagement
- I Other / unclear: compound, transitional, ambiguous, or insufficiently visible reaction

Rules:
- Select one dominant reaction type family from A-I.
- You may select co-occurring reaction type families if clearly supported.
- If the clip contains a clear temporal transition between families, use I and
  list the ordered families in transition_order.
- If evidence is insufficient, use I and explain why.
- Do not guess labels that are not visually supported.

Return only valid JSON with this schema:

{
  "clip_id": "<clip_id>",
  "dominant_reaction_type": "<A|B|C|D|E|F|G|H|I>",
  "co_occurring_reaction_types": ["<A|B|C|D|E|F|G|H|I>", "..."],
  "transition_order": ["<A|B|C|D|E|F|G|H|I>", "..."],
  "visual_evidence": "<one or two sentences naming the visible facial/head cues>",
  "stimulus_relation": "<one sentence explaining what stimulus event or content the reaction appears to respond to>",
  "confidence": "<high|medium|low>",
  "uncertainty_reason": "<required if confidence is low or dominant_reaction_type is I; otherwise empty string>"
}
```

---

## Notes for Human Validation

VLM-generated labels should be treated as draft annotations. Human validators
should check whether the visual evidence supports the dominant label, whether
co-occurring labels are necessary, and whether ambiguous clips should be moved to
I rather than forced into a specific reaction type.
