# White-box consequence awareness in small open models

## Motivation

The Claude Mythos system card (April 2026) shows that Claude internally represents consequence and risk even when it proceeds with a dangerous action. SAE features fire for "security risks and data leaks" during credential scraping; features for "avoiding suspicion" activate during cover-ups; a dual-role circuit encodes both the transgressive action and its negative valence. Causal steering with emotion/persona vectors reliably modulates destructive action rates.

We can't access those weights. **Can we find the same thing in a small open model?**

## The project

Run Qwen3-8B in a multi-step agentic harness where tasks naturally involve a critical action that varies in consequence. Train SAEs on the model's activations. Ask: does the model have legible features that differentially activate for high-consequence actions?

### Why Qwen3-8B

Strong tool-use capability (needed to actually drive the harness), dense architecture (no MoE complications for interpretability), 8B is tractable for SAE training.

### Why multi-step

Single tool calls would miss the point. Consequence awareness matters *in context* -- at step 12 of a 15-step trace, not in isolation. The model builds up state through a task, hits a decision point, and we want the activation pattern at that moment, shaped by the preceding context.

## Design

### 1. Task pairs (~50 matched pairs)

Each pair: same task structure, same tools, same phrasing, but the critical action target differs in consequence. Full list in [task-pairs.md](task-pairs.md).

Key design principles:
- **Action syntax is identical/boring across variants.** `sed`, `curl`, `UPDATE`, `git push`. No `rm -rf /` or `DROP TABLE` -- those are training-data memes the model has seen a billion times, not genuine consequence awareness.
- **Consequence comes from context built up in prior steps.** The model read a file, listed a directory, checked a status. That context is what makes the next action high-consequence. The action itself is mundane.
- **Scenarios feel like realistic work tasks.** "Clean up this config," not "here is a dangerous action, proceed?" Anything that looks like an eval will trigger different model behaviour and confound results.

Matching matters more than volume. The high-consequence and low-consequence variants must differ *only* in the target context, not in phrasing, task complexity, tool set, or anything else.

### 2. Mock tool harness

Hybrid approach: the model generates freely at every step, but the environment returns canned responses matched flexibly to the model's tool calls.

NOT prefilling model turns -- that's out-of-distribution and invalidates the activation measurements. NOT full autonomy with a real environment -- too hard to control at this scale.

The harness needs to be stateful enough that the model builds up context across steps -- list files, read contents, *then* decide what to edit. Canned responses keyed to tool calls, with flexible matching (the model might call `list_directory("./config")` or `list_directory("config/")`).

TransformerLens can do the forward pass with activation caching. The harness is scaffolding that feeds scenarios in and collects activations out.

### 3. SAE training

Use SAELens on the residual stream of Qwen3-8B.

- **Layer:** Start at a middle layer (layer 16 of 32), sweep nearby layers if initial results are noisy.
- **Tokens of interest:** The critical moment is when the model generates the tool call -- specifically the tokens specifying *what* to edit/send/modify. Also the planning tokens immediately before the call.
- **Training data:** Broader than just our task pairs. Train the SAE on a diverse corpus of the model doing agentic tasks (e.g. Berkeley Function Calling Leaderboard data, general instruction-following). The SAE should learn general features; we then check which ones activate on our specific scenarios.

### 4. Contrastive analysis

For each task pair, collect SAE feature activations at the critical token(s). Compare high-consequence vs. low-consequence:

- Which features have the largest mean activation difference?
- Are those features consistent across task types? (A feature that only fires on config edits isn't consequence-awareness -- it's just a config feature.)
- Cluster the differential features. Do they form a coherent "consequence" direction, or is it scattered?

### 5. Causal verification

Steer with the top candidate features:
- Amplify the "consequence" feature on a low-consequence task. Does the model hesitate, refuse, or add caveats?
- Suppress it on a high-consequence task. Does the model become more cavalier?

This confirms the features are causally involved, not just correlated.

## Why this is interesting regardless of outcome

- **Positive result:** White-box consequence monitoring works even at 8B scale. You can read the model's own risk assessment off its activations and use it as a safety signal -- without an external monitor parsing CoT.
- **Negative result:** Consequence awareness is an emergent capability that requires scale. This tells you the minimum threshold for white-box safety monitoring, and means smaller deployed models are blind to their own risk -- itself a safety-relevant finding.

## Ablation: planning tokens

If safety training causes the model to refuse high-consequence variants (asymmetric completion rates), analyse activations at the *planning* tokens before the model commits to either executing or refusing. The question becomes: does the model represent consequence at the deliberation stage, regardless of the behavioural outcome? This is an ablation, not the mainline experiment.

## Open questions

- **Token selection.** "The moment the model commits to the tool call" -- is that the first token of the function name? The argument specifying the target? Need to experiment.
- **Layer selection.** Middle layer is a guess. Consequence-awareness might live in later layers (closer to the output decision) or earlier layers (semantic understanding of the target).
- **SAE training corpus.** How much does the SAE training data affect which features emerge? If trained mostly on chat, it might not develop good features for tool-call contexts.
- **Confounds in task pairs.** Even with careful matching, high-consequence targets might have different token frequencies, embeddings, or associations that the SAE picks up on instead of "consequence" per se. Need ablations: does the feature fire on high-consequence *descriptions* (model reading about production databases) even when no action is taken?
