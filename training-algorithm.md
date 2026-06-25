# Rollout Policy Iteration

KumpelBrain trains by repeatedly improving a frozen champion policy. Instead
of assigning one terminal `0` or `1` to whichever action happened to be taken,
the trainer recreates the same state, forces several alternatives, and plays
each branch to completion multiple times.

## Policy-Relative Values

For state \(s\), action \(a\), and continuation-policy mixture \(\mu\), the
learned quantity is

\[
Q^\mu(s,a)=P(\text{current player wins}\mid s,a,\mu).
\]

A single terminal result is a valid Bernoulli sample of this probability, but
it has high variance. Repeated forced continuations provide an aggregate
estimate and counterfactual information about actions that the seed game did
not choose.

Recreating a state reshuffles hidden decks. Stochastic policy sampling also
changes later choices. These effects provide the variation needed for repeated
rollouts to estimate a probability rather than reproduce one deterministic
game.

The value is policy-relative: it describes play against the configured
champion/history mixture, not perfect-play game-theoretic value.

## Rollout Targets

For \(w\) wins, \(d\) draws, \(l\) losses, and
\(n=w+d+l\), the trainer uses a uniform Beta prior:

\[
\hat Q(s,a)=\frac{w+0.5d+1}{n+2}.
\]

The prior prevents small rollout sets from creating exact zero or one targets.
The confidence weight

\[
c(n)=\frac{n}{n+2}
\]

reduces the influence of estimates supported by few successful continuations.

Rollout values are converted into an improved policy:

\[
\pi^*(a\mid s)=
\operatorname{softmax}\left(
\frac{\operatorname{logit}(
\operatorname{clip}(\hat Q,10^{-4},1-10^{-4})
)}{0.25}
\right).
\]

Only evaluated choices participate in this distribution. Unevaluated legal
choices receive no direct value or policy loss.

## Model And Loss

The state and interaction encoders are shared. Every legal interaction and
target candidate receives:

- a value logit \(v_\theta\), interpreted as
  \(Q_\theta=\sigma(v_\theta)\);
- a policy logit \(p_\theta\), used for play, exploration, and branch ranking.

Interaction and target selection are hierarchical. A target context is
identified by the interaction, the already-forced target prefix, the current
candidate deck IDs, and whether stopping is legal. Target alternatives are
evaluated by recreating the root and replaying the complete forced prefix.

The objective is

\[
L =
L^{interaction}_{value}
+ L^{interaction}_{policy}
+ L^{target}_{value}
+ L^{target}_{policy}.
\]

Value components use confidence-weighted soft-target binary cross entropy:

\[
L_{value} =
\frac{\sum_i c(n_i)\,
\operatorname{BCEWithLogits}(v_i,\hat Q_i)}
{\sum_i c(n_i)}.
\]

Policy components use cross entropy against the rollout-improved distribution:

\[
L_{policy} =
-\sum_{a\in A_{evaluated}}\pi^*(a\mid s)
\log\operatorname{softmax}(p)_a.
\]

Each present component is normalized independently and then summed. Target
components are omitted for batches without target evaluations. Optimization
uses AdamW with learning rate \(10^{-4}\), weight decay \(10^{-5}\), and
gradient norm clipping at `1.0`.

## Branch Budgets

States from all game phases are sampled every iteration. Root-action labels are
sampled-policy estimates, not exhaustive branch comparisons.

For each sampled root state:

1. apply the phase-based candidate mask below;
2. sample `root_action_rollouts` forced root interactions from
   `softmax(policy_logits / root_action_temperature)` over that mask;
3. force the sampled interaction and continue the game with the existing
   stochastic continuation policy;
4. aggregate successful outcomes by interaction index;
5. leave actions that were never sampled without a label for that root.

Default rollout budgets:

| Setting | Default |
|---|---:|
| `root_action_rollouts` | 10 |
| `root_action_temperature` | 1.0 |
| `target_contexts_per_action` | 2 |
| `target_choices_per_context` | 3 |
| `target_rollouts_per_choice` | 1 |

Phase-based action caps are used only as the candidate mask before softmax
sampling:

| Phase | Plies from end | Candidate mask |
|---|---:|---|
| terminal | 0–2 | all legal actions |
| late | 3–7 | all legal actions, capped at 8 |
| mid-late | 8–15 | policy top 4 + 2 random |
| mid | 16–31 | policy top 3 + 1 random |
| early | 32+ | policy top 3 + 1 random |

Target learning is lightweight and non-recursive. During the root-action
rollouts, the evaluator records observed target contexts. Per root action it
keeps at most `target_contexts_per_action` distinct contexts, evaluates policy
top 2 plus sampled candidates up to `target_choices_per_context`, and runs each forced target choice
`target_rollouts_per_choice` time.

Replay mismatches are recorded as missing observations instead of producing a
label. These mismatches are expected when hidden-card reshuffling changes a
determination. Engine, inference, timeout, and other runtime exceptions remain
fatal and stop the iteration immediately.

## Opponents And Promotion

Continuation policies use epsilon `0.10` and policy temperature `0.5`.
Opponents are sampled 75% from the current champion and 25% uniformly from the
four newest historical champions. This reduces overfitting and strategic
cycling.

Every candidate starts from champion weights with a fresh optimizer. It trains
from the five newest rollout shards. Root IDs are stable hashes of state bytes
and player perspective; 10% are permanently assigned to validation.

Candidates are promoted by playing strength, not validation loss. Candidate
and champion play equal numbers on each player side in blocks of 100 games.
After at least 200 games:

- promote when the one-sided 90% Wilson lower bound exceeds `0.5`;
- reject when the upper bound is at most `0.5`;
- otherwise continue to at most 800 games and reject if still inconclusive.

Promotion atomically replaces `champion.pt`, archives the previous champion,
and retains the four newest historical champions.

## Operational Loop

One `train_self_play.py` invocation:

1. generates stochastic seed games;
2. samples phase-balanced recreatable roots;
3. evaluates sampled forced root-action rollouts and capped target branches;
4. writes one aggregated rollout shard;
5. trains a fresh candidate;
6. gates it against the champion;
7. promotes or records rejection.

`automate_training.py` repeats complete iterations. Its runtime limit is
checked before starting the next iteration.

Rollout generation is expected to dominate runtime. Increasing rollouts
improves target precision approximately with \(1/\sqrt n\), so doubling
precision requires roughly four times as many continuations.

## Future Extension

The current implementation always rolls out to terminal. A later version can
add a state-value head and truncate expensive early-game continuations,
bootstrapping the remainder. That value head can also support batched MCTS.
Those extensions should be introduced only after forced-action rollout targets
and champion gating are empirically reliable.
