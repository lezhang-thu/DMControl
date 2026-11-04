# Integrating the ReSPRIT world model into `spr_sac`

The world-model extension itself fits this codebase. The real input gap is which latent the actor and the critic read, and the real on-policy gap is that a squashed Gaussian has no exact sum over actions.

Scaled-by-Resetting is out of scope. The only piece of that schedule to keep is the first warmup.

## What is being ported

In `resprit`, the world model is the existing SPR transition plus two readout heads. `ConvTMCell` already consumes the action. `reward_head` and `continue_head` read only the latent. Their losses use `stop_gradient` on the features, so the encoder and the transition are supervised only by the SPR loss. The heads are trained on both the open-loop rollout latents and the real future encoder latents.

Imagination is a second loop, `imagine_from_latent`. From a real replay latent it samples the current policy for `imag_horizon` steps (5 in `BBF-100K.gin`), predicts reward and continuation, and builds a λ-return. Every latent, reward, continuation, and return is stop-gradient. That loss trains only the policy and the Q head.

The warmup is a multiplier on those two imagination losses, not a reset:

```python
# Imagination is gated off right after each shrink-and-perturb reset
# (the Q target the imagined values come from is also reset), then
# ramped back in linearly over another imag_warmup gradient steps.
imag_ramp = float(
    np.clip(
        (self.cycle_grad_steps - self.imag_warmup) /
        max(1, self.imag_warmup), 0.0, 1.0))
imag_actor_mult = self.imag_actor_weight * imag_ramp
imag_value_mult = self.imag_value_weight * imag_ramp
```

With `imag_warmup = 2000`, the imagination weights are 0 for 2000 gradient steps, then rise linearly to their full values over the next 2000. Reward loss, continue loss, SPR, and the real Q update stay on the whole time. `cycle_grad_steps` is cleared inside `reset_weights`. Keeping only the first warmup means one counter from the first gradient step, never cleared, and no shrink-and-perturb, no parameter copy, and no repeated ramp.

## 1. State and action as inputs

The transition does not need a new input interface. `DeterministicTransitionModel` already maps `cat(latent, action)` to the next latent, and the replay actions are the squashed actions in `[-1, 1]`. A continuous action is a vector in that concatenation. In ReSPRIT the same slot is a one-hot. Reward and continue stay functions of the latent alone, with ReSPRIT's arrival alignment: the latent of `s_{k+1}` predicts `r_k` and the continuation at `s_{k+1}`. These six Control Suite rewards are state rewards (pose, speed, distance), summed over `action_repeat` by `ControlSuiteEnv.step`, so a latent readout matches the stored target.

SAC's `(state, action)` requirement is the critic, `QFunction.forward`. In ReSPRIT the Q head maps a latent to a value for every discrete action, so `V(z) = Σ_a π(a|z) Q(z, a)` never receives an action tensor. Here the action has to be concatenated. In the imaginary loop that action is the one just sampled, so the tensor exists. The call is `Q(z_t, a_t)` on the latent, not `critic(pixels, a_t)`.

Three input mismatches are real:

**The actor and the critic do not share a latent.** `SPR.encoder` is `critic.encoder`. The actor encoder shares only the conv layers, through `copy_conv_weights_from`. Its linear layer and LayerNorm are trained by the actor loss. The critic's are trained by the critic loss and by SPR. `Actor.forward` always runs its own encoder, and `Critic.forward` always runs the critic encoder. An imagined state is a transition output, not pixels, so neither encoder can be applied. The trunk that samples `a` and the Q trunk that scores `(z, a)` have to accept the SPR latent directly. Today the actor trunk was trained on a different 50-dimensional vector than the one the transition rolls out.

**SPR matches a projection, not the raw latent the control heads see.** `do_spr_loss` trains `predictor(projector(z_pred))` against `target_projector(target_encoder(image))`, after L2 normalization. ReSPRIT then applies Q and the policy through that same `project` module. Here the Q MLP and the actor trunk are separate, and they are trained on `LayerNorm(fc(conv(image)))`. The transition output is an unbounded linear map. Nothing in `DeterministicTransitionModel` renormalizes it. ReSPRIT min-max renormalizes every transition cell, so a short rollout stays on a similar scale to encoder features. A 5- or 6-step rollout here can leave the feature scale that Q and the actor have seen, even when the SPR projection loss is small.

**The disabled reward head is not the ReSPRIT head.** `reward_loss_weight` is 0. The module emits one unit, `update_spr` applies `log_softmax` over that unit, and the target is `to_categorical(..., limit=0)`, which builds a length-1 distribution. Gradients from that loss would also enter the transition, because the latent is not detached. The continue target is not wired either: `sample_spr` loads `not_dones` and then omits it from `spr_kwargs`, and it drops every window that contains a `real_done`. A new detached scalar reward head and a new continue head are the extension. The current predictor is not a starting point.

The continue target has a domain problem on top of the wiring. `not_dones` is 0 only for a true termination (`timestep.discount == 0`). A time limit sets `truncated`, and the SAC critic still bootstraps. `sample_spr` never shows the heads a boundary. On these six tasks a true terminal is rare, so a continue head trained like the current SPR batch collapses to 1, and the λ-return reduces to a plain discounted sum. That is acceptable if it is intentional. It is not a learned termination model until windows that contain endings are kept, with time limits labeled as continue.

## 2. On-policy updates with continuous actions

Sampling the action is not the hard part. `Actor` already draws a squashed Gaussian and `squash` already corrects `log π`. The imaginary step is: read the latent, sample `a ∈ [-1, 1]`, step the transition with that squashed action, read reward and continue from the detached next latent. One latent step is one agent step, so the discount stays `0.99`, the same discount the real critic uses.

The hard part is the value ReSPRIT computes by summing over actions. Its imagined actor is REINFORCE,

`-log π(a|z) * stopgrad((λ-return - V(z)) / scale) - β * entropy`,

with `V(z) = Σ_a π(a|z) Q_target(z, a)` and the categorical entropy exact. A diagonal Gaussian on cheetah or walker has 6 dimensions. There is no exact sum. A sampled baseline is a high-variance estimator of that same loss.

The actor update already in `update_actor_and_alpha` is the continuous replacement that avoids the sum: sample `a = tanh(μ + σ ε)` and minimize `α log π(a|z) - min(Q1, Q2)`. It is on-policy in the same sense as the real SAC update. The imagined critic analogue of ReSPRIT's distributional value loss is an MSE of the online twin Qs at `(z_t, a_t)` toward a stop-gradient λ-return. The bootstrap inside that return is the target critic on a fresh policy sample, which is the SAC backup `min Q_target - α log π` if the two critics are to agree. ReSPRIT bootstraps `E[Q]` and puts entropy only in the actor. Using that backup here would disagree with `update_critic`, which subtracts `α log π`.

One leak exists only in the continuous graph. ReSPRIT's action is an integer index, so the imagination loss cannot reach the transition through the action. SAC's Q is differentiable in `a`, and `a` is differentiable in the actor. If the latent still has a gradient, the actor loss trains the transition and the encoder. ReSPRIT prevents that by stop-gradient on the latent at every imagined step, including the real start state, and again on the predicted rewards, continuations, and returns. The same cuts are required here. Reward and continue heads stay trained only by their own detached regression losses.

`α` can stay the temperature learned on real replay. ReSPRIT's entropy coefficient is a schedule shared by the real and imagined actors. This code learns one `log_alpha`. The imagined actor can use that value without a second temperature.

The imagination weights are also on a different clock than in ReSPRIT. This learner takes one gradient step per agent action. A 100k-step run is about 12.5k updates for cartpole (`action_repeat` 8), 25k for cheetah, and 50k for walker. ReSPRIT's 2000-step warmup was written for a replay ratio of 64, and it is repeated after every reset. Copied once as 2000 updates, the off period plus the ramp is a large share of a 100k cartpole run. `init_steps: 1000` is a different warmup: random actions before any gradient. The imagination gate starts at the first call to `update`.

## Decisions to settle before implementation

1. Share one encoder feature between the SPR transition, the actor trunk, and both Q heads, so imagination can query `π(z)` and `Q(z, a)` without a pixel encoder.
2. Keep reward and continue as detached latent readouts with arrival alignment, and decide whether episode boundaries are included in their batches.
3. Use the reparameterized SAC actor loss and a scalar λ-return for the twin Qs inside the rollout, with the latent detached at every step.
4. Decide whether the λ-return bootstraps the soft target `min Q_target - α log π` or the plain expected Q used in ReSPRIT.
5. Run the imagination weight gate once: 0 for `imag_warmup` gradient steps, linear ramp over the next `imag_warmup`, and choose that count against the one-update-per-action budget.

## Updated plan for the two representation mismatches

This replaces the earlier proposal that kept the actor loss off the shared encoder. Imagination, reward heads, and continue heads stay out of this step. The SPR rollout uses replay actions.

One pixel encoder remains: `critic.encoder`. The actor encoder and `copy_conv_weights_from` go away. Call its output `z`.

The transition stays in that raw space. With the replay action `a_k` from `sample_spr`,

`z_0 = encoder(o_0)`, then `z_{k+1} = T(z_k, a_k)`.

Each control head gets its own projection of `z`, with its own predictor and its own target projection:

| Branch | Online projection | Predictor | Target projection | Who reads the projection |
| --- | --- | --- | --- | --- |
| Q | `q_proj` | `q_predictor` | `q_proj_target` | critic |
| Policy | `pi_proj` | `pi_predictor` | `pi_proj_target` | actor trunk |

Each module has the same shape as today's `global_classifier`: `feature_dim → latent_dim → feature_dim`. The single `global_classifier`, `global_target_classifier`, and `global_final_classifier` are removed. Branch width stays `encoder_feature_dim`, so the Q input width and the trunk input width stay as they are.

`q_proj(z)` is the critic feature. `pi_proj(z)` is the actor feature. Both are functions of the same `z`. The heads do not read the raw transition output, and they do not read the predictor output.

For every latent in the open-loop rollout, including `z_0`:

- Q term: normalize `q_predictor(q_proj(z_k))` and match `q_proj_target(target_encoder(o_k))`.
- Policy term: normalize `pi_predictor(pi_proj(z_k))` and match `pi_proj_target(target_encoder(o_k))`.

Each target projection is stop-gradient, as `global_target_classifier` is today. Each branch is normalized on its own. The branch loss is the squared error of those unit vectors, the same cosine form as `spr_loss`. The SPR loss is the sum of the two branch losses.

The existing split between the first step and the later steps stays. `t0_spr_loss_weight` and `model_spr_weight` multiply that sum. Both branches have equal weight inside the sum.

On a real critic update, `Q` takes `q_proj(encoder(obs))` concatenated with the replay action. The Bellman target takes `q_proj_target(critic_target.encoder(next_obs))` concatenated with the target action. The target action still comes from the current actor, as in `update_critic`.

On a real actor update, the trunk takes `pi_proj(z)` and the gradient continues into the shared encoder. This matches the real-data policy path in ReSPRIT: `policy_logits_from_feature` reads the online encoding with no stop-gradient, and only the advantage Q values are stopped. `detach_encoder=True` in `update_actor_and_alpha` is not used on this path. Sampling stays inside `update_actor_and_alpha`. It is the existing SAC actor sample, and it is not an input to the transition.

`q_proj_target` and `pi_proj_target` are exponential-moving-average copies of the online projections, updated with `momentum_tau` in the same place as `global_target_classifier`. The predictors are online only and are not copied into a target.

Parameter updates:

- The transition is trained only by the two SPR terms.
- The predictors are trained only by their own SPR term.
- `q_proj` is trained by the Q-branch SPR term and by the real critic loss.
- `pi_proj` is trained by the policy-branch SPR term and by the real actor loss.
- The shared encoder is trained by the real critic loss, the two SPR terms, and the real actor loss.

The real actor loss reaches the encoder from the current frame, through `pi_proj`. It does not enter the transition. `update_critic` steps `q_proj` before `update_spr` clears the SPR optimizer. `update_actor_and_alpha` steps `pi_proj` and the encoder after the SPR step. The predictors stay in the SPR optimizer only.

The disabled reward predictor is left untouched.

## Follow-up: one encoder update per SPR loss

Review found that `SPR.encoder` and `critic.encoder` are the same module, so collecting all `SPR.parameters()` also put the shared encoder in `spr_optimizer`. Together with `encoder_optimizer`, this applied two Adam updates to the encoder from the same SPR backward pass. This overlap predates `fa12903`.

The shared encoder parameters are now excluded from `spr_optimizer` by parameter identity. During `update_spr`, `encoder_optimizer` alone applies the encoder update at its existing learning rate of `2 * encoder_lr`. The transition, both online projections, and both predictors continue to receive their SPR updates through `spr_optimizer` at `auxiliary_task_lr`. The disabled reward predictor is unchanged.

The real critic and real actor optimizers still include the shared encoder and train it from their respective losses. Only the duplicate encoder update from the SPR loss is removed. No imaginary learning is added; the SPR rollout continues to use replay actions.

The regression check verifies that all shared encoder parameters belong to `encoder_optimizer`, remain in the actor and critic optimizers, and are absent from `spr_optimizer`. It also verifies that the encoder still changes during the SPR update.
