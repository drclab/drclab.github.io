+++
title = "GRPO 101: Calculating Loss in Group Relative Policy Optimization"
date = "2026-01-08"
type = "post"
math = true
draft = false
tags = ["deep learning", "reinforcement learning", "GRPO", "LLM", "fine-tuning"]
categories = ["deep_learning"]
description = "Understanding GRPO loss calculation—from policy ratios to clipping and KL divergence for LLM fine-tuning."
+++

Group Relative Policy Optimization (GRPO) is a reinforcement learning technique for fine-tuning large language models. It builds on PPO (Proximal Policy Optimization) but adds improvements for training stability. This post walks through the loss calculation step-by-step, using a small model (BabyLlama) for illustration.

## Why GRPO?

Traditional supervised fine-tuning updates model weights to maximize likelihood of preferred responses. GRPO instead:

1. **Compares** the policy model against a frozen reference model
2. **Scales** updates by an advantage function (how good is this response?)
3. **Clips** probability ratios to prevent destructive updates
4. **Penalizes** divergence from the reference model via KL divergence

This combination keeps the model improving while preventing catastrophic forgetting.

## Setup: Reference and Policy Models

The **reference model** is the base LLM, frozen throughout training. The **policy model** is a copy with a LoRA adapter—only the adapter weights get updated.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from peft import LoraConfig, get_peft_model

# Load base model
model_str = 'babylm/babyllama-100m-2024'
base_model = AutoModelForCausalLM.from_pretrained(model_str)
tokenizer = AutoTokenizer.from_pretrained(model_str)

# Reference model: frozen copy
ref_model = copy.deepcopy(base_model)

# Policy model: base + trainable LoRA adapter
lora_config = LoraConfig(
    r=8,  # Rank of update matrices
    lora_alpha=32,  # Alpha scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    init_lora_weights=False,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
```

The LoRA adapter adds low-rank matrices to attention projections, keeping most parameters frozen while enabling efficient fine-tuning.

## Input Preparation

For loss calculation, we need to:
1. Tokenize both prompt and completion
2. Concatenate them into a single sequence
3. Create a mask identifying which tokens are from the completion

```python
def prepare_inputs(prompt, completion):
    # Tokenization
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    completion_tokens = tokenizer(completion, return_tensors="pt")

    # Combined input
    input_ids = torch.cat(
        [prompt_tokens["input_ids"], completion_tokens["input_ids"]], 
        dim=1
    )
    attention_mask = torch.cat(
        [prompt_tokens["attention_mask"], completion_tokens["attention_mask"]],
        dim=1
    )

    prompt_length = prompt_tokens["input_ids"].shape[1]
    completion_length = completion_tokens["input_ids"].shape[1]
    total_length = prompt_length + completion_length

    # Mask for completion tokens only
    completion_mask = torch.zeros(total_length, dtype=torch.float32)
    completion_mask[prompt_length:] = 1.0

    return input_ids, attention_mask, completion_mask
```

The completion mask is crucial—we only compute loss over tokens the model generated, not the prompt.

## Computing Log Probabilities

The core of GRPO compares how likely each token is under the policy vs. reference model:

```python
import torch.nn.functional as F

def compute_log_probs(model, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # Convert logits to log-probabilities
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    
    # Extract log-prob for the actual token at each position
    return log_probs.gather(
        dim=-1, 
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)
```

This returns the log-probability of each token in the sequence, given the preceding context.

## The Basic GRPO Loss

The fundamental loss combines the probability ratio with an advantage signal:

$$\text{ratio} = \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)} = \exp(\log \pi_\theta - \log \pi_{\text{ref}})$$

$$\mathcal{L} = -\text{ratio} \times \text{advantage}$$

```python
def grpo_loss(model, ref_model, prompt, completion, advantage):
    input_ids, attention_mask, completion_mask = prepare_inputs(
        prompt, completion
    )

    # Compute log-probs for both models
    token_log_probs = compute_log_probs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(
            ref_model, input_ids, attention_mask
        )

    # Probability ratio: p_model / p_ref
    ratio = torch.exp(token_log_probs - ref_token_log_probs)

    # Scale by advantage
    policy_loss = ratio * advantage

    # Negative because optimizers minimize
    per_token_loss = -policy_loss

    # Average over completion tokens only
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return loss
```

**Key insight:** At the first training step, both models are identical, so the ratio equals 1 everywhere. The loss reduces to just the negative advantage:

```python
# At step 1: ratio = 1, so loss = -advantage
grpo_loss(ref_model, ref_model, prompt, "fence and", advantage=2.0)
# Output: tensor(-2.)
```

Positive advantages push the model toward that completion; negative advantages push away.

## Adding Clipping for Stability

Large probability ratios can destabilize training. PPO/GRPO addresses this by clipping:

$$\mathcal{L}_{\text{clip}} = \min\Big(\text{ratio} \times A, \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \times A\Big)$$

With $\epsilon = 0.2$, ratios are clamped to $[0.8, 1.2]$.

```python
def grpo_loss_with_clip(model, ref_model, prompt, completion, 
                        advantage, epsilon=0.2):
    input_ids, attention_mask, completion_mask = prepare_inputs(
        prompt, completion
    )

    token_log_probs = compute_log_probs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(
            ref_model, input_ids, attention_mask
        )

    ratio = torch.exp(token_log_probs - ref_token_log_probs)

    # Unclipped and clipped objectives
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage

    # Take the minimum (most conservative)
    policy_loss = torch.min(unclipped, clipped)

    per_token_loss = -policy_loss
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return loss
```

**Why minimum?** 
- For positive advantages: clipping prevents the ratio from growing too large
- For negative advantages: clipping prevents the ratio from shrinking too small

The minimum ensures we always take the more conservative update.

## Adding KL Divergence Regularization

Clipping alone doesn't prevent the model from drifting too far from the reference over many updates. KL divergence acts as a "gravitational pull" back toward the reference:

$$D_{KL}(\pi_{\text{ref}} \| \pi) = \exp(-\Delta) + \Delta - 1$$

where $\Delta = \log \pi_\theta - \log \pi_{\text{ref}}$.

```python
def grpo_loss_with_kl(model, ref_model, prompt, completion, 
                      advantage, epsilon=0.2, beta=0.1):
    input_ids, attention_mask, completion_mask = prepare_inputs(
        prompt, completion
    )

    token_log_probs = compute_log_probs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(
            ref_model, input_ids, attention_mask
        )

    ratio = torch.exp(token_log_probs - ref_token_log_probs)

    # Clipped policy loss
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    policy_loss = torch.min(unclipped, clipped)

    # KL divergence penalty
    delta = token_log_probs - ref_token_log_probs
    per_token_kl = torch.exp(-delta) + delta - 1

    # Combined loss: policy - beta * KL
    per_token_loss = -(policy_loss - beta * per_token_kl)

    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return loss
```

The $\beta$ hyperparameter controls the strength of regularization:

| Beta | Effect |
|------|--------|
| 0.0  | No KL penalty—model can drift freely |
| 0.1  | Mild regularization—common default |
| 0.5  | Strong pull toward reference |

## Understanding the KL Penalty Shape

The KL penalty is asymmetric and always non-negative:

- **$\Delta > 0$** (model more confident than reference): Penalty grows exponentially—the "overconfident region"
- **$\Delta < 0$** (model less confident): Penalty grows more slowly—the "conservative region"
- **$\Delta = 0$**: Zero penalty when models agree

This asymmetry encourages the policy to stay close to the reference's probability distribution while still allowing learning.

## Putting It Together

A complete GRPO training step:

1. **Sample** completions from the policy model
2. **Score** them with a reward function to compute advantages
3. **Calculate** the GRPO loss with clipping and KL regularization
4. **Backpropagate** and update LoRA weights

```python
# Example: training on a good completion (positive advantage)
loss = grpo_loss_with_kl(
    model,
    ref_model,
    prompt="The quick brown fox jumped over the ",
    completion="fence and",
    advantage=2.0,
    epsilon=0.2,
    beta=0.1
)

# Backprop and update
loss.backward()
optimizer.step()
```

## Key Takeaways

1. **Two models**: Reference (frozen) provides the baseline; policy (LoRA) learns
2. **Probability ratios**: Measure how much the policy has diverged from reference
3. **Advantages**: Scale updates—positive reinforces, negative discourages
4. **Clipping**: Prevents catastrophically large updates in single steps
5. **KL divergence**: Long-term regularization to prevent drift

GRPO builds these components into a stable training signal for aligning LLMs with human preferences or other reward signals.

## Further Reading

- [PPO Paper](https://arxiv.org/abs/1707.06347) - The foundation for clipped objectives
- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization specifics
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-rank adaptation for efficient fine-tuning
