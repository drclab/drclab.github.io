+++
title = "StatsClaw 111: Researcher Control vs AI Automation"
date = "2026-05-07"
tags = ["statsclaw", "ai-governance", "research-methods", "decision-making"]
categories = ["posts", "stats"]
series = ["StatsClaw"]
type = "post"
draft = true
math = false
description = "A decision framework for what should be delegated to AI and what must remain researcher-owned."
+++

AI assistance is most useful when boundaries are explicit. The question is not whether to automate. The question is what to automate.

## Tasks Suitable For Delegation

These are usually safe to delegate with review:

1. Boilerplate scaffolding.
2. Refactoring with preserved semantics.
3. Test harness generation from predefined criteria.
4. Documentation formatting and structure.

## Tasks That Must Stay Researcher-Owned

These decisions are methodological, not mechanical:

1. Estimand definition.
2. Identification assumptions.
3. Inference interpretation.
4. Tradeoffs between bias, variance, and robustness.

Delegating these without oversight risks scientific validity.

## Governance Checkpoints

A lightweight governance model can include:

1. Human sign-off for methodological changes.
2. Mandatory artifact review for release gates.
3. Audit logs linking prompts, code changes, and outcomes.

Governance should increase clarity, not bureaucracy.

## Key Takeaway

Use AI for execution leverage, but keep scientific judgment under explicit human control.
