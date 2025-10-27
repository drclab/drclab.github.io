+++
title = "Instrumental Variables, Visualized"
type = "post"
date = 2025-10-27
draft = false
summary = "A guided tour of instrumental variables with diagrams that highlight the identifying assumptions, 2SLS workflow, and pitfalls to watch before trusting an instrument."
tags = ["causal-inference", "instrumental-variables", "dag"]
+++

Instrumental variables (IVs) are the workhorses of causal inference when randomized experiments are out of reach and confounding blocks the backdoor path. Yet the identifying assumptions often feel abstract until you see how instruments reshape the causal graph and estimation steps. This post mirrors the collider walkthrough with fresh diagrams to emphasize what the instrument must—and must not—do.

## Visualizing the IV setup

The canonical IV story introduces an observed instrument `Z` that nudges treatment `X` without touching the outcome `Y` except through `X`. Any unobserved confounder `U` may still influence both `X` and `Y`, but it must remain disconnected from `Z`.

{{< post-figure src="images/posts/instrumental-variable-dag.svg" alt="DAG showing Z pointing to X pointing to Y, with an unobserved U affecting X and Y but not Z." caption="**Instrumental variable structure**: A valid instrument `Z` (left) affects the treatment `X`, which then changes the outcome `Y`. Unobserved confounders `U` may bias the direct `X→Y` link, but because `Z` breaks the confounding tie, it recovers variation in `X` that is orthogonal to `U`." >}}

This structure encodes the three core conditions in every textbook—from [Angrist & Pischke](https://masteringmetrics.com) to [Wooldridge](https://econ.msu.edu)—that keep IV designs credible:

- **Relevance**: `Z` shifts `X` in the data. Without a first stage, the instrument is moot.
- **Exclusion**: `Z` influences `Y` only through `X`. Any direct `Z → Y` arrow breaks identification.
- **Independence**: `Z` is as-if randomized with respect to the unobserved causes of `Y`, keeping `Z` and `U` independent.

## Diagnosing strength and exclusion in practice

Two-stage least squares (2SLS) makes the IV workflow explicit: isolate the part of `X` explained by `Z`, then use only that predicted component to estimate the effect on `Y`. Visualizing the pipeline highlights where diagnostics belong.

{{< post-figure src="images/posts/two-stage-least-squares.svg" alt="Two panels showing Stage 1 regressing X on Z to obtain X-hat, and Stage 2 regressing Y on X-hat." caption="**2SLS workflow**: Stage 1 regresses the treatment on the instrument (and controls) to produce the fitted values `X̂`. Stage 2 regresses the outcome on `X̂`. Weak-instrument tests (like the first-stage F-statistic) live in Stage 1, while overidentification checks probe exclusion by comparing multiple instruments against the Stage 2 fit." >}}

When reviewing an IV design:

- Inspect the Stage 1 fit—F-statistics above 10 are the conventional cutoff, but context matters.
- Probe exclusion restrictions with subject-matter knowledge or overidentification tests when you have multiple instruments.
- Trace alternative pathways in the DAG to ensure `Z` is not a proxy for a hidden policy, trend, or selection effect that touches `Y` directly.

## Workflow checklist

1. Map the causal story as a DAG and call out every path that connects `Z` to `Y`.
2. Justify why each non-`Z→X→Y` path stays closed (policy rules, timing, institutional design).
3. Quantify the first-stage relationship and communicate diagnostics, not just point estimates.
4. Report sensitivity or robustness checks (placebo outcomes, alternative controls) that stress-test the exclusion restriction.
5. Revisit the design whenever the instrument changes scope—new cohorts or jurisdictions can break independence.

## Further reading

- [Angrist & Pischke, *Mostly Harmless Econometrics*](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics) for econometric intuition and the 2SLS playbook.
- [Joshua Angrist’s MIT lecture notes](https://economics.mit.edu) for graphical IV examples and diagnostics.
- [Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*](https://www.cambridge.org) for LATE interpretations and monotonicity.
- [Miguel Hernán & James Robins, *Causal Inference: What If*](https://www.hsph.harvard.edu) for potential-outcomes treatments of instruments alongside DAG reasoning.
