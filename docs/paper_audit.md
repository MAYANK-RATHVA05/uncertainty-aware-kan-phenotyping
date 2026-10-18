# Source paper and implementation audit

This project adapts ideas from **Mai Ali, Zhenghao Ni, and Deepa Kundur,
“Interpretable-MTLNet: A Kolmogorov–Arnold Network for Multitask Mental Health
Prediction,” NeurIPS 2025 Workshop: Foundation Models for the Brain and Body**,
as supplied in `hci paper 4.pdf`. Page references below use the PDF's printed
pages, 1–9. This audit distinguishes the paper's stated design, unresolved
implementation details, and the code actually used with the uploaded data.

The trained workflow in this repository is an **exploratory, retrospective,
tabular regression of measured PHQ-9 change**. It is an adaptation, not a
replication of the paper's temporal, multitask classification experiment.
The temporal KAN is a separately tested, untrained reference module.

## What the paper specifies

| Component | Evidence in supplied PDF | Stated design |
|---|---|---|
| Source population and observations | Section 4.1, p. 4 | Approximately 40,000 **All of Us** participants, with 60-day Fitbit streams containing daily steps, calories, heart-rate/HRV, and sleep-stage information. These are not the uploaded PSYCHE-D aggregate rows. |
| Outcomes | Section 3.5, p. 3, Eq. (5); Section 4.1, p. 4 | Two binary tasks, depression and anxiety, with task-specific scalar logits and sigmoid probabilities. The paper describes labels from standardized interviews and medical records. It does not define a PHQ-change threshold for this upload. |
| KAN layer | Section 3.2, p. 3, Eqs. (1)–(2) | Each output sums learned univariate input-edge functions and one neuron bias. Each edge combines a B-spline expansion with a weighted SiLU base term. |
| Temporal convolution | Section 3.3, p. 3, Eq. (3) | Three temporal scales, with kernels **3, 5, and 7**. An output channel sums edge functions over input features and lag offsets. The equation uses `t - j`, a past/current indexing convention. |
| Temporal aggregation | Section 3.4, p. 3, Eq. (4) | Global average pooling over time for each scale; concatenate pooled scale vectors and apply shared KAN mixing. |
| Channels and mixing width | Section 4.2, p. 4; Appendix B, p. 9 | Appendix B clarifies **64 total channels** after pooling/concatenation and one shared hidden mixing layer of width 64. It does not specify the three branch channel counts. |
| Static covariates and heads | Section 3.1, pp. 2–3; Section 3.5, p. 3, Eq. (5) | The mathematical/textual description concatenates the shared embedding with static covariates before task-specific KAN heads. Parameters before those heads are shared between the two tasks. |
| Classification objective | Section 3.5, p. 4, Eqs. (6)–(7); Appendix B, p. 9 | Weighted task-wise binary focal loss. Appendix B gives `alpha = 0.25` and `gamma = 2.0`; Section 5.2, p. 6, gives selected task weights `w_dep = 1.2`, `w_anx = 0.8`. |
| Split and preprocessing | Section 4.2, p. 4; Appendix B, p. 9 | Subject-level 70%/10%/20% training/validation/test splits; training-set median imputation, missingness indicators, and training-set mean/standard-deviation normalization. |
| Optimization and selection | Appendix B, p. 9 | AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 128, and early stopping on validation **Macro-AUROC** with patience 10. A maximum epoch budget is not stated. |
| Evaluation and ablations | Appendix A, p. 9; Section 5.2 and Table 2, p. 6 | Binary AUROC/AUC-PR, task macro/micro aggregation, and five training seeds. Ablations include single task, MLP-based, single scale `k = 5`, and removal of temporal features. |
| Interpretation | Introduction, pp. 1–2; Section 5.1, pp. 4–5 | Learned univariate spline functions and feature interpretation are central. The paper also mentions activation-weighted curves and symbolic surrogates, but does not give a complete reproducible procedure for these in the supplied PDF. |

The paper's reported performance belongs to its All of Us experiment. It is
not a measurement from this repository, a target for PSYCHE-D regression, or
evidence of clinical utility here. No paper performance number is imported
into the project's measured result summaries.

## Details the PDF does not determine

These choices must be disclosed instead of being presented as recovered
paper defaults:

| Unresolved detail | Evidence boundary | Repository choice |
|---|---|---|
| Spline degree, knot grid, range, and adaptation | Eq. (2), p. 3, introduces `p`, `G`, and knots but does not give numeric values or an adaptive-grid algorithm. | Cubic degree 3, 8 intervals over `[-3, 3]`, fixed uniform knots extended by 3 knots at each end. No adaptive grid or input clipping. |
| Exact basis count | The sum in Eq. (2) is typeset from `k = 0` through `G + p`; the knot-vector convention and relationship of `G` to interval count are not fixed. The inclusive notation alone does not establish this implementation's coefficient count. | With 8 intervals and degree 3, the chosen extended-knot convention has **11 basis functions** per input edge. Cox–de Boor evaluation is tested against an independent scalar reference, including partition of unity at the default grid boundaries. This is an implementation convention, not a literal reconstruction of the typeset sum bounds. |
| Initialization | No initialization distribution or coefficient scale is specified. | Kaiming-uniform base weights, modest uniform spline coefficients scaled by input width, and zero-initialized learnable bias. |
| Padding, boundaries, missing timesteps, and stride | Eq. (3), p. 3, uses past lags but does not describe boundary padding, sequence-length handling, or mask-aware pooling. | Reference convolution uses stride 1, causal left padding, preserved length, and explicit masks. Padded edges contribute neither their base nor spline response; invalid timesteps are excluded from pooling. Empty valid histories are rejected. |
| Allocation of the 64 channels | Appendix B, p. 9, gives only the combined total. | Reference branches allocate **22 / 21 / 21** channels to kernels 3 / 5 / 7. It is not 64 channels per branch. |
| Shared embedding and task-head dimensions | Sections 3.4–3.5 and Appendix B specify a shared width of 64 and final task outputs, without all intermediate head dimensions. | Reference uses one KAN mixing map from the concatenated 64 channels to width 64; static features are appended once; one scalar KAN output map follows. No hidden head widths are claimed to come from the paper. |
| Figure 1's extra paths | Figure 1, p. 2, visibly includes a **Feature Engineering Processor** feeding the shared feature path and a separate **Static KAN Processor**. Their dimensions, functions, fusion operation, and relationship to the simpler text/Eq. (5) description are not fully specified. | The temporal reference follows the text and Eq. (5): pooled KAN mixing followed by one static concatenation. It does not invent an unspecified engineering skip path or separate static KAN processor. The discrepancy prevents an exact architecture-reproduction claim. |
| Full interpretation procedure | The abstract/introduction name activation weighting and symbolic surrogates; Section 5.1 and Figure 2 do not specify a complete algorithm, normalization, edge-selection rule, or symbolic fitting protocol. | Expose the actual learned SiLU and spline edge components; keep them separate from validation permutation reliance and partial-dependence-style responses. Shared neuron bias is excluded from each individual edge curve. No symbolic-recovery or paper-identical ranking claim is made. |

The reference's masking and initialization safeguards are engineering choices.
They do not supply missing scientific evidence about observation timing.

## Adaptation to the actual PSYCHE-D upload

The [dataset audit](dataset_audit.md) documents **35,694 rows, 155 columns**,
and **10,866 rows with both PHQ-9 score endpoints**. The accompanying dictionary
defines `phq9_score_start` and `phq9_score_end` as scores at the start and end
of a three-month period. Consequently, the selected outcome is the continuous
quantity:

```text
PHQ change = phq9_score_end - phq9_score_start
```

No depression diagnosis, binary change threshold, or anxiety target is inferred.
The four PHQ fields are excluded from predictors, including the start score.
The index, undocumented predictors, 84 undocumented regression-window summaries,
and 10 Dynamic intervention/change fields are also excluded. The current
predictor schema retains 29 described Static and 27 described Statistics fields;
train-fitted imputation, missingness encoding, scaling, and categorical encoding
determine the final transformed width saved in each run.

The composite index has a repeated-prefix pattern, but authoritative participant
identity and chronology are unavailable. Prefixes are kept together using a
70%/10%/20% **provisional group split**. This preserves the paper's motivation
for separating repeated groups without claiming verified subject-level splitting.
It does not verify that included covariates precede the PHQ outcome period.

The actual model is `TabularDepressionKAN`: transformed covariates → KAN width
64 → scalar KAN output. It receives a two-dimensional feature matrix and has
no temporal convolution. Its output is an unbounded change estimate. Training
uses regression loss and selects checkpoints by validation RMSE; evaluation
reports MAE, RMSE, and R². Applying sigmoid, focal loss, AUROC, or probability
calibration to this continuous target would change the problem without a supplied
definition, so those are not used by the PSYCHE-D command-line workflow.

The repository retains a separately named, conventional MLP baseline and explicit
tabular KAN/MLP-mixing variants using the same saved split and preprocessing.
Comparisons between them describe this adapted regression problem; they are not
replications of the paper's multitask or temporal ablation table. Initialization
seeds and run budgets are recorded locally. Repeated seeds measure training
variability on the saved split, not population confidence or external validity.

`TemporalDepressionKAN` implements a reference with the disclosed choices above
and passes synthetic numerical, causality, and padding tests. It remains
**untrained on real PSYCHE-D sequences**. The forecasting helper is gated on
explicit confirmation of index semantics and measurement timing. Tensor shape,
suffix appearance, or successful model execution cannot establish valid time
sequences or a forecast horizon.

## Interpretation and result boundaries

Learned edge curves show individual model terms in transformed coordinates;
they are not automatically raw-feature effects or feature-importance scores.
Permutation and feature-response analyses use validation data, keep the model
frozen, and retain caveats about correlated inputs, encoded categories, and
repeated provisional groups. Repeated-shuffle spread is not a sampling confidence
interval. An edge curve from a hidden layer describes a learned representation.

All generated results concern retrospective association with observed PHQ change.
Unverified identity/timing, missing outcomes, confounding, and sample selection
limit interpretation. Neither the source paper nor this implementation establishes
causal effects, diagnosis, treatment recommendations, or prospective clinical
performance on this upload.
