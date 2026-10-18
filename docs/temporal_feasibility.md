# Temporal feasibility: historical observations → future PHQ-9 severity

Reviewed **2026-10-18**, against project commit
`f4db9ac959c722abfa87cae01445a39ad21be3ca`.

**Decision: conditionally supported at the relative study-month level.**

The two supplied papers resolve the earlier lack of documented sample-month
and wearable-window alignment. Together with the original code and the CSV,
they support investigating measured SM0 PHQ-9 plus historical SM1/SM2 inputs
→ measured SM3 PHQ-9 severity, excluding all SM3 wearable/intervention inputs.
The previous blocked decision is therefore revised. This is **not** a claim
that actual available-at timing, exact elapsed-day horizons, or the export's
feature-generation provenance have been validated.

The research sequence remains **KAN → validated future forecasting → KAN-ODE
→ conformal prediction**. Neither cutoff is selected for implementation here.
No model is trained, no existing training code is changed, and no later stage
is implemented.

## 1. Sources and page-reference convention

All PDF page numbers below are **physical viewer pages, counted from 1**,
including cover pages/front matter. They are not inferred manuscript page
numbers. Both Methods text and timeline/filtering figures were inspected.

| Ref | Inspected source | Identification |
|---|---|---|
| A | User-supplied `preprint-34148-accepted.pdf`, 32 pages | Accepted-version file supplied by the user; [paper landing page](https://preprints.jmir.org/preprint/34148). PDF SHA-256: `c3d36fb14bff076d8d3c4c7ed5ef1afe10fbedd4863c502f647aab3b01fa7b82`. The PDF labels itself an unpublished, peer-reviewed preprint; this review does not invent final-journal pagination. |
| B | User-supplied `ssrn-3889996.pdf`, 24 pages | Earlier version, marked not peer reviewed; [SSRN source](https://ssrn.com/abstract=3889996). PDF SHA-256: `e287c9a68e9dcbb72ac8f7a204b097d30e0f5968233df6890901ffb40387d6e1`. |
| C | [Original Evidation PSYCHE-D repository](https://github.com/evidation-opensource/PSYCHE-D/tree/444d3a6594988c6730a1cb5d24f2bdb8d1a971ba) | Inspected revision pinned to `444d3a6594988c6730a1cb5d24f2bdb8d1a971ba`; source was read, not executed. |
| D | Uploaded CSV and feature dictionary, retained outside Git at `/workspace/private-data/psyche-d/` | CSV SHA-256: `3d96afaa768bcd50c56524b931ac2e6acdd5f08c95db16c1e3b13d7dd6ce3934`; TSV SHA-256: `e346b57eb81c8c7d22a6ff0bda71214e527d8bed72eef1c2218d7e5826612e3e`. |

The earlier [dataset audit](dataset_audit.md), [schema](data_schema.md),
[paper audit](paper_audit.md), and [experiment report](experiment_report.md)
remain unchanged. Their retrospective results are not reinterpreted as
forecasting results. The supplied PDFs are now readable locally despite the
previous network denial for publisher/release destinations.

## 2. Paper evidence, distinguished by version

| Finding | Accepted PDF A | Earlier SSRN PDF B | Supporting passage / scope |
|---|---|---|---|
| Participant/month organization | pp. 8–10 describe the longitudinal study and relative sample months | pp. 5–7 give the same design | **Neither PDF explicitly defines `__index_level_0__` or its serialized prefix/suffix.** That syntax comes from official code C, not a fabricated PDF citation. |
| Quarterly PHQ-9 requests | p. 8, Methods → Data collection | p. 5, Methods → Data collection | “every 3 months, participants were requested to complete the PHQ-9.” This is the protocol schedule, not proof of actual submission dates. |
| Baseline demographics | p. 8; enrollment interpretation reiterated p. 16 | pp. 5–6; enrollment interpretation reiterated p. 13 | Screener survey: “prior to the study, participants self-reported socio-demographic information, as well as comorbidities.” Static features are “defined once” and remain constant across a participant's samples. |
| Monthly intervention/lifestyle survey | p. 8 | p. 5 | LMC requested “every month,” reporting changes “over the past month.” |
| Three-month samples and measured endpoints | p. 9, Dataset construction, continuing p. 10 | p. 7, Dataset construction | “3 month long, non-overlapping” samples; SM0–SM3 are relative time points; “PHQ-9 survey responses at SM0 and SM3.” |
| Optional intermediate observations | pp. 9–10 | p. 7 | LMC and wearable PGHD at SM3, and SM1/SM2 “if available.” The papers do not promise complete histories. |
| Wearable alignment | p. 10, first paragraph | p. 7 | “The wearable PGHD included data collected from 8 to 14 days prior to the PHQ-9 label generation date (SM1 and SM2 in phase 1, SM3 in phase 2).” |
| Target's recall-window overlap | p. 9, filtering; Figure 2 on pp. 9/25 | p. 6; Figure 2 on p. 7 | Filtering uses activity density “in the two weeks matching the PHQ-9 recall period”; figure criterion is ≥10 hours wear/day for ≥4 days/week in the two-week interval. |
| Original final-stage inputs near the final assessment | p. 10, Phase 2c | pp. 7–8, Phase 2 | Along with SM0 PHQ and generated intermediate categories/probabilities, the pipeline uses LMC and wearable PGHD “collected over the 2 weeks prior to final PHQ-9 completion at SM3.” |
| Intermediate PHQ categories are generated | Figure 1, embedded p. 8 and enlarged p. 24; text pp. 10/12 | Figure 1, p. 5 and duplicate p. 23; text p. 7 | The timeline marks intermediate PHQ categories **G**, not **C**; the observed SM0 category is **C**. SM1/SM2 outputs are model-generated labels, not additional measured questionnaire responses. |
| Missing intermediate histories occur | p. 16 | Optional SM1/SM2 observations already stated p. 7 | A notes that some samples have two intermediate labels, but sometimes one or none; missing PGHD/intermediate labels were tolerated by the original model. |

### Timeline interpretation and window wording

The inspected timeline in both versions matches the official repository
figure: LMC/wearable inputs are collected at SM1, SM2, **and SM3**. PHQ at
SM0 is collected, PHQ categories at SM1/SM2 are generated, and the final
change output is generated at SM3. The figure does not explicitly draw a
collected PHQ marker at SM3; the **Dataset construction text** supplies that
measured endpoint.

The statement “8 to 14 days prior” must be preserved without silently turning
it into exact inclusive/exclusive day bounds. Both versions also describe a
two-week interval, its data-density criterion, and final-stage data in the two
weeks before SM3 PHQ completion. The main text alone does not settle whether
8–14 describes usable-data duration, a particular lag band, or all individual
feature computations. It does not establish fixed days −14 through −8.

This does establish substantially more than the earlier README-only review:
wearable packets are aligned to **monthly label-generation/assessment points**,
rather than arbitrary feature-column order. It does **not** make the CSV a
daily stream or show that each feature summarizes an entire month. The local
dictionary's four-/seven-day summaries may be shorter summaries within the
observation packet; their exact relationship to that packet still needs the
feature-engineering specification.

### Version differences and shared limitations

The temporal construction passages agree: neither version provides a different
SM0/SM3 assessment schedule or removes SM3 inputs. There is no temporal
contradiction to resolve by preferring the accepted paper.

The modeling/validation descriptions do differ:

- **SSRN B p. 7:** phase 1 uses XGBoost. **B p. 8:** random sample assignment
  in an 80:20 split is followed by a participant-based re-evaluation. **B p. 15**
  explicitly acknowledges possible leakage between its two phases.
- **Accepted A pp. 10–12:** phases are combined, participant train/test
  partitioning occurs once at the pipeline's start, and phase 1 uses LightGBM
  DART. This is the combined approach reflected by the inspected source code.

The accepted validation change addresses participant/phase separation; it is
not proof of an earlier forecast cutoff. Both versions' original final stage
still uses target-proximal SM3 inputs. Their original category-change
classification results are not evidence for our proposed future continuous
severity task.

## 3. Official-code mapping to the uploaded CSV

[Original `combined_pipeline.py`, lines 27–33](https://github.com/evidation-opensource/PSYCHE-D/blob/444d3a6594988c6730a1cb5d24f2bdb8d1a971ba/combined_pipeline.py#L27-L33)
parses the original Parquet DataFrame index into:

```python
user_id = index_prefix
user_mth = integer_index_suffix
user_qtr = (user_mth - 1) // 3
```

Its [participant splitter](https://github.com/evidation-opensource/PSYCHE-D/blob/444d3a6594988c6730a1cb5d24f2bdb8d1a971ba/utils.py#L88-L95)
groups by `user_id`.
[Endpoint assembly, `utils.py` lines 98–114](https://github.com/evidation-opensource/PSYCHE-D/blob/444d3a6594988c6730a1cb5d24f2bdb8d1a971ba/utils.py#L98-L114)
renames start/end categories to `cat_m0`/`cat_m3` and obtains SM1 and SM2 from
rows with suffix `m−2` and `m−1` for endpoint suffix `m`.
[Lines 118–125](https://github.com/evidation-opensource/PSYCHE-D/blob/444d3a6594988c6730a1cb5d24f2bdb8d1a971ba/utils.py#L118-L125)
merge the endpoint's own sleep/activity/LMC fields. Thus the released code
corroborates the papers' use of SM3 inputs in the original model; it does not
implement our proposed exclusion of those inputs.

The local dictionary defines `phq9_score_start` and `phq9_score_end` as
scores at the start and end of the three-month period. Combined with the
papers' measured endpoint descriptions, this connects the code's category
mapping to the exported score fields; no generated category is converted
into a measured severity score.

Under the documented release convention, the proposed mapping is:

| Sample item | CSV source for a participant and quarter ending at suffix `m` |
|---|---|
| SM0 measured PHQ-9 severity | `phq9_score_start` on endpoint row `m`, representing the earlier observed assessment—not a predictor measured at SM3 |
| SM1 historical wearable packet | Same prefix, suffix `m−2` |
| SM2 historical wearable packet | Same prefix, suffix `m−1` |
| SM3 measured target severity | `phq9_score_end` on endpoint row `m` |
| Enrollment demographics, if included | Original screener values, not a later-updated measurement; verify they were available by cutoff |

For endpoint suffixes **3 / 6 / 9 / 12**, SM1 suffixes are **1 / 4 / 7 / 10**
and SM2 suffixes **2 / 5 / 8 / 11**. These are relative study-month labels,
not invented calendar timestamps. A separate CSV row for SM0 is not required:
there is no row 0, while the first quarter's earlier measured PHQ is retained
in `phq9_score_start`. Storage on the endpoint row does not make it a future
measurement—the paper explicitly documents measurement at SM0—but storage
alone would not establish actual availability in the presence of survey delays.

### Export corroboration and its remaining qualification

The export has 35,694 rows, 155 columns, and unique composite keys. It has
10,866 measured endpoint pairs across 4,036 prefixes at suffixes 3/6/9/12.
All have integer scores in 0–27; none of these existing endpoint rows lack
either score. All four PHQ fields are missing on intermediate rows.

More strongly, its labeled-quarter count distribution **exactly matches
accepted Figure 2 (A pp. 9/25)**:

| Labeled quarters per prefix | CSV prefixes | Paper participants |
|---:|---:|---:|
| 1 | 929 | 929 |
| 2 | 774 | 774 |
| 3 | 943 | 943 |
| 4 | 1,390 | 1,390 |

Where the preceding endpoint row exists, all **6,471** previous end scores
match the next start score. The official 29 `SCREENER_COLS` exactly match all
29 dictionary Static names and are invariant within prefixes. These checks
strongly corroborate the documented mapping. They are not an authenticated
Parquet-to-CSV export manifest or actual availability timestamps. Participant
counts below mean distinct prefixes **under that source-supported mapping**.

## 4. Candidate forecasting samples, without implementing them

Both candidates keep the same target: **measured continuous PHQ-9 severity at
SM3**, `y = phq9_score_end`, not `end − start`, a generated category, or a
thresholded diagnosis.

- **Cutoff SM1:** after the SM1 packet and measured SM0 PHQ are available,
  before SM3; use the SM1 historical wearable row and measured SM0 PHQ.
  Horizon: **two relative study-month intervals**, not a claimed 60 days.
- **Cutoff SM2:** after the SM2 packet and measured SM0 PHQ are available,
  before SM3; the complete-history variant uses both SM1 and SM2 wearable
  rows plus measured SM0 PHQ. Horizon: **one relative study-month interval**,
  not a claimed 30 days.

These definitions are conditional on the exported packets being computable
from observations available by the stated cutoff. Enrollment covariates may
be considered only as enrollment measurements. Historical LMC inclusion
would require the same availability checks; it is not necessary for the
wearable-history counts below.

**Exclude all SM3 wearable and intervention features**, and any derived
predictor incorporating them. Do not pass the whole endpoint row to the
model: select only its documented earlier baseline score and the measured
future score as the target. `phq9_cat_end`, SM3 LMC/PGHD, and the target score
must not become predictors. SM1/SM2 generated PHQ categories/probabilities
must never become ground truth. This audit does not generate or impute any
PHQ label.

The SM0 baseline score is supported as an earlier measured input by the
papers, unlike the earlier README-only evidence. Its actual completion and
arrival before the candidate cutoff remain required for an operational study.
Reading the start score from a later export row is acceptable only because of
that documented assessment meaning, not because all endpoint-row fields are
safe inputs.

## 5. Conditional aggregate eligibility and missing histories

These are **structural candidate counts**, not certified real-time availability
counts. They assume the original participant/month mapping applies, require
both measured endpoint scores, and do not impose a new completeness or raw
wear-time threshold. They cover existing released endpoints only; no four
quarter opportunities are invented for every prefix.

| Candidate | Required historical rows | Candidate samples | Distinct participants under mapping | Endpoints missing required history | Base participants with no eligible sample |
|---|---|---:|---:|---:|---:|
| Reference endpoint pool | None | 10,866 | 4,036 | — | — |
| SM1 cutoff | SM1 present | **9,280** | **3,593** | **1,586** | 443 |
| SM2 cutoff, complete history | SM1 and SM2 present | **8,494** | **3,375** | **2,372** | 661 |

If SM1 is allowed to be missing but SM2 must be observed, the **different,
relaxed** SM2 candidate has **9,393 samples / 3,635 participants**, with 1,473
endpoints missing SM2. This is not silently substituted for the two-row
complete-history candidate; a later protocol would have to choose explicitly.

The mutually exclusive endpoint-history partition is:

| History present | Endpoint samples | Distinct participant prefixes |
|---|---:|---:|
| Both SM1 and SM2 | 8,494 | 3,375 |
| SM1 only | 786 | 725 |
| SM2 only | 899 | 815 |
| Neither | 687 | 639 |

Sample counts sum to 10,866. Participant counts in these rows overlap because
one participant can have different missing-history patterns in different
quarters; they must not be summed or called wholly excluded participants.

Per-quarter reproducibility check:

| Endpoint suffix | Measured endpoint pool | SM1 candidate | Complete SM1+SM2 candidate |
|---:|---:|---:|---:|
| 3 | 3,244 | 2,729 | 2,488 |
| 6 | 2,832 | 2,372 | 2,166 |
| 9 | 2,732 | 2,367 | 2,157 |
| 12 | 2,058 | 1,812 | 1,683 |
| Total | 10,866 | 9,280 | 8,494 |

Row absence and within-row missing values are separate issues. As a diagnostic
only, complete-case counts across all required history rows are:

| Candidate | All 27 Statistics fields complete | The 11 dictionary four-/seven-day fields complete |
|---|---|---|
| SM1 | 915 samples / 572 participants | 8,760 samples / 3,407 participants |
| SM1+SM2 | 412 samples / 271 participants | 7,941 samples / 3,169 participants |

No required observed history row is entirely missing across either bundle;
many have individual missing values. These diagnostic counts are not a choice
to drop incomplete cases or evidence that all 27 fields have verified window
semantics. No imputation was fitted. The 11-field bundle is the dictionary's
four active/sedentary/hyper-/hyposomnia counts marked “last 7 days” plus seven
recent step/sleep summaries marked “last 4 days”; its names are in the
aggregate audit JSON.

Counts were independently checked with a vectorized participant/suffix join.
Audit script and aggregate JSON, containing no participant records, are saved
outside Git at `/workspace/scratch/temporal_feasibility_v2/candidate_counts.py`
and `candidate_counts.json`.

## 6. Conditions that remain before calling forecasting validated

1. **Export correspondence.** Confirm that the CSV's `__index_level_0__`
   preserves the source release index and original participant/month anchors.
   The exact cohort distribution and endpoint continuity are strong
   corroboration, not a release manifest.
2. **Available-at timing.** The papers give scheduled relative months, not
   actual survey deadlines, late completion, wearable synchronization,
   processing/ingestion delays, or an as-of guarantee for exported features.
   A packet arriving after a cutoff cannot be treated as available at it.
3. **Feature construction.** The main text describes observation packets,
   not every column's formula. The 84 regression descriptions, trailing-
   underscore variants, and possible interpolation/imputation need a
   no-look-ahead check for any features selected. No undocumented feature is
   admitted merely because its row precedes the target. It is not necessary
   to resolve every excluded feature to validate a smaller documented bundle.
4. **Target recall period.** SM3 PHQ-9 reflects the preceding two weeks;
   the original SM3 wearable packet explicitly overlaps that period. Removing
   SM3 PGHD/LMC removes this known overlap. Without actual dates or reliable
   relative-time bounds, SM1/SM2 labels alone cannot prove nonoverlap with the
   target's recall period, especially if surveys were delayed. Distinguish
   forecasting a future questionnaire response from forecasting an entirely
   future symptom interval. Neither exact day lead nor recall separation is
   certified here.
5. **Baseline availability.** Prior-to-study screener collection is documented
   (A p. 8; B p. 5), and each sample's measured PHQ is documented at SM0
   (A p. 9; B p. 7). Confirm these are the corresponding earlier observations
   in the export and arrived before cutoff, without later replacement or
   backfilling. Sample SM0 is not the same as original enrollment for every
   quarter; demographics remain the enrollment measurements.
6. **Selection and coverage.** The published endpoint cohort was filtered
   using final-quarter PHQ/LMC and final-assessment wearable coverage
   (A p. 9; B pp. 6–7). That does not establish equivalent SM1/SM2 coverage
   or prospective cohort inclusion known at an earlier cutoff. Counts apply
   to the released, future-observed endpoint cohort and may be selected.
7. **Repeated participants and architecture.** Multiple quarters from the same
   participant remain dependent; any later split must keep participants
   together. One or two monthly summary rows are not a daily series or proof
   that the reference temporal KAN's receptive fields are appropriate. These
   counts establish data candidates, not model readiness.

The smallest remaining evidence is a release/export timing note or the
feature-construction supplement/code covering the chosen historical bundle,
including ordering/availability bounds and provenance. The accepted PDF
**p. 30** links a supplementary DOCX rather than embedding its contents:
<http://asset.jmir.pub/assets/7d764e9680f0db44dc53769df569ada2.docx>.
An HTTPS retrieval attempt was denied by the proxy with `403 Forbidden`.
No content from that uninspected supplement is assumed. Absolute identifiable
calendar timestamps are not essential if reliable relative bounds can prove
the required ordering; exact elapsed-day claims would require suitable dates.

## 7. Decision and preservation

**Conditionally supported:** the papers, original code, and export jointly
support the nominal SM1/SM2 → measured SM3 severity candidates and the
conditional structural counts above. **Fully supported/validated forecasting
is not yet established** because provenance, per-feature availability, and
recall-window separation remain unresolved. Calling the original task
“prediction” is not used as proof.

The existing guarded forecasting helper still computes `end − start`, not
future severity, and prespecifies a different seven-month history. Its flags
were not enabled; neither it nor the current retrospective experiment code
was modified. Any later implementation must separately implement the approved
future-severity protocol after these conditions are addressed.

Only this feasibility document is updated. Existing source, tests, the other
audits, retrospective results, and artifacts are preserved. No new model,
ODE, or conformal code was added or trained; no commit or push was made.
