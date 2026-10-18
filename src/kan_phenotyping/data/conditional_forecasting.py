"""Explicitly conditional SM1/SM2 → measured SM3 severity sample construction.

This module does not establish available-at timestamps. Its opt-in assumptions
are deliberately separate from the existing, more restrictive forecasting API.
See ``docs/temporal_feasibility.md`` for the source-supported relative mapping
and the remaining provenance, feature-window, and questionnaire-timing limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import INDEX_COLUMN


# Fixed reviewed names: a new dictionary column is never silently admitted.
BASELINE_DEMOGRAPHIC_FEATURES = (
    "sex", "race_white", "race_black", "race_hispanic", "race_asian",
    "race_other", "birthyear", "educ", "height", "weight", "bmi",
    "pregnant", "birth", "trauma", "insurance", "money",
    "money_assistance", "household", "comorbid_cancer",
    "comorbid_diabetes_typ1", "comorbid_diabetes_typ2", "comorbid_gout",
    "comorbid_migraines", "comorbid_ms", "comorbid_osteoporosis",
    "comorbid_neuropathic", "comorbid_arthritis", "num_migraine_days",
    "meds_migraine",
)
WEARABLE_STATISTICS_FEATURES = (
    "steps_awake_mean", "steps_mvpa_iqr", "steps_lpa_iqr",
    "steps_awake_sum_iqr", "steps__active_day_count_",
    "steps__sedentary_day_count_", "steps_mvpa_sum_recent",
    "steps_lpa_sum_recent", "steps_rolling_6_median_recent",
    "steps_rolling_6_max_recent", "sleep_asleep_weekday_mean",
    "sleep_asleep_weekend_mean", "sleep_in_bed_weekday_mean",
    "sleep_in_bed_weekend_mean", "sleep_ratio_asleep_in_bed_weekday_mean",
    "sleep_ratio_asleep_in_bed_weekend_mean", "sleep_in_bed_iqr",
    "sleep_asleep_iqr", "sleep_ratio_asleep_in_bed_iqr",
    "sleep_main_start_hour_adj_median", "sleep_main_start_hour_adj_iqr",
    "sleep_main_start_hour_adj_range", "sleep__hypersomnia_count_",
    "sleep__hyposomnia_count_", "sleep_asleep_mean_recent",
    "sleep_in_bed_mean_recent", "sleep_ratio_asleep_in_bed_mean_recent",
)
DEMOGRAPHIC_CATEGORICAL_FEATURES = (
    "sex", "educ", "money", "household", "num_migraine_days",
)
_START = "phq9_score_start"
_END = "phq9_score_end"
_ENDPOINT_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class ConditionalForecastAssumptions:
    """Acknowledged working assumptions, never claims of verified timestamps.

    Every assumption defaults to false, so callers must explicitly opt in and
    retain an evidence/limitations reference with any generated artifacts.
    """

    assume_release_index_mapping: bool = False
    assume_historical_features_available_by_sm2: bool = False
    assume_sm0_phq_available_by_sm2: bool = False
    assume_enrollment_demographics: bool = False
    evidence_reference: str = ""

    def validate(self) -> None:
        names = (
            "assume_release_index_mapping",
            "assume_historical_features_available_by_sm2",
            "assume_sm0_phq_available_by_sm2",
            "assume_enrollment_demographics",
        )
        if not all(getattr(self, name) is True for name in names):
            raise ValueError(
                "Conditional relative-month forecasting requires explicit opt-in "
                "to all four assumptions; these are not verified timestamp flags."
            )
        if not isinstance(self.evidence_reference, str) or not self.evidence_reference.strip():
            raise ValueError("A nonempty evidence_reference describing the conditional assumptions is required.")


@dataclass(frozen=True)
class ForecastPredictors:
    """Predictor-only payload: deliberately contains no future PHQ target.

    Wearable rows are stacked sample-major in the order SM1 then SM2. Missing
    individual predictor values are preserved; no imputation occurs here.
    Identifiers and source row positions live separately on the dataset.
    """

    wearable: pd.DataFrame
    demographics: pd.DataFrame
    sm0_phq: np.ndarray
    relative_months: np.ndarray
    study_months: np.ndarray
    feature_names: dict[str, list[str]] = field(default_factory=dict)
    exclusion_metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sm0_phq)


@dataclass(frozen=True)
class ConditionalForecastDataset:
    """Samples with internal mapping data and targets outside predictors.

    Participant labels/row positions are for joining, splitting, and local
    provenance only. Persist them only outside tracked source/public reports.
    Quarters use the original code's zero-based ``(month - 1) // 3`` rule.
    """

    predictors: ForecastPredictors
    targets: np.ndarray
    participants: np.ndarray
    quarters: np.ndarray
    endpoint_months: np.ndarray
    history_row_positions: np.ndarray
    baseline_row_positions: np.ndarray
    target_row_positions: np.ndarray
    audit: dict[str, Any]
    assumptions: ConditionalForecastAssumptions
    feature_allowlist: dict[str, Any]

    def __len__(self) -> int:
        return len(self.targets)

    def export_metadata(self) -> dict[str, Any]:
        """Aggregate metadata only: no individual records, labels, or IDs."""
        return {
            "study_type": "conditional_relative_month_forecasting",
            "available_at_timestamps_verified": False,
            "prospective_clinical_validation": False,
            "assumptions": asdict(self.assumptions),
            "cutoff": "SM2 after conditional availability of the SM2 historical packet",
            "horizon": "one relative study-month interval; not a fixed elapsed-day duration",
            "target": "measured SM3 PHQ-9 severity (phq9_score_end), not change",
            "baseline": "measured SM0 phq9_score_start stored on the same quarter's endpoint row",
            "demographics": "enrollment screener copy from SM1 only; no future-row backfill",
            "baseline_demographics_source": "SM1 copy of assumed enrollment responses",
            "history": "exactly two observed packets at relative SM1 and SM2; no synthetic days",
            "n_samples": len(self),
            "n_participants_under_assumed_mapping": len(np.unique(self.participants)),
            "feature_allowlist": self.feature_allowlist,
            "predictor_exclusions": self.predictors.exclusion_metadata,
            "eligibility": self.audit,
            "raw_shapes": {
                "wearable": [len(self), 2, len(WEARABLE_STATISTICS_FEATURES)],
                "demographics": list(self.predictors.demographics.shape),
                "sm0_phq": list(self.predictors.sm0_phq.shape),
                "relative_months": list(self.predictors.relative_months.shape),
                "study_months": list(self.predictors.study_months.shape),
                "targets": list(self.targets.shape),
            },
            "selection_limitations": [
                "Released endpoint cohort already conditions on future-observed questionnaire and wearable coverage.",
                "Requiring both historical rows selects participants/quarters with more continuous observation.",
                "Within-row missing predictors are retained; targets are never imputed.",
                "Relative labels do not certify exact availability or target recall-window separation.",
            ],
        }


def _validated_dictionary(dictionary: pd.DataFrame | str | Path) -> dict[str, dict[str, str]]:
    table = (
        dictionary.copy(deep=True) if isinstance(dictionary, pd.DataFrame)
        else pd.read_csv(dictionary, sep="\t", dtype=str)
    )
    required = {"Feature name", "Category", "Subcategory", "Description", "Notes"}
    if not required.issubset(table.columns):
        raise ValueError("Feature dictionary lacks required schema columns.")
    table = table.fillna("")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    if table["Feature name"].duplicated().any():
        raise ValueError("Feature dictionary contains duplicate feature names.")
    details = {
        row["Feature name"]: {
            "category": row["Category"], "subcategory": row["Subcategory"],
            "description": row["Description"], "notes": row["Notes"],
        }
        for row in table.to_dict(orient="records")
    }
    for name in BASELINE_DEMOGRAPHIC_FEATURES + WEARABLE_STATISTICS_FEATURES + (_START, _END):
        spec = details.get(name)
        if spec is None or not spec["description"]:
            raise ValueError(f"Required feature {name} lacks a documented description.")
        if name in BASELINE_DEMOGRAPHIC_FEATURES:
            expected = ("Demographic", "Static")
        elif name in WEARABLE_STATISTICS_FEATURES:
            expected = ("Steps" if name.startswith("steps_") else "Sleep", "Statistics")
        else:
            expected = ("PHQ-9", "PHQ-9 score")
        if (spec["category"], spec["subcategory"]) != expected:
            raise ValueError(f"Required feature {name} does not match its reviewed dictionary category/subcategory.")
    return details


def _scores(frame: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(frame[name], errors="coerce")
    if (frame[name].notna() & values.isna()).any():
        raise ValueError(f"{name} contains nonnumeric nonmissing values.")
    observed = values.dropna()
    if not np.isfinite(observed).all() or not observed.between(0, 27).all():
        raise ValueError(f"{name} must be a finite measured score in 0..27 or missing.")
    if not (observed == np.floor(observed)).all():
        raise ValueError(f"{name} must contain integer measured scores or missing values.")
    return values


def _assemble_predictors(
    historical_wearable: pd.DataFrame,
    sm1_demographics: pd.DataFrame,
    measured_sm0_scores: np.ndarray,
    study_months: np.ndarray,
    exclusions: dict[str, str],
) -> ForecastPredictors:
    """Accept only already-selected predictors; no endpoint target is available."""
    count = len(measured_sm0_scores)
    return ForecastPredictors(
        wearable=historical_wearable.reset_index(drop=True).copy(deep=True),
        demographics=sm1_demographics.reset_index(drop=True).copy(deep=True),
        sm0_phq=np.asarray(measured_sm0_scores, dtype=np.float32).reshape(count, 1),
        relative_months=np.tile(np.array([1, 2], dtype=np.int64), (count, 1)),
        study_months=study_months.copy(),
        feature_names={
            "wearable": list(WEARABLE_STATISTICS_FEATURES),
            "demographics": list(BASELINE_DEMOGRAPHIC_FEATURES),
            "sm0_phq": ["measured_sm0_phq9"],
        },
        exclusion_metadata=exclusions,
    )


def build_conditional_forecasting_dataset(
    frame: pd.DataFrame,
    dictionary: pd.DataFrame | str | Path,
    assumptions: ConditionalForecastAssumptions | None = None,
) -> ConditionalForecastDataset:
    """Build the conditional complete-history SM2 → SM3 severity cohort.

    For each endpoint suffix m in 3/6/9/12, join ONLY the same participant's
    rows m−2 and m−1. The start score on row m is documented as the measured
    SM0 assessment for that quarter; no row 0 or previous-quarter endpoint is
    required. This interpretation and its pre-cutoff availability are explicit
    assumptions, not implications of where the value is stored in the export.

    This function reads no generated PHQ labels and never uses endpoint
    wearables/interventions/demographics. Individual predictor missingness is
    retained. Missing measured endpoints and missing required history rows
    exclude a sample; scores are never imputed.
    """
    assumptions = assumptions or ConditionalForecastAssumptions()
    assumptions.validate()
    details = _validated_dictionary(dictionary)
    required = {
        INDEX_COLUMN, _START, _END,
        *BASELINE_DEMOGRAPHIC_FEATURES, *WEARABLE_STATISTICS_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset lacks required conditional forecasting columns: {', '.join(missing)}.")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")
    source = frame.reset_index(drop=True)
    parts = source[INDEX_COLUMN].astype("string").str.extract(r"^([0-9]+)_([0-9]+)\Z")
    if parts.isna().any(axis=None):
        raise ValueError("Conditional release mapping requires composite digits_digits indices.")
    participants = parts[0].astype(str)
    # Convert only bounded suffix strings, avoiding overflow on malformed keys.
    suffix = parts[1].str.lstrip("0").replace("", "0")
    if suffix.str.len().gt(2).any():
        raise ValueError("Relative study month suffix must be in 1..12.")
    months = suffix.astype(np.int64)
    if not months.between(1, 12).all():
        raise ValueError("Relative study month suffix must be in 1..12.")
    keys = pd.DataFrame({"participant": participants, "month": months})
    if keys.duplicated().any():
        raise ValueError("Duplicate normalized participant-month indices are ambiguous; no aggregation is inferred.")
    baseline_scores = _scores(source, _START)
    target_scores = _scores(source, _END)
    is_endpoint = months.isin(_ENDPOINT_MONTHS)
    if ((baseline_scores.notna() | target_scores.notna()) & ~is_endpoint).any():
        raise ValueError("Measured PHQ endpoints on a non-quarter-end row contradict the assumed release mapping.")

    lookup = {(participant, int(month)): position for position, (participant, month) in enumerate(zip(participants, months))}
    endpoint_positions = np.flatnonzero(is_endpoint.to_numpy())
    # Stable sample identity/order independent of input CSV row ordering.
    endpoint_positions = sorted(endpoint_positions, key=lambda pos: (participants.iloc[pos], int(months.iloc[pos])))
    measured_positions = [
        pos for pos in endpoint_positions
        if pd.notna(baseline_scores.iloc[pos]) and pd.notna(target_scores.iloc[pos])
    ]
    history_positions: list[list[int]] = []
    kept_endpoints: list[int] = []
    history_patterns = {"both": 0, "sm1_only": 0, "sm2_only": 0, "neither": 0}
    pattern_participants: dict[str, set[str]] = {name: set() for name in history_patterns}
    quarter_counts = {
        str(month): {"endpoint_rows": 0, "measured_endpoint_pairs": 0, "eligible": 0, "missing_required_history": 0}
        for month in _ENDPOINT_MONTHS
    }
    for pos in endpoint_positions:
        quarter_counts[str(int(months.iloc[pos]))]["endpoint_rows"] += 1
    for pos in measured_positions:
        participant = participants.iloc[pos]
        month = int(months.iloc[pos])
        row1 = lookup.get((participant, month - 2))
        row2 = lookup.get((participant, month - 1))
        pattern = "both" if row1 is not None and row2 is not None else (
            "sm1_only" if row1 is not None else "sm2_only" if row2 is not None else "neither"
        )
        history_patterns[pattern] += 1
        pattern_participants[pattern].add(participant)
        quarter_counts[str(month)]["measured_endpoint_pairs"] += 1
        if pattern != "both":
            quarter_counts[str(month)]["missing_required_history"] += 1
            continue
        history_positions.append([int(row1), int(row2)])
        kept_endpoints.append(pos)
        quarter_counts[str(month)]["eligible"] += 1

    endpoint_rows = np.asarray(kept_endpoints, dtype=np.int64)
    history_rows = np.asarray(history_positions, dtype=np.int64).reshape(-1, 2)
    endpoint_months = months.iloc[endpoint_rows].to_numpy(dtype=np.int64)
    history_months = endpoint_months[:, None] + np.array([-2, -1], dtype=np.int64)
    sample_participants = participants.iloc[endpoint_rows].to_numpy(dtype=str)
    excluded_columns = {
        name: (
            "Identifier: internal joining/splitting only; never a predictor."
            if name == INDEX_COLUMN else
            "Measured SM3 target only, stored separately from predictors."
            if name == _END else
            "Earlier measured SM0 scalar selected only from the paired endpoint; no historical PHQ columns."
            if name == _START else
            "PHQ category/generated label/proxy excluded; never ground truth or an input."
            if name.lower().startswith("phq") else
            "Undocumented regression summary excluded."
            if details.get(name, {}).get("subcategory") == "Linear regression" else
            "Intervention/change features excluded at every month."
            if details.get(name, {}).get("subcategory") == "Dynamic" else
            "Not in the fixed reviewed feature allowlist."
        )
        for name in source.columns
        if name not in BASELINE_DEMOGRAPHIC_FEATURES + WEARABLE_STATISTICS_FEATURES
    }
    # Payload construction receives only selected historical columns and SM0.
    # In particular, neither the endpoint frame nor phq9_score_end is passed.
    predictors = _assemble_predictors(
        source.iloc[history_rows.ravel()][list(WEARABLE_STATISTICS_FEATURES)],
        source.iloc[history_rows[:, 0]][list(BASELINE_DEMOGRAPHIC_FEATURES)],
        baseline_scores.iloc[endpoint_rows].to_numpy(dtype=np.float32),
        history_months,
        excluded_columns,
    )
    targets = target_scores.iloc[endpoint_rows].to_numpy(dtype=np.float32)
    base_participants = set(participants.iloc[measured_positions].tolist())
    eligible_participants = set(sample_participants.tolist())
    endpoint_start = baseline_scores.iloc[endpoint_positions]
    endpoint_end = target_scores.iloc[endpoint_positions]
    audit = {
        "input_rows": len(source),
        "input_participants_under_assumed_mapping": int(participants.nunique()),
        "candidate_endpoint_rows": len(endpoint_positions),
        "missing_measured_endpoint_pair": len(endpoint_positions) - len(measured_positions),
        "missing_sm0_score": int(endpoint_start.isna().sum()),
        "missing_sm3_score": int(endpoint_end.isna().sum()),
        "missing_both_scores": int((endpoint_start.isna() & endpoint_end.isna()).sum()),
        "measured_endpoint_pool": len(measured_positions),
        "measured_endpoint_participants": len(base_participants),
        "history_patterns_samples": history_patterns,
        "history_patterns_participants_overlapping": {name: len(values) for name, values in pattern_participants.items()},
        "excluded_missing_required_history": len(measured_positions) - len(endpoint_rows),
        "eligible_samples": len(endpoint_rows),
        "eligible_participants": len(eligible_participants),
        "measured_endpoint_participants_without_eligible_sample": len(base_participants - eligible_participants),
        "per_endpoint_month": quarter_counts,
        "individual_missing_predictor_values_retained": {
            "wearable": int(predictors.wearable.isna().sum().sum()),
            "demographics": int(predictors.demographics.isna().sum().sum()),
        },
        "sample_order": "participant prefix lexicographic, then endpoint study month numeric",
        "quarter_numbering": "zero-based (endpoint_month - 1) // 3",
        "participant_prefix_spelling": "preserved; integer normalization applies to month suffix only",
    }
    return ConditionalForecastDataset(
        predictors=predictors,
        targets=targets,
        participants=sample_participants,
        quarters=(endpoint_months - 1) // 3,
        endpoint_months=endpoint_months,
        history_row_positions=history_rows,
        baseline_row_positions=endpoint_rows.copy(),
        target_row_positions=endpoint_rows,
        audit=audit,
        assumptions=assumptions,
        feature_allowlist={
            "wearable_sm1_sm2": list(WEARABLE_STATISTICS_FEATURES),
            "enrollment_demographics_from_sm1": list(BASELINE_DEMOGRAPHIC_FEATURES),
            "measured_sm0_scalar": _START,
            "categorical_demographics": list(DEMOGRAPHIC_CATEGORICAL_FEATURES),
            "feature_details": {
                name: details[name]
                for name in BASELINE_DEMOGRAPHIC_FEATURES + WEARABLE_STATISTICS_FEATURES + (_START,)
            },
        },
    )
