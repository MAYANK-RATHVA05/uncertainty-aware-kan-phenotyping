"""Dictionary-driven PSYCHE-D loading and conservative forecasting cohorts.

The anonymized index is deliberately *not* interpreted until its semantics have
been independently confirmed. Pattern matching alone cannot establish timing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INDEX_COLUMN = "__index_level_0__"
PHQ_COLUMNS = (
    "phq9_score_start", "phq9_score_end", "phq9_cat_start", "phq9_cat_end"
)
CATEGORICAL_COLUMNS = frozenset({
    "sex", "educ", "money", "household", "num_migraine_days"
})


@dataclass(frozen=True)
class FeatureSchema:
    """Allowed predictors and explicit reasons for all excluded columns."""

    temporal_columns: tuple[str, ...]
    static_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    feature_details: dict[str, dict[str, str]] = field(default_factory=dict)

    def export_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastConfig:
    verified_index_semantics: bool = False
    index_semantics_authority: str = ""
    target_month: int = 12
    history_months: int = 7
    horizon_months: int = 3
    verified_outcome_alignment: bool = False
    verified_feature_windows: bool = False

    def validate(self) -> None:
        if not (
            self.verified_index_semantics and self.verified_outcome_alignment
            and self.verified_feature_windows and self.index_semantics_authority.strip()
        ):
            raise ValueError(
                "Forecasting is blocked until the anonymized index is confirmed to "
                "mean participant_id_month and rows are confirmed to summarize only "
                "their recorded month; outcome endpoints must also be confirmed to "
                "align to t-3 and t, and every feature window must be available at the "
                "cutoff. Set all three verification flags and record the user or "
                "documentation authority covering these claims; index appearance "
                "is insufficient."
            )
        if self.history_months != 7:
            raise ValueError("The prespecified forecasting workflow requires exactly 7 observed months.")
        if self.horizon_months != 3:
            raise ValueError("The documented PHQ outcome interval is exactly 3 months.")
        if self.target_month not in (3, 6, 9, 12):
            raise ValueError("target_month must be 3, 6, 9, or 12, matching the observed suffix schedule in this file; calendar alignment still requires external verification.")


@dataclass
class ForecastExamples:
    """Guarded reference cohort; valid only after external timing verification.

    ``groups`` contains identifiers in memory exclusively for splitting. Never use
    it as a model input or include it in public experiment artifacts. Row positions
    refer to the input CSV, and may be persisted only under ignored local artifacts.
    """

    temporal: pd.DataFrame
    static: pd.DataFrame
    tabular: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    history_row_positions: np.ndarray
    cutoff_row_positions: np.ndarray
    target_row_positions: np.ndarray
    history_months: np.ndarray
    outcome_months: np.ndarray
    schema: FeatureSchema
    config: ForecastConfig
    eligibility_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def history_length(self) -> int:
        return self.config.history_months

    def export_metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "target": "phq9_score_end - phq9_score_start",
            "task": "continuous_regression",
            "n_examples": len(self),
            "n_participants": int(len(np.unique(self.groups))),
            "eligibility_counts": self.eligibility_counts,
            "schema": self.schema.export_metadata(),
            "prediction_cutoff": "target_month - 3",
            "history": "7 consecutive observed months ending at prediction cutoff",
            "static_covariates": "values in the cutoff row only",
            "tabular_ablation": "cutoff-row temporal features plus the same static features",
            "stratification": "none: the target is continuous PHQ-9 score change",
        }


@dataclass
class TabularExamples:
    """Retrospective covariates with conservative provisional-prefix grouping.

    ``temporal`` is the Statistics feature block for shared preprocessing only;
    these rows are not a sequence and do not establish any forecast horizon.
    ``groups`` must remain internal. Its prefix relationship is observed, while
    its relationship to actual participant identity remains unverified.
    """

    temporal: pd.DataFrame
    static: pd.DataFrame
    tabular: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    row_positions: np.ndarray
    schema: FeatureSchema
    eligibility_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def history_length(self) -> int:
        return 1

    @property
    def history_row_positions(self) -> np.ndarray:
        """Compatibility row map; the singleton axis is not a time sequence."""
        return self.row_positions[:, None]

    def export_metadata(self) -> dict[str, Any]:
        return {
            "target": "phq9_score_end - phq9_score_start",
            "task": "continuous_regression",
            "study_type": "retrospective_association",
            "temporal_semantics_verified": False,
            "index_grouping": "composite_prefix_provisional",
            "group_identity_verified": False,
            "grouping_limitation": "Repeated composite-index prefixes are kept together as a conservative grouping rule; participant identity is not confirmed.",
            "timing_limitation": "Feature availability relative to PHQ outcomes is unverified; this is not prospective prediction or causal inference.",
            "n_examples": len(self),
            "n_provisional_groups": int(len(np.unique(self.groups))),
            "eligibility_counts": self.eligibility_counts,
            "schema": self.schema.export_metadata(),
            "stratification": "none: the target is continuous PHQ-9 score change",
        }


def load_feature_schema(dictionary_path: str | Path, columns: list[str]) -> FeatureSchema:
    dictionary = pd.read_csv(dictionary_path, sep="\t", dtype=str).fillna("")
    required = {"Feature name", "Category", "Subcategory", "Description", "Notes"}
    if not required.issubset(dictionary.columns):
        raise ValueError("Feature dictionary lacks required schema columns.")
    dictionary["Feature name"] = dictionary["Feature name"].str.strip()
    if dictionary["Feature name"].duplicated().any():
        raise ValueError("Feature dictionary contains duplicate feature names.")
    details = {
        row["Feature name"]: {
            "category": row["Category"], "subcategory": row["Subcategory"],
            "description": row["Description"], "notes": row["Notes"],
        }
        for row in dictionary.to_dict(orient="records")
    }
    temporal, static, excluded = [], [], {}
    for name in columns:
        spec = details.get(name)
        if name == INDEX_COLUMN:
            excluded[name] = "Identifier; grouping and time validation only."
        elif name in PHQ_COLUMNS or name.lower().startswith("phq") or (spec and spec["category"] == "PHQ-9"):
            excluded[name] = "PHQ outcome or proxy; excluded from all predictors."
        elif spec is None:
            excluded[name] = "Undocumented feature: availability at the cutoff is unverified."
        elif spec["subcategory"] == "Linear regression":
            excluded[name] = "Derived regression window/timing is undocumented."
        elif not spec["description"].strip():
            excluded[name] = "Feature lacks a documented description."
        elif spec["subcategory"] == "Static":
            static.append(name)
        elif spec["subcategory"] in ("Statistics", "Dynamic"):
            temporal.append(name)
        else:
            excluded[name] = "Unsupported or undocumented feature timing."
    included = temporal + static
    return FeatureSchema(
        tuple(temporal), tuple(static),
        tuple(name for name in included if name in CATEGORICAL_COLUMNS),
        excluded, {name: details[name] for name in included},
    )


def load_dataset(csv_path: str | Path, dictionary_path: str | Path) -> tuple[pd.DataFrame, FeatureSchema]:
    """Load local anonymized data without making unverified temporal claims."""
    frame = pd.read_csv(csv_path, dtype={INDEX_COLUMN: str}, low_memory=False)
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if INDEX_COLUMN not in frame:
        raise ValueError(f"Missing required index column {INDEX_COLUMN}.")
    if not set(PHQ_COLUMNS).issubset(frame.columns):
        raise ValueError("The dataset must include all four documented PHQ outcome columns.")
    return frame, load_feature_schema(dictionary_path, frame.columns.tolist())


def inspect_dataset(frame: pd.DataFrame, schema: FeatureSchema) -> dict[str, Any]:
    """Return aggregate audit information only; do not expose records or IDs."""
    matching = frame[INDEX_COLUMN].astype("string").str.fullmatch(r"\d+_\d+").fillna(False)
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "temporal_features": len(schema.temporal_columns),
        "static_features": len(schema.static_columns),
        "excluded_features": len(schema.exclusion_reasons),
        "nonmissing_phq_score_pairs": int(frame[["phq9_score_start", "phq9_score_end"]].notna().all(axis=1).sum()),
        "index_matches_digits_underscore_digits": bool(matching.all()),
        "index_semantics_verified": False,
        "exclusion_reasons": schema.exclusion_reasons,
    }


def prepare_tabular_examples(frame: pd.DataFrame, schema: FeatureSchema) -> TabularExamples:
    """Use labeled rows for exploratory association, without inferred timing.

    The feature dictionary supports ordinary Static and Statistics covariates.
    Intervention/change fields in Dynamic are excluded because their timing is
    uncertain. All PHQ columns, identifiers, undocumented features, and derived
    regression columns are excluded regardless of correlations with the label.
    """
    frame = frame.reset_index(drop=True)
    parts = frame[INDEX_COLUMN].astype("string").str.extract(r"^(\d+)_(\d+)$")
    if parts.isna().any(axis=None):
        raise ValueError("Provisional grouping requires a composite digits_digits index; no participant identity is inferred.")
    if frame[INDEX_COLUMN].duplicated().any():
        raise ValueError("Duplicate composite indices could duplicate examples and must be resolved explicitly.")
    scores = _numeric_scores(frame)
    selected = scores[["phq9_score_start", "phq9_score_end"]].notna().all(axis=1)
    positions = np.flatnonzero(selected.to_numpy())
    if not len(positions):
        raise ValueError("No rows have both documented PHQ score endpoints.")
    exclusions = dict(schema.exclusion_reasons)
    statistics = []
    for name in schema.temporal_columns:
        details = schema.feature_details.get(name, {})
        if details.get("subcategory") != "Statistics":
            exclusions[name] = "Dynamic intervention/change timing relative to PHQ outcomes is unverified."
        elif not details.get("description", "").strip():
            exclusions[name] = "Statistics feature lacks a documented description."
        else:
            statistics.append(name)
    static_columns = []
    for name in schema.static_columns:
        details = schema.feature_details.get(name, {})
        if details.get("subcategory") == "Static" and details.get("description", "").strip():
            static_columns.append(name)
        else:
            exclusions[name] = "Static feature lacks a documented description or schema."
    included = statistics + static_columns
    if any(name == INDEX_COLUMN or name.lower().startswith("phq") for name in included):
        raise ValueError("Predictor schema contains an identifier or PHQ leakage column.")
    restricted = FeatureSchema(
        tuple(statistics), tuple(static_columns),
        tuple(name for name in included if name in schema.categorical_columns),
        exclusions, {name: schema.feature_details[name] for name in included},
    )
    if not included:
        raise ValueError("No described Statistics or Static features are available.")
    statistics_frame = frame.iloc[positions][statistics].reset_index(drop=True)
    static_frame = frame.iloc[positions][static_columns].reset_index(drop=True)
    y = (scores.loc[positions, "phq9_score_end"] - scores.loc[positions, "phq9_score_start"]).to_numpy(dtype=np.float32)
    return TabularExamples(
        statistics_frame, static_frame, pd.concat([statistics_frame, static_frame], axis=1),
        y, parts.iloc[positions, 0].to_numpy(dtype=str), positions, restricted,
        {"total_rows": len(frame), "missing_score_pair": int((~selected).sum()), "eligible": len(positions)},
    )


def _numeric_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for name in PHQ_COLUMNS:
        values = pd.to_numeric(frame[name], errors="coerce")
        if (frame[name].notna() & values.isna()).any():
            raise ValueError(f"{name} contains nonnumeric nonmissing values.")
        maximum = 4 if "cat" in name else 27
        observed = values.dropna()
        if not observed.between(0, maximum).all() or not np.isfinite(observed).all():
            raise ValueError(f"{name} is outside its documented range 0..{maximum}.")
        if not (observed == np.floor(observed)).all():
            raise ValueError(f"{name} must contain integer questionnaire scores or missing values.")
        result[name] = values
    return result


def prepare_forecasting_examples(
    frame: pd.DataFrame, schema: FeatureSchema, config: ForecastConfig | None = None,
) -> ForecastExamples:
    """Construct genuinely prior observations; never pad or invent time points.

    For an outcome recorded at month 12, PHQ change describes months 9..12.
    Predictors stop at month 9 and use exactly months 3..9. The whole outcome
    interval is excluded. The boundary convention assumes each recorded monthly
    summary is available at that month end, which requires external confirmation.
    """
    config = config or ForecastConfig()
    config.validate()
    if not schema.temporal_columns:
        raise ValueError("No documented monthly predictors are available.")
    unsafe = {INDEX_COLUMN, *PHQ_COLUMNS}
    included = set(schema.temporal_columns) | set(schema.static_columns)
    if included & unsafe or any(name.lower().startswith("phq") for name in included):
        raise ValueError("Predictor schema contains an identifier or PHQ leakage column.")
    frame = frame.reset_index(drop=True)
    parts = frame[INDEX_COLUMN].astype("string").str.extract(r"^(\d+)_(\d+)$")
    if parts.isna().any(axis=None):
        raise ValueError("Index must follow the confirmed participant_id_month format.")
    groups = parts[0].astype(str)
    months = parts[1].astype(int)
    if not months.between(1, 12).all():
        raise ValueError("Observed months must be in 1..12.")
    keys = pd.DataFrame({"participant": groups, "month": months})
    if keys.duplicated().any():
        raise ValueError("Duplicate participant-month observations are ambiguous; no aggregation is inferred.")
    scores = _numeric_scores(frame)
    cutoff = config.target_month - config.horizon_months
    required_months = np.arange(cutoff - config.history_months + 1, cutoff + 1)
    if required_months[0] < 1:
        raise ValueError("The target month cannot provide seven pre-interval observed months.")
    by_key = {(participant, int(month)): int(pos) for pos, (participant, month) in enumerate(zip(groups, months))}
    outcomes = np.flatnonzero((months == config.target_month).to_numpy())
    history_positions, target_positions, example_groups = [], [], []
    counts = {"candidate_outcome_rows": len(outcomes), "missing_score_pair": 0, "incomplete_history": 0, "eligible": 0}
    for pos in outcomes:
        if scores.loc[pos, ["phq9_score_start", "phq9_score_end"]].isna().any():
            counts["missing_score_pair"] += 1
            continue
        participant = groups.iloc[pos]
        positions = [by_key.get((participant, int(month))) for month in required_months]
        if any(position is None for position in positions):
            counts["incomplete_history"] += 1
            continue
        history_positions.append(positions)
        target_positions.append(int(pos))
        example_groups.append(participant)
    counts["eligible"] = len(target_positions)
    if not target_positions:
        raise ValueError("No eligible outcomes have a complete seven-month pre-interval history.")
    history = np.asarray(history_positions, dtype=np.int64)
    target = np.asarray(target_positions, dtype=np.int64)
    cutoff_positions = history[:, -1]
    temporal = frame.iloc[history.ravel()][list(schema.temporal_columns)].reset_index(drop=True)
    static = frame.iloc[cutoff_positions][list(schema.static_columns)].reset_index(drop=True)
    tabular = pd.concat([
        frame.iloc[cutoff_positions][list(schema.temporal_columns)].reset_index(drop=True), static,
    ], axis=1)
    y = (scores.loc[target, "phq9_score_end"] - scores.loc[target, "phq9_score_start"]).to_numpy(dtype=np.float32)
    return ForecastExamples(
        temporal, static, tabular, y, np.asarray(example_groups, dtype=str), history,
        cutoff_positions, target, np.tile(required_months, (len(target), 1)),
        np.full(len(target), config.target_month, dtype=np.int64), schema, config, counts,
    )
