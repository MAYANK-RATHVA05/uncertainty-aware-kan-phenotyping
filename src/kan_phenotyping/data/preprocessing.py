"""Train-only preprocessing shared by all models and ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import FeatureSchema, ForecastExamples, TabularExamples

Examples = ForecastExamples | TabularExamples


@dataclass(frozen=True)
class TransformedFeatures:
    temporal: np.ndarray
    static: np.ndarray
    tabular: np.ndarray
    feature_names: dict[str, list[str]]


def _indices(indices: np.ndarray, total: int) -> np.ndarray:
    result = np.asarray(indices)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("Example indices must be a one-dimensional integer array.")
    if len(result) == 0 or (result < 0).any() or (result >= total).any():
        raise ValueError("Example indices must be nonempty and within the dataset.")
    if len(np.unique(result)) != len(result):
        raise ValueError("Repeated example indices would change preprocessing weights.")
    return result.astype(np.int64)


def _categorical_value(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".15g")
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return format(float(text), ".15g")
    except ValueError:
        return text


def _normalise_block(frame: pd.DataFrame, categorical: tuple[str, ...]) -> pd.DataFrame:
    """Normalize documented numeric/boolean types without inventing values."""
    result = pd.DataFrame(index=frame.index)
    for name in frame.columns:
        values = frame[name]
        if name in categorical:
            result[name] = values.map(_categorical_value).astype(object)
            continue
        if pd.api.types.is_bool_dtype(values.dtype):
            result[name] = values.astype(float)
            continue
        if pd.api.types.is_object_dtype(values.dtype) or isinstance(values.dtype, pd.StringDtype):
            values = values.map(
                lambda value: {"true": 1.0, "false": 0.0}.get(value.strip().lower(), value)
                if isinstance(value, str) else value
            )
        numeric = pd.to_numeric(values, errors="coerce")
        if (values.notna() & numeric.isna()).any():
            raise ValueError(f"Documented numeric feature {name} contains nonnumeric values.")
        result[name] = numeric.replace([np.inf, -np.inf], np.nan).astype(float)
    return result


def _transformer(columns: tuple[str, ...], categorical: tuple[str, ...]) -> ColumnTransformer | None:
    numeric = [name for name in columns if name not in categorical]
    category = [name for name in columns if name in categorical]
    steps = []
    if numeric:
        steps.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if category:
        steps.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]), category))
    return ColumnTransformer(steps, remainder="drop", sparse_threshold=0, verbose_feature_names_out=True) if steps else None


class ForecastPreprocessor:
    """Shared Statistics/monthly and Static transforms, fitted on training only.

    For retrospective tabular examples, the Statistics block has one covariate
    row per example. Its singleton tensor axis is an implementation convenience,
    not a claimed temporal observation. For verified forecasting examples, the
    same transform is fitted on unique training person-month rows and shared
    across all seven positions; static variables are never duplicated over time.
    """

    def __init__(self, schema: FeatureSchema):
        self.schema = schema
        self.temporal_transformer: ColumnTransformer | None = None
        self.static_transformer: ColumnTransformer | None = None
        self._fitted = False

    def fit(self, examples: Examples, train_indices: np.ndarray) -> "ForecastPreprocessor":
        if examples.schema != self.schema:
            raise ValueError("Preprocessor schema must match the eligible example schema.")
        indices = _indices(train_indices, len(examples))
        length = examples.history_length
        flattened = (indices[:, None] * length + np.arange(length)[None, :]).ravel()
        actual_rows = examples.history_row_positions[indices].ravel()
        _, first = np.unique(actual_rows, return_index=True)
        fit_rows = flattened[np.sort(first)]
        temporal = _normalise_block(examples.temporal.iloc[fit_rows], self.schema.categorical_columns)
        static = _normalise_block(examples.static.iloc[indices], self.schema.categorical_columns)
        self.temporal_transformer = _transformer(self.schema.temporal_columns, self.schema.categorical_columns)
        self.static_transformer = _transformer(self.schema.static_columns, self.schema.categorical_columns)
        if self.temporal_transformer is not None:
            self.temporal_transformer.fit(temporal)
        if self.static_transformer is not None:
            self.static_transformer.fit(static)
        self.n_training_examples_ = len(indices)
        self.n_training_covariate_rows_ = len(fit_rows)
        self.study_type_ = examples.export_metadata().get("study_type", "verified_forecasting")
        self._fitted = True
        return self

    @property
    def feature_names(self) -> dict[str, list[str]]:
        if not self._fitted:
            raise RuntimeError("Preprocessing must be fitted on the training partition first.")
        temporal = self.temporal_transformer.get_feature_names_out().tolist() if self.temporal_transformer is not None else []
        static = self.static_transformer.get_feature_names_out().tolist() if self.static_transformer is not None else []
        return {"temporal": temporal, "static": static, "tabular": temporal + static}

    @property
    def permutation_groups(self) -> dict[str, list[int]]:
        """Map each raw covariate to its whole encoded tabular block.

        Numeric measurements and their fitted missingness indicators travel
        together; every categorical feature's one-hot columns travel together.
        Derivation uses fitted column positions, never parsing output names that
        may contain ambiguous underscores. No refit or new stored state is needed,
        including when this method is used on an earlier serialized preprocessor.
        """
        names = self.feature_names
        groups: dict[str, list[int]] = {}
        for transformer, block_offset in (
            (self.temporal_transformer, 0),
            (self.static_transformer, len(names["temporal"])),
        ):
            if transformer is None:
                continue
            for kind, pipeline, columns in transformer.transformers_:
                if kind == "remainder":
                    continue
                output = transformer.output_indices_[kind]
                start = block_offset + output.start
                if kind == "numeric":
                    imputer = pipeline.named_steps["imputer"]
                    # Our fitted pipeline retains every numeric input, even an
                    # all-missing feature, before appending fitted indicators.
                    indicator_features = imputer.indicator_.features_
                    expected_width = len(columns) + len(indicator_features)
                    if output.stop - output.start != expected_width:
                        raise ValueError("Unexpected numeric transform layout for grouped permutation.")
                    for position, raw_name in enumerate(columns):
                        groups[raw_name] = [start + position]
                    for indicator_position, raw_position in enumerate(indicator_features):
                        groups[columns[int(raw_position)]].append(start + len(columns) + indicator_position)
                elif kind == "categorical":
                    encoder = pipeline.named_steps["encoder"]
                    # No category grouping/min-frequency option is enabled by
                    # this project's encoder. Account for a dropped category if
                    # an otherwise compatible saved encoder used that option.
                    if encoder.min_frequency is not None or encoder.max_categories is not None:
                        raise ValueError("Grouped permutation requires the project's unfused categorical encoding.")
                    cursor = start
                    for position, (raw_name, categories) in enumerate(zip(columns, encoder.categories_)):
                        dropped = encoder.drop_idx_ is not None and encoder.drop_idx_[position] is not None
                        width = len(categories) - int(dropped)
                        groups[raw_name] = list(range(cursor, cursor + width))
                        cursor += width
                    if cursor != block_offset + output.stop:
                        raise ValueError("Unexpected categorical transform layout for grouped permutation.")
                else:
                    raise ValueError("Unsupported transform block for grouped permutation.")
        encoded_indices = [index for indices in groups.values() for index in indices]
        if sorted(encoded_indices) != list(range(len(names["tabular"]))):
            raise ValueError("Raw-feature permutation groups must cover every encoded column exactly once.")
        return groups

    def transform(self, examples: Examples, indices: np.ndarray) -> TransformedFeatures:
        if not self._fitted:
            raise RuntimeError("Preprocessing must be fitted on the training partition first.")
        if examples.schema != self.schema:
            raise ValueError("Transform schema differs from the training schema.")
        indices = _indices(indices, len(examples))
        length = examples.history_length
        flattened = (indices[:, None] * length + np.arange(length)[None, :]).ravel()
        temporal_frame = _normalise_block(examples.temporal.iloc[flattened], self.schema.categorical_columns)
        static_frame = _normalise_block(examples.static.iloc[indices], self.schema.categorical_columns)
        temporal = (
            self.temporal_transformer.transform(temporal_frame).astype(np.float32)
            if self.temporal_transformer is not None else np.empty((len(flattened), 0), dtype=np.float32)
        )
        static = (
            self.static_transformer.transform(static_frame).astype(np.float32)
            if self.static_transformer is not None else np.empty((len(indices), 0), dtype=np.float32)
        )
        temporal = temporal.reshape(len(indices), length, -1)
        tabular = np.concatenate([temporal[:, -1, :], static], axis=1)
        if not np.isfinite(tabular).all() or not np.isfinite(temporal).all():
            raise ValueError("Preprocessing produced nonfinite model inputs.")
        return TransformedFeatures(temporal, static, tabular, self.feature_names)

    def export_metadata(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "permutation_groups": self.permutation_groups,
            "schema": self.schema.export_metadata(),
            "study_type": self.study_type_,
            "fitted_partition": "training_only",
            "training_examples": self.n_training_examples_,
            "training_covariate_rows": self.n_training_covariate_rows_,
            "numeric_transform": "training median imputation with missingness indicators, followed by training StandardScaler",
            "empty_numeric_features": "retained with imputed zero and training missingness indicator",
            "categorical_transform": "explicit missing category, one-hot encoding, unknown categories ignored",
            "categorical_features": list(self.schema.categorical_columns),
            "numeric_units": "standardized; contributions and curves must be labeled as transformed units",
        }


# The primary workflow is retrospective tabular regression; retain the earlier
# class name for the independently gated forecasting reference API.
TabularPreprocessor = ForecastPreprocessor
