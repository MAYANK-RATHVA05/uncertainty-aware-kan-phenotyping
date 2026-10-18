"""Four participant-disjoint partitions for conditional relative-month data.

Calibration is reserved for later research. This module neither uses outcomes
nor computes conformal scores; it only partitions participant identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


PARTITION_NAMES = ("train", "validation", "calibration", "test")


def _participant_ids(participants: np.ndarray) -> np.ndarray:
    result = np.asarray(participants)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("Participants must be a nonempty one-dimensional array.")
    for participant in result:
        if not np.isscalar(participant) or pd.isna(participant):
            raise ValueError("Participant identifiers must be nonmissing scalar values.")
        if isinstance(participant, (float, np.floating)) and not np.isfinite(participant):
            raise ValueError("Participant identifiers must be finite.")
        if not str(participant).strip():
            raise ValueError("Participant identifiers must not be blank.")
    # The builder supplies strings. A homogeneous numeric array is also useful
    # for synthetic validation, but mixed types could collapse distinct IDs.
    kinds = {"string" if isinstance(value, (str, np.str_)) else "numeric"
             if isinstance(value, (int, float, np.integer, np.floating)) else "other"
             for value in result}
    if "other" in kinds or len(kinds) != 1:
        raise ValueError("Participant identifiers must be uniformly strings or numeric scalars.")
    return result.astype(str)


@dataclass(frozen=True)
class ForecastPartitions:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def validate(self, participants: np.ndarray) -> None:
        groups = _participant_ids(participants)
        parts = []
        for name in PARTITION_NAMES:
            part = np.asarray(getattr(self, name))
            if part.ndim != 1 or not np.issubdtype(part.dtype, np.integer):
                raise ValueError("Partition indices must be one-dimensional integer arrays.")
            if len(part) == 0:
                raise ValueError("Every partition must be nonempty.")
            if (part < 0).any() or (part >= len(groups)).any():
                raise ValueError("Partition indices must be within the dataset.")
            parts.append(part)
        joined = np.concatenate(parts)
        if not np.array_equal(np.sort(joined), np.arange(len(groups))):
            raise ValueError("Partitions must cover every sample exactly once, without repeated indices.")
        grouped = [set(groups[part].tolist()) for part in parts]
        if any(left & right for left, right in combinations(grouped, 2)):
            raise ValueError("Participant leakage across forecasting partitions.")

    def metadata(self, participants: np.ndarray) -> dict:
        """Aggregate counts only: no participant identifiers or row records."""
        self.validate(participants)
        groups = _participant_ids(participants)
        return {
            name: {"samples": int(len(getattr(self, name))),
                   "participants": int(len(np.unique(groups[getattr(self, name)])))}
            for name in PARTITION_NAMES
        } | {
            "participant_overlap": False,
            "every_sample_assigned_once": True,
            "identity_basis": "source-supported participant prefix, conditional export correspondence",
            "calibration_role": "reserved; not used to fit preprocessing or any model",
            "stratification": "none; no target values used",
        }


def participant_forecast_split(
    participants: np.ndarray, seed: int = 42,
    fractions: tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2),
) -> ForecastPartitions:
    """Sort IDs, seed-shuffle, allocate 60/10/10/20 by participant count.

    Largest-remainder allocation breaks ties in train/validation/calibration/
    test order. If a small positive share rounds to zero, move one participant
    from the largest partition (same stable tie order) so each share is nonempty.
    Every eligible quarter from a participant follows that participant's share.
    """
    groups = _participant_ids(participants)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("The split seed must be a nonnegative integer.")
    shares = np.asarray(fractions, dtype=float)
    if (shares.shape != (4,) or not np.isfinite(shares).all()
            or (shares <= 0).any() or not np.isclose(shares.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Four positive split fractions must sum to one.")
    unique = np.unique(groups)
    if len(unique) < 4:
        raise ValueError("At least four participants are required for four disjoint partitions.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    exact = len(unique) * shares
    sizes = np.floor(exact).astype(np.int64)
    remaining = int(len(unique) - sizes.sum())
    for index in np.argsort(-(exact - sizes), kind="stable")[:remaining]:
        sizes[index] += 1
    for empty in np.flatnonzero(sizes == 0):
        donor = int(np.argmax(sizes))
        sizes[donor] -= 1
        sizes[empty] += 1
    blocks = np.split(shuffled, np.cumsum(sizes)[:-1])
    result = ForecastPartitions(*(np.flatnonzero(np.isin(groups, block)) for block in blocks))
    result.validate(groups)
    return result
