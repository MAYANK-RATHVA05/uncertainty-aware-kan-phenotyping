"""Deterministic group-exclusive development and held-out partitions.

For the unverified PSYCHE-D composite index, groups are provisional prefixes;
these functions do not establish that those prefixes identify participants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticipantSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, groups: np.ndarray) -> None:
        groups = np.asarray(groups)
        indices = [np.asarray(part) for part in (self.train, self.validation, self.test)]
        joined = np.concatenate(indices)
        if len(joined) != len(groups) or not np.array_equal(np.sort(joined), np.arange(len(groups))):
            raise ValueError("Partitions must cover each example exactly once.")
        participants = [set(groups[part].tolist()) for part in indices]
        if participants[0] & participants[1] or participants[0] & participants[2] or participants[1] & participants[2]:
            raise ValueError("Participant leakage across data partitions.")
        if any(len(part) == 0 for part in indices):
            raise ValueError("Each data partition must be nonempty.")

    def export_metadata(self, groups: np.ndarray) -> dict:
        self.validate(groups)
        return {
            name: {"examples": int(len(part)), "groups": int(len(np.unique(groups[part])))}
            for name, part in (("train", self.train), ("validation", self.validation), ("test", self.test))
        } | {"group_overlap": False, "group_identity_verified_by_splitter": False, "stratification": "none (continuous regression)"}


def group_split(
    groups: np.ndarray, seed: int = 42, train_fraction: float = 0.7,
    validation_fraction: float = 0.1, test_fraction: float = 0.2,
) -> ParticipantSplit:
    """Shuffle unique participants, then allocate approximately 70/10/20.

    Rounding uses largest remainders and requires at least one participant in
    each partition. No synthetic binary classes or outcome stratification occur.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("groups must be a nonempty one-dimensional array.")
    if any(group is None or (isinstance(group, float) and np.isnan(group)) for group in groups):
        raise ValueError("Participant groups must not be missing.")
    fractions = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to one.")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three participants are required for disjoint train/validation/test partitions.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    exact = len(unique) * fractions
    sizes = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - sizes), kind="stable")[:len(unique) - sizes.sum()]:
        sizes[index] += 1
    for empty in np.flatnonzero(sizes == 0):
        donor = int(np.argmax(sizes))
        sizes[donor] -= 1
        sizes[empty] += 1
    first, second = int(sizes[0]), int(sizes[0] + sizes[1])
    parts = np.split(shuffled, [first, second])
    result = ParticipantSplit(*(np.flatnonzero(np.isin(groups, part)) for part in parts))
    result.validate(groups)
    return result
