"""Shared total-case budgeting for all held-out acquisitions."""
from __future__ import annotations


def planned_counts(total_masks: int, acquisitions: int) -> list[int | None]:
    """Allocate a global mask budget without multiplying it by the test count."""
    if acquisitions < 1:
        raise ValueError("At least one acquisition is required")
    if total_masks <= 0:
        return [None] * acquisitions
    each, remainder = divmod(int(total_masks), acquisitions)
    return [each + (index < remainder) for index in range(acquisitions)]
