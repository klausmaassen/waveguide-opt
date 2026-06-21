from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

from .models import Candidate


@dataclass(frozen=True)
class BemppValidationStatus:
    available: bool
    message: str


def bempp_status() -> BemppValidationStatus:
    if find_spec("bempp") or find_spec("bempp_cl"):
        return BemppValidationStatus(True, "Bempp-cl package is importable")
    return BemppValidationStatus(False, "Bempp-cl is not installed in this Python environment")


def validate_top_candidate(candidate: Candidate) -> None:
    """Placeholder for the later Bempp-cl Top-N validation pipeline.

    The optimizer deliberately does not call Bempp-cl. This hook keeps the
    interface explicit for the later cross-check stage: mesh generation,
    Helmholtz solve, far-field extraction and Fast-BEM-vs-Bempp comparison.
    """
    status = bempp_status()
    if not status.available:
        raise RuntimeError(status.message)
    raise NotImplementedError(f"Bempp-cl validation is prepared but not implemented for {candidate.name}")
