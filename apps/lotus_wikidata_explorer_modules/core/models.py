"""
Data models and structures for LOTUS Wikidata Explorer.
"""

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["ElementRange", "FormulaFilters"]


@dataclass(frozen=True)
class ElementRange:
    """Range for element count in molecular formula."""

    min_val: Optional[int] = None
    max_val: Optional[int] = None

    def is_active(self) -> bool:
        """Check if range filter is active."""
        return self.min_val is not None or self.max_val is not None

    def matches(self, count: int) -> bool:
        """Check if count is within range."""
        if not self.is_active():
            return True
        if self.min_val is not None and count < self.min_val:
            return False
        if self.max_val is not None and count > self.max_val:
            return False
        return True


@dataclass(frozen=True)
class FormulaFilters:
    """Molecular formula filtering criteria."""

    exact_formula: Optional[str] = None
    c: ElementRange = field(default_factory=ElementRange)
    h: ElementRange = field(default_factory=ElementRange)
    n: ElementRange = field(default_factory=ElementRange)
    o: ElementRange = field(default_factory=ElementRange)
    p: ElementRange = field(default_factory=ElementRange)
    s: ElementRange = field(default_factory=ElementRange)
    f_state: str = "allowed"
    cl_state: str = "allowed"
    br_state: str = "allowed"
    i_state: str = "allowed"

    def is_active(self) -> bool:
        """Check if any filter is active."""
        if self.exact_formula and self.exact_formula.strip():
            return True
        if any(r.is_active() for r in [self.c, self.h, self.n, self.o, self.p, self.s]):
            return True
        if any(
            s != "allowed"
            for s in [self.f_state, self.cl_state, self.br_state, self.i_state]
        ):
            return True
        return False
