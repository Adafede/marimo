"""ElementRange dataclass for formula filtering."""

__all__ = ["ElementRange"]

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementRange:
    """Range for element count in molecular formula."""

    min_val: int | None = None
    max_val: int | None = None

    def is_active(self) -> bool:
        """Check if range filter is active.

        Returns
        -------
        bool
            ``True`` if active; otherwise ``False``.
        """
        return self.min_val is not None or self.max_val is not None

    def matches(self, count: int) -> bool:
        """Check if count is within range.

        Parameters
        ----------
        count : int
            Count.

        Returns
        -------
        bool
            Result matches.
        """
        if not self.is_active():
            return True
        if self.min_val is not None and count < self.min_val:
            return False
        if self.max_val is not None and count > self.max_val:
            return False
        return True
