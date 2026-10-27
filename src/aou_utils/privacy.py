"""Helpers for following the All of Us dissemination policy on small counts."""

SMALL_COUNT = 20


def suppress(counts, threshold: int = SMALL_COUNT):
    """Replace counts from 1 to `threshold` with '<=20' in a DataFrame or Series of counts.

    Zero is left as is. Apply this to anything that will be displayed or shared.
    """
    small = (counts >= 1) & (counts <= threshold)
    return counts.astype(object).mask(small, f"<={threshold}")


def mask_fractions(fractions, counts, threshold: int = SMALL_COUNT):
    """Blank out fractions whose underlying count is from 1 to `threshold`.

    A fraction of a known group size reveals the count, so it needs the same suppression.
    """
    small = (counts >= 1) & (counts <= threshold)
    return fractions.mask(small)
