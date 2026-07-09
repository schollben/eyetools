from .config import DATA_DIR
from .parse_session_name import parse_session_name

_ANALYZABLE_OUTPUTS = DATA_DIR


def get_sessions_by_eo(eo_low: int, eo_high: int | None = None) -> list[str]:
    """Return session directory names with EO numbers in the given range, across all ferrets.

    Args:
        eo_low: EO number (if eo_high is None) or the low end of an inclusive range.
        eo_high: High end of an inclusive EO range, e.g. get_sessions_by_eo(0, 4).

    Returns:
        List of session directory name strings, sorted by EO number.

    Raises:
        ValueError: If no sessions match the given EO number or range.
    """
    lo, hi = (eo_low, eo_high) if eo_high is not None else (eo_low, eo_low)

    all_dirs = [d.name for d in _ANALYZABLE_OUTPUTS.iterdir() if d.is_dir()]
    matched = sorted(
        (name for name in all_dirs if lo <= parse_session_name(name)["eo"] <= hi),
        key=lambda name: parse_session_name(name)["eo"],
    )
    if not matched:
        eo_desc = f"{lo}" if eo_high is None else f"{lo}-{hi}"
        raise ValueError(f"No sessions found for EO: {eo_desc}")

    return matched
