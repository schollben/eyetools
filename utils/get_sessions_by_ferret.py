import re
from pathlib import Path

_ANALYZABLE_OUTPUTS = Path(
    "/Users/benjaminscholl/Dropbox/projects/VisBehavDev/data/analyzable_outputs"
)
_FERRET_ID_RE = re.compile(r"_ferret_(\d+)_")


def get_sessions_by_ferret(*ferret_ids: int) -> list[str]:
    """Return session directory names matching one or more ferret IDs.

    Args:
        *ferret_ids: One or more integer ferret IDs (e.g. 757, 753).

    Returns:
        Sorted list of session directory name strings for all requested IDs.

    Raises:
        ValueError: If any requested ID has no matching sessions.
    """
    all_dirs = [d.name for d in _ANALYZABLE_OUTPUTS.iterdir() if d.is_dir()]

    id_to_sessions: dict[int, list[str]] = {fid: [] for fid in ferret_ids}
    for name in all_dirs:
        m = _FERRET_ID_RE.search(name)
        if m:
            fid = int(m.group(1))
            if fid in id_to_sessions:
                id_to_sessions[fid].append(name)

    missing = [fid for fid, sessions in id_to_sessions.items() if not sessions]
    if missing:
        raise ValueError(f"No sessions found for ferret ID(s): {missing}")

    result = []
    for fid in ferret_ids:
        result.extend(sorted(id_to_sessions[fid]))
    return result
