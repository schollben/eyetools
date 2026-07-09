from .config import DATA_DIR
from .parse_session_name import parse_session_name

_ANALYZABLE_OUTPUTS = DATA_DIR


def get_sessions(*args: int | str) -> list[str]:
    """Return session directory names matching ferret IDs or full session names.

    Args:
        *args: One or more ferret IDs (int) or full session name strings.
               Types can be mixed, e.g.:
                 get_sessions(753)
                 get_sessions("session_2026-03-16_ferret_403_P49_E7_analyzable_output")
                 get_sessions(402, 420)

    Returns:
        List of session directory name strings, sorted within each ferret ID group,
        in the order arguments were provided.

    Raises:
        ValueError: If any ferret ID has no matching sessions, or a named session
                    does not exist in the data directory.
    """
    all_dirs = {d.name for d in _ANALYZABLE_OUTPUTS.iterdir() if d.is_dir()}

    result: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            if arg not in all_dirs:
                raise ValueError(f"Session not found: {arg}")
            if arg not in result:
                result.append(arg)
        else:
            fid = arg
            matched = sorted(
                name for name in all_dirs
                if parse_session_name(name)["id"] == fid
            )
            if not matched:
                raise ValueError(f"No sessions found for ferret ID: {fid}")
            result.extend(matched)

    return result
