from .get_sessions import get_sessions


def get_sessions_by_ferret(*ferret_ids: int) -> list[str]:
    return get_sessions(*ferret_ids)
