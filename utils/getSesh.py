from types import SimpleNamespace

from .get_sessions import get_sessions
from .get_sessions_by_ferret import get_sessions_by_ferret
from .get_sessions_by_eo import get_sessions_by_eo

getSesh = SimpleNamespace(
    by_name=get_sessions,
    by_ferret=get_sessions_by_ferret,
    by_eo=get_sessions_by_eo,
)
