import re

# Expected format: session_YYYY-MM-DD_ferret_ID_[EyeCameras_][PAGE_]EON_analyzable_output
_PATTERN = re.compile(
    r"session_(?P<date>\d{4}-\d{2}-\d{2})_ferret_(?P<id>\d+)_(?:EyeCameras_)?(?:P(?P<age>\d+)_)?EO?(?P<eo>\d+)(?:_+\d+)?_analyzable_output"
)

def parse_session_name(session: str) -> dict:
    m = _PATTERN.match(session)
    assert m, f"Session name did not match expected format: {session}"
    return {
        "session":      session,
        "date":     m.group("date"),
        "id":       int(m.group("id")),
        "age":      int(m.group("age")) if m.group("age") else None,
        "eo":       int(m.group("eo")),
    }
