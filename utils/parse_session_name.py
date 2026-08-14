import re

# Expected: session_<date>_ferret_<id>_[<descriptor>_...][P<age>_]E[O]<eo>[__<run>]_analyzable_output
_PATTERN = re.compile(
    r"session_(?P<date>\d{4}-\d{2}-\d{2})_ferret_(?P<id>\d+)_"
    r"(?:[^_]+_)*?"          # optional descriptor tokens: EyeCameras / EyeCamera / future variants
    r"(?:P(?P<age>\d+)_)?"   # optional postnatal age
    r"EO?(?P<eo>\d+)"        # EO / E / E0 + number  (int() collapses leading zeros: E011 -> 11)
    r"(?:_+\d+)?"            # optional trailing run index: __1 / __2
    r"_analyzable_output"
)

def parse_session_name(session: str) -> dict:
    m = _PATTERN.match(session)
    if not m:
        raise ValueError(f"Session name did not match expected format: {session}")
    return {
        "session":      session,
        "date":     m.group("date"),
        "id":       int(m.group("id")),
        "age":      int(m.group("age")) if m.group("age") else None,
        "eo":       int(m.group("eo")),
    }
