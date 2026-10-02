from pathlib import Path
import yaml

from .adapter import DATA_DIR
from utils.get_sessions import get_sessions
from utils.parse_session_name import parse_session_name

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "hmm.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def eligible_sessions(cfg: dict) -> list[str]:
    c = cfg["sessions"]
    eo_max = c["eo_max"] if c["eo_max"] is not None else float("inf")

    ids = []
    for s in get_sessions(*c["ferrets"]):
        eo = parse_session_name(s)["eo"]
        if c["eo_min"] <= eo <= eo_max and (DATA_DIR / s / "skull_kinematics").exists():
            ids.append(s)
    return ids
