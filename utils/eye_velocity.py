import numpy as np

# The loaders store eye velocity SWAPPED: session.{eye}_vx is vertical and
# session.{eye}_vy is horizontal, with per-eye signs (corr = +-1.00 against d(x)/dt and
# d(y)/dt in all 33 sessions checked). This maps them back.
_VEL = {("LE", "vx"): ("vy", -1.0), ("LE", "vy"): ("vx", -1.0),
        ("RE", "vx"): ("vy", 1.0), ("RE", "vy"): ("vx", -1.0)}


def eye_velocity(session, eye, key, head_frame=False):
    """Eye velocity (deg/s): key "vx" = d(x)/dt (horizontal), "vy" = d(y)/dt (vertical).

    head_frame=True negates LE horizontal so + = head yaw + for both eyes (LE_x points
    against yaw, RE_x along it) — needed whenever eye velocity is compared with yaw_v.
    """
    attr, sign = _VEL[(eye, key)]
    v = sign * np.asarray(getattr(session, f"{eye}_{attr}"), float)
    return -v if head_frame and eye == "LE" and key == "vx" else v
