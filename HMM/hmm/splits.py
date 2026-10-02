def leave_one_session_out(segs: list[dict]) -> list[tuple[list[str], list[str]]]:
    ids = sorted({g["session_id"] for g in segs})
    return [([s for s in ids if s != test], [test]) for test in ids]


def leave_one_animal_out(segs: list[dict]) -> list[tuple[list[str], list[str]]]:
    animals = sorted({g["animal_id"] for g in segs})
    folds = []
    for a in animals:
        train = sorted({g["session_id"] for g in segs if g["animal_id"] != a})
        test = sorted({g["session_id"] for g in segs if g["animal_id"] == a})
        folds.append((train, test))
    return folds


def select(segs: list[dict], ids: list[str]) -> list[dict]:
    return [g for g in segs if g["session_id"] in ids]
