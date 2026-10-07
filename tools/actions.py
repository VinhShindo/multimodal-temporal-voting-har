"""Định nghĩa 20 hành động — dùng chung cho collector và annotation."""

ACTIONS = [
    # Nhóm 1: DI CHUYỂN (Movement) — 0..9
    (0,  "standing",         "movement"),
    (1,  "sitting",          "movement"),
    (2,  "lying",            "movement"),
    (3,  "walking",          "movement"),
    (4,  "running",          "movement"),
    (5,  "standing_up",      "movement"),
    (6,  "sitting_down",     "movement"),
    (7,  "jumping",          "movement"),
    (8,  "turning",          "movement"),
    (9,  "bending",          "movement"),

    # Nhóm 2: CỬ CHỈ / CẦM NẮM (Gesture / Grasp) — 10..19
    (10, "waving_hand",      "gesture"),
    (11, "clapping",         "gesture"),
    (12, "pointing",         "gesture"),
    (13, "drinking",         "gesture"),
    (14, "using_phone",      "gesture"),
    (15, "picking_object",   "gesture"),
    (16, "carrying_object",  "gesture"),
    (17, "opening_door",     "gesture"),
    (18, "writing",          "gesture"),
    (19, "throwing",         "gesture"),
]

NO_ACTION_ID = 255


def get_by_group(group: str):
    return [a for a in ACTIONS if a[2] == group]


def get_by_id(aid: int):
    for a in ACTIONS:
        if a[0] == aid:
            return a
    return None