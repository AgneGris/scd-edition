"""Electrode geometry definitions used by the editing interface."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

GRID_POSITIONS_13x5 = {
    1: (1, 0),
    2: (2, 0),
    3: (3, 0),
    4: (4, 0),
    5: (5, 0),
    6: (6, 0),
    7: (7, 0),
    8: (8, 0),
    9: (9, 0),
    10: (10, 0),
    11: (11, 0),
    12: (12, 0),
    25: (0, 1),
    24: (1, 1),
    23: (2, 1),
    22: (3, 1),
    21: (4, 1),
    20: (5, 1),
    19: (6, 1),
    18: (7, 1),
    17: (8, 1),
    16: (9, 1),
    15: (10, 1),
    14: (11, 1),
    13: (12, 1),
    26: (0, 2),
    27: (1, 2),
    28: (2, 2),
    29: (3, 2),
    30: (4, 2),
    31: (5, 2),
    32: (6, 2),
    33: (7, 2),
    34: (8, 2),
    35: (9, 2),
    36: (10, 2),
    37: (11, 2),
    38: (12, 2),
    51: (0, 3),
    50: (1, 3),
    49: (2, 3),
    48: (3, 3),
    47: (4, 3),
    46: (5, 3),
    45: (6, 3),
    44: (7, 3),
    43: (8, 3),
    42: (9, 3),
    41: (10, 3),
    40: (11, 3),
    39: (12, 3),
    52: (0, 4),
    53: (1, 4),
    54: (2, 4),
    55: (3, 4),
    56: (4, 4),
    57: (5, 4),
    58: (6, 4),
    59: (7, 4),
    60: (8, 4),
    61: (9, 4),
    62: (10, 4),
    63: (11, 4),
    64: (12, 4),
}
GRID_POSITIONS_8x8 = {
    8: (0, 0),
    7: (1, 0),
    6: (2, 0),
    5: (3, 0),
    4: (4, 0),
    3: (5, 0),
    2: (6, 0),
    1: (7, 0),
    16: (0, 1),
    15: (1, 1),
    14: (2, 1),
    13: (3, 1),
    12: (4, 1),
    11: (5, 1),
    10: (6, 1),
    9: (7, 1),
    24: (0, 2),
    23: (1, 2),
    22: (2, 2),
    21: (3, 2),
    20: (4, 2),
    19: (5, 2),
    18: (6, 2),
    17: (7, 2),
    32: (0, 3),
    31: (1, 3),
    30: (2, 3),
    29: (3, 3),
    28: (4, 3),
    27: (5, 3),
    26: (6, 3),
    25: (7, 3),
    40: (0, 4),
    39: (1, 4),
    38: (2, 4),
    37: (3, 4),
    36: (4, 4),
    35: (5, 4),
    34: (6, 4),
    33: (7, 4),
    48: (0, 5),
    47: (1, 5),
    46: (2, 5),
    45: (3, 5),
    44: (4, 5),
    43: (5, 5),
    42: (6, 5),
    41: (7, 5),
    56: (0, 6),
    55: (1, 6),
    54: (2, 6),
    53: (3, 6),
    52: (4, 6),
    51: (5, 6),
    50: (6, 6),
    49: (7, 6),
    64: (0, 7),
    63: (1, 7),
    62: (2, 7),
    61: (3, 7),
    60: (4, 7),
    59: (5, 7),
    58: (6, 7),
    57: (7, 7),
}
GRID_POSITIONS_20x2 = {
    # Key = 0-based OTBio channel index, value = (row, col) physical position
    # Col 0 = Side A (pads 1-20), Col 1 = Side B (pads 21-40)
    # Row 0 = pad 1/21 (proximal end), Row 19 = pad 20/40 (distal end)
    # Derived from DEMOVE->OTBio connector mapping table
    0: (18, 1),  # OTBio 1  -> DEMOVE 39 -> Side B pad 19
    1: (17, 1),  # OTBio 2  -> DEMOVE 38 -> Side B pad 18
    2: (16, 1),  # OTBio 3  -> DEMOVE 37 -> Side B pad 17
    3: (19, 1),  # OTBio 4  -> DEMOVE 40 -> Side B pad 20
    4: (12, 1),  # OTBio 5  -> DEMOVE 33 -> Side B pad 13
    5: (15, 1),  # OTBio 6  -> DEMOVE 36 -> Side B pad 16
    6: (14, 1),  # OTBio 7  -> DEMOVE 35 -> Side B pad 15
    7: (13, 1),  # OTBio 8  -> DEMOVE 34 -> Side B pad 14
    8: (10, 1),  # OTBio 9  -> DEMOVE 31 -> Side B pad 11
    9: (9, 1),  # OTBio 10 -> DEMOVE 30 -> Side B pad 10
    10: (6, 1),  # OTBio 11 -> DEMOVE 27 -> Side B pad 7
    11: (5, 1),  # OTBio 12 -> DEMOVE 26 -> Side B pad 6
    12: (2, 1),  # OTBio 13 -> DEMOVE 23 -> Side B pad 3
    13: (1, 1),  # OTBio 14 -> DEMOVE 22 -> Side B pad 2
    14: (18, 0),  # OTBio 15 -> DEMOVE 19 -> Side A pad 19
    15: (17, 0),  # OTBio 16 -> DEMOVE 18 -> Side A pad 18
    16: (14, 0),  # OTBio 17 -> DEMOVE 15 -> Side A pad 15
    17: (13, 0),  # OTBio 18 -> DEMOVE 14 -> Side A pad 14
    18: (10, 0),  # OTBio 19 -> DEMOVE 11 -> Side A pad 11
    19: (9, 0),  # OTBio 20 -> DEMOVE 10 -> Side A pad 10
    20: (6, 0),  # OTBio 21 -> DEMOVE 7  -> Side A pad 7
    21: (5, 0),  # OTBio 22 -> DEMOVE 6  -> Side A pad 6
    22: (2, 0),  # OTBio 23 -> DEMOVE 3  -> Side A pad 3
    23: (1, 0),  # OTBio 24 -> DEMOVE 2  -> Side A pad 2
    24: (0, 0),  # OTBio 25 -> DEMOVE 1  -> Side A pad 1
    25: (3, 0),  # OTBio 26 -> DEMOVE 4  -> Side A pad 4
    26: (4, 0),  # OTBio 27 -> DEMOVE 5  -> Side A pad 5
    27: (7, 0),  # OTBio 28 -> DEMOVE 8  -> Side A pad 8
    28: (8, 0),  # OTBio 29 -> DEMOVE 9  -> Side A pad 9
    29: (11, 0),  # OTBio 30 -> DEMOVE 12 -> Side A pad 12
    30: (12, 0),  # OTBio 31 -> DEMOVE 13 -> Side A pad 13
    31: (15, 0),  # OTBio 32 -> DEMOVE 16 -> Side A pad 16
    32: (16, 0),  # OTBio 33 -> DEMOVE 17 -> Side A pad 17
    33: (19, 0),  # OTBio 34 -> DEMOVE 20 -> Side A pad 20
    34: (0, 1),  # OTBio 35 -> DEMOVE 21 -> Side B pad 1
    35: (3, 1),  # OTBio 36 -> DEMOVE 24 -> Side B pad 4
    36: (4, 1),  # OTBio 37 -> DEMOVE 25 -> Side B pad 5
    37: (7, 1),  # OTBio 38 -> DEMOVE 28 -> Side B pad 8
    38: (8, 1),  # OTBio 39 -> DEMOVE 29 -> Side B pad 9
    39: (11, 1),  # OTBio 40 -> DEMOVE 32 -> Side B pad 12
}

GRID_POSITIONS_HD02MM0808 = {
    # Col 0
    53: (0, 0),
    54: (0, 1),
    55: (0, 2),
    56: (0, 3),
    64: (0, 4),
    63: (0, 5),
    62: (0, 6),
    61: (0, 7),
    52: (1, 0),
    51: (1, 1),
    50: (1, 2),
    49: (1, 3),
    60: (1, 4),
    57: (1, 5),
    58: (1, 6),
    59: (1, 7),
    48: (2, 0),
    47: (2, 1),
    46: (2, 2),
    45: (2, 3),
    33: (2, 4),
    34: (2, 5),
    35: (2, 6),
    36: (2, 7),
    44: (3, 0),
    43: (3, 1),
    42: (3, 2),
    41: (3, 3),
    37: (3, 4),
    38: (3, 5),
    39: (3, 6),
    40: (3, 7),
    32: (4, 0),
    31: (4, 1),
    30: (4, 2),
    29: (4, 3),
    28: (4, 4),
    27: (4, 5),
    26: (4, 6),
    25: (4, 7),
    24: (5, 0),
    23: (5, 1),
    22: (5, 2),
    21: (5, 3),
    20: (5, 4),
    19: (5, 5),
    18: (5, 6),
    17: (5, 7),
    1: (6, 0),
    2: (6, 1),
    3: (6, 2),
    4: (6, 3),
    5: (6, 4),
    6: (6, 5),
    7: (6, 6),
    8: (6, 7),
    16: (7, 0),
    15: (7, 1),
    14: (7, 2),
    13: (7, 3),
    12: (7, 4),
    11: (7, 5),
    10: (7, 6),
    9: (7, 7),
}

GRID_POSITIONS_HD04MM1305 = {
    # Col 0
    52: (0, 0),
    53: (0, 1),
    54: (0, 2),
    55: (0, 3),
    56: (0, 4),
    57: (0, 5),
    58: (0, 6),
    59: (0, 7),
    60: (0, 8),
    61: (0, 9),
    62: (0, 10),
    63: (0, 11),
    64: (0, 12),
    39: (1, 0),
    40: (1, 1),
    41: (1, 2),
    42: (1, 3),
    43: (1, 4),
    44: (1, 5),
    45: (1, 6),
    46: (1, 7),
    47: (1, 8),
    48: (1, 9),
    49: (1, 10),
    50: (1, 11),
    51: (1, 12),
    26: (2, 0),
    27: (2, 1),
    28: (2, 2),
    29: (2, 3),
    30: (2, 4),
    31: (2, 5),
    32: (2, 6),
    33: (2, 7),
    34: (2, 8),
    35: (2, 9),
    36: (2, 10),
    37: (2, 11),
    38: (2, 12),
    13: (3, 0),
    14: (3, 1),
    15: (3, 2),
    16: (3, 3),
    17: (3, 4),
    18: (3, 5),
    19: (3, 6),
    20: (3, 7),
    21: (3, 8),
    22: (3, 9),
    23: (3, 10),
    24: (3, 11),
    25: (3, 12),
    1: (4, 1),
    2: (4, 2),
    3: (4, 3),
    4: (4, 4),
    5: (4, 5),
    6: (4, 6),
    7: (4, 7),
    8: (4, 8),
    9: (4, 9),
    10: (4, 10),
    11: (4, 11),
    12: (4, 12),
}

GRID_POSITIONS_HD04MM1606 = {
    # Col 0: ch 81–96 (rows 0–15)
    81: (0, 0),
    82: (1, 0),
    83: (2, 0),
    84: (3, 0),
    85: (4, 0),
    86: (5, 0),
    87: (6, 0),
    88: (7, 0),
    89: (8, 0),
    90: (9, 0),
    91: (10, 0),
    92: (11, 0),
    93: (12, 0),
    94: (13, 0),
    95: (14, 0),
    96: (15, 0),
    # Col 1: ch 65–80
    65: (0, 1),
    66: (1, 1),
    67: (2, 1),
    68: (3, 1),
    69: (4, 1),
    70: (5, 1),
    71: (6, 1),
    72: (7, 1),
    73: (8, 1),
    74: (9, 1),
    75: (10, 1),
    76: (11, 1),
    77: (12, 1),
    78: (13, 1),
    79: (14, 1),
    80: (15, 1),
    # Col 2: ch 49–64
    49: (0, 2),
    50: (1, 2),
    51: (2, 2),
    52: (3, 2),
    53: (4, 2),
    54: (5, 2),
    55: (6, 2),
    56: (7, 2),
    57: (8, 2),
    58: (9, 2),
    59: (10, 2),
    60: (11, 2),
    61: (12, 2),
    62: (13, 2),
    63: (14, 2),
    64: (15, 2),
    # Col 3: ch 33–48
    33: (0, 3),
    34: (1, 3),
    35: (2, 3),
    36: (3, 3),
    37: (4, 3),
    38: (5, 3),
    39: (6, 3),
    40: (7, 3),
    41: (8, 3),
    42: (9, 3),
    43: (10, 3),
    44: (11, 3),
    45: (12, 3),
    46: (13, 3),
    47: (14, 3),
    48: (15, 3),
    # Col 4: ch 17–32
    17: (0, 4),
    18: (1, 4),
    19: (2, 4),
    20: (3, 4),
    21: (4, 4),
    22: (5, 4),
    23: (6, 4),
    24: (7, 4),
    25: (8, 4),
    26: (9, 4),
    27: (10, 4),
    28: (11, 4),
    29: (12, 4),
    30: (13, 4),
    31: (14, 4),
    32: (15, 4),
    # Col 5: ch 1–16
    1: (0, 5),
    2: (1, 5),
    3: (2, 5),
    4: (3, 5),
    5: (4, 5),
    6: (5, 5),
    7: (6, 5),
    8: (7, 5),
    9: (8, 5),
    10: (9, 5),
    11: (10, 5),
    12: (11, 5),
    13: (12, 5),
    14: (13, 5),
    15: (14, 5),
    16: (15, 5),
}

# HD10MM0804 / HD05MM0804: 8 rows × 4 cols = 32 channels
# Sequential layout (NOT serpentine): channels increase monotonically
# top-to-bottom within each physical column, columns left-to-right.
# grid_shape=(8,4) → positions as (row, col), matching the (rows,cols) convention
# used by GR08MM1305 and GR10MM0808.
GRID_POSITIONS_8x4 = {
    1: (0, 0),
    2: (1, 0),
    3: (2, 0),
    4: (3, 0),
    5: (4, 0),
    6: (5, 0),
    7: (6, 0),
    8: (7, 0),
    9: (0, 1),
    10: (1, 1),
    11: (2, 1),
    12: (3, 1),
    13: (4, 1),
    14: (5, 1),
    15: (6, 1),
    16: (7, 1),
    17: (0, 2),
    18: (1, 2),
    19: (2, 2),
    20: (3, 2),
    21: (4, 2),
    22: (5, 2),
    23: (6, 2),
    24: (7, 2),
    25: (0, 3),
    26: (1, 3),
    27: (2, 3),
    28: (3, 3),
    29: (4, 3),
    30: (5, 3),
    31: (6, 3),
    32: (7, 3),
}


# SIM10X32: simulated 10 rows x 32 cols = 320 channels.
# Row-major channel order (ch = row * 32 + col), matching the ch_map dataset in
# the simulation HDF5 files, so port-local indices are already grid keys.
GRID_POSITIONS_SIM10x32 = {i: (i // 32, i % 32) for i in range(320)}

# ULTRAHD 4X4: 16-channel ultra-high-density array, 4 rows x 4 cols, 250 um
# pitch. Channel numbering (ch1..ch16 in file order) wraps around the array
# rather than running row-major:
#
#     ch1   ch2   ch15  ch16
#     ch3   ch4   ch13  ch14
#     ch5   ch6   ch11  ch12
#     ch7   ch8   ch9   ch10
GRID_POSITIONS_ULTRAHD4x4 = {
    1: (0, 0),
    2: (0, 1),
    15: (0, 2),
    16: (0, 3),
    3: (1, 0),
    4: (1, 1),
    13: (1, 2),
    14: (1, 3),
    5: (2, 0),
    6: (2, 1),
    11: (2, 2),
    12: (2, 3),
    7: (3, 0),
    8: (3, 1),
    9: (3, 2),
    10: (3, 3),
}


ELECTRODE_GRIDS = {
    "GR04MM1305": {
        "grid_shape": (13, 5),
        "ied_mm": 4,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_13x5,
    },
    "GR08MM1305": {
        "grid_shape": (13, 5),
        "ied_mm": 8,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_13x5,
    },
    "GR10MM0808": {
        "grid_shape": (8, 8),
        "ied_mm": 10,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_8x8,
    },
    "Thin-film": {
        "grid_shape": (20, 2),
        "ied_mm": 5,
        "n_channels": 40,
        "muap_mapping": {i: i for i in range(40)},
        "positions": GRID_POSITIONS_20x2,
    },
    "HD02MM0808": {
        "grid_shape": (8, 8),
        "ied_mm": 2,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_HD02MM0808,
    },
    "HD04MM1305": {
        "grid_shape": (5, 13),
        "ied_mm": 4,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_HD04MM1305,
    },
    "HD04MM1606": {
        "grid_shape": (16, 6),
        "ied_mm": 4,
        "n_channels": 96,
        "muap_mapping": {i: i + 1 for i in range(96)},
        "positions": GRID_POSITIONS_HD04MM1606,
    },
    # Same 16x6 electrode with hardware channels 1-16 disconnected.  The
    # remaining channels occupy five complete columns; re-key them locally so
    # an 80-channel decomposition retains the correct physical arrangement.
    "HD08MM1606, CHANNELS 17-96": {
        "grid_shape": (16, 5),
        "ied_mm": 8,
        "n_channels": 80,
        "muap_mapping": {i: i + 1 for i in range(80)},
        "positions": {
            local_ch: GRID_POSITIONS_HD04MM1606[local_ch + 16]
            for local_ch in range(1, 81)
        },
    },
    "HD08MM1606": {
        "grid_shape": (16, 6),
        "ied_mm": 8,
        "n_channels": 96,
        "muap_mapping": {i: i + 1 for i in range(96)},
        "positions": GRID_POSITIONS_HD04MM1606,
    },
    "HD08MM1305": {
        "grid_shape": (5, 13),
        "ied_mm": 8,
        "n_channels": 64,
        "muap_mapping": {i: i + 1 for i in range(64)},
        "positions": GRID_POSITIONS_HD04MM1305,
    },
    "HD10MM0804": {
        "grid_shape": (8, 4),
        "ied_mm": 10,
        "n_channels": 32,
        "muap_mapping": {i: i + 1 for i in range(32)},
        "positions": GRID_POSITIONS_8x4,
    },
    "HD05MM0804": {
        "grid_shape": (8, 4),
        "ied_mm": 5,
        "n_channels": 32,
        "muap_mapping": {i: i + 1 for i in range(32)},
        "positions": GRID_POSITIONS_8x4,
    },
    "SIM10X32": {
        "grid_shape": (10, 32),
        "ied_mm": 4,
        "n_channels": 320,
        "muap_mapping": {i: i for i in range(320)},
        "positions": GRID_POSITIONS_SIM10x32,
    },
    "ULTRAHD 4X4": {
        "grid_shape": (4, 4),
        "ied_mm": 0.25,
        "n_channels": 16,
        "muap_mapping": {i: i + 1 for i in range(16)},
        "positions": GRID_POSITIONS_ULTRAHD4x4,
    },
}


def get_grid_config(electrode_type: str | None) -> dict | None:
    if electrode_type is None:
        return None
    key = electrode_type.upper()
    for name, cfg in ELECTRODE_GRIDS.items():
        if name.upper() in key:
            return cfg
    return None
