from pathlib import Path

SHIFT_MIN = 850
SHIFT_MAX = 1750

DIR_DATA = Path("./dots")
DIR_GRAPHS = Path("./graphs")

COLORS = {
    "kb": "red",
    "smb": "orange",
    "pb": "yellow",
    "svb": "purple",
    "sb": "royalblue",
    "kz": "red",
    "smz": "orange",
    "pz": "yellow",
    "svz": "purple",
    "sz": "royalblue",
}

NAMES = []
MEAN_NAMES = {
    "kb": [r"k\db-p\d"],
    "pb": [r"p\db-p\d"],
    "smb": [r"sm\db-p\d"],
    "svb": [r"sv\db-p\d"],
    "sb": [r"s\db-p\d"]
}

NAMES = ["k1b-p1", "sm1b-p1", "p1b-p1",
         "sv1b-p1", "s1b-p1"]

MEAN_NAMES_ALBUMIN = {
    "kb": [r"k\db-p\d"],
    "pb": [r"p\db-p\d"],
    "smb": [r"sm\db-p\d"],
    "svb": [r"sv\db-p\d"],
    "sb": [r"s\db-p\d"],
}
MEAN_NAMES_YOLK = {
    "kz": [r"k\dz-p\d"],
    "pz": [r"p\dz-p\d"],
    "smz": [r"sm\dz-p\d"],
    "svz": [r"sv\dz-p\d"],
    # "sz": [r"s\dz-p\d"]
}