from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SOCCERBENCH_ROOT = DATA_ROOT / "raw" / "datasets" / "SoccerBench"
QA_DIR = SOCCERBENCH_ROOT / "qa"

PROCESSED_DIR = DATA_ROOT / "processed"
CLEAN_DIR = PROCESSED_DIR / "clean"
INDEX_DIR = PROCESSED_DIR / "index"
AUDIT_DIR = PROCESSED_DIR / "audit"

CACHE_DIR = DATA_ROOT / "cache"
FRAME_CACHE_DIR = CACHE_DIR / "frames"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"

CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ten file json lech 1 so voi ky hieu trong paper (q8.json la Q7, ...)
TASK_IDS = ["q8", "q9", "q10", "q11", "q13", "q14"]

TASK_META = {
    "q8":  {"paper_id": "Q7",  "name": "Camera Status Switching",     "branch": "B", "n_clips": 1},
    "q9":  {"paper_id": "Q8",  "name": "Replay Grounding",            "branch": "C", "n_clips": 5},
    "q10": {"paper_id": "Q9",  "name": "Action Classification",       "branch": "A", "n_clips": 1},
    "q11": {"paper_id": "Q10", "name": "Commentary Generation",       "branch": "A", "n_clips": 1},
    "q13": {"paper_id": "Q12", "name": "Jersey Color Relevant QA",    "branch": "B", "n_clips": 1},
    "q14": {"paper_id": "Q13", "name": "Multi-view Foul Recognition", "branch": "C", "n_clips": None},
}

EXPECTED_COUNTS = {"q8": 400, "q9": 400, "q10": 1000, "q11": 1000, "q13": 700, "q14": 300}
EXPECTED_COUNTS_CLEAN = {"q8": 400, "q9": 400, "q10": 1000, "q11": 1000, "q13": 700, "q14": 276}
EXPECTED_OPTION_COUNTS = {
    "q8": {4: 400}, "q9": {4: 400}, "q10": {4: 1000}, "q11": {4: 1000},
    "q13": {4: 700}, "q14": {2: 174, 3: 53, 4: 73},
}
TASKS_WITH_LIST_OPENA = ["q9"]

# loi annotation da biet (index theo file json goc)
Q14_DROP_INDICES = [6, 8, 45, 46, 53, 72, 77, 83, 98, 115, 121, 128,
                    140, 145, 160, 167, 168, 180, 191, 223, 255, 266, 281, 285]
Q14_CLOSEA_FIX = {41: "O4", 70: "O2", 86: "O2", 148: "O4", 203: "O2", 208: "O3", 280: "O3"}
Q13_CLOSEA_FIX = {163: "O2"}
Q13_OPENA_EMPTY_INDICES = [225, 297, 484, 688]

LABEL_ALIASES = {"st. tackling": "standing tackling", "foul (no card)": "foul with no card",
                 "start of game(half)": "start of half game"}

ORDINAL_SCALES = {
    "severity": {"no card": 0.0, "no card or yellow card": 0.5, "yellow card": 1.0,
                 "yellow card or red card": 1.5, "red card": 2.0},
    "offence": {"no offence": 0.0, "between": 0.5, "offence": 1.0},
}

# 13 nhan camera cua SoccerNet-v2
CAMERA_LABELS = [
    "Main camera center", "Close-up player or field referee", "Close-up side staff",
    "Main camera left", "Main behind the goal", "Close-up behind the goal",
    "Spider camera", "Main camera right", "Public", "Goal line technology camera",
    "Close-up corner", "Inside the goal", "Other",
]

OPTION_KEY_PATTERN = r"O\d+"
