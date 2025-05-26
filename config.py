# config.py  ─ central place for environment & constants
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load .env if present (never committed because of .gitignore)
load_dotenv(override=False)

## -------- Public constants -------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "Environment variable OPENROUTER_API_KEY is missing. "
        "Create a .env file with OPENROUTER_API_KEY='sk-...'"
    )

REPO_ROOT = Path(__file__).resolve().parent
PROMPT_TEMPLATE_PATH = REPO_ROOT / "prompt.txt"
MODELS_YAML_PATH = REPO_ROOT / "models.yaml"

# Read model map once at import time
with MODELS_YAML_PATH.open("r") as fh:
    MODEL_ID_MAP: dict[str, str] = yaml.safe_load(fh)

# Regex we expect in model output (strict lowercase labels)
VALID_LABELS = ("low", "medium", "high", "very high")
