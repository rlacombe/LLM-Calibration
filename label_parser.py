# parser.py  ─ extracts the label from model responses
import re
from typing import Optional
from config import VALID_LABELS

# Pre-compile once - order matters: "very high" must come before "high"
_LABEL_RE = re.compile(r"(I don't know|very high|high|medium|low|not classifiable|probably not carcinogenic|probably carcinogenic|possibly carcinogenic|carcinogenic to humans)", re.IGNORECASE)

def extract_label(text: str) -> Optional[str]:
    """
    Extract the label from the response text.
    Returns "(empty)" if text is empty, None if no valid label is found.
    If multiple hits, only the last one counts, with "high" in "very high" counted as "very high".
    """
    if not text:
        return "(empty)"

    # Find all matches, order matters: "very high" before "high"
    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        return None

    # Only the last match counts
    last_label = matches[-1].group(1).lower()
    return last_label
