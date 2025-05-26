# parser.py  ─ extracts the label from model responses
import re
from typing import Optional
from config import VALID_LABELS

# Pre-compile once
_LABEL_RE = re.compile(r"\b(low|medium|high|very high)\b", re.IGNORECASE)

def extract_label(text: str) -> Optional[str]:
    """
    Return one of 'low'|'medium'|'high'|'very high' found in `text`
    or None if nothing matches.
    """
    m = _LABEL_RE.search(text)
    if not m:
        return None
    return m.group(1).lower()

