"""Report formatting and readability checks."""

import re
import unicodedata
from pathlib import Path

MEDICAL = re.compile(
    r"carcinoma|histolog|tumou?r|differenti|malignan|specimen|diagnos|lung|breast|lymph|biopsy|patholog",
    re.I,
)


def quality(text):
    tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)
    visible = [c for c in text if not c.isspace()]
    long = sum(len(w) >= 3 for w in tokens)
    alpha = sum(c.isalpha() for c in visible) / max(1, len(visible))
    ratio = long / max(1, len(tokens))
    reasons = []
    if len(tokens) < 20:
        reasons.append("too_few_words")
    if ratio < 0.45:
        reasons.append("fragmented_words")
    if alpha < 0.55:
        reasons.append("mostly_symbols_or_digits")
    if "\ufffd" in text or "\x00" in text:
        reasons.append("invalid_characters")
    if not MEDICAL.search(text):
        reasons.append("no_clinical_terms")
    return {
        "words": len(tokens),
        "long_word_fraction": ratio,
        "alphabetic_fraction": alpha,
        "status": "review_required" if reasons else "pass",
        "reasons": reasons,
    }


def normalize_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return (
        "\n".join(re.sub(r"[ \t]+", " ", s).strip() for s in text.splitlines()).strip()
        + "\n"
    )


def load_training_report(row):
    if (
        not row.get("report_trainable")
        or row.get("report_validation_status") != "validated"
    ):
        raise ValueError(
            "Report has not passed source-only checks: " + row["patient_id"]
        )
    text = Path(row["clean_report_path"]).read_text(encoding="utf-8")
    if quality(text)["status"] != "pass":
        raise ValueError("Unreadable report: " + row["patient_id"])
    return text
