"""Report formatting and readability / 报告格式规范化与可读性检查。"""

import re
import unicodedata
from pathlib import Path

MEDICAL = re.compile(
    r"carcinoma|histolog|tumou?r|differenti|malignan|specimen|diagnos|lung|breast|lymph|biopsy|patholog",
    re.I,
)


def quality(text):
    """Assess report readability / 判断报告可读性。

    Input / 输入:
        text: report string.
        text：报告字符串。
    Processing / 处理:
        Check word count, long-word and alphabetic fractions, invalid characters, and clinical terms.
        检查词数、长词占比、字母占比、异常字符及临床关键词。
    Output / 输出:
        Dictionary of measurements, pass/review_required status, and failure reasons.
        包含统计值、pass/review_required 状态及失败原因的字典。
    """
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
    """Normalize report formatting / 规范化报告格式。

    Input / 输入:
        text: original report string.
        text：原始报告字符串。
    Processing / 处理:
        Apply NFC, normalize line endings, collapse spaces/tabs, and strip line edges.
        应用 NFC，统一换行、合并空格与制表符、去除行首行尾空白。
    Output / 输出:
        Formatted string ending with a newline; clinical wording is preserved.
        以换行结尾的格式化文本，保留临床文字内容。
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return (
        "\n".join(re.sub(r"[ \t]+", " ", s).strip() for s in text.splitlines()).strip()
        + "\n"
    )


def load_training_report(row):
    """Load a report approved for supervision / 读取可用于监督的报告。

    Input / 输入:
        Patient row with report status and clean_report_path.
        包含报告可用状态与 clean_report_path 的患者记录。
    Processing / 处理:
        Check availability, read the copy, and recheck readability.
        检查可用状态，读取清洗副本并复查可读性。
    Output / 输出:
        Report text; raises ValueError for unapproved or unreadable reports.
        返回报告文本；未通过检查或不可读时抛出 ValueError。
    """
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
