"""JSON output helpers / JSON 与 JSONL 写入工具。"""

import json
from pathlib import Path


def write_json(path, value):
    """Write one JSON document / 写入一个 JSON 文件。

    Input / 输入:
        Destination path and JSON-serializable value.
        目标路径和可 JSON 序列化的数据。
    Processing / 处理:
        Write formatted JSON to a temporary sibling file, then replace the target.
        先写入同目录临时文件，再替换目标文件；拒绝 NaN 等非标准值。
    Output / 输出:
        File on disk; returns None.
        生成磁盘文件，无返回值。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def write_jsonl(path, values):
    """Write one record per line / 按行写入 JSON 记录。

    Input / 输入:
        Destination path and iterable of JSON-serializable records.
        目标路径及可 JSON 序列化的记录集合。
    Processing / 处理:
        Create parent directories and serialize each record on its own line.
        创建父目录，每条记录序列化为一行。
    Output / 输出:
        JSONL file on disk; returns None.
        生成 JSONL 文件，无返回值。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values)
    )
