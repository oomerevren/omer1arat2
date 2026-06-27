"""Ortak dosya G/Ç yardımcıları (JSON okuma/yazma)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def read_json(path: PathLike) -> Any:
    """UTF-8 bir JSON dosyasını okuyup içeriğini döndürür."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    path: PathLike,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Veriyi UTF-8 bir JSON dosyasına yazar."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
