"""PDF raporları için platformdan bağımsız Unicode font kaydı.

Rapor scriptleri daha önce her birinde sabit Windows yollarını
(``C:\\Windows\\Fonts\\arial*.ttf``) tekrar ediyordu; bu da Windows dışı
ortamlarda (CI, Linux) FPDF'in çökmesine yol açıyordu. Bu yardımcı, mevcut
ilk TTF'i bularak (Windows Arial, ardından Linux/macOS'ta Arial metriğiyle
uyumlu Liberation Sans ve DejaVu Sans) fontları ``ArialUni``/``CourierUni``
aileleri altında kaydeder.
"""

from __future__ import annotations

import os
from typing import List, Optional

REGULAR_FAMILY = "ArialUni"
COURIER_FAMILY = "CourierUni"

_REGULAR_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]

_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

_ITALIC_CANDIDATES = [
    r"C:\Windows\Fonts\ariali.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/Library/Fonts/Arial Italic.ttf",
]

_COURIER_CANDIDATES = [
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Courier New.ttf",
]


def _first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def register_report_fonts(pdf, *, courier: bool = False) -> str:
    """Unicode fontları ``pdf`` üzerine kaydeder ve ana aile adını döndürür.

    Regular/Bold/Italic stilleri ``ArialUni`` ailesi altında kaydedilir.
    ``courier=True`` verilirse ek olarak ``CourierUni`` ailesi de kaydedilir.
    """
    regular = _first_existing(_REGULAR_CANDIDATES)
    if regular is None:
        raise FileNotFoundError(
            "Rapor için uygun bir Unicode TTF font bulunamadı. "
            "Liberation veya DejaVu fontlarını kurun "
            "(ör. Debian/Ubuntu: 'apt-get install fonts-liberation')."
        )

    pdf.add_font(REGULAR_FAMILY, "", regular)
    pdf.add_font(REGULAR_FAMILY, "B", _first_existing(_BOLD_CANDIDATES) or regular)
    pdf.add_font(REGULAR_FAMILY, "I", _first_existing(_ITALIC_CANDIDATES) or regular)

    if courier:
        pdf.add_font(
            COURIER_FAMILY, "", _first_existing(_COURIER_CANDIDATES) or regular
        )

    return REGULAR_FAMILY
