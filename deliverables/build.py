"""Dựng slide HTML và báo cáo PDF từ template + số liệu trong outputs_v2.

Usage:
    python deliverables/build.py            # slide + báo cáo HTML + PDF
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs_v2"
CLASSES = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other"]
BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]


def data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def drift_points() -> str:
    v = pd.read_csv(OUT / "data_audit" / "video_audit.csv")
    pts = [[round(d, 2), round(x, 3), int(round(f, 2) == 23.98)]
           for d, x, f in zip(v.duration, v.legacy_drift_end_s, v.fps)]
    return json.dumps(pts, separators=(",", ":"))


def class_shift() -> str:
    cd = pd.read_csv(OUT / "data_audit" / "class_distribution.csv").set_index("group")
    pct = lambda g: [round(100 * cd.loc[g, c] / cd.loc[g, CLASSES].sum(), 1) for c in CLASSES]
    return json.dumps({"syn": pct("OF-Syn"), "itw": pct("ItW test")})


def build_slides() -> Path:
    src = (HERE / "src" / "slides.src.html").read_text(encoding="utf-8")
    src = (src.replace("{{CSS}}", (HERE / "src" / "space.css").read_text(encoding="utf-8"))
              .replace("{{JS}}", (HERE / "src" / "space.js").read_text(encoding="utf-8")))
    html = (src.replace("{{FRAME}}", data_uri(HERE / "assets" / "frame_fallen.jpg"))
               .replace("{{APP}}", data_uri(HERE / "assets" / "app_screen.jpg"))
               .replace("{{DRIFT}}", drift_points())
               .replace("{{SHIFT}}", class_shift()))
    assert "{{" not in html, "còn placeholder chưa thay"
    # bản độc lập mở bằng trình duyệt
    out = HERE / "OmniFall_slides.html"
    out.write_text('<!doctype html>\n<html lang="vi"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                   f'</head><body>\n{html}\n</body></html>\n', encoding="utf-8")
    return out


def build_report() -> Path | None:
    src_path = HERE / "src" / "report.src.html"
    if not src_path.exists():
        return None
    html = src_path.read_text(encoding="utf-8")
    figs = {
        "FIG_CLASS": OUT / "data_audit" / "fig_class_distribution.png",
        "FIG_TRANS": OUT / "data_audit" / "fig_transition_matrices.png",
        "FIG_VIDEO": OUT / "data_audit" / "fig_video_audit.png",
        "FIG_DUR": OUT / "data_audit" / "fig_segment_durations.png",
        "FIG_CONF": OUT / "final" / "fig_confusion_full.png",
        "FIG_TRADE": OUT / "final" / "fig_stream_tradeoff.png",
        "FIG_FRAME": HERE / "assets" / "frame_fallen.jpg",
        "FIG_APP": HERE / "assets" / "app_screen.jpg",
    }
    for k, p in figs.items():
        html = html.replace("{{" + k + "}}", data_uri(p))
    assert "{{" not in html, "còn placeholder chưa thay"
    out_html = HERE / "OmniFall_report.html"
    out_html.write_text(html, encoding="utf-8")
    browser = next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        print("Không tìm thấy Edge/Chrome để xuất PDF")
        return out_html
    pdf = HERE / "OmniFall_report.pdf"
    if pdf.exists():
        pdf.unlink()
    import tempfile
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={prof}", "--virtual-time-budget=20000",
                        f"--print-to-pdf={pdf}", out_html.as_uri()], timeout=240)
        # Edge/Chrome trên Windows trả về trước khi tiến trình con ghi xong PDF
        import time
        for _ in range(120):
            if pdf.exists() and pdf.stat().st_size > 0:
                time.sleep(1.0)
                break
            time.sleep(0.5)
    if not pdf.exists():
        raise RuntimeError("Trình duyệt không xuất được PDF")
    return pdf


if __name__ == "__main__":
    print("slides:", build_slides())
    print("report:", build_report())
    sys.exit(0)
