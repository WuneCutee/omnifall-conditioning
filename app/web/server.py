"""Web demo: mở camera realtime trên trình duyệt hoặc upload video để chạy mô hình phát hiện té ngã.

Chạy:
    python app/web/server.py            # mở http://localhost:8000

  * Camera: trình duyệt gửi frame JPEG qua WebSocket (/ws/camera), server chạy FallEngine
    và trả trạng thái (nhãn, xác suất 10 lớp, cảnh báo) cho mỗi frame được lấy mẫu.
  * Upload: video được xử lý ở nền (mọi frame, không bỏ frame), trả video đã gắn nhãn
    + dòng thời gian dự đoán để xem lại.
Camera và upload dùng 2 engine riêng nên chạy song song; video upload xử lý lần lượt.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.annotate_video import annotate, build_engine
from tcfall.config import CLASS_NAMES, OUTPUTS
from tcfall.realtime import MODES

STATIC = Path(__file__).parent / "static"
JOBS_DIR = OUTPUTS / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fall detection demo")
print("Đang tải VideoMAE + classifier + conditioner ...")
# 2 engine riêng: camera và upload chạy song song, không phải chờ nhau
ENGINE = build_engine()          # camera realtime
UP_ENGINE = build_engine()       # xử lý video upload
DEFAULT_MODE = ENGINE.cfg.mode
UP_LOCK = threading.Lock()       # các video upload xử lý lần lượt
CAM_SESSION = {"id": None}       # phiên camera mới nhất được dùng engine, phiên cũ tự dừng
JOBS: dict[str, dict] = {}


def engine_snapshot() -> dict:
    st = ENGINE.state
    return {"t": round(st.t, 2), "ready": st.ready, "status": ENGINE.status, "label": st.label,
            "probs": [round(float(p), 4) for p in st.probs],
            "baseline": [round(float(p), 4) for p in st.baseline_probs],
            "alarm": st.alarm, "alarms": [round(x, 2) for x in st.alarm_events],
            "latency_ms": round(st.latency_ms, 1)}


@app.get("/api/info")
def info():
    return {"classes": CLASS_NAMES, "modes": list(MODES), "default_mode": DEFAULT_MODE,
            "device": ENGINE.device, "spatial_tta": ENGINE.cfg.spatial_tta}


# ------------------------------------------------------------------ camera realtime
@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await ws.accept()
    sid = uuid.uuid4().hex
    CAM_SESSION["id"] = sid
    try:
        ENGINE.cfg.mode = ws.query_params.get("mode", DEFAULT_MODE)
        ENGINE.reset()
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if CAM_SESSION["id"] != sid:
                await ws.send_json({"error": "Camera đã được mở ở tab hoặc máy khác, phiên này dừng lại."})
                await ws.close(); break
            if msg.get("text"):
                cmd = json.loads(msg["text"])
                if cmd.get("mode") in MODES:
                    ENGINE.cfg.mode = cmd["mode"]
                if cmd.get("reset"):
                    ENGINE.reset()
                continue
            data = msg.get("bytes")
            if not data or len(data) < 9:
                continue
            t = float(np.frombuffer(data[:8], dtype="<f8")[0])           # 8 byte đầu: thời điểm (giây)
            bgr = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            sampled = await asyncio.to_thread(lambda: ENGINE.push_frame(rgb, t) and [ENGINE.step() for _ in range(4)])
            await ws.send_json({**engine_snapshot(), "sampled": bool(sampled)})
    except (WebSocketDisconnect, RuntimeError):
        pass


# ------------------------------------------------------------------ upload video
def run_job(job_id: str, src: Path, mode: str):
    job = JOBS[job_id]
    with UP_LOCK:
        job["state"] = "running"
        try:
            UP_ENGINE.cfg.mode = mode
            # chỉ dự đoán (không vẽ lên video); giao diện web tự hiển thị nhãn và xác suất
            res = annotate(UP_ENGINE, str(src), None, progress=lambda p: job.update(progress=round(p, 3)))
            out = src.with_name("video.mp4")
            ff = shutil.which("ffmpeg")
            if ff:  # video gốc, cắt bỏ đoạn mô hình khởi động, mã hoá H.264 để trình duyệt phát được
                subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(res["start"]), "-i", str(src),
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-c:a", "aac",
                                "-movflags", "+faststart", str(out)], check=True)
            else:
                shutil.copy(src, out); res["start"] = 0.0  # không cắt được: video giữ nguyên từ giây 0
            (src.parent / "result.json").write_text(json.dumps(res), encoding="utf-8")
            job.update(state="done", progress=1.0, result=res)
        except Exception as e:  # báo lỗi cho giao diện thay vì treo
            job.update(state="error", error=str(e))
        finally:
            UP_ENGINE.cfg.mode = DEFAULT_MODE


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), mode: str = Form(None)):
    mode = mode if mode in MODES else DEFAULT_MODE
    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id; d.mkdir(parents=True)
    src = d / ("input" + (Path(file.filename or "").suffix or ".mp4"))
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    JOBS[job_id] = {"state": "queued", "progress": 0.0, "name": file.filename, "mode": mode, "created": time.time()}
    threading.Thread(target=run_job, args=(job_id, src, mode), daemon=True).start()
    return {"job": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Không có job này")
    out = {k: v for k, v in job.items() if k != "result"}
    if job["state"] == "done":
        out["result"] = job["result"]; out["video"] = f"/api/job/{job_id}/video"
    return out


@app.get("/api/job/{job_id}/video")
def job_video(job_id: str):
    p = JOBS_DIR / job_id / "video.mp4"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
