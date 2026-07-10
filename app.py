#!/usr/bin/env python3
import os
import uuid
import json
import shutil
import threading
import subprocess
import time
import io
import zipfile
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response, render_template

import numpy as np
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
HISTORY_FILE = Path("history.json")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# job_id -> {status, progress, output_path, output_name, output_format, original_name, error}
jobs: dict = {}
history: list = []

VIDEO_CODEC_MAP = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}
AUDIO_CODEC_MAP = {"aac": "aac", "mp3": "libmp3lame", "ac3": "ac3"}
CHANNEL_MAP = {"mono": "1", "stereo": "2", "5.1": "6"}
MIME_MAP = {"mp4": "video/mp4", "mov": "video/quicktime", "mkv": "video/x-matroska", "webm": "video/webm"}


def load_history():
    global history
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            history = []


def save_history():
    try:
        HISTORY_FILE.write_text(json.dumps(history[-200:], indent=2))
    except Exception:
        pass


def get_video_info(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(r.stdout)
    except Exception:
        return {}

    info = {}
    try:
        info["duration"] = float(data["format"].get("duration", 0))
        info["size"] = int(data["format"].get("size", 0))
        info["format_name"] = data["format"].get("format_name", "")
        info["overall_bitrate"] = data["format"].get("bit_rate", "")
    except Exception:
        pass

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and "video" not in info:
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = round(float(num) / float(den), 3) if float(den) else 0
            except Exception:
                fps = 0
            info["video"] = {
                "codec": stream.get("codec_name", ""),
                "profile": stream.get("profile", ""),
                "width": stream.get("width", 0),
                "height": stream.get("height", 0),
                "fps": fps,
                "bitrate": stream.get("bit_rate", ""),
                "pix_fmt": stream.get("pix_fmt", ""),
                "color_space": stream.get("color_space", ""),
            }
        elif stream.get("codec_type") == "audio" and "audio" not in info:
            info["audio"] = {
                "codec": stream.get("codec_name", ""),
                "sample_rate": stream.get("sample_rate", ""),
                "channels": stream.get("channels", 0),
                "channel_layout": stream.get("channel_layout", ""),
                "bitrate": stream.get("bit_rate", ""),
            }

    return info


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def build_compress_cmd(input_path: Path, output_path: Path, p: dict) -> list[str]:
    codec = p.get("codec", "h264")
    crf = int(p.get("crf", 23))
    speed = p.get("speed", "medium")
    output_format = p.get("output_format", "mp4")
    audio_mode = p.get("audio_mode", "copy")

    vc = "libx265" if codec == "h265" else "libx264"

    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-progress", "pipe:1", "-nostats"]
    cmd += ["-c:v", vc, "-crf", str(crf), "-preset", speed]

    if codec == "h265" and output_format in ("mp4", "mov"):
        cmd += ["-tag:v", "hvc1"]

    if audio_mode == "copy":
        cmd += ["-c:a", "copy"]
    else:
        abr = int(p.get("audio_bitrate", 128))
        cmd += ["-c:a", "aac", "-b:a", f"{abr}k"]

    cmd.append(str(output_path))
    return cmd


def run_compress_job(job_id: str, input_path: Path, output_path: Path, params: dict) -> None:
    stderr_lines: list[str] = []
    try:
        duration = get_duration(input_path)

        cmd = build_compress_cmd(input_path, output_path, params)
        jobs[job_id].update({"status": "processing", "progress": 0.0})

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        def drain_stderr() -> None:
            for line in proc.stderr:
                stderr_lines.append(line)

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()

        for line in proc.stdout:
            parts = line.strip().split("=", 1)
            if len(parts) == 2 and parts[0] == "out_time_us" and duration > 0:
                try:
                    pct = min(99.0, float(parts[1]) / 1_000_000 / duration * 100)
                    jobs[job_id]["progress"] = round(pct, 1)
                except ValueError:
                    pass

        proc.wait()
        t.join(timeout=5)

        if proc.returncode == 0:
            output_size = output_path.stat().st_size if output_path.exists() else 0
            jobs[job_id].update({"status": "done", "progress": 100.0, "output_size": output_size})
            entry = {
                "id": job_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "original_name": jobs[job_id].get("original_name", ""),
                "output_name": jobs[job_id]["output_name"],
                "settings": params,
                "output_size": output_size,
                "source_duration": duration,
            }
            history.insert(0, entry)
            save_history()
        else:
            tail = "".join(stderr_lines[-15:]).strip()
            jobs[job_id].update({
                "status": "error",
                "error": f"FFmpeg exited with code {proc.returncode}.\n\n{tail}",
            })
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


def build_cmd(input_path: Path, output_path: Path, p: dict) -> list[str]:
    cmd = ["ffmpeg", "-y"]

    trim_start = p.get("trim_start")
    trim_end = p.get("trim_end")
    if trim_start:
        cmd += ["-ss", str(trim_start)]

    cmd += ["-i", str(input_path), "-progress", "pipe:1", "-nostats"]

    vc = VIDEO_CODEC_MAP.get(p.get("video_codec", "h264"), "libx264")
    cmd += ["-c:v", vc]

    if vc == "libx265" and p.get("output_format") in ("mp4", "mov"):
        cmd += ["-tag:v", "hvc1"]

    res = p.get("resolution", "original")
    if res and res != "original":
        w, h = res.split("x")
        cmd += ["-vf", f"scale={w}:{h}"]

    vbr = p.get("video_bitrate")
    if vbr:
        cmd += ["-b:v", f"{vbr}k"]

    fr = p.get("frame_rate")
    if fr:
        cmd += ["-r", str(fr)]

    if trim_start and trim_end:
        cmd += ["-t", str(float(trim_end) - float(trim_start))]
    elif trim_end and not trim_start:
        cmd += ["-to", str(trim_end)]

    if p.get("color_space") == "hdr":
        cmd += ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084"]
    else:
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

    ac = AUDIO_CODEC_MAP.get(p.get("audio_codec", "aac"), "aac")
    cmd += ["-c:a", ac]

    abr = p.get("audio_bitrate")
    if abr:
        cmd += ["-b:a", f"{abr}k"]

    sr = p.get("audio_sample_rate")
    if sr:
        cmd += ["-ar", str(sr)]

    ch = CHANNEL_MAP.get(p.get("audio_channels", "stereo"), "2")
    cmd += ["-ac", ch]

    cmd.append(str(output_path))
    return cmd


def run_job(job_id: str, input_path: Path, output_path: Path, params: dict) -> None:
    stderr_lines: list[str] = []
    try:
        duration = get_duration(input_path)
        ts = float(params.get("trim_start") or 0)
        te = float(params.get("trim_end") or 0)
        if te > 0:
            duration = te - ts
        elif ts > 0:
            duration = max(0, duration - ts)

        cmd = build_cmd(input_path, output_path, params)
        jobs[job_id].update({"status": "processing", "progress": 0.0})

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        def drain_stderr() -> None:
            for line in proc.stderr:
                stderr_lines.append(line)

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()

        for line in proc.stdout:
            parts = line.strip().split("=", 1)
            if len(parts) == 2 and parts[0] == "out_time_us" and duration > 0:
                try:
                    pct = min(99.0, float(parts[1]) / 1_000_000 / duration * 100)
                    jobs[job_id]["progress"] = round(pct, 1)
                except ValueError:
                    pass

        proc.wait()
        t.join(timeout=5)

        if proc.returncode == 0:
            output_size = output_path.stat().st_size if output_path.exists() else 0
            jobs[job_id].update({"status": "done", "progress": 100.0, "output_size": output_size})
            entry = {
                "id": job_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "original_name": jobs[job_id].get("original_name", ""),
                "output_name": jobs[job_id]["output_name"],
                "settings": params,
                "output_size": output_size,
                "source_duration": duration,
            }
            history.insert(0, entry)
            save_history()
        else:
            tail = "".join(stderr_lines[-15:]).strip()
            jobs[job_id].update({
                "status": "error",
                "error": f"FFmpeg exited with code {proc.returncode}.\n\n{tail}",
            })
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


# ── Seam Carving ─────────────────────────────────────────────────────────────

def _energy(arr: np.ndarray) -> np.ndarray:
    """Dual-gradient energy map (grayscale perceived luminance)."""
    if arr.ndim == 3:
        gray = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2])
    else:
        gray = arr.astype(np.float64)
    dy = np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    dx = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    return dx + dy


def _find_seam(energy: np.ndarray) -> np.ndarray:
    """Dynamic-programming vertical seam with fully vectorised row updates."""
    h, w = energy.shape
    dp = energy.copy()
    back = np.zeros((h, w), dtype=np.int32)
    cols = np.arange(w)
    for i in range(1, h):
        prev = dp[i - 1]
        p = np.pad(prev, (1, 1), mode="edge")
        stacked = np.stack([p[:-2], p[1:-1], p[2:]], axis=1)   # (w, 3)
        choice = np.argmin(stacked, axis=1) - 1                 # -1, 0, 1
        prev_col = np.clip(cols + choice, 0, w - 1)
        back[i] = prev_col
        dp[i] += prev[prev_col]
    seam = np.empty(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for i in range(h - 2, -1, -1):
        seam[i] = back[i + 1, seam[i + 1]]
    return seam


def _remove_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    if img.ndim == 2:
        return img[mask].reshape(h, w - 1)
    c = img.shape[2]
    return img[np.broadcast_to(mask[:, :, None], (h, w, c))].reshape(h, w - 1, c)


def _insert_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    squeeze = img.ndim == 2
    if squeeze:
        img = img[:, :, np.newaxis]
    c = img.shape[2]
    out = np.empty((h, w + 1, c), dtype=img.dtype)
    rows = np.arange(h)
    l_idx = np.clip(seam - 1, 0, w - 1)
    r_idx = np.clip(seam,     0, w - 1)
    avg = ((img[rows, l_idx].astype(np.int32) + img[rows, r_idx].astype(np.int32)) // 2).astype(img.dtype)
    for i in range(h):
        j = seam[i]
        out[i, :j]     = img[i, :j]
        out[i, j]      = avg[i]
        out[i, j + 1:] = img[i, j:]
    return out[:, :, 0] if squeeze else out


def _map_seams_to_original(seam_list: list) -> list:
    """Translate seams found in progressively-compressed images to original coords."""
    orig = [s.copy() for s in seam_list]
    for i in range(1, len(seam_list)):
        seam = seam_list[i].copy()
        for j in range(i - 1, -1, -1):
            seam[seam_list[j] <= seam] += 1
        orig[i] = seam
    return orig


def _carve_width(arr: np.ndarray, target_w: int, progress_cb=None, p_start=0, p_end=100) -> np.ndarray:
    h, w = arr.shape[:2]
    delta = target_w - w
    total = abs(delta) or 1

    if delta < 0:
        for n in range(-delta):
            arr = _remove_seam(arr, _find_seam(_energy(arr.astype(np.float64))))
            if progress_cb:
                progress_cb(int(p_start + (n + 1) / total * (p_end - p_start)))
    elif delta > 0:
        k = delta
        temp = arr.copy()
        seam_list = []
        for n in range(k):
            seam = _find_seam(_energy(temp.astype(np.float64)))
            seam_list.append(seam)
            temp = _remove_seam(temp, seam)
            if progress_cb:
                progress_cb(int(p_start + (n + 1) / (k * 2) * (p_end - p_start)))
        orig_seams = _map_seams_to_original(seam_list)
        for n in range(k):
            seam = orig_seams[n].copy()
            for j in range(n):
                seam += (orig_seams[j] <= seam).astype(np.int32)
            arr = _insert_seam(arr, seam)
            if progress_cb:
                progress_cb(int(p_start + (k + n + 1) / (k * 2) * (p_end - p_start)))

    return arr


def seam_carve(arr: np.ndarray, target_w: int, target_h: int, progress_cb=None) -> np.ndarray:
    arr = _carve_width(arr, target_w, progress_cb, p_start=0, p_end=50)
    # Transpose → carve height as width → transpose back
    arr = np.swapaxes(arr, 0, 1)
    arr = _carve_width(arr, target_h, progress_cb, p_start=50, p_end=98)
    arr = np.swapaxes(arr, 0, 1)
    return arr


# ── Image resize jobs ─────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
MAX_IMG_PX = 2000   # cap each dimension before seam carving


def run_image_job(job_id: str, input_path: Path, output_path: Path,
                  target_w: int, target_h: int, out_fmt: str,
                  crop=None) -> None:
    try:
        pil_img = Image.open(input_path)
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        # Apply user crop before seam carving
        if crop:
            cx, cy = int(crop.get("x", 0)), int(crop.get("y", 0))
            cw, ch = int(crop.get("w", pil_img.width)), int(crop.get("h", pil_img.height))
            pil_img = pil_img.crop((cx, cy, cx + cw, cy + ch))

        orig_w, orig_h = pil_img.size
        scale = min(1.0, MAX_IMG_PX / max(orig_w, orig_h))
        if scale < 1.0:
            pil_img = pil_img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
            target_w = max(1, round(target_w * scale))
            target_h = max(1, round(target_h * scale))

        arr = np.array(pil_img)
        jobs[job_id].update({"status": "processing", "progress": 0.0})

        def cb(pct: int):
            jobs[job_id]["progress"] = float(pct)

        result = seam_carve(arr, target_w, target_h, progress_cb=cb)

        out_pil = Image.fromarray(result.astype(np.uint8))

        # JPEG cannot store alpha — composite onto white
        if out_fmt.lower() in ("jpeg", "jpg") and out_pil.mode == "RGBA":
            bg = Image.new("RGB", out_pil.size, (255, 255, 255))
            bg.paste(out_pil, mask=out_pil.split()[3])
            out_pil = bg
        elif out_fmt.lower() in ("jpeg", "jpg") and out_pil.mode != "RGB":
            out_pil = out_pil.convert("RGB")

        out_pil.save(str(output_path), format=out_fmt.upper())

        jobs[job_id].update({
            "status": "done",
            "progress": 100.0,
            "output_size": output_path.stat().st_size,
        })
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
        input_path.unlink(missing_ok=True)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    upload_id = str(uuid.uuid4())
    suffix = Path(f.filename).suffix.lower() or ".mp4"
    dest = UPLOAD_DIR / f"{upload_id}{suffix}"
    f.save(dest)

    info = get_video_info(dest)

    return jsonify({
        "upload_id": upload_id,
        "original_name": f.filename,
        "size": dest.stat().st_size,
        "info": info,
    })


@app.route("/serve/<upload_id>")
def serve_upload(upload_id: str):
    matches = list(UPLOAD_DIR.glob(f"{upload_id}.*"))
    if not matches:
        return jsonify({"error": "Not found"}), 404
    path = matches[0]
    suffix = path.suffix.lower().lstrip(".")
    mime = MIME_MAP.get(suffix, "video/mp4")
    return send_file(path, mimetype=mime, conditional=True)


@app.route("/thumbnail/<upload_id>")
def thumbnail(upload_id: str):
    matches = list(UPLOAD_DIR.glob(f"{upload_id}.*"))
    if not matches:
        return jsonify({"error": "Not found"}), 404

    t = request.args.get("t", "0")
    safe_t = t.replace(".", "_").replace("-", "")
    thumb_path = OUTPUT_DIR / f"thumb_{upload_id}_{safe_t}.jpg"

    if not thumb_path.exists():
        cmd = [
            "ffmpeg", "-y", "-ss", str(t), "-i", str(matches[0]),
            "-frames:v", "1", "-q:v", "2", str(thumb_path),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not thumb_path.exists():
            return jsonify({"error": "Thumbnail extraction failed"}), 500

    return send_file(thumb_path, mimetype="image/jpeg")


@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json(force=True)
    upload_id = data.get("upload_id", "")
    params = data.get("params", {})
    original_name = data.get("original_name", "video")
    output_format = params.get("output_format", "mp4")

    matches = list(UPLOAD_DIR.glob(f"{upload_id}.*"))
    if not matches:
        return jsonify({"error": "Upload not found — please re-upload the file"}), 404

    input_path = matches[0]
    job_id = str(uuid.uuid4())
    stem = Path(original_name).stem
    output_path = OUTPUT_DIR / f"{job_id}_{stem}.{output_format}"

    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "output_path": str(output_path),
        "output_name": f"{stem}_converted.{output_format}",
        "output_format": output_format,
        "original_name": original_name,
        "error": "",
    }

    threading.Thread(
        target=run_job, args=(job_id, input_path, output_path, params), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/compress", methods=["POST"])
def compress():
    data = request.get_json(force=True)
    upload_id = data.get("upload_id", "")
    params = data.get("params", {})
    original_name = data.get("original_name", "video")
    output_format = params.get("output_format", "mp4")

    matches = list(UPLOAD_DIR.glob(f"{upload_id}.*"))
    if not matches:
        return jsonify({"error": "Upload not found — please re-upload the file"}), 404

    input_path = matches[0]
    input_size = input_path.stat().st_size
    job_id = str(uuid.uuid4())
    stem = Path(original_name).stem
    output_path = OUTPUT_DIR / f"{job_id}_{stem}_compressed.{output_format}"

    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "output_path": str(output_path),
        "output_name": f"{stem}_compressed.{output_format}",
        "output_format": output_format,
        "original_name": original_name,
        "input_size": input_size,
        "error": "",
    }

    threading.Thread(
        target=run_compress_job, args=(job_id, input_path, output_path, params), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id: str):
    def generate():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found', 'progress': 0, 'error': ''})}\n\n"
                return
            payload = {
                "status": job["status"],
                "progress": job.get("progress", 0),
                "error": job.get("error", ""),
                "output_size": job.get("output_size"),
                "input_size": job.get("input_size"),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if job["status"] in ("done", "error"):
                return
            time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/preview/<job_id>")
def preview(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Job not ready"}), 404
    path = Path(job["output_path"])
    if not path.exists():
        return jsonify({"error": "Output file missing"}), 404
    fmt = job.get("output_format", "mp4")
    mime = MIME_MAP.get(fmt, "video/mp4")
    return send_file(path, mimetype=mime, conditional=True)


@app.route("/download/<job_id>")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Job not ready"}), 404
    path = Path(job["output_path"])
    if not path.exists():
        return jsonify({"error": "Output file missing"}), 404
    return send_file(path, as_attachment=True, download_name=job["output_name"])


@app.route("/history")
def get_history():
    return jsonify(history[:50])


@app.route("/cleanup/<job_id>", methods=["DELETE"])
def cleanup(job_id: str):
    job = jobs.pop(job_id, None)
    if job:
        try:
            Path(job["output_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/image-upload", methods=["POST"])
def image_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    suffix = Path(f.filename).suffix.lower()
    if suffix not in IMG_EXTS:
        return jsonify({"error": "Unsupported image format"}), 400
    uid = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{uid}{suffix}"
    f.save(dest)
    try:
        with Image.open(dest) as im:
            w, h = im.size
            mode = im.mode
    except Exception as e:
        dest.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 400
    return jsonify({"upload_id": uid, "width": w, "height": h,
                    "mode": mode, "size": dest.stat().st_size,
                    "original_name": f.filename})


@app.route("/image-resize", methods=["POST"])
def image_resize():
    data = request.get_json(force=True)
    uid = data.get("upload_id", "")
    target_w = int(data.get("target_w", 0))
    target_h = int(data.get("target_h", 0))
    out_fmt = data.get("format", "jpeg").lower()
    original_name = data.get("original_name", "image")

    crop = data.get("crop") or None   # {x, y, w, h} in original image pixels

    if not target_w or not target_h:
        return jsonify({"error": "target_w and target_h are required"}), 400

    matches = list(UPLOAD_DIR.glob(f"{uid}.*"))
    if not matches:
        return jsonify({"error": "Upload not found"}), 404

    job_id = str(uuid.uuid4())
    ext = "jpg" if out_fmt == "jpeg" else out_fmt
    stem = Path(original_name).stem
    output_path = OUTPUT_DIR / f"{job_id}_{stem}_reframed.{ext}"

    jobs[job_id] = {
        "status": "queued", "progress": 0.0,
        "output_path": str(output_path),
        "output_name": f"{stem}_reframed.{ext}",
        "output_format": out_fmt,
        "original_name": original_name,
        "error": "",
        "job_type": "image",
    }

    threading.Thread(
        target=run_image_job,
        args=(job_id, matches[0], output_path, target_w, target_h, out_fmt, crop),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/image-progress/<job_id>")
def image_progress(job_id: str):
    def generate():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status':'not_found','progress':0,'error':''})}\n\n"
                return
            payload = {"status": job["status"], "progress": job.get("progress", 0),
                       "error": job.get("error", "")}
            yield f"data: {json.dumps(payload)}\n\n"
            if job["status"] in ("done", "error"):
                return
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/image-download/<job_id>")
def image_download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    path = Path(job["output_path"])
    if not path.exists():
        return jsonify({"error": "File missing"}), 404
    return send_file(path, as_attachment=True, download_name=job["output_name"])


@app.route("/image-zip", methods=["POST"])
def image_zip():
    job_ids = request.get_json(force=True).get("job_ids", [])
    buf = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen_names: dict = {}
        for jid in job_ids:
            job = jobs.get(jid)
            if not job or job.get("status") != "done":
                continue
            path = Path(job["output_path"])
            if not path.exists():
                continue
            name = job["output_name"]
            # Deduplicate filenames inside the zip
            if name in seen_names:
                seen_names[name] += 1
                stem, ext = name.rsplit(".", 1) if "." in name else (name, "")
                name = f"{stem}_{seen_names[name]}.{ext}" if ext else f"{stem}_{seen_names[name]}"
            else:
                seen_names[name] = 0
            zf.write(path, name)
            added += 1
    if added == 0:
        return jsonify({"error": "No completed jobs found"}), 404
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="reframed_images.zip")


@app.route("/image-preview/<job_id>")
def image_preview(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    path = Path(job["output_path"])
    fmt = job.get("output_format", "jpeg")
    mime = "image/png" if fmt == "png" else "image/webp" if fmt == "webp" else "image/jpeg"
    return send_file(path, mimetype=mime)


if __name__ == "__main__":
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe must be installed and on PATH")
    load_history()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
