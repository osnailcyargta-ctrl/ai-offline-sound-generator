#!/usr/bin/env python3
"""Steam Workshop Downloader — a small Flask front end around SteamCMD.

Every path is configurable through the environment so the exact same file runs
inside the container and on a laptop against a stub SteamCMD.
"""

import os
import re
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent

STEAMCMD_DIR = Path(os.environ.get("STEAMCMD_DIR", "/steamcmd"))
STEAMCMD_BIN = os.environ.get("STEAMCMD_BIN", str(STEAMCMD_DIR / "steamcmd.sh"))
WORKSHOP_ROOT = Path(
    os.environ.get(
        "WORKSHOP_ROOT", str(STEAMCMD_DIR / "steamapps" / "workshop" / "content")
    )
)

PORT = int(os.environ.get("PORT", "9976"))
# A big item on a slow line can legitimately take a while; past this we assume
# SteamCMD is wedged and kill it rather than leak the job forever.
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "3600"))
MAX_LOG_LINES = int(os.environ.get("MAX_LOG_LINES", "3000"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "50"))

app = Flask(__name__, static_folder=None)

# SteamCMD keeps mutable state in its own install directory and does not cope
# with two copies running against it at once, so downloads are serialised.
STEAMCMD_LOCK = threading.Lock()
JOBS: "OrderedDict[str, Job]" = OrderedDict()
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# input parsing
# --------------------------------------------------------------------------- #

_LONG_DIGITS = re.compile(r"\d{4,}")
_URL_ID = re.compile(r"[?&]id=(\d+)")


def parse_app_id(raw):
    """App IDs are always plain numbers."""
    raw = (raw or "").strip()
    return raw if raw.isdigit() else None


def parse_item_id(raw):
    """Accept a bare ID or any Workshop URL the user pasted."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return raw
    m = _URL_ID.search(raw)
    if m:
        return m.group(1)
    # Last resort: the first long run of digits, which covers URL shapes that
    # put the ID in the path instead of the query string.
    m = _LONG_DIGITS.search(raw)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# filesystem helpers
# --------------------------------------------------------------------------- #


def item_dir(appid, itemid):
    return WORKSHOP_ROOT / appid / itemid


def list_files(directory):
    """Every file under the item directory, as posix-relative paths."""
    out = []
    if not directory.is_dir():
        return out
    for root, _dirs, files in os.walk(directory):
        for name in files:
            full = Path(root) / name
            try:
                size = full.stat().st_size
            except OSError:
                continue
            out.append(
                {"name": full.relative_to(directory).as_posix(), "size": size}
            )
    out.sort(key=lambda f: f["name"])
    return out


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #


class Job:
    def __init__(self, appid, itemid):
        self.id = uuid.uuid4().hex[:12]
        self.appid = appid
        self.itemid = itemid
        self.status = "queued"  # queued | running | done | failed
        self.message = "Waiting for a free SteamCMD slot..."
        self.returncode = None
        self.created = time.time()
        self.finished = None
        self._lock = threading.Lock()
        # Log is trimmed from the front; _offset is how many lines were dropped
        # so the UI can keep asking for an absolute line index.
        self._lines = []
        self._offset = 0

    def append(self, line):
        with self._lock:
            self._lines.append(line)
            excess = len(self._lines) - MAX_LOG_LINES
            if excess > 0:
                del self._lines[:excess]
                self._offset += excess

    def set_status(self, status, message=None):
        with self._lock:
            self.status = status
            if message is not None:
                self.message = message
            if status in ("done", "failed"):
                self.finished = time.time()

    def snapshot(self, from_index=0):
        with self._lock:
            start = max(from_index - self._offset, 0)
            lines = self._lines[start:]
            next_index = self._offset + len(self._lines)
            return {
                "job_id": self.id,
                "appid": self.appid,
                "itemid": self.itemid,
                "status": self.status,
                "message": self.message,
                "returncode": self.returncode,
                "log": lines,
                "next_index": next_index,
                "elapsed": round((self.finished or time.time()) - self.created, 1),
            }


def register(job):
    with JOBS_LOCK:
        JOBS[job.id] = job
        while len(JOBS) > MAX_JOBS:
            JOBS.popitem(last=False)


def get_job(job_id):
    with JOBS_LOCK:
        return JOBS.get(job_id)


def diagnose(job, log_text):
    """Explain an empty download directory in the user's terms."""
    lowered = log_text.lower()
    if "invalid platform" in lowered:
        return (
            "SteamCMD reported 'Invalid platform'. That usually means this App ID "
            "has no Workshop content for Linux, or the App ID does not match the item."
        )
    if "access denied" in lowered:
        return (
            "SteamCMD reported 'Access Denied'. The item is most likely private, "
            "friends-only, or restricted by its creator — anonymous login cannot reach it."
        )
    if "file not found" in lowered or "failure" in lowered:
        return (
            "SteamCMD could not fetch the item. The two usual causes are a Workshop "
            "item that Steam has removed/taken down, or an App ID that does not match "
            "the item. Removed items generally stay unavailable even with a real account."
        )
    if "timeout" in lowered:
        return "SteamCMD timed out talking to Steam. Try again in a moment."
    return (
        "SteamCMD finished but downloaded nothing. Check that the App ID belongs to "
        "this item; if it does, the item is probably removed, private, or needs a real "
        "Steam account instead of anonymous login."
    )


def run_job(job):
    # Queue behind any download already in flight.
    with STEAMCMD_LOCK:
        job.set_status("running", "Starting SteamCMD...")
        target = item_dir(job.appid, job.itemid)

        cmd = [
            STEAMCMD_BIN,
            "+login",
            "anonymous",
            "+workshop_download_item",
            job.appid,
            job.itemid,
            "+quit",
        ]
        job.append("$ " + " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(STEAMCMD_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            job.append(f"!! could not launch SteamCMD: {exc}")
            job.set_status("failed", f"Could not launch SteamCMD at {STEAMCMD_BIN}: {exc}")
            return

        # Watchdog: a silently wedged SteamCMD would otherwise block the lock
        # for every later job too.
        timed_out = threading.Event()

        def _kill():
            timed_out.set()
            try:
                proc.kill()
            except Exception:
                pass

        killer = threading.Timer(JOB_TIMEOUT, _kill)
        killer.daemon = True
        killer.start()

        try:
            # text=True gives universal-newline handling, so SteamCMD's \r
            # progress updates arrive as ordinary lines.
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line:
                    job.append(line)
            returncode = proc.wait()
        finally:
            killer.cancel()
            try:
                proc.stdout.close()
            except Exception:
                pass

        job.returncode = returncode

        if timed_out.is_set():
            job.set_status("failed", f"SteamCMD was killed after {JOB_TIMEOUT}s without finishing.")
            return

        files = list_files(target)
        if files:
            total = human_size(sum(f["size"] for f in files))
            job.set_status("done", f"Downloaded {len(files)} file(s), {total}.")
        else:
            log_text = "\n".join(job.snapshot()["log"])
            job.set_status("failed", diagnose(job, log_text))


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/api/health")
def api_health():
    return jsonify(
        ok=True,
        steamcmd=STEAMCMD_BIN,
        steamcmd_present=os.path.isfile(STEAMCMD_BIN),
        workshop_root=str(WORKSHOP_ROOT),
        busy=STEAMCMD_LOCK.locked(),
    )


@app.post("/api/download")
def api_download():
    data = request.get_json(silent=True) or request.form or {}
    appid = parse_app_id(data.get("appid"))
    itemid = parse_item_id(data.get("itemid"))

    if not appid:
        return jsonify(error="App ID must be a number, e.g. 431960."), 400
    if not itemid:
        return jsonify(
            error="Could not read an Item ID. Paste the numeric ID or the full "
                  "Workshop URL (…/sharedfiles/filedetails/?id=123456789)."
        ), 400

    job = Job(appid, itemid)
    register(job)
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return jsonify(job_id=job.id, appid=appid, itemid=itemid), 202


@app.get("/api/status/<job_id>")
def api_status(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify(error="Unknown job id."), 404
    try:
        from_index = int(request.args.get("from", 0))
    except ValueError:
        from_index = 0
    return jsonify(job.snapshot(from_index))


@app.get("/api/files/<appid>/<itemid>")
def api_files(appid, itemid):
    if not (appid.isdigit() and itemid.isdigit()):
        return jsonify(error="appid and itemid must be numeric."), 400
    directory = item_dir(appid, itemid)
    files = list_files(directory)
    return jsonify(
        appid=appid,
        itemid=itemid,
        path=str(directory),
        count=len(files),
        files=[
            {
                "name": f["name"],
                "size": f["size"],
                "size_human": human_size(f["size"]),
                "url": f"/api/get/{appid}/{itemid}/{f['name']}",
            }
            for f in files
        ],
    )


@app.get("/api/get/<appid>/<itemid>/<path:filename>")
def api_get(appid, itemid, filename):
    if not (appid.isdigit() and itemid.isdigit()):
        abort(400)
    directory = item_dir(appid, itemid)
    if not directory.is_dir():
        abort(404)
    # send_from_directory refuses to escape the directory, which is what keeps
    # a crafted filename from reaching the rest of the filesystem.
    return send_from_directory(directory, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
