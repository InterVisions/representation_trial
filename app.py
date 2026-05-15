import base64
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests as http
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file

load_dotenv()

app = Flask(__name__)

IMAGES_DIR = os.path.join(app.static_folder, "images")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "logs")
ADMIN_LOG  = os.path.join(LOGS_DIR, "admin.jsonl")
PUBLIC_LOG = os.path.join(LOGS_DIR, "public.jsonl")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ── Model presets ──────────────────────────────────────────────────────────────
MODEL_PRESETS = [
    # ── fal.ai hosted models ──────────────────────────────────────────
    {
        "id": "fal-nano-banana-2",
        "label": "fal.ai — Nano Banana 2",
        "api_format": "fal",
        "base_url": "https://fal.run/fal-ai/nano-banana-2",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-flux-schnell",
        "label": "fal.ai — Flux Schnell",
        "api_format": "fal",
        "base_url": "https://fal.run/fal-ai/flux/schnell",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-openai-gpt-image-2",
        "label": "fal.ai — OpenAI gpt-image-2",
        "api_format": "fal",
        "base_url": "https://fal.run/openai/gpt-image-2",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-nano-banana-pro",
        "label": "fal.ai — Nano Banana Pro",
        "api_format": "fal",
        "base_url": "https://fal.run/fal-ai/nano-banana-pro",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-flux-dev",
        "label": "fal.ai — Flux Dev",
        "api_format": "fal",
        "base_url": "https://fal.run/fal-ai/flux/dev",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-flux-pro-2",
        "label": "fal.ai — Flux Pro 2",
        "api_format": "fal",
        "base_url": "https://fal.run/fal-ai/flux-pro/v1.1",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-nano-banana",
        "label": "fal.ai — Nano Banana",
        "api_format": "fal",
        "base_url": "https://fal.run/fal-ai/nano-banana",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    {
        "id": "fal-grok-image",
        "label": "fal.ai — xAI Grok Image",
        "api_format": "fal",
        "base_url": "https://fal.run/xai/grok-imagine-image",
        "image_size": "square_hd",
        "api_key_env": "FAL_API_KEY",
        "auth_scheme": "Key",
    },
    # ── OpenAI direct ─────────────────────────────────────────────────
    {
        "id": "openai-gpt-image-2",
        "label": "OpenAI — gpt-image-2",
        "api_format": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-image-2",
        "size": "1024x1024",
        "api_key_env": "OPENAI_API_KEY",
        "auth_scheme": "Bearer",
    },
]
PRESETS_BY_ID = {p["id"]: p for p in MODEL_PRESETS}

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "status":      "idle",   # idle | generating | done
    "question_id": None,
    "total":       0,
    "images":      [],       # [{"id": str, "prompt": str, "url": str|None}]
    "votes":       [],
}
lock     = threading.Lock()
log_lock = threading.Lock()


# ── Logging helpers ────────────────────────────────────────────────────────────
def _append_log(path, entry):
    with log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Image generation ───────────────────────────────────────────────────────────
def _generate_one(prompt, img_id, preset, api_key):
    url = None
    try:
        headers = {
            "Authorization": f"{preset['auth_scheme']} {api_key}",
            "Content-Type": "application/json",
        }

        if preset["api_format"] == "fal":
            payload = {"prompt": prompt, "image_size": preset["image_size"], "num_images": 1}
            resp = http.post(preset["base_url"], headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "images" in data:
                url = data["images"][0]["url"]
            elif "data" in data:
                img = data["data"][0]
                if img.get("b64_json"):
                    path = os.path.join(IMAGES_DIR, f"{img_id}.png")
                    with open(path, "wb") as fh:
                        fh.write(base64.b64decode(img["b64_json"]))
                    url = f"/static/images/{img_id}.png"
                else:
                    url = img.get("url")
            else:
                print(f"[generation error] unexpected response shape: {list(data.keys())}")

        else:  # openai
            payload = {"model": preset["model"], "prompt": prompt, "n": 1, "size": preset["size"]}
            resp = http.post(
                f"{preset['base_url'].rstrip('/')}/images/generations",
                headers=headers, json=payload, timeout=120,
            )
            resp.raise_for_status()
            img_data = resp.json()["data"][0]
            if img_data.get("b64_json"):
                path = os.path.join(IMAGES_DIR, f"{img_id}.png")
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(img_data["b64_json"]))
                url = f"/static/images/{img_id}.png"
            else:
                url = img_data.get("url")

    except Exception as exc:
        print(f"[generation error] {prompt!r}: {exc}")

    return {"id": img_id, "prompt": prompt, "url": url}


def _generate(prompts, preset_id, question_id):
    preset    = PRESETS_BY_ID.get(preset_id, MODEL_PRESETS[0])
    api_key   = os.environ.get(preset["api_key_env"], "")
    started   = datetime.now().isoformat()

    # Pre-assign image IDs in prompt order before parallel dispatch
    tasks = [(prompt, f"{question_id}_{i+1:02d}") for i, prompt in enumerate(prompts)]

    results = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {executor.submit(_generate_one, p, img_id, preset, api_key): img_id
                   for p, img_id in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with lock:
                state["images"].append(result)

    with lock:
        state["status"] = "done"

    # Sort results by image ID (prompt order) for the log
    results.sort(key=lambda r: r["id"])
    _append_log(ADMIN_LOG, {
        "datetime":    started,
        "question_id": question_id,
        "model":       preset["label"],
        "prompts":     prompts,
        "results": [
            {"image": r["id"], "prompt": r["prompt"], "status": "OK" if r["url"] else "ERR"}
            for r in results
        ],
    })


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/api/presets")
def api_presets():
    return jsonify([{"id": p["id"], "label": p["label"]} for p in MODEL_PRESETS])


@app.route("/api/state")
def api_state():
    with lock:
        return jsonify({
            "status":      state["status"],
            "question_id": state["question_id"],
            "total":       state["total"],
            "images":      state["images"],
        })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data      = request.json or {}
    prompts   = [p.strip() for p in data.get("prompts", []) if p.strip()]
    preset_id = data.get("preset", MODEL_PRESETS[0]["id"])
    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400
    with lock:
        if state["status"] == "generating":
            return jsonify({"error": "Already generating"}), 409
        question_id = datetime.now().strftime("Q%Y%m%d_%H%M%S")
        state["status"]      = "generating"
        state["question_id"] = question_id
        state["total"]       = len(prompts)
        state["images"]      = []
        state["votes"]       = []
    threading.Thread(
        target=_generate, args=(prompts, preset_id, question_id), daemon=True
    ).start()
    return jsonify({"ok": True, "question_id": question_id})


@app.route("/api/vote", methods=["POST"])
def api_vote():
    data              = request.json or {}
    scores            = data.get("scores", {})
    client_session_id = data.get("client_session_id", "unknown")
    with lock:
        question_id = state["question_id"]
        state["votes"].append({"client_session_id": client_session_id, "scores": scores})
    _append_log(PUBLIC_LOG, {
        "datetime":         datetime.now().isoformat(),
        "question_id":      question_id,
        "client_session_id": client_session_id,
        "votes":            scores,
    })
    return jsonify({"ok": True})


@app.route("/admin/logs/<log>")
def download_log(log):
    paths = {"admin": ADMIN_LOG, "public": PUBLIC_LOG}
    if log not in paths:
        abort(404)
    path = paths[log]
    if not os.path.exists(path):
        abort(404, description="No log entries yet.")
    return send_file(path, as_attachment=True, download_name=f"{log}.jsonl")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with lock:
        old_images = state["images"][:]
        state["status"]      = "idle"
        state["question_id"] = None
        state["total"]       = 0
        state["images"]      = []
        state["votes"]       = []
    for img in old_images:
        if img.get("url", "").startswith("/static/images/"):
            try:
                os.remove(os.path.join(app.static_folder, img["url"].removeprefix("/static/")))
            except OSError:
                pass
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
