import os
import io
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

# ── Tensorflow / Keras ──────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH", "best_emotion_model.h5")
IMG_SIZE    = (128, 128)
EMOTIONS    = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_META = {
    "Angry":    {"emoji": "😠", "level": "frustrated",  "color": "#ef4444"},
    "Disgust":  {"emoji": "🤢", "level": "frustrated",  "color": "#f97316"},
    "Fear":     {"emoji": "😨", "level": "struggling",  "color": "#a855f7"},
    "Happy":    {"emoji": "😊", "level": "engaged",     "color": "#22c55e"},
    "Neutral":  {"emoji": "😐", "level": "engaged",     "color": "#3b82f6"},
    "Sad":      {"emoji": "😢", "level": "struggling",  "color": "#6366f1"},
    "Surprise": {"emoji": "😲", "level": "confused",    "color": "#eab308"},
}

# ── Build model & load weights (avoids all config-version issues) ───────────
model = None

def build_and_load():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file not found: {MODEL_PATH}")
        return

    print(f"Building model architecture...")
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights=None)
    base.trainable = False
    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(len(EMOTIONS), activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)

    print(f"Loading weights from {MODEL_PATH}...")
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    # Warm up
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("✅ Model ready.")

build_and_load()

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="EmotiLearn API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict = {}

class FrameRequest(BaseModel):
    image: str
    session_id: str
    timestamp: float

def preprocess(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: FrameRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")
    try:
        x = preprocess(req.image)
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    preds      = model.predict(x, verbose=0)[0]
    idx        = int(np.argmax(preds))
    emotion    = EMOTIONS[idx]
    confidence = float(preds[idx])
    meta       = EMOTION_META[emotion]

    result = {
        "emotion": emotion, "confidence": round(confidence, 4),
        "emoji": meta["emoji"], "level": meta["level"], "color": meta["color"],
        "all": {EMOTIONS[i]: round(float(preds[i]), 4) for i in range(len(EMOTIONS))},
        "timestamp": req.timestamp,
    }

    sessions.setdefault(req.session_id, {"frames": [], "started_at": req.timestamp})
    sessions[req.session_id]["frames"].append({
        "timestamp": req.timestamp, "emotion": emotion,
        "confidence": confidence, "level": meta["level"],
    })
    return result

@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")
    data, frames = sessions[session_id], sessions[session_id]["frames"]
    if not frames:
        return {"session_id": session_id, "frames": [], "summary": {}}

    emotion_counts: dict = {}
    level_counts = {"engaged": 0, "confused": 0, "struggling": 0, "frustrated": 0}
    for f in frames:
        emotion_counts[f["emotion"]] = emotion_counts.get(f["emotion"], 0) + 1
        level_counts[f["level"]] += 1

    total    = len(frames)
    dominant = max(emotion_counts, key=emotion_counts.get)
    eng_pct  = round(level_counts["engaged"] / total * 100, 1)

    t0, buckets = data["started_at"], {}
    for f in frames:
        b = int((f["timestamp"] - t0) // 30)
        buckets.setdefault(b, []).append(f["emotion"])

    timeline = []
    for b in sorted(buckets):
        most = max(set(buckets[b]), key=buckets[b].count)
        timeline.append({"bucket": b, "time_label": f"{b*30}s",
                         "dominant_emotion": most,
                         "color": EMOTION_META[most]["color"],
                         "emoji": EMOTION_META[most]["emoji"]})
    return {
        "session_id": session_id, "total_frames": total,
        "started_at": data["started_at"], "frames": frames,
        "summary": {"dominant_emotion": dominant, "emotion_counts": emotion_counts,
                    "level_counts": level_counts, "engagement_pct": eng_pct},
        "timeline": timeline,
    }

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
