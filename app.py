import os
import io
import base64

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(APP_DIR, "best_emotion_model.h5"))
STATIC_DIR = os.path.join(APP_DIR, "static")
IMG_SIZE = (128, 128)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_META = {
    "Angry": {"emoji": "angry", "level": "frustrated", "color": "#ef4444"},
    "Disgust": {"emoji": "disgust", "level": "frustrated", "color": "#f97316"},
    "Fear": {"emoji": "fear", "level": "struggling", "color": "#a855f7"},
    "Happy": {"emoji": "happy", "level": "engaged", "color": "#22c55e"},
    "Neutral": {"emoji": "neutral", "level": "engaged", "color": "#3b82f6"},
    "Sad": {"emoji": "sad", "level": "struggling", "color": "#6366f1"},
    "Surprise": {"emoji": "surprise", "level": "confused", "color": "#eab308"},
}

model = None


def update_img_size_from_model() -> None:
    global IMG_SIZE

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 4 and input_shape[1] and input_shape[2]:
        IMG_SIZE = (int(input_shape[1]), int(input_shape[2]))


def warm_up_model() -> None:
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)


def build_fallback_model() -> Model:
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights=None)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(EMOTIONS), activation="softmax")(x)

    return Model(inputs=base.input, outputs=out)


def build_and_load() -> None:
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        update_img_size_from_model()
        warm_up_model()
        print("Model ready.")
        return
    except Exception as e:
        print(f"Could not load full model directly: {e}")
        print("Trying to load it as model weights instead...")

    try:
        model = build_fallback_model()
        print(f"Loading weights from {MODEL_PATH}...")
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        warm_up_model()
        print("Model ready.")
    except Exception as e:
        print(f"Model could not be loaded: {e}")
        model = None


build_and_load()

app = FastAPI(title="EmotiLearn API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}


@app.post("/predict")
def predict(req: FrameRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    try:
        x = preprocess(req.image)
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}") from e

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    emotion = EMOTIONS[idx]
    confidence = float(preds[idx])
    meta = EMOTION_META[emotion]

    result = {
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "emoji": meta["emoji"],
        "level": meta["level"],
        "color": meta["color"],
        "all": {EMOTIONS[i]: round(float(preds[i]), 4) for i in range(len(EMOTIONS))},
        "timestamp": req.timestamp,
    }

    sessions.setdefault(req.session_id, {"frames": [], "started_at": req.timestamp})
    sessions[req.session_id]["frames"].append(
        {
            "timestamp": req.timestamp,
            "emotion": emotion,
            "confidence": confidence,
            "level": meta["level"],
        }
    )
    return result


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    data = sessions[session_id]
    frames = data["frames"]
    if not frames:
        return {"session_id": session_id, "frames": [], "summary": {}}

    emotion_counts: dict = {}
    level_counts = {"engaged": 0, "confused": 0, "struggling": 0, "frustrated": 0}
    for frame in frames:
        emotion_counts[frame["emotion"]] = emotion_counts.get(frame["emotion"], 0) + 1
        level_counts[frame["level"]] += 1

    total = len(frames)
    dominant = max(emotion_counts, key=emotion_counts.get)
    engagement_pct = round(level_counts["engaged"] / total * 100, 1)

    t0 = data["started_at"]
    buckets = {}
    for frame in frames:
        bucket = int((frame["timestamp"] - t0) // 30)
        buckets.setdefault(bucket, []).append(frame["emotion"])

    timeline = []
    for bucket in sorted(buckets):
        most_common = max(set(buckets[bucket]), key=buckets[bucket].count)
        timeline.append(
            {
                "bucket": bucket,
                "time_label": f"{bucket * 30}s",
                "dominant_emotion": most_common,
                "color": EMOTION_META[most_common]["color"],
                "emoji": EMOTION_META[most_common]["emoji"],
            }
        )

    return {
        "session_id": session_id,
        "total_frames": total,
        "started_at": data["started_at"],
        "frames": frames,
        "summary": {
            "dominant_emotion": dominant,
            "emotion_counts": emotion_counts,
            "level_counts": level_counts,
            "engagement_pct": engagement_pct,
        },
        "timeline": timeline,
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
