# EmotiLearn — Real-Time Student Engagement Monitor

Detect student emotions via webcam during online sessions using your trained MobileNetV2 model.

---

## Project Structure

```
emotion_app/
├── app.py               ← FastAPI backend (model inference + session storage)
├── requirements.txt
├── best_emotion_model.h5   ← ⬅ Place your trained model here
└── static/
    └── index.html       ← Full frontend (Student + Teacher views)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model
Copy your trained `best_emotion_model.h5` into the `emotion_app/` folder.

Or set a custom path:
```bash
export MODEL_PATH=/path/to/your/model.h5
```

### 3. Run the server
```bash
cd emotion_app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the app
Visit **http://localhost:8000** in your browser.

---

## How to Use

### Student
1. Open the app → **Student View**
2. Click **▶ Start Session** — allow camera access
3. The model runs every 2 seconds, detecting your emotion live
4. Click **■ Stop** when done
5. Note your **Session ID** (shown in the sidebar) — share it with your teacher
6. Click **View Report** to see your own session

### Teacher
1. Open the app → **Teacher Dashboard**
2. Enter the student's **Session ID**
3. Click **Load** to view:
   - Engagement %, dominant emotion, duration
   - Emotion distribution donut chart
   - Engagement level bar chart
   - 30-second emotion timeline heatmap
   - Full frame-by-frame log

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET    | `/health` | Check server + model status |
| POST   | `/predict` | Send a base64 frame, get emotion back |
| GET    | `/session/{id}` | Get full session report |
| DELETE | `/session/{id}` | Clear a session |

---

## Emotion → Engagement Mapping

| Emotion  | Engagement Level |
|----------|-----------------|
| Happy    | Engaged ✅ |
| Neutral  | Engaged ✅ |
| Surprise | Confused 🤔 |
| Fear     | Struggling ⚠️ |
| Sad      | Struggling ⚠️ |
| Angry    | Frustrated ❌ |
| Disgust  | Frustrated ❌ |

---

## Notes
- Sessions are stored **in-memory** — they reset when the server restarts. Add a database (SQLite/PostgreSQL) for persistence.
- The model expects **128×128 RGB** images normalized to [0, 1].
- Inference runs every **2 seconds** — adjust `setInterval(captureAndPredict, 2000)` in `index.html` to change the rate.
