import requests
import numpy as np
from PIL import Image
import io, base64

# Make a random test image
img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format='JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()

# Test /health
r = requests.get("http://127.0.0.1:8000/health")
print("Health:", r.json())

# Test /predict
r = requests.post("http://127.0.0.1:8000/predict", json={
    "image": b64,
    "session_id": "test_session",
    "timestamp": 1000.0
})
print("Prediction:", r.json())
