from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import base64
from flask_cors import CORS
import os

# ---------------------------
# Create Flask app
# ---------------------------

app = Flask(__name__, static_folder="../dist")

# ---------------------------
# Fix CORS properly
# ---------------------------
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",       # Vite default
    "http://localhost:8080",       # React dev server
    "https://space-detection-ai.netlify.app"  # Netlify frontend
]}})

# ---------------------------
# Load YOLO model
# ---------------------------
model = YOLO("last.pt")

# ---------------------------
# Health check endpoint
# ---------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "success": True,
        "status": "ok",
        "message": "Backend is running"
    }), 200

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        results = model.predict(img_np, imgsz=640, conf=0.25, device="cpu")

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        detection_counts = {}
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                bbox = box.xyxy.tolist()[0]
                cls_name = model.names[cls_id]

                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 2) for x in bbox]
                })

                draw.rectangle(bbox, outline="red", width=3)
                text = f"{cls_name} {conf:.2f}"

                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]

                draw.rectangle(
                    [bbox[0], bbox[1]-text_h, bbox[0]+text_w, bbox[1]],
                    fill="red"
                )
                draw.text((bbox[0], bbox[1]-text_h), text, fill="white", font=font)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "image": img_base64,
            "counts": detection_counts,
            "detections": detections
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------
# Serve React frontend
# ---------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

