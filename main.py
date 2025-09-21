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
app = Flask(__name__, static_folder="frontend/dist")
CORS(app, origins=[
    "http://localhost:5173",   # Vite default
    "http://localhost:8080",   # React dev server
    "https://your-frontend.com"  # replace with your Netlify/Vercel domain
])

# ---------------------------
# Load YOLO model
# ---------------------------
# Make sure last.pt is in the backend folder
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
        # Read and convert image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Run YOLO prediction
        results = model.predict(img_np, imgsz=640, conf=0.25, device="cpu")

        # Prepare drawing
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
                bbox = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                cls_name = model.names[cls_id]

                # Update counts
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

                # Store detection details
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 2) for x in bbox]
                })

                # Draw bounding box + label
                draw.rectangle(bbox, outline="red", width=3)
                text = f"{cls_name} {conf:.2f}"

                # Use textbbox() instead of textsize()
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]

                draw.rectangle(
                    [bbox[0], bbox[1]-text_h, bbox[0]+text_w, bbox[1]],
                    fill="red"
                )
                draw.text((bbox[0], bbox[1]-text_h), text, fill="white", font=font)

        # Convert annotated image to base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Return JSON
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
