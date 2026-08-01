import os
import cv2
import numpy as np

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load trained model
model = load_model("flood_model.keras")

IMG_SIZE = 128


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = img.copy()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0]

    # Binary mask
    mask = (pred > 0.5).astype(np.uint8) * 255

    mask = mask.squeeze()

    mask_path = os.path.join(OUTPUT_FOLDER, "mask.png")
    cv2.imwrite(mask_path, mask)

    # Resize mask to original image size
    mask_big = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Create blue overlay
    overlay = original.copy()

    overlay[mask_big == 255] = [0, 0, 255]

    overlay = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

    overlay_path = os.path.join(OUTPUT_FOLDER, "overlay.png")

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    flood_percent = np.sum(mask == 255) / mask.size * 100

    if flood_percent < 10:
        risk = "LOW"
        damage = "Minor Flooding"

    elif flood_percent < 30:
        risk = "MEDIUM"
        damage = "Moderate Flooding"

    else:
        risk = "HIGH"
        damage = "Severe Flooding"

    return render_template(
        "index.html",
        original="uploads/" + file.filename,
        mask="output/mask.png",
        overlay="output/overlay.png",
        flood=round(flood_percent, 2),
        risk=risk,
        damage=damage
    )


if __name__ == "__main__":
    app.run(debug=True)