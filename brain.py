import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS 
app = Flask(__name__)
CORS(app)

# Load the YOLO model
def get_model():
    global model
    model = YOLO("model/best.pt")
    print(" * Model loaded!")

get_model()

def draw_segmented_box(image, box, label, confidence):
    """
    Draw bounding box and segmentation overlay on the image.
    """
    x_min, y_min, x_max, y_max = map(int, box)
    label_text = f"{label}: {confidence:.2f}"

    # Draw bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add label
    cv2.putText(
        image, label_text, (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

    # Create a segmentation overlay (for demonstration, this can be a binary mask or color overlay)
    overlay = image.copy()
    alpha = 0.4  # Transparency factor
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Decode the input image
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = np.array(Image.open(BytesIO(decoded)).convert("RGB"))

    # Perform prediction
    results = model.predict(source=image)
    predictions = []
    
    for result in results:
        for box in result.boxes:
            coordinates = box.xyxy[0].tolist()  # Bounding box coordinates
            label = result.names[int(box.cls)]  # Class label
            confidence = float(box.conf)  # Confidence score
            
            # Draw segmentation and bounding box on the image
            image = draw_segmented_box(image, coordinates, label, confidence)
            
            # Add prediction to response
            predictions.append({
                "box": coordinates,
                "class": label,
                "confidence": confidence
            })

    # Convert the modified image back to Base64
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    segmented_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return predictions and segmented image
    response = {
        "predictions": predictions,
        "segmented_image": segmented_image_base64
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
