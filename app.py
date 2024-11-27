from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
from vehicle_detector import VehicleDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by default

count_json_path = 'count.json'

# Function to read the counts from count.json
def read_counts():
    if os.path.exists(count_json_path):
        with open(count_json_path, 'r') as f:
            return json.load(f)
    else:
        # If the file doesn't exist, return default values
        return {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Train": 0,
            "Truck": 0
        }

# Function to write updated counts to count.json
def write_counts(count_data):
    with open(count_json_path, 'w') as f:
        json.dump(count_data, f, indent=4)
	
# Initialize the vehicle detector
vehicle_detector = VehicleDetector()

# Ensure the images folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Define the route for image upload and processing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided!"}), 400

    image_file = request.files['image']

    # Validate image data
    try:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400


    # Perform vehicle detection
    try:
        vehicles = vehicle_detector.detect_vehicles(img)
    except Exception as e:
        return jsonify({"error": f"Vehicle detection failed: {str(e)}"}), 500

    # Count vehicles and save data
    vehicle_counts = {vehicle: 0 for vehicle in vehicle_detector.class_names.values()}
    for vehicle_type, _ in vehicles:
        vehicle_counts[vehicle_type] += 1
        
    # Ensure 'images' folder exists
    # os.makedirs('images', exist_ok=True)
    original_image_path = 'images/original_image.jpg'

    try:
        cv2.imwrite(original_image_path, img)
    except Exception as e:
        return jsonify({"error": f"Failed to save original image: {str(e)}"}), 500

    try:
        write_counts(vehicle_counts)
    except Exception as e:
        return jsonify({"error": f"Failed to update vehicle counts: {str(e)}"}), 500

    # Annotate image
    try:
        for vehicle_type, box in vehicles:
            x, y, w, h = box
            color = vehicle_detector.class_colors[vehicle_type]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        output_path = 'images/processed_image.jpg'
        cv2.imwrite(output_path, img)
    except Exception as e:
        return jsonify({"error": f"Failed to annotate or save processed image: {str(e)}"}), 500

    # Return JSON response
    return jsonify({
        'vehicle_counts': vehicle_counts,
        'original_image_url': url_for('serve_image', filename='original_image.jpg'),
        'output_image_url': url_for('serve_image', filename='processed_image.jpg')
    })

# Route to serve images for displaying
@app.route('/image/<filename>')
def serve_image(filename):
    return send_file(os.path.join('images', filename))

if __name__ == '__main__':
    app.run(debug=True)
