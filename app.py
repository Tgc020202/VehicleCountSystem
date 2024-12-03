from flask import Flask, request, jsonify, send_file, url_for, render_template, redirect, session
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
from vehicle_detector import VehicleDetector

app = Flask(__name__)
app.secret_key = 'secret_key'  # Required for session handling
CORS(app)

count_json_path = 'count.json'

# Function to read the counts from count.json
def read_counts():
    if os.path.exists(count_json_path):
        with open(count_json_path, 'r') as f:
            return json.load(f)
    return {vehicle: 0 for vehicle in ["Car", "Motorcycle", "Bus", "Train", "Truck"]}

# Function to write updated counts to count.json
def write_counts(count_data):
    with open(count_json_path, 'w') as f:
        json.dump(count_data, f, indent=4)

# Initialize the vehicle detector
vehicle_detector = VehicleDetector()

# Ensure the new images directory exists
image_dir = 'images/ProcessImages'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for uploading and processing images
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided!"}), 400

    image_file = request.files['image']

    try:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    # Reset counts
    vehicle_counts = {vehicle: 0 for vehicle in vehicle_detector.class_names.values()}
    write_counts(vehicle_counts)

    # Perform vehicle detection
    try:
        vehicles = vehicle_detector.detect_vehicles(img)
    except Exception as e:
        return jsonify({"error": f"Vehicle detection failed: {str(e)}"}), 500

    # Count detected vehicles
    for vehicle_type, _ in vehicles:
        vehicle_counts[vehicle_type] += 1

    # Save the counts
    write_counts(vehicle_counts)

    # Save original image
    original_image_path = os.path.join(image_dir, 'original_image.jpg')
    cv2.imwrite(original_image_path, img)

    # Annotate image
    for vehicle_type, box in vehicles:
        x, y, w, h = box
        color = vehicle_detector.class_colors[vehicle_type]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    processed_image_path = os.path.join(image_dir, 'processed_image.jpg')
    cv2.imwrite(processed_image_path, img)

    # Store data in session
    session['vehicle_counts'] = vehicle_counts
    session['original_image_url'] = url_for('serve_image', filename='original_image.jpg')
    session['output_image_url'] = url_for('serve_image', filename='processed_image.jpg')

    # Redirect to the output page
    return redirect(url_for('output'))

# Route for the output page
@app.route('/output')
def output():
    # Retrieve data from the session
    vehicle_counts = session.get('vehicle_counts', {})
    original_image_url = session.get('original_image_url', '')
    output_image_url = session.get('output_image_url', '')

    # Render the output page
    return render_template(
        'output.html',
        vehicle_counts=vehicle_counts,
        original_image_url=original_image_url,
        output_image_url=output_image_url
    )

# Route to serve images
@app.route('/image/<filename>')
def serve_image(filename):
    return send_file(os.path.join(image_dir, filename))

if __name__ == '__main__':
    app.run(debug=True)
