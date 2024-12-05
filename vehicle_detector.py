import cv2
import numpy as np
import os
import gdown

class VehicleDetector:

    def __init__(self):
        # Paths and Google Drive file IDs
        self.model_dir = "dnn_model"
        self.cfg_file = os.path.join(self.model_dir, "yolov4.cfg")
        self.weights_file = os.path.join(self.model_dir, "yolov4.weights")

        # Google Drive file IDs
        self.cfg_file_id = "1kDsdBkrMx0vC8Zb7qfHNDyn1omYm5CvP"
        self.weights_file_id = "1v8G6m2P7V9pCg2hnzPeoY_wBRGjqL3cn"

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Download files if missing
        self.download_file(self.cfg_file_id, self.cfg_file)
        self.download_file(self.weights_file_id, self.weights_file)

        # Load Network
        net = cv2.dnn.readNet(self.weights_file, self.cfg_file)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)

        # Class names and color mappings for vehicles
        self.class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 6: "Train", 7: "Truck"}
        self.class_colors = {  # Unique colors for each type
            "Car": (0, 255, 0),         # Green
            "Motorcycle": (255, 0, 0),  # Blue
            "Bus": (0, 0, 255),         # Red
            "Train": (255, 0, 255),     # Magenta
            "Truck": (255, 255, 0)      # Cyan
        }
        self.classes_allowed = list(self.class_names.keys())  # Filter for these classes only

    def download_file(self, file_id, output_path):
        """Downloads a file from Google Drive if it doesn't already exist."""
        if not os.path.exists(output_path):
            print(f"Downloading {output_path} from Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            print(f"Downloaded {output_path}.")
        else:
            print(f"{output_path} already exists.")

    def detect_vehicles(self, img):
        # Detect Objects
        vehicles = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:  # Skip low-confidence detections
                continue
            if class_id in self.classes_allowed:
                vehicle_type = self.class_names[class_id]
                vehicles.append((vehicle_type, box))  # Append type and bounding box
        return vehicles
