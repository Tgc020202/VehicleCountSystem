import cv2
import numpy as np

class VehicleDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
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
