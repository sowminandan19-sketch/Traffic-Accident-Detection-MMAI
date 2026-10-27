import cv2
import numpy as np
import os
from ultralytics import YOLO
import gradio as gr

# Load the YOLOv11 model (auto-downloads weights if present)
model = YOLO("yolov11n.pt") 

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to 640x640 while preserving aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for collision severity assessment."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def process_traffic_video(input_video_path):
    """
    Executes the multi-stage pipeline:
    1. Preprocessing (640x640 resizing & [0.0, 1.0] intensity normalization).
    2. Temporal key-frame extraction (10 FPS).
    3. YOLOv11 + ByteTrack multi-object tracking.
    4. Traffic density estimation & 3-tier severity classification.
    5. Localized vehicle impact quantification.
    """
    if not input_video_path or not os.path.exists(input_video_path):
        return None

    cap = cv2.VideoCapture(input_video_path)
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    if fps_native <= 0: 
        fps_native = 30.0
    
    # 10 FPS temporal sub-sampling step
    k = max(1, int(round(fps_native / 10.0)))

    # Saves the output video directly into your working directory (Downloads folder)
    output_path = "output_traffic_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (640, 640))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % k != 0:
            continue

        # Preprocessing: Spatial resizing & pixel intensity scaling
        resized_frame, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))
        normalized_frame = resized_frame.astype(np.float32) / 255.0  

        # YOLOv11 feature extraction & ByteTrack tracking
        results = model.track(resized_frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.3)
        
        vehicle_boxes = []
        track_ids = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            t_ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            vehicle_classes = [2, 3, 5, 7] # COCO classes: car, motorcycle, bus, truck
            for box, t_id, cls in zip(boxes, t_ids, classes):
                if int(cls) in vehicle_classes:
                    vehicle_boxes.append(box)
                    track_ids.append(int(t_id))

        # Traffic Density Estimation
        vehicle_count = len(vehicle_boxes)
        if vehicle_count <= 5:
            density_status = "Light"
            density_color = (0, 255, 0)
        elif vehicle_count <= 15:
            density_status = "Moderate"
            density_color = (0, 165, 255)
        else:
            density_status = "Heavy"
            density_color = (0, 0, 255)

        # 3-Tier Severity Assessment & Impact Quantification
        max_iou = 0.0
        entangled_vehicles = 0
        severity = "Normal"
        severity_color = (255, 255, 255)

        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                iou = calculate_iou(vehicle_boxes[i], vehicle_boxes[j])
                if iou > max_iou:
                    max_iou = iou

        if max_iou >= 0.25:
            severity = "Severe"
            severity_color = (0, 0, 255)
            entangled_vehicles = min(vehicle_count, 3)
        elif max_iou >= 0.10:
            severity = "Moderate"
            severity_color = (0, 140, 255)
            entangled_vehicles = min(vehicle_count, 2)
        elif max_iou >= 0.02:
            severity = "Minor"
            severity_color = (0, 255, 255)
            entangled_vehicles = 1

        # Visualization & HUD Annotation
        annotated_frame = resized_frame.copy()
        
        for box, t_id in zip(vehicle_boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"ID: {t_id}", (x1, max(15, y1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(annotated_frame, f"Density: {density_status} ({vehicle_count} Veh)", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2)
        cv2.putText(annotated_frame, f"Severity: {severity}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, severity_color, 2)
        cv2.putText(annotated_frame, f"Impact Count: {entangled_vehicles}", (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()
    return output_path

# Gradio Web Dashboard Interface
demo = gr.Interface(
    fn=process_traffic_video,
    inputs=gr.Video(label="Upload Surveillance / Dashcam Video"),
    outputs=gr.Video(label="Processed Output with Traffic Analytics & Accident Alerts"),
    title="Automated Traffic Density, Accident Detection & Severity Assessment System",
    description="Powered by YOLOv11 backbone, ByteTrack multi-object tracking engine, and Gradio web dashboard.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)