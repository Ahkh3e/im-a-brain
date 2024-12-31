import cv2
import torch
import matplotlib.pyplot as plt
import pandas

CONFIDENCE_THRESHOLD = 0.45
# Load MiDaS model
model_type = "DPT_Large"  # Change model type as needed
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Check for GPU support
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use YOLOv5 small model
yolo_model.to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Loop to process webcam frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transformation
        input_batch = transform(img_rgb).to(device)

        # Run depth estimation
        with torch.no_grad():
            prediction = midas(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        # Normalize the depth map for visualization
        depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map_colored = (depth_map_normalized * 255).astype("uint8")  # Convert to 8-bit for drawing
        depth_map_colored = cv2.applyColorMap(depth_map_colored, cv2.COLORMAP_JET)

        # Resize depth map to match original frame dimensions
        depth_map_colored_resized = cv2.resize(depth_map_colored, (frame.shape[1], frame.shape[0]))

        # Run YOLO object detection
        results = yolo_model(img_rgb)
        detections = results.pandas().xyxy[0]  # Get bounding boxes

        # Draw YOLO detections on the original frame and resized depth map
        for _, det in detections.iterrows():
            if det['confidence'] > CONFIDENCE_THRESHOLD:
                # Extract bounding box and label
                x_min, y_min, x_max, y_max = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                label = f"{det['name']} {det['confidence']:.2f}"

                # Draw on the original frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw on the resized depth map
                cv2.rectangle(depth_map_colored_resized, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(depth_map_colored_resized, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display results
        cv2.imshow("Original Frame with YOLO", frame)
        cv2.imshow("Depth Map with YOLO", depth_map_colored_resized)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream stopped manually")

finally:
    cap.release()
    cv2.destroyAllWindows()
