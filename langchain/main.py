import os
import cv2
import torch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import time
import getpass
import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Set environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

# Initialize ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

CONFIDENCE_THRESHOLD = 0.45

# Load MiDaS model
try:
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
except Exception as e:
    raise RuntimeError(f"Failed to load MiDaS model: {e}")

# Check for GPU support
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load YOLO model
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    yolo_model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# Function to process frame data and get navigation decision from LLM
async def process_frame_with_llm(frame, depth_map, yolo_detections):
    """
    Processes frame data, sends it directly to the LLM, and returns navigation decisions.
    """
    context_message = (
        "You are receiving the following data: "
        "1. Image data from a webcam. "
        "2. Depth map indicating relative distances. "
        "3. Object detections from YOLO including labels and bounding boxes. "
        "Make a decision to navigate: turn left, turn right, tilt up, or tilt down, look still"
        "Only respond with json format: Pan:<direction(left(angle),right(angle),still)>, Tilt:<up(angle),down(angle),still>,General:<General comment of what your seeing>, Reason:<reason>, Depth:<in <close>, <distant>,"
        "Persons are important and its good to track them"
        "only make a movement decisicion when there is no person in the frame, try to have the person centered kinda centered is fine, they dont need to be perfectly in the middle but close to it"
    )

    # Create messages for the LLM
    messages = [
        AIMessage(content="System ready to process navigation instructions."),
        HumanMessage(content=context_message),
        HumanMessage(
            content=(
                f"Image shape: {frame}, "
                f"Depth map shape: {depth_map}, "
                f"YOLO detections: {yolo_detections}"
            )
        ),
    ]

    try:
        # Directly invoke the LLM
        response = llm.invoke(messages)
        decision = response.content.strip().lower()
        print(f"Navigation decision: {decision}")
        print("\n")
        return decision
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return "unknown"

# Main loop to process frames
async def main_loop():
    """
    Continuously processes frames from the webcam, sending image data to the LLM for decisions.
    """
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Preprocess frame for depth estimation
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                depth_map = prediction.squeeze().cpu().numpy()

            depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
            depth_map_colored = (depth_map_normalized * 255).astype("uint8")
            depth_map_colored_resized = cv2.resize(depth_map_colored, (frame.shape[1], frame.shape[0]))

            # Run YOLO for object detection
            results = yolo_model(img_rgb)
            yolo_detections = results.pandas().xyxy[0]

            # Send data to the LLM and get the navigation decision
            decision = await process_frame_with_llm(frame, depth_map, yolo_detections)

            # Draw YOLO detections on the frame
            for _, det in yolo_detections.iterrows():
                if det['confidence'] > CONFIDENCE_THRESHOLD:
                    x_min, y_min, x_max, y_max = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                    label = f"{det['name']} {det['confidence']:.2f}"
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame and depth map
            cv2.imshow("Original Frame with YOLO", frame)
            cv2.imshow("Depth Map", depth_map_colored_resized)

            # Print the decision on the frame
            cv2.putText(
                frame, f"Decision: {decision}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

            # Exit loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Delay 1 second for the next frame
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Stream stopped manually")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    asyncio.run(main_loop())
