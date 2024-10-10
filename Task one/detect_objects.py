import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('../yolov8n.pt')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream from camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Filter detections for specific classes
    filtered_results = results[0].boxes  # YOLOv8 detections (bounding boxes)

    # Annotate the frame with filtered results
    annotated_frame = frame.copy()
    
    for box in filtered_results:
        # Extract the bounding box coordinates
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        conf = box.conf[0].cpu().numpy()  # confidence score
        cls = int(box.cls[0].cpu().numpy())  # class ID
        
        # Draw the bounding box
        cv2.rectangle(annotated_frame, 
                      (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), 
                      (0, 255, 0), 2)  # green box
        
        # Draw the label text with the confidence score
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.putText(annotated_frame, label, 
                    (int(xyxy[0]), int(xyxy[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with filtered detections
    cv2.imshow('YOLOv8 Detection (Filtered)', annotated_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
