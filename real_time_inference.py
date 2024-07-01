from ultralytics import RTDETR
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = RTDETR("/Users/amenijomaa/Downloads/best.pt")

# Object classes 
classNames = ["hogweed"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence:", confidence)

            # Class name
            cls = int(box.cls[0])
            if cls < len(classNames):
                print("Class name:", classNames[cls])
            else:
                print(f"Unknown class index: {cls}")

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            if cls < len(classNames):
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            else:
                cv2.putText(img, "Unknown", org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

