#Write a program to implement a Forward Collision Warning System that alerts the driver when the distance to the vehicle ahead becomes unsafe. 
from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("test.mp4")

FOCAL_LENGTH = 700
KNOWN_WIDTH = 1.8
previous_distance = None
previous_time = None
smoothed_ttc = None

def estimate_distance(box_width):
    box_width = max(box_width, 1)
    return (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame, verbose=False)[0]

    best_distance = None
    best_box = None

    for box in results.boxes:
        name = model.names[int(box.cls[0])]
        if name not in ["car", "truck", "bus", "motorcycle"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        distance = estimate_distance(width)

        # Choose the closest vehicle
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_box = (x1, y1, x2, y2)

    ttc = float("inf")

    if best_distance is not None:
        if previous_distance is not None and previous_time is not None:
            dt = max(current_time - previous_time, 1e-3)

            # Positive relative speed means the gap is closing
            relative_speed = (previous_distance - best_distance) / dt

            if relative_speed > 0.1:
                ttc = best_distance / relative_speed

                # Simple temporal smoothing
                if smoothed_ttc is None:
                    smoothed_ttc = ttc
                else:
                    smoothed_ttc = 0.7 * smoothed_ttc + 0.3 * ttc

        previous_distance = best_distance
        previous_time = current_time

        if best_box:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(frame, f"Distance: {best_distance:.1f} m",
                        (x1, max(y1-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    displayed_ttc = smoothed_ttc if smoothed_ttc is not None else float("inf")

    if displayed_ttc < 2:
        warning = "CRITICAL: COLLISION WARNING!"
    elif displayed_ttc < 4:
        warning = "CAUTION: REDUCE SPEED"
    else:
        warning = "SAFE"

    cv2.putText(frame, warning, (25,45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,0,255) if "CRITICAL" in warning else
                (0,165,255) if "CAUTION" in warning else (0,255,0), 3)

    cv2.putText(frame, f"TTC: {displayed_ttc:.2f} s",
                (25,85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)

    cv2.imshow("Forward Collision Warning", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




