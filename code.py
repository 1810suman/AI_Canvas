import cv2
import mediapipe as mp
import os
import time
import numpy as np

# Initialize webcam with retry mechanism
cap = None
for i in range(3):  # Try different camera indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera initialized with index {i}")
        break
else:
    print("Error: Unable to access webcam.")
    exit()

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(10, 150)  # Brightness

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

pasttime = 0

# Define color options (manually defined)
color_palette = [
    ((0, 0, 0), "Eraser"),  # Black (Eraser)
    ((255, 0, 0), "Blue"),
    ((0, 255, 0), "Green"),
    ((0, 0, 255), "Red"),
    ((255, 255, 0), "Turquoise"),
]

# Create a blank header for color palette
header = np.zeros((100, 640, 3), np.uint8)

# Display distinct colors in separate boxes
for i, (color, name) in enumerate(color_palette):
    x_start = i * 128  # Space out colors evenly
    cv2.rectangle(header, (x_start, 0), (x_start + 128, 100), color, cv2.FILLED)
    cv2.putText(header, name, (x_start + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Default drawing color (red) 
col = (0, 0, 255)
xp, yp = 0, 0

# Create a blank canvas
canvas = np.zeros((480, 640, 3), np.uint8)

# Shape drawing mode
shape_mode = None  # None, "circle", "rectangle", "line"
start_point = None

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Unable to capture video. Check your webcam.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1], landmarks[8][2]
        x2, y2 = landmarks[12][1], landmarks[12][2]

        if landmarks[8][2] < landmarks[6][2] and landmarks[12][2] < landmarks[10][2]:
            xp, yp = 0, 0
            if y1 < 100:
                for i, (color, _) in enumerate(color_palette):
                    if (i * 128) < x1 < ((i + 1) * 128):
                        col = color
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

        elif landmarks[8][2] < landmarks[6][2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if shape_mode is None:
                thickness = 25 if col != (0, 0, 0) else 100
                cv2.line(frame, (xp, yp), (x1, y1), col, thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), col, thickness)
                xp, yp = x1, y1
            
            elif shape_mode == "circle":
                cv2.circle(canvas, (x1, y1), 30, col, -1)
            
            elif shape_mode == "rectangle":
                if start_point is None:
                    start_point = (x1, y1)
                else:
                    cv2.rectangle(canvas, start_point, (x1, y1), col, -1)
                    start_point = None
            
            elif shape_mode == "line":
                if start_point is None:
                    start_point = (x1, y1)
                else:
                    cv2.line(canvas, start_point, (x1, y1), col, 5)
                    start_point = None
    
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)
    frame[0:100, 0:640] = header

    ctime = time.time()
    fps = 1 / (ctime - pasttime) if pasttime else 0
    pasttime = ctime
    cv2.putText(frame, f"FPS: {int(fps)}", (490, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow("Camera", frame)
    cv2.imshow("Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        shape_mode = "circle"
    elif key == ord("r"):
        shape_mode = "rectangle"
    elif key == ord("l"):
        shape_mode = "line"
    elif key == ord("d"):
        shape_mode = None  # Back to freehand drawing

cap.release()
cv2.destroyAllWindows()
