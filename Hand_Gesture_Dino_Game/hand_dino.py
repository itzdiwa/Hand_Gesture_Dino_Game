import cv2
import numpy as np
import math
import pyautogui

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI) where we look for hand
    roi = frame[100:400, 100:400]                                  
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range (tune if needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
               
    # Blur to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 100)                   

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    try:
        # Find contour with max area (likely the hand)
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        
        # Convex hull & convexity defects                                   
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Calculate angle using cosine rule
                a = math.dist(end, start)
                b = math.dist(far, start)
                c = math.dist(end, far)
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / math.pi

                # If angle ≤ 90°, count as defect (gap between fingers)
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(roi, far, 4, [0, 0, 255], -1)

                # Draw lines around fingers
                cv2.line(roi, start, end, [0, 255, 0], 2)

        # If enough defects, simulate jump
        if count_defects >= 4:
            pyautogui.press('space')
            cv2.putText(frame, "JUMP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    except:
        pass

    # Show frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()                               
cv2.destroyAllWindows()                                                                                                                                                                                                                             