import cv2
import numpy as np
import pyautogui
import time
import math

# Delay to let you switch to the Dino game window
print("Switch to the Dino game window... Starting in 5 seconds")
time.sleep(5)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Define ROI
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get max contour
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

        # Convexity defects to count fingers
        hull = cv2.convexHull(cnt, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                count_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    a = math.dist(start, end)
                    b = math.dist(start, far)
                    c = math.dist(end, far)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-5)) * (180 / math.pi)

                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(roi, far, 5, (255, 0, 0), -1)

                # 0 defects = 1 finger up
                if count_defects == 0:
                    print("Jump!")
                    pyautogui.press('space')
                    time.sleep(0.5)

    cv2.imshow("Dino Gesture Control", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to quit
        break

cap.release()  
cv2.destroyAllWindows()
