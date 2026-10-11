import cv2
import numpy as np
cap = cv2.VideoCapture("people.mp4")

line_up, line_down, min_size = 600, 600, 70  #line

algo = cv2.createBackgroundSubtractorMOG2()

up_count, down_count = 0, 0
prev = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    mask = algo.apply(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0))


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cur = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w >= min_size and h >= min_size:
            cur.append((x, y))
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    for i in cur:
        for j in prev:
            if abs(i[0]-j[0]) < 30:
                if j[1] < line_up <= i[1] :
                    down_count += 1
                elif j[1] > line_down >= i[1]:
                    up_count += 1
    prev = cur

    cv2.imshow("video", frame)
    if cv2.waitKey(25) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()