import cv2
import numpy as np

# Load the template image in grayscale
template = cv2.imread('images', 1)
if template is None:
    print("Error: Template image not found.")
    exit()

template_w, template_h = template.shape[::-1]

# Open the video file
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error: Video file not found or cannot be opened.")
    exit()

# Set a threshold for template matching
threshold = 0.8  # Adjust between 0 and 1 for matching sensitivity

total_matches = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)

    # Get locations of matches above threshold
    locations = np.where(result >= threshold)

    detected_this_frame = 0
    # Draw rectangles around matched regions
    for pt in zip(*locations[::-1]):  # Switch columns and rows
        detected_this_frame += 1
        bottom_right = (pt[0] + template_w, pt[1] + template_h)
        cv2.rectangle(frame, pt, bottom_right, (0, 255, 0), 2)

    total_matches += detected_this_frame

    # Display number of matches in the current frame
    cv2.putText(frame, f'Matches this frame: {detected_this_frame}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video frame with detections
    cv2.imshow('Template Matching', frame)

    # Quit if 'p' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'Total matches detected in video: {total_matches}')
