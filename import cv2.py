"""
It demonstrates real-time image segmentation using K-Means clustering
as an initial exploration of separating foreground regions from a
webcam feed.

Current Implementation:
- Captures live video using OpenCV
- Applies K-Means clustering for color-based image segmentation
- Displays original and segmented frames side-by-side
"""
import cv2
import numpy as np
# 1. Segment catalogue items 2. run clip on catalogue items. 3. segment the user. 4. Run smpl on user 5. Recommend based on query 6. Integrate
def segment_frame_kmeans(frame, k=3):
    # Resize for speed
    frame_small = cv2.resize(frame, (320, 240))

    # Reshape & convert to float32
    pixel_values = frame_small.reshape((-1, 3)).astype(np.float32)

    # K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8
    centers = np.uint8(centers)

    # Replace pixels with cluster centers
    segmented = centers[labels.flatten()].reshape(frame_small.shape)

    return segmented

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    segmented = segment_frame_kmeans(frame, k=3)

    # Show original and segmented side-by-side
    combined = np.hstack((cv2.resize(frame, (320, 240)), segmented))
    cv2.imshow('Original (Left) | Segmented (Right)', combined)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
