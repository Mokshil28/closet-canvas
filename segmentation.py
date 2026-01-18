import cv2
import torch
import numpy as np

image_path ="fashionpedia_subset/0a2bbd07cd557932cd3c6a27fd54ad5b.jpg"
def segment_frame_kmeans_torch(frame, k=3, max_iter=10):
    # Resize for speed
    frame_small = cv2.resize(frame, (320, 240))

    # Convert to torch tensor (N, 3)
    pixel_values = torch.from_numpy(frame_small.reshape((-1, 3))).float().to(device)

    # Initialize cluster centers randomly
    indices = torch.randperm(pixel_values.size(0))[:k]
    centers = pixel_values[indices]

    for i in range(max_iter):
        # Compute distances (N x K)
        distances = torch.cdist(pixel_values, centers)

        # Assign clusters
        labels = torch.argmin(distances, dim=1)

        # Update centers
        new_centers = torch.stack([
            pixel_values[labels == i].mean(dim=0) if (labels == i).any() else centers[i]
            for i in range(k)
        ])

        # Check for convergence (optional)
        if torch.allclose(centers, new_centers, atol=1e-2):
            break
        centers = new_centers

    # Map pixels to cluster centers
    segmented = centers[labels].cpu().numpy().astype(np.uint8)
    segmented = segmented.reshape(frame_small.shape)

    return segmented

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    segmented = segment_frame_kmeans_torch(frame, k=3, max_iter=10)

    # Show original and segmented side-by-side
    combined = np.hstack((cv2.resize(frame, (320, 240)), segmented))
    cv2.imshow('Original (Left) | Segmented (Right)', combined)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()