# Closet Canvas – K-Means Segmentation Prototype

This repository contains an early computer vision prototype for the Closet Canvas project.  
The goal of this prototype is to explore real-time image segmentation using K-Means clustering
as a first step toward separating foreground regions from a live webcam feed.

## What This Prototype Does

- Captures live video from a webcam using OpenCV  
- Applies K-Means clustering to group pixels based on color similarity  
- Displays the original frame and the segmented output side-by-side in real time  

This implementation focuses on understanding image preprocessing and segmentation behavior
rather than producing garment-level segmentation.

## Demo Output

The image below shows the output of the script when running on a live webcam feed.

- **Left:** Original webcam frame  
- **Right:** Color-based segmented frame using K-Means clustering

  
