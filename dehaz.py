import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('Foog.mp4')

# Get the video's frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a VideoWriter object to write the dehazed video
out = cv2.VideoWriter('dehazed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to YUV color space
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Stretch the histogram
    frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])

    # Convert the frame back to RGB color space
    frame_rgb = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

    # Apply a bilateral filter to reduce noise
    frame_filtered = cv2.bilateralFilter(frame_rgb, 9, 75, 75)

    # Write the dehazed frame to the new video file
    out.write(frame_filtered)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
