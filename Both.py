import cv2
import numpy as np
import pytesseract

def dark_channel(image, window_size=15):
    """Calculate the dark channel of an image."""
    min_channel = np.min(image, axis=2)
    return cv2.erode(min_channel, np.ones((window_size, window_size)))

def estimate_atmosphere(image, dark_channel, percentile=0.001):
    """Estimate the atmosphere light of the image."""
    flat_dark_channel = dark_channel.flatten()
    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]
    num_pixels_to_keep = int(num_pixels * percentile)
    indices = np.argpartition(flat_dark_channel, -num_pixels_to_keep)[-num_pixels_to_keep:]
    atmosphere = np.max(flat_image[indices], axis=0)
    return atmosphere

def dehaze(image, tmin=0.1, omega=0.95, window_size=15):
    """Dehaze the input image using the Dark Channel Prior algorithm."""
    if image is None:
        return None

    image = image.astype(np.float64) / 255.0
    dark_ch = dark_channel(image, window_size)
    atmosphere = estimate_atmosphere(image, dark_ch)
    transmission = 1 - omega * dark_ch
    transmission = np.maximum(transmission, tmin)
    dehazed = np.zeros_like(image)
    for channel in range(3):
        dehazed[:, :, channel] = (image[:, :, channel] - atmosphere[channel]) / transmission + atmosphere[channel]
    dehazed = np.clip(dehazed, 0, 1)
    dehazed = (dehazed * 255).astype(np.uint8)
    return dehazed

# Load the pre-trained cascade classifier for license plate detection
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

plate_cascade = cv2.CascadeClassifier("D:\\Chrome\\haarcascade_russian_plate_number.xml")

# Specify the path to the video file
video_path = r"C:\\Users\\hp\\Desktop\\Dehaze\\Foog.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

while True:
    # Read the next frame from the video file
    ret, frame = cap.read()

    # If the frame is not empty, dehaze it
    if ret:
        dehazed_frame = dehaze(frame)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2GRAY)

        # Apply some preprocessing to improve OCR accuracy
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform number plate detection
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop over detected plates and perform OCR
        for (x, y, w, h) in plates:
            plate_roi = gray[y:y + h, x:x + w]

            # Apply OCR using pytesseract with a specific config
            config = '--oem 3 --psm 11'
            plate_text = pytesseract.image_to_string(plate_roi, config=config)

            # You can add additional validation to filter out false positives
            if len(plate_text) >= 6:
                print(f"Detected Plate: {plate_text}")

                # Draw a rectangle around the detected plate
                cv2.rectangle(dehazed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the dehazed frame
        cv2.imshow('Dehazed Video', dehazed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the frame is empty, break the loop
    else:
        print("Can't receive frame (stream end?). Exiting ...")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
