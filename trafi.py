import cv2
import pytesseract

def detect_number_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade model for number plate detection
    plate_cascade = cv2.CascadeClassifier('path/to/haarcascade_plate.xml')

    # Detect number plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected plates
    for (x, y, w, h) in plates:
        # Extract the region of interest (ROI) containing the number plate
        plate_roi = gray[y:y+h, x:x+w]

        # Perform OCR on the ROI using pytesseract
        number_plate_text = pytesseract.image_to_string(plate_roi, config='--psm 7')

        # Print the detected number plate
        print('Detected Number Plate:', number_plate_text.strip())

    # Return the first detected number plate (if any)
    if len(plates) > 0:
        (x, y, w, h) = plates[0]
        plate_roi = gray[y:y+h, x:x+w]
        number_plate_text = pytesseract.image_to_string(plate_roi, config='--psm 7')
        return number_plate_text.strip()
    else:
        return None

# Provide the path to your image
image_path = '/path/to/your/image.jpg'

# Call the function to detect the number plate
number_plate = detect_number_plate(image_path)

# Print the detected number plate
if number_plate is not None:
    print('Detected Number Plate:', number_plate)
else:
    print('No number plate detected.')
