import cv2
import numpy as np


def process_image(image_path):
    """
    Process an image to detect and highlight individual bricks based on contours.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    None
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert the image to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blur = cv2.GaussianBlur(gray, (55, 55), 0)

    # Apply Otsu's thresholding to create a binary image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Define a kernel for morphological operations to remove small noise
    kernel = np.ones((40, 40), np.uint8)

    # Perform morphological opening to clean up the binary image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours of the objects in the cleaned binary image
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    image_number = 1
    # Loop over each contour found
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around each detected object (brick)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        # # Extract the region of interest (ROI) corresponding to each brick
        # ROI = original[y:y + h, x:x + w]

        # # Create image of each brick
        # cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        # image_number += 1

    # Display the original image with detected bricks highlighted
    cv2.imshow('Original Image', image)

    # Display intermediate results for debugging or visualization
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Blurred', blur)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Opening', opening)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
process_image('Jonah/brick.jpg')  # Fix file direction if needed
