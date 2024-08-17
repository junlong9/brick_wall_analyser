import cv2
import numpy as np

def process_image(image_path):
    """
    Process an image to detect and highlight individual bricks based on contours.
    If the image has more black areas than white, it inverts the image.

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
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Apply Otsu's thresholding to create a binary image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Define a kernel for morphological operations to remove small noise
    kernel = np.ones((35, 35), np.uint8)

    # Perform morphological opening to clean up the binary image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Check if the image should be inverted
    if np.mean(opening) < 150:  # If the average intensity is less than 127, invert
        opening = cv2.bitwise_not(opening)

    # Find contours of the objects in the cleaned binary image
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    image_number = 1
    # Loop over each contour found
    for contour in contours:
        # Get the minimum area bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Convert the box points to integers

        # Convert the box points to a list of (x, y) tuples with simple integers
        box_points = [(int(point[0]), int(point[1])) for point in box]
        print(f"Rectangle {image_number} points: {box_points}")

        # Draw the rectangle using the box points
        cv2.drawContours(image, [box], 0, (36, 255, 12), 2)

        # Draw circles at each point in the rectangle
        for point in box_points:
            cv2.circle(image, point, 10, (0, 0, 255), -1)  # Red circle with radius 10

        image_number += 1

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
process_image('Jonah/brick.jpg')
