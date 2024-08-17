import cv2
import numpy as np

count = 0  # Initialize the count variable

def process_image(image_path):
    global count  # Declare count as global to modify it inside the function
    
    """
    Process an image to detect, highlight, and count individual bricks based on contours.
    If the image has more black areas than white, it inverts the image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    int: Number of detected bricks.
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return 0

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
    if np.mean(opening) < 150:  # If the average intensity is less than 150, invert
        opening = cv2.bitwise_not(opening)

    # Find contours of the objects in the cleaned binary image
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Filter out small contours that are unlikely to be bricks
    brick_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this value based on the expected size of bricks
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is approximately rectangular (4 sides)
            if len(approx) == 4:
                brick_contours.append(contour)

    # Count the number of bricks
    num_bricks = len(brick_contours)

    image_number = 1
    
    # Loop over each contour found
    for contour in brick_contours:
        count += 1  # Increment the count for each detected brick
        
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

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Number of bricks detected: {num_bricks}")
    return num_bricks

# Example usage:
num_bricks = process_image('David/brick.jpg')
print(f"Total bricks detected: {count}")
