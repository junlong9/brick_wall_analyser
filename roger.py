import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


class Brick:
    def __init__(self, coord1, coord2, coord3, coord4, image):
        coords = [coord1, coord2, coord3, coord4]

        # Separate coordinates into x and y lists
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        # Find the extreme coordinates
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        self.top_left = (min_x,min_y)
        self.top_right = (max_x, min_y)
        self.bottom_left = (min_x, max_y)
        self.bottom_right = (max_x, max_y)

        self.image = image

    def calculate_brightness(self):
        # Extract the region defined by the brick's coordinates
        roi = self.image[self.top_left[1]:self.bottom_left[1], self.top_left[0]:self.top_right[0]]

        # Check if the ROI is empty
        if roi.size == 0:
            raise ValueError("ROI is empty. Check brick coordinates.")

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calculate the average brightness
        average_brightness = np.mean(gray_roi)

        return average_brightness






def process_image(image_path):
    """
    Process an image to detect and return the corner coordinates of bricks.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    list: List of corner coordinates for each detected brick in the image.
    """
    # Load the image from the given path
    image = cv2.imread(image_path)

    # Convert the image to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blur = cv2.GaussianBlur(gray, (55, 55), 0)

    # Apply Otsu's thresholding to create a binary image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Define a kernel for morphological operations to remove small noise
    kernel = np.ones((30, 30), np.uint8)

    # Perform morphological opening to clean up the binary image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours of the objects in the cleaned binary image
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Extract corner coordinates of each detected brick
    brick_coords = []
    for contour in contours:
        # Get the minimum area bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Convert the box points to integers

        # Convert the box points to a list of (x, y) tuples with simple integers
        box_points = [(int(point[0]), int(point[1])) for point in box]
        brick_coords.append(box_points)

    return brick_coords




# Read the image once
image = cv2.imread("jonah.jpg")

# Get brick coordinates from the image
all_boxes = process_image("jonah.jpg")

# Create Brick objects
bricks = [Brick(*box, image=image) for box in all_boxes]

# Print top-left coordinates and brightness of each brick
for brick in bricks:
    try:
        print(f"Brightness: {brick.calculate_brightness()}")
    except ValueError as e:
        print(e)
