import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


class Brick:
    def __init__(self, top_left, top_right, bottom_left, bottom_right, image):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.image = image

    def get_midpoint(self):
        x_coords = [self.top_left[0], self.top_right[0], self.bottom_left[0], self.bottom_right[0]]
        y_coords = [self.top_left[1], self.top_right[1], self.bottom_left[1], self.bottom_right[1]]
        midpoint = (sum(x_coords) / 4, sum(y_coords) / 4)
        return midpoint

    def average_color(self):
        """
        Calculate the average color of the brick region using all pixels within the brick's bounding box.

        Returns:
        tuple: Average color (R, G, B)
        """
        pass


class Wall:
    def __init__(self, bricks, image):
        self.midpoints = []
        self.average_colors = []
        self.bricks = bricks
        self.image = image

    def calculate_brick_properties(self):
        self.midpoints = [brick.get_midpoint() for brick in self.bricks]
        self.average_colors = [brick.average_color() for brick in self.bricks]

        # Debug: Print average colors
        for i, color in enumerate(self.average_colors):
            print(f"Brick {i + 1} Average Color: {color}")

    def get_darkest_brick_index(self):
        brightness = [sum(color) / 3 for color in self.average_colors]
        print(f"Brightness values: {brightness}")  # Debug: Print brightness values
        return np.argmin(brightness)

    def get_lightest_brick_index(self):
        brightness = [sum(color) / 3 for color in self.average_colors]
        print(f"Brightness values: {brightness}")  # Debug: Print brightness values
        return np.argmax(brightness)

    def outline_bricks(self):
        """
        Outline the darkest and lightest bricks with different colors.
        Display the image with outlines.
        """
        # Calculate the maximum dimension of the image
        height, width = self.image.shape[:2]
        max_dim = max(width, height)

        # Set the outline width to 1% of the maximum dimension
        outline_width = int(0.01 * max_dim)

        # Get the darkest and lightest bricks
        darkest_index = self.get_darkest_brick_index()
        lightest_index = self.get_lightest_brick_index()

        darkest_brick = self.bricks[darkest_index]
        lightest_brick = self.bricks[lightest_index]

        # Debug: Print coordinates
        print(f"Darkest Brick: Top Left: {darkest_brick.top_left}, Bottom Right: {darkest_brick.bottom_right}")
        print(f"Lightest Brick: Top Left: {lightest_brick.top_left}, Bottom Right: {lightest_brick.bottom_right}")

        # Outline the darkest brick
        cv2.rectangle(self.image, darkest_brick.top_left, darkest_brick.bottom_right, (0, 255, 0), outline_width)

        # Outline the lightest brick
        cv2.rectangle(self.image, lightest_brick.top_left, lightest_brick.bottom_right, (255, 0, 0), outline_width)

        # Display the image with outlines
        cv2.namedWindow('Wall with Darkest and Lightest Bricks Outlined', cv2.WINDOW_NORMAL)
        cv2.imshow('Wall with Darkest and Lightest Bricks Outlined', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_image(image_path):
    """
    Process an image to detect and highlight individual bricks based on contours,
    and annotate each brick with its bounding box.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    List[Brick]: List of Brick objects.
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert the image to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blur = cv2.GaussianBlur(gray, (55, 55), 0)

    # Apply Otsu's thresholding to create a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a kernel for morphological operations to remove small noise
    kernel = np.ones((40, 40), np.uint8)

    # Perform morphological opening to clean up the binary image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours of the objects in the cleaned binary image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    brick_coordinates = []

    # Loop over each contour found
    for contour in contours:
        # Get the minimum area bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Convert the box points to integers

        # Compute the center of the rectangle
        center = np.mean(box, axis=0)

        # Sort points based on their relative position to the center
        box_sorted = sorted(box,
                            key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0]) + 2 * np.pi) % (2 * np.pi))

        # Ensure correct ordering: top-left, top-right, bottom-right, bottom-left
        top_left, top_right, bottom_right, bottom_left = box_sorted

        # Convert the box points to a numpy array of points
        box_array = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

        # Convert the box points to a list of (x, y) tuples
        box_points = [tuple(point) for point in [top_left, top_right, bottom_left, bottom_right]]
        brick_coordinates.append(box_points)

        # Draw the rectangle using the box points
        cv2.drawContours(image, [box_array], 0, (36, 255, 12), 2)

        # Draw circles at each point in the rectangle
        for point in box_points:
            cv2.circle(image, point, 10, (0, 0, 255), -1)  # Red circle with radius 10

        # Annotate the brick with its index
        x, y = box_points[0]  # Use the top-left corner for annotation
        cv2.putText(image, f'Brick {len(brick_coordinates)}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the image with all bricks annotated
    cv2.namedWindow('Annotated Brick Wall', cv2.WINDOW_NORMAL)
    cv2.imshow('Annotated Brick Wall', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert coordinates to Brick objects
    bricks = [Brick(*coords, original) for coords in brick_coordinates]

    return bricks


def main():
    # Hide the root window
    Tk().withdraw()

    # Ask the user to select an image file
    filename = askopenfilename()

    # Process the image and detect bricks
    bricks = process_image(filename)

    # Create a Wall object with the detected bricks
    wall = Wall(bricks, cv2.imread(filename))

    # Calculate properties for the bricks
    wall.calculate_brick_properties()

    # Outline the darkest and lightest bricks
    wall.outline_bricks()


if __name__ == '__main__':
    main()
