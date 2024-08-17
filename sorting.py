import cv2
import numpy as np

def sort_and_display_bricks(image_path):
    """
    Detects bricks in the image, sorts them from darkest to lightest, and displays them in order.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    None
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    original = image.copy()

    # Convert the image to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blur = cv2.GaussianBlur(gray, (25, 25), 0)

    # Apply adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Define a kernel for morphological operations to remove small noise
    kernel = np.ones((20, 20), np.uint8)

    # Perform morphological opening to clean up the binary image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Check if the image should be inverted
    if np.mean(opening) < 150:  # If the average intensity is less than 150, invert
        opening = cv2.bitwise_not(opening)

    # Find contours of the objects in the cleaned binary image
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Filter contours to remove small ones that are unlikely to be bricks
    brick_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    # Initialize a list to hold the bricks and their intensities
    bricks = []

    # Loop over each detected brick contour
    for contour in brick_contours:
        # Get the bounding box for the brick
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the brick region from the grayscale image
        brick_gray = gray[y:y+h, x:x+w]
        brick_color = original[y:y+h, x:x+w]

        # Calculate the average intensity of the brick
        avg_intensity = np.mean(brick_gray)

        # Store the brick and its intensity in the list
        bricks.append((avg_intensity, brick_color))

    # Sort the bricks by intensity (darkest to lightest)
    bricks.sort(key=lambda x: x[0])

    # Display the bricks in order from darkest to lightest
    for i, (intensity, brick) in enumerate(bricks):
        cv2.imshow(f'Brick {i+1} (Intensity: {intensity:.2f})', brick)
        cv2.waitKey(0)  # Wait for a key press to show the next brick

    cv2.destroyAllWindows()

# Example usage:
sort_and_display_bricks('David/bricks.jpg')
