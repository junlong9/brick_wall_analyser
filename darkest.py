import cv2
import numpy as np

def find_and_save_darkest_brick(image_path, output_path='darkest_brick.jpg'):
    """
    Detects bricks in the image and saves the darkest brick to a file.

    Parameters:
    image_path (str): Path to the image file.
    output_path (str): Path where the darkest brick image will be saved.

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

    # Initialize variables to track the darkest brick
    darkest_brick_intensity = float('inf')
    darkest_brick_image = None

    # Loop over each detected brick contour
    for contour in brick_contours:
        # Get the bounding box for the brick
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the brick region from the grayscale image
        brick_gray = gray[y:y+h, x:x+w]

        # Calculate the average intensity of the brick
        avg_intensity = np.mean(brick_gray)

        # Check if this brick is the darkest one so far
        if avg_intensity < darkest_brick_intensity:
            darkest_brick_intensity = avg_intensity
            darkest_brick_image = original[y:y+h, x:x+w]

    # Save the darkest brick image if one was found
    if darkest_brick_image is not None:
        cv2.imwrite(output_path, darkest_brick_image)
        print(f"The darkest brick was saved to {output_path}")
        # Optionally, display the darkest brick
        cv2.imshow('Darkest Brick', darkest_brick_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No bricks were detected.")

# Example usage:
find_and_save_darkest_brick('David/bricks.jpg')
