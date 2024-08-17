import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, Button, Label, filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class Brick:
    def __init__(self, coord1, coord2, coord3, coord4, image):
        coords = [coord1, coord2, coord3, coord4]

        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        self.top_left = (min_x, min_y)
        self.top_right = (max_x, min_y)
        self.bottom_left = (min_x, max_y)
        self.bottom_right = (max_x, max_y)

        self.image = image

        if not self.is_valid():
            raise ValueError("Invalid brick coordinates.")

    def is_valid(self):
        height, width = self.image.shape[:2]
        x_coords = [self.top_left[0], self.top_right[0], self.bottom_left[0], self.bottom_right[0]]
        y_coords = [self.top_left[1], self.top_right[1], self.bottom_left[1], self.bottom_right[1]]

        if (min(x_coords) < 0 or max(x_coords) >= width or
                min(y_coords) < 0 or max(y_coords) >= height):
            return False

        if not (self.top_left[0] < self.top_right[0] and
                self.top_left[1] < self.bottom_left[1]):
            return False

        return True

    def calculate_brightness(self):
        roi = self.image[self.top_left[1]:self.bottom_left[1], self.top_left[0]:self.top_right[0]]
        if roi.size == 0:
            raise ValueError("ROI is empty. Check brick coordinates.")
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        average_brightness = np.mean(gray_roi)
        return average_brightness

    def calculate_average_color(self):
        roi = self.image[self.top_left[1]:self.bottom_left[1], self.top_left[0]:self.top_right[0]]
        if roi.size == 0:
            raise ValueError("ROI is empty. Check brick coordinates.")
        average_color = np.mean(roi, axis=(0, 1))
        return tuple(map(int, average_color))


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (55, 55), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((30, 30), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    brick_coords = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        box_points = [(int(point[0]), int(point[1])) for point in box]
        brick_coords.append(box_points)

    return brick_coords


def plot_rgb_colors(bricks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    r_values = []
    g_values = []
    b_values = []

    for brick in bricks:
        try:
            avg_color = brick.calculate_average_color()
            b_values.append(avg_color[2])
            g_values.append(avg_color[1])
            r_values.append(avg_color[0])
        except ValueError as e:
            print(e)

    rgb_values = np.array([r_values, g_values, b_values]).T / 255.0
    scatter = ax.scatter(r_values, g_values, b_values, c=rgb_values, marker='o')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    plt.show()


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class BrickAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brick Analyzer")
        self.root.state('zoomed')  # Maximize window
        self.image = None

        self.label = Label(root, text="Upload an image to analyze bricks.", font=('Arial', 16))
        self.label.pack(pady=20)

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image, font=('Arial', 14))
        self.upload_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image.")
                return

            all_boxes = process_image(self.image)
            bricks = []
            for box in all_boxes:
                try:
                    brick = Brick(*box, image=self.image)
                    bricks.append(brick)
                except ValueError as e:
                    print(e)

            darkest_brick = None
            lightest_brick = None
            min_brightness = float('inf')
            max_brightness = float('-inf')

            for brick in bricks:
                try:
                    brightness = brick.calculate_brightness()
                    if brightness < min_brightness:
                        min_brightness = brightness
                        darkest_brick = brick
                    if brightness > max_brightness:
                        max_brightness = brightness
                        lightest_brick = brick
                except ValueError as e:
                    print(e)

            height, width = self.image.shape[:2]
            outline_thickness = int(0.01 * min(width, height))
            annotated_image = self.image.copy()

            def draw_rectangle(image, brick, color):
                cv2.polylines(image,
                              [np.array([brick.top_left, brick.top_right, brick.bottom_right, brick.bottom_left],
                                        dtype=np.int32)],
                              isClosed=True, color=color, thickness=outline_thickness)

            if darkest_brick:
                draw_rectangle(annotated_image, darkest_brick, (0, 0, 255))  # Red for darkest

            if lightest_brick:
                draw_rectangle(annotated_image, lightest_brick, (0, 255, 0))  # Green for lightest

            self.show_image(annotated_image)
            plot_rgb_colors(bricks)

    def show_image(self, image):
        # Resize the image to fit the screen/window
        screen_res = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height)

        new_dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        # Show the resized image
        cv2.imshow("Image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    app = BrickAnalyzerApp(root)
    root.mainloop()
