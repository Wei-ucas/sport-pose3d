import numpy as np
import cv2

court_width = 28.1  # meters
court_height = 15.1  # meters

court_ground_corners = np.asarray([
    [0 - 0.5, 0 - 0.5, 0],
    [0 - 0.5, 15.1 + 0.5, 0],
    [28.1 + 0.5, 15.1 + 0.5, 0],
    [28.1 + 0.5, 0 + 0.5, 0]
])

court_top_corners = np.asarray([
    [0, 0, 2.5],
    [0, 15.1, 2.5],
    [28.1, 15.1, 2.5],
    [28.1, 0, 2.5]
])

image_raw = cv2.imread("assets/tactical-board.png")  # 1024 x 580

# def get_court_background_image():
court_left = 51
court_right = 972
court_top = 42
court_bottom = 537

width_rate = (court_right - court_left + 1) / court_width
height_rate = (court_bottom - court_top + 1) / court_height


def project_point_to_image(point: np.ndarray):
    x = int(point[0] * width_rate + court_left)
    y = int(court_bottom - point[1] * height_rate)
    return x, y
