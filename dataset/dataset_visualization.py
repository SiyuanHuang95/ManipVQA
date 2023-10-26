import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_bounding_box(image_path, bounding_box):
    image = cv2.imread(image_path)

    x1, y1, x2, y2 = map(float, bounding_box)

    color = (0, 255, 0)
    thickness = 2 
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image with Horizontal Bounding Box")
    plt.axis('off')
    plt.show()
    
def visualize_minimum_rotated_bounding_box(mask_path, box_points_center_size_angle):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    x1, y1, x2, y2, x3, y3, x4, y4, center, size, angle = box_points_center_size_angle

    visualization_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(visualization_image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)

    # Display the visualization
    plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
    plt.title("Image with Rotated Bounding Box")
    plt.axis('off')
    plt.show()