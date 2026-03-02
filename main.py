import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("cam.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)
cv2.imwrite("edges.jpg", edges)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=30,
    minRadius=50,
    maxRadius=200
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    outer_circle = circles[0][0]

    center_x, center_y, outer_radius = outer_circle

    # Draw detected circle
    output = image.copy()
    cv2.circle(output, (center_x, center_y), outer_radius, (0, 255, 0), 2)
    cv2.imwrite("detected_circles.jpg", output)

    # Assume inner radius = outer - 10 pixels
    inner_radius = outer_radius - 10

    angles = np.arange(0, 360, 5)
    thickness_values = []

    for angle in angles:
        thickness = outer_radius - inner_radius
        thickness_values.append(thickness)

    # Save CSV
    df = pd.DataFrame({
        "Angle": angles,
        "Thickness": thickness_values
    })

    df.to_csv("thickness_data.csv", index=False)

    # Plot graph
    plt.plot(angles, thickness_values)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Thickness (pixels)")
    plt.title("Cam Ring Thickness vs Angle")
    plt.savefig("thickness_graph.png")
    plt.show()

    print("Thickness is uniform:", thickness_values[0], "pixels")

else:
    print("No circle detected")
