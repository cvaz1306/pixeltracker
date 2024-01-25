import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

# Open the video file
cap = cv2.VideoCapture('IMG_0223.mp4')

# Create Tkinter main window
root = tk.Tk()
root.title("OpenCV + Tkinter")

# Initialize variables
num_frames_back = 2  # Number of frames to go back for image averaging
frame_buffer = [None] * (num_frames_back + 1)

# Create four sliders
slider1 = ttk.Scale(root, from_=0, to=300, length=200, orient="horizontal")
slider2 = ttk.Scale(root, from_=0, to=300, length=200, orient="horizontal")
slider3 = ttk.Scale(root, from_=0, to=300, length=200, orient="horizontal")
slider4 = ttk.Scale(root, from_=0, to=300, length=200, orient="horizontal")
slider1.set(300)

# Create labels to display video streams
combined_label = ttk.Label(root)
combined_label.pack()

# Function to filter contours based on shape
def is_hexagonal_contour(approx):
    return len(approx) == 6 and cv2.isContourConvex(approx)

# Start the continuous frame update
def update_frame():
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use Canny edge detector
        edges = cv2.Canny(gray_frame, int(slider1.get()), int(slider2.get()))

        # Find contours in the edge-detected frame
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on shape (hexagon with rounded corners)
        hexagonal_contours = [contour for contour in contours if is_hexagonal_contour(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True))]

        if hexagonal_contours:
            # Find the largest hexagonal contour based on area
            largest_hexagonal_contour = max(hexagonal_contours, key=cv2.contourArea)

            # Draw contours on the edge-detected frame
            edges_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(edges_frame, [largest_hexagonal_contour], -1, (0, 255, 0), 2)

            # Draw contours on the original frame
            frame_with_contours = frame.copy()
            cv2.drawContours(frame_with_contours, [largest_hexagonal_contour], -1, (0, 255, 0), 2)

            # Extract bounding rectangle parameters
            (x, y, w, h) = cv2.boundingRect(largest_hexagonal_contour)

            # Locate the object and draw a green circle and rectangle around it
            center = (int(x + w / 2), int(y + h / 2))
            radius = int(max(w, h) / 2)

            # Draw a green circle
            cv2.circle(frame_with_contours, center, radius, (0, 255, 0), 2)

            # Draw a rectangle below the circle
            rect_height = int(h * 0.5)
            cv2.rectangle(frame_with_contours, (x, y + h), (x + w, y + h + rect_height), (0, 255, 0), 2)

            # Concatenate the two frames side by side
            combined_frame = np.concatenate((edges_frame, frame_with_contours), axis=1)

            # Convert the combined frame to RGB for displaying in Tkinter
            rgb_combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

            # Convert the combined frame to a Tkinter-compatible image
            combined_img = Image.fromarray(rgb_combined_frame)
            combined_imgtk = ImageTk.PhotoImage(image=combined_img)

            # Update the Tkinter label with the new combined image
            combined_label.imgtk = combined_imgtk
            combined_label.config(image=combined_imgtk)

        # Schedule the function to run again after a short delay
        root.after(10, update_frame)

# Arrange sliders using the grid layout manager
slider1.pack(pady=10)
slider2.pack(pady=10)
slider3.pack(pady=10)
slider4.pack(pady=10)

# Start the continuous frame update
update_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the video capture object when the Tkinter window is closed
cap.release()
