import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('IMG_0222.mp4')

# Initialize variables
num_frames_back = 2  # Number of frames to go back for image averaging
frame_buffer = [None] * (num_frames_back + 1)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Apply Gaussian blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Save the blurred frame in the buffer
    frame_buffer.pop(0)
    frame_buffer.append(blurred_frame.copy())

    # Convert the blurred frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detector
    
    edges = cv2.Canny(gray_frame, 50, 275)

    # Find contours in the edge-detected frame
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    frame_with_contours = frame.copy()
    cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)

    # Add motion blur to the frame
    kernel_size = 15
    motion_blur_kernel = np.zeros((kernel_size, kernel_size))
    motion_blur_kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    motion_blur_kernel /= kernel_size
    frame_with_contours = cv2.filter2D(frame_with_contours, -1, motion_blur_kernel)

    # Average the frame with the previous frames in the buffer
    if all(frame is not None for frame in frame_buffer):
        for i in range(num_frames_back):
            weight = 1 / (i + 1)
            frame_with_contours = cv2.addWeighted(frame_with_contours, 1 - weight, frame_buffer[i], weight, 0)

    # Locate the object and draw a green circle and rectangle around it
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        center = (int(x + w/2), int(y + h/2))
        radius = int(max(w, h) / 2)
        
        # Draw a green circle
        cv2.circle(frame_with_contours, center, radius, (0, 255, 0), 2)
        
        # Draw a rectangle below the circle
        rect_height = int(h * 0.5)
        cv2.rectangle(frame_with_contours, (x, y + h), (x + w, y + h + rect_height), (0, 255, 0), 2)

    # Display the frame with contours, circles, rectangles, motion blur, and image averaging
    cv2.imshow('Video with Contours, Circles, Rectangles, Blur, and Image Averaging', edges)#frame_with_contours)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
