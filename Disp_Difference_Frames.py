import numpy as np
import cv2

def difference(video_data):
    Previousframe = None  # Initialize Previousframe as None
    # Assuming the shape is (num_frames, height, width, channels)
    for frame in video_data:
        # Apply Gaussian blur to reduce noise
        blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if Previousframe is not None:
            # Compute absolute difference between frames
            frame_diff = cv2.absdiff(blur_frame, Previousframe)

            # Display the frame difference
            cv2.namedWindow('tEST', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('tEST', 800, 600)
            cv2.imshow('tEST', 255 - frame_diff)


        # Update the Previousframe
        Previousframe = blur_frame

        # Wait for 25ms before moving to the next frame (approx. 40 FPS)
        if cv2.getWindowProperty('tEST', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

    # Release the display window
    cv2.destroyAllWindows()
