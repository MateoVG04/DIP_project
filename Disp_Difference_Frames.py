import numpy as np
import cv2

def load_video_lazy():
    # Lazy loading of the video data
    return np.lib.format.open_memmap('Data/Ballenwerper_sync_380fps_006.npy', mode='r+')

video_data = load_video_lazy()

Previousframe = None  # Initialize Previousframe as None
# Assuming the shape is (num_frames, height, width, channels)
for frame in video_data:
    # Apply Gaussian blur to reduce noise
    blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    if Previousframe is not None:
        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(blur_frame, Previousframe)

        # Display the frame difference
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 800, 600)
        cv2.imshow('Video', 255 - frame_diff)


    # Update the Previousframe
    Previousframe = blur_frame

    # Wait for 25ms before moving to the next frame (approx. 40 FPS)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# Release the display window
cv2.destroyAllWindows()
