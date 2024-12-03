import numpy as np
import cv2

def load_video_eager():
    # Eager loaded data file (all in memory before continuing)
    return np.load("Data/Ballenwerper_sync_380fps_006.npy")

def load_video_lazy():
    # Lazy loading means the data is only loaded into memory when it is needed (frame-by-frame)
    return np.lib.format.open_memmap('Data/Ballenwerper_sync_380fps_006.npy', mode='r+')

video_data = load_video_lazy()


# Assuming the shape is (num_frames, height, width, channels)
for frame in video_data:
    # Display the frame
    cv2.imshow('Video', frame)

    # Wait for 25ms before moving to the next frame (40 FPS)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed.")
        break

# Release the display window
cv2.destroyAllWindows()