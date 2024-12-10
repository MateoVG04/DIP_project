import numpy as np
import cv2

def load_video_eager():
    # Eager loaded data file (all in memory before continuing)
    return np.load("Data/Ballenwerper_sync_380fps_006.npy")

def load_video_lazy():
    # Lazy loading means the data is only loaded into memory when it is needed (frame-by-frame)
    return np.lib.format.open_memmap('Data/Ballenwerper_sync_380fps_006.npy', mode='r+')

def get_screen_resolution():
    screen = cv2.namedWindow("Temp", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Temp",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    screen_width = cv2.getWindowImageRect("Temp")[2]
    screen_height = cv2.getWindowImageRect("Temp")[3]
    cv2.destroyWindow("Temp")
    return screen_width,screen_height


video_data = load_video_lazy()

screen_width, screen_height = get_screen_resolution()

# Assuming the shape is (num_frames, height, width, channels)
for frame in video_data:
    # Display the frame
    resizedFrame = cv2.resize(frame,(screen_width,screen_height),interpolation=cv2.INTER_AREA)
    cv2.imshow('Video', resizedFrame)
    oldframe = resizedFrame

    # Wait for 25ms before moving to the next frame (40 FPS)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed.")
        break

# Release the display window
cv2.destroyAllWindows()